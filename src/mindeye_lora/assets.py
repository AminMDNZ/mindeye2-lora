"""Acquire only the bytes we actually need from `pscotti/mindeyev2`.

The full HuggingFace dataset is ~215 GB, which is a non-starter on Colab (and on a
free 15 GB Drive). Two tricks keep the footprint at roughly 3 GB:

1. **Remote HDF5 row slicing.** `coco_images_224_float16.hdf5` is 22 GB but a 1-session
   fine-tuning experiment touches fewer than 2,000 distinct images. h5py can read from
   any file-like object, so we open the file over HTTPS with fsspec and pull only the
   rows we need (a few hundred MB of range requests), then write a local subset. The
   same trick reduces the 1.89 GB betas file to the ~4,000 trials we use.

2. **Checkpoint slimming.** The published `last.pth` files are ~2.9 GB because they
   carry DeepSpeed optimizer state. We strip that once and keep a ~1 GB init file.

Everything is written through a manifest so re-running after a Colab reset is a no-op.
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import human_bytes, log, read_json, write_json

REPO_ID = "pscotti/mindeyev2"
REPO_TYPE = "dataset"
RESOLVE = "https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/"

# --- file names on the hub (verified against the repo listing) -------------------------
BETAS_TPL = "betas_all_subj{subj:02d}_fp32_renorm.hdf5"
COCO_IMAGES = "coco_images_224_float16.hdf5"
WDS_TRAIN_TPL = "wds/subj{subj:02d}/train/{shard}.tar"
WDS_TEST_CANDIDATES = ["wds/subj{subj:02d}/new_test/0.tar", "wds/subj{subj:02d}/test/0.tar"]
UNCLIP_CKPT = "unclip6_epoch0_step110000.ckpt"          # 18 GB, only for image decoding
PRETRAIN_CKPTS = {
    # Colab-friendly shared-subject pretrain: hidden_dim=1024, low-level module disabled.
    "multisubject_1024": "train_logs/multisubject_subj01_1024hid_nolow_300ep/last.pth",
    # Paper-scale shared-subject pretrain: hidden_dim=4096, needs a big GPU to fine-tune.
    "multisubject_4096": "train_logs/final_multisubject_subj{subj:02d}/last.pth",
    # Published single-subject 1-session fine-tune — used as an evaluation sanity check.
    "reference_1sess": "train_logs/final_subj{subj:02d}_pretrained_1sess_24bs/last.pth",
}

# behav column indices, from the upstream README
BEHAV_COCO_IDX = 0
BEHAV_SUBJECT = 1
BEHAV_SESSION = 2
BEHAV_GLOBAL_TRIAL = 5
BEHAV_IS_SHARED = 16


# --------------------------------------------------------------------------------------
# generic download
# --------------------------------------------------------------------------------------
def hf_download(filename: str, local_dir: Path, force: bool = False) -> Path:
    """Download one file from the mindeyev2 dataset repo, resuming if interrupted."""
    from huggingface_hub import hf_hub_download

    target = Path(local_dir) / filename
    if target.exists() and not force:
        log.info("cached: %s (%s)", filename, human_bytes(target.stat().st_size))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s ...", filename)
    out = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
    )
    return Path(out)


def repo_files() -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(REPO_ID, repo_type=REPO_TYPE)


def resolve_test_shard(subj: int, available: Sequence[str] | None = None) -> str:
    available = available if available is not None else repo_files()
    for tpl in WDS_TEST_CANDIDATES:
        name = tpl.format(subj=subj)
        if name in available:
            return name
    raise FileNotFoundError(f"No test webdataset shard found for subj{subj:02d}")


# --------------------------------------------------------------------------------------
# remote HDF5 access
# --------------------------------------------------------------------------------------
@dataclass
class RemoteH5:
    """Read rows of a hub-hosted HDF5 without downloading the whole thing."""

    filename: str
    block_size: int = 8 * 1024 * 1024

    def __enter__(self):
        import fsspec
        import h5py

        url = RESOLVE + self.filename
        self._fs_file = fsspec.open(url, "rb", block_size=self.block_size).open()
        self._h5 = h5py.File(self._fs_file, "r")
        return self

    def __exit__(self, *exc):
        try:
            self._h5.close()
        finally:
            self._fs_file.close()

    @property
    def key(self) -> str:
        import h5py

        for k in self._h5.keys():
            if isinstance(self._h5[k], h5py.Dataset):
                return k
        raise KeyError(f"No dataset inside {self.filename}")

    def take(self, indices: Sequence[int], batch: int = 256) -> np.ndarray:
        """Gather `indices` rows (sorted internally, returned in the given order)."""
        ds = self._h5[self.key]
        order = np.argsort(indices)
        sorted_idx = np.asarray(indices, dtype=np.int64)[order]
        out = np.empty((len(indices), *ds.shape[1:]), dtype=ds.dtype)
        for start in range(0, len(sorted_idx), batch):
            chunk = sorted_idx[start : start + batch]
            out[order[start : start + batch]] = ds[chunk]
            log.info("  fetched %d/%d rows of %s", min(start + batch, len(sorted_idx)),
                     len(sorted_idx), self.filename)
        return out


def local_h5_take(path: Path, indices: Sequence[int]) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx)
        out = np.empty((len(idx), *f[key].shape[1:]), dtype=f[key].dtype)
        out[order] = f[key][idx[order]]
    return out


def gather_rows(filename: str, indices: Sequence[int], local_copy: Path | None) -> np.ndarray:
    """Prefer a local full copy if the user already has one, else stream over HTTPS."""
    if local_copy is not None and Path(local_copy).exists():
        return local_h5_take(Path(local_copy), indices)
    with RemoteH5(filename) as rh:
        return rh.take(indices)


# --------------------------------------------------------------------------------------
# webdataset behaviour tables
# --------------------------------------------------------------------------------------
def read_behav_tar(path: Path) -> np.ndarray:
    """Extract the `behav` arrays from a MindEye2 webdataset shard -> (N, 17) float32.

    Uses plain `tarfile` rather than the `webdataset` package: fewer dependencies, and
    we only need one field per sample.
    """
    rows: list[np.ndarray] = []
    with tarfile.open(path, "r") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.name.endswith("behav.npy") and "past" not in m.name
             and "future" not in m.name and "old" not in m.name),
            key=lambda m: m.name,
        )
        for m in members:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            arr = np.load(io.BytesIO(fh.read()), allow_pickle=False)
            rows.append(np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1]))
    if not rows:
        raise ValueError(f"No behav arrays inside {path}")
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------------------
# checkpoint slimming
# --------------------------------------------------------------------------------------
class _Placeholder:
    """Stand-in for a class we cannot import. Only tensors survive slimming anyway."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TolerantUnpickler:
    """Unpickle checkpoints referencing classes we do not have installed.

    The published checkpoints were saved under DeepSpeed, so the pickle names
    `deepspeed.runtime.zero.config.ZeroStageEnum` and friends. Installing DeepSpeed
    purely to reconstruct objects we discard a moment later would be absurd, so
    unknown classes become placeholders and are dropped when we keep only tensors.
    """

    def __init__(self):
        import pickle

        outer = self

        class Unpickler(pickle.Unpickler):
            def find_class(self, mod, name):
                try:
                    return super().find_class(mod, name)
                except (ModuleNotFoundError, AttributeError, ImportError):
                    outer.substituted.add(f"{mod}.{name}")
                    return type(name, (_Placeholder,), {})

        self.substituted: set[str] = set()
        self.Unpickler = Unpickler
        self.load = pickle.load
        self.Pickler = pickle.Pickler
        self.dump = pickle.dump


def slim_checkpoint(src: Path, dst: Path) -> Path:
    """Keep only fp32 model weights; drop DeepSpeed optimizer/scheduler state."""
    import torch

    if dst.exists():
        return dst
    log.info("Slimming %s (%s) ...", src.name, human_bytes(src.stat().st_size))

    tolerant = _TolerantUnpickler()
    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=False,
                          pickle_module=tolerant)
    except TypeError:  # torch too old to accept pickle_module here
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if tolerant.substituted:
        log.info("  ignored %d unavailable classes (e.g. %s)",
                 len(tolerant.substituted), sorted(tolerant.substituted)[0])

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "module", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:  # pragma: no cover
        raise TypeError(f"Unexpected checkpoint type {type(ckpt)}")

    clean = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if torch.is_tensor(v):        # placeholders and scalars are discarded here
            clean[k] = v.float()

    if not clean:
        raise RuntimeError(f"No tensors recovered from {src}; checkpoint layout changed.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": clean, "source": src.name}, dst)
    log.info("  -> %s (%s, %d tensors)", dst.name,
             human_bytes(dst.stat().st_size), len(clean))
    return dst


# --------------------------------------------------------------------------------------
# top-level asset preparation
# --------------------------------------------------------------------------------------
@dataclass
class AssetPaths:
    behav_train: Path
    behav_test: Path
    voxels: Path
    voxel_index: Path
    images: Path
    image_index: Path
    pretrain_ckpt: Path
    meta: Path


def prepare_assets(
    ws,
    subj: int = 1,
    num_sessions: int = 1,
    pretrain: str = "multisubject_1024",
    full_local_hdf5: bool = False,
    force: bool = False,
) -> AssetPaths:
    """Download / derive every artifact needed for training and evaluation."""
    assets, data = ws["assets"], ws["data"]
    meta_path = data / f"subj{subj:02d}_sess{num_sessions}_meta.json"
    paths = AssetPaths(
        behav_train=data / f"subj{subj:02d}_behav_train_{num_sessions}sess.npy",
        behav_test=data / f"subj{subj:02d}_behav_test.npy",
        voxels=data / f"subj{subj:02d}_voxels_{num_sessions}sess.npy",
        voxel_index=data / f"subj{subj:02d}_voxel_trials_{num_sessions}sess.npy",
        images=data / f"subj{subj:02d}_images_{num_sessions}sess.npy",
        image_index=data / f"subj{subj:02d}_coco_idx_{num_sessions}sess.npy",
        pretrain_ckpt=assets / f"{pretrain}_subj{subj:02d}_init.pth",
        meta=meta_path,
    )
    if meta_path.exists() and not force and all(
        p.exists() for p in [paths.behav_train, paths.behav_test, paths.voxels, paths.images,
                             paths.pretrain_ckpt]
    ):
        log.info("Assets already prepared for subj%02d / %d session(s).", subj, num_sessions)
        return paths

    files = repo_files()

    # 1. behaviour tables ------------------------------------------------------------
    train_behav = []
    for shard in range(num_sessions):
        name = WDS_TRAIN_TPL.format(subj=subj, shard=shard)
        if name not in files:
            raise FileNotFoundError(f"{name} not in repo — subj{subj:02d} has fewer sessions.")
        train_behav.append(read_behav_tar(hf_download(name, assets)))
    behav_train = np.concatenate(train_behav, 0)
    behav_test = read_behav_tar(hf_download(resolve_test_shard(subj, files), assets))
    np.save(paths.behav_train, behav_train)
    np.save(paths.behav_test, behav_test)
    log.info("behav: %d train trials, %d test trials", len(behav_train), len(behav_test))

    # 2. voxel betas ------------------------------------------------------------------
    trials = np.concatenate(
        [behav_train[:, BEHAV_GLOBAL_TRIAL], behav_test[:, BEHAV_GLOBAL_TRIAL]]
    ).astype(np.int64)
    uniq_trials = np.unique(trials)
    betas_name = BETAS_TPL.format(subj=subj)
    local_betas = hf_download(betas_name, assets) if full_local_hdf5 else None
    voxels = gather_rows(betas_name, uniq_trials, local_betas)
    np.save(paths.voxels, voxels.astype(np.float32))
    np.save(paths.voxel_index, uniq_trials)
    log.info("voxels: %s -> %s", voxels.shape, human_bytes(paths.voxels.stat().st_size))

    # 3. stimulus images ---------------------------------------------------------------
    coco = np.concatenate(
        [behav_train[:, BEHAV_COCO_IDX], behav_test[:, BEHAV_COCO_IDX]]
    ).astype(np.int64)
    uniq_coco = np.unique(coco)
    local_coco = hf_download(COCO_IMAGES, assets) if full_local_hdf5 else None
    images = gather_rows(COCO_IMAGES, uniq_coco, local_coco)
    np.save(paths.images, images)  # float16 [N, 3, 224, 224] in [0, 1]
    np.save(paths.image_index, uniq_coco)
    log.info("images: %s -> %s", images.shape, human_bytes(paths.images.stat().st_size))

    # 4. pretrained shared-subject checkpoint -------------------------------------------
    tpl = PRETRAIN_CKPTS[pretrain]
    remote = tpl.format(subj=subj)
    raw = hf_download(remote, assets)
    slim_checkpoint(raw, paths.pretrain_ckpt)
    if paths.pretrain_ckpt.exists() and raw.exists() and raw != paths.pretrain_ckpt:
        raw.unlink()  # reclaim ~2 GB; the slim file is all we ever need
        log.info("removed raw checkpoint after slimming")

    write_json(
        meta_path,
        {
            "subj": subj,
            "num_sessions": num_sessions,
            "pretrain": pretrain,
            "num_voxels": int(voxels.shape[1]),
            "n_train_trials": int(len(behav_train)),
            "n_test_trials": int(len(behav_test)),
            "n_unique_images": int(len(uniq_coco)),
            "files": {k: str(v) for k, v in paths.__dict__.items()},
        },
    )
    ws.mark_done(f"assets_subj{subj:02d}_{num_sessions}sess")
    return paths


def load_asset_meta(ws, subj: int, num_sessions: int) -> dict:
    meta = read_json(ws["data"] / f"subj{subj:02d}_sess{num_sessions}_meta.json")
    if meta is None:
        raise FileNotFoundError("Assets not prepared. Run `mindeye-lora assets` first.")
    return meta
