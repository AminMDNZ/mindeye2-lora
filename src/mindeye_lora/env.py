"""Persistent workspace management.

The single most annoying thing about Colab is that `/content` is wiped every time the
runtime is recycled. Everything expensive in this project (HuggingFace downloads, torch
model zoo weights, precomputed CLIP embeddings, checkpoints, results) therefore lives
under one root directory on Google Drive, and every cache environment variable is
pointed at it *before* torch / huggingface are ever imported.

Design rules
------------
1. Big immutable blobs   -> Drive (`cache/`, `assets/`).
2. Derived small caches  -> Drive (`data/`).
3. Run state + results   -> Drive (`runs/`, `results/`).
4. Python site-packages  -> **local disk** (importing from Drive is painfully slow),
   but the pip *wheel cache* lives on Drive so reinstalls are fast and mostly offline.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .utils import disk_free, human_bytes, log, read_json, write_json

SUBDIRS = {
    "cache_hf": "cache/huggingface",
    "cache_torch": "cache/torch",
    "cache_pip": "cache/pip",
    "cache_misc": "cache/misc",
    "upstream": "upstream",
    "assets": "assets",
    "data": "data",
    "runs": "runs",
    "results": "results",
    "figures": "results/figures",
    "reports": "results/reports",
    "logs": "logs",
}


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


@dataclass
class Workspace:
    root: Path
    paths: dict[str, Path] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Path:  # ws.data, ws.runs, ...
        paths = self.__dict__.get("paths", {})
        if item in paths:
            return paths[item]
        raise AttributeError(item)

    def __getitem__(self, item: str) -> Path:
        return self.paths[item]

    # -- manifest -----------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "MANIFEST.json"

    def manifest(self) -> dict:
        return read_json(self.manifest_path, default={"stages": {}, "assets": {}}) or {
            "stages": {},
            "assets": {},
        }

    def mark_done(self, stage: str, **info) -> None:
        m = self.manifest()
        m["stages"][stage] = {"done": True, **info}
        write_json(self.manifest_path, m)

    def is_done(self, stage: str) -> bool:
        return bool(self.manifest()["stages"].get(stage, {}).get("done"))

    def clear_stage(self, stage: str) -> None:
        m = self.manifest()
        m["stages"].pop(stage, None)
        write_json(self.manifest_path, m)

    def run_dir(self, run_name: str) -> Path:
        d = self.paths["runs"] / run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def summary(self) -> str:
        lines = [f"workspace root : {self.root}", f"free space     : {human_bytes(disk_free(self.root))}"]
        m = self.manifest()
        done = [k for k, v in m["stages"].items() if v.get("done")]
        lines.append(f"completed      : {', '.join(done) if done else '(nothing yet)'}")
        return "\n".join(lines)


def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Mount Google Drive if we are on Colab and it is not already mounted."""
    if not in_colab():
        return False
    if os.path.ismount(mountpoint) or os.path.exists(f"{mountpoint}/MyDrive"):
        return True
    try:
        from google.colab import drive  # type: ignore

        drive.mount(mountpoint)
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("Could not mount Drive (%s); falling back to local storage.", exc)
        return False


def default_root() -> Path:
    if os.environ.get("MINDEYE_LORA_ROOT"):
        return Path(os.environ["MINDEYE_LORA_ROOT"])
    if in_colab() and os.path.exists("/content/drive/MyDrive"):
        return Path("/content/drive/MyDrive/mindeye2_lora")
    return Path.cwd() / "workspace"


def setup_environment(
    root: str | Path | None = None,
    use_drive: bool = True,
    min_free_gb: float = 6.0,
) -> Workspace:
    """Create the workspace and redirect every library cache into it.

    Must be called *before* importing torch / transformers / huggingface_hub for the
    cache redirection to take effect.
    """
    if use_drive:
        mount_drive()
    root = Path(root) if root else default_root()
    root.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, rel in SUBDIRS.items():
        p = root / rel
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p

    os.environ["MINDEYE_LORA_ROOT"] = str(root)
    # HuggingFace
    os.environ["HF_HOME"] = str(paths["cache_hf"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths["cache_hf"] / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(paths["cache_hf"] / "datasets")
    os.environ.setdefault("HF_HUB_ENABLE_HXET", "1")  # fast xet transfers where available
    # torch.hub / torchvision model zoo (AlexNet, Inception, EfficientNet, ... for eval)
    os.environ["TORCH_HOME"] = str(paths["cache_torch"])
    os.environ["XDG_CACHE_HOME"] = str(paths["cache_misc"])
    # pip wheel cache -> reinstalls after a runtime reset are near-instant
    os.environ["PIP_CACHE_DIR"] = str(paths["cache_pip"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Reduces allocator fragmentation, which matters on a 16 GB card where the diffusion
    # prior's attention allocates and frees large blocks every step. Only takes effect if
    # set before CUDA initialises, which is why it lives here rather than in the trainer.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ws = Workspace(root=root, paths=paths)

    free = disk_free(root)
    if free < min_free_gb * 1024**3:
        log.warning(
            "Only %s free at %s. The minimal asset set needs ~%.0f GB "
            "(betas subset + image subset + pretrained backbone).",
            human_bytes(free), root, min_free_gb,
        )
    log.info("Workspace ready at %s (free: %s)", root, human_bytes(free))
    return ws


def mirror_data_locally(ws: Workspace, local_root: str | Path = "/content/mindeye_cache") -> Workspace:
    """Copy the memory-mapped caches to local disk and read them from there.

    The voxel, image and CLIP-embedding caches are read with random access, one sample
    at a time. On Drive that is a network round trip per sample and the GPU spends most
    of its time idle; on Colab's local disk it is not. Results and checkpoints stay on
    Drive, so nothing is lost when the runtime resets — only the mirror is rebuilt, and
    that is a sequential copy, which Drive is fine at.

    No-op off Colab, or if the files are already mirrored.
    """
    import shutil

    if not in_colab():
        return ws
    local = Path(local_root) / "data"
    local.mkdir(parents=True, exist_ok=True)
    source = ws.paths["data"]
    copied = 0
    for f in sorted(source.glob("*.npy")):
        dst = local / f.name
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        log.info("mirroring %s (%s) to local disk", f.name, human_bytes(f.stat().st_size))
        shutil.copy2(f, dst)
        copied += 1
    for f in sorted(source.glob("*.json")):
        shutil.copy2(f, local / f.name)
    ws.paths["data"] = local
    if copied:
        log.info("mirrored %d file(s) -> %s", copied, local)
    return ws


def get_workspace() -> Workspace:
    """Re-attach to an already-configured workspace without touching env vars."""
    root = Path(os.environ.get("MINDEYE_LORA_ROOT", default_root()))
    paths = {k: root / v for k, v in SUBDIRS.items()}
    return Workspace(root=root, paths=paths)
