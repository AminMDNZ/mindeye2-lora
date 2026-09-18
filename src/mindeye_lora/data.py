"""Datasets built from the cached subsets produced by `assets.py`.

MindEye2's webdataset shards contain only behavioural metadata; the voxels and the
stimulus images are looked up from HDF5 by index. We keep that structure but operate on
small local `.npy` subsets, with two index remaps:

    behav[:, 5]  (global trial) -> row in the cached voxel array
    behav[:, 0]  (COCO 73k id)  -> row in the cached image / CLIP-embedding arrays

Test-set handling follows the paper: each of the 1,000 shared images was seen three
times, and the three beta patterns are averaged before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .assets import BEHAV_COCO_IDX, BEHAV_GLOBAL_TRIAL, AssetPaths
from .utils import log, worker_seed_fn


def _index_map(values: np.ndarray) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(values)}


@dataclass
class Subset:
    voxels: np.ndarray          # (T, V) float32, one row per trial
    coco_ids: np.ndarray        # (T,) int64
    image_rows: np.ndarray      # (T,) int64 -> row in the shared image/embedding cache
    repeats: np.ndarray | None  # (T,) int64 group id when trials were averaged

    def __len__(self) -> int:
        return len(self.voxels)


class MindEyeDataset(Dataset):
    """Yields (voxel, clip_target, image, image_row)."""

    def __init__(
        self,
        subset: Subset,
        images_path: Path,
        embeddings_path: Path | None,
        return_images: bool = False,
    ):
        self.subset = subset
        self.images_path = Path(images_path)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.return_images = return_images
        self._images = None
        self._embs = None

    # lazily memory-map inside each worker process
    def _img(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    def _emb(self) -> np.ndarray | None:
        if self.embeddings_path is None:
            return None
        if self._embs is None:
            self._embs = np.load(self.embeddings_path, mmap_mode="r")
        return self._embs

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, i: int):
        row = int(self.subset.image_rows[i])
        voxel = torch.from_numpy(np.asarray(self.subset.voxels[i], dtype=np.float32))
        emb = self._emb()
        clip_target = (
            torch.from_numpy(np.asarray(emb[row], dtype=np.float32))
            if emb is not None
            else torch.zeros(1)
        )
        image = (
            torch.from_numpy(np.asarray(self._img()[row], dtype=np.float32))
            if self.return_images
            else torch.zeros(1)
        )
        return voxel, clip_target, image, row


def build_subsets(
    paths: AssetPaths,
    average_test_repeats: bool = True,
) -> tuple[Subset, Subset, np.ndarray]:
    """Return (train_subset, test_subset, unique_coco_ids)."""
    behav_train = np.load(paths.behav_train)
    behav_test = np.load(paths.behav_test)
    voxels = np.load(paths.voxels, mmap_mode="r")
    trial_ids = np.load(paths.voxel_index)
    coco_ids = np.load(paths.image_index)

    trial_to_row = _index_map(trial_ids)
    coco_to_row = _index_map(coco_ids)

    def rows_for(behav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.array([trial_to_row[int(x)] for x in behav[:, BEHAV_GLOBAL_TRIAL]], dtype=np.int64)
        c = np.array([coco_to_row[int(x)] for x in behav[:, BEHAV_COCO_IDX]], dtype=np.int64)
        return t, c

    tr_trials, tr_imgrows = rows_for(behav_train)
    train = Subset(
        voxels=np.asarray(voxels[tr_trials], dtype=np.float32),
        coco_ids=behav_train[:, BEHAV_COCO_IDX].astype(np.int64),
        image_rows=tr_imgrows,
        repeats=None,
    )

    te_trials, te_imgrows = rows_for(behav_test)
    if average_test_repeats:
        uniq_rows, inverse = np.unique(te_imgrows, return_inverse=True)
        acc = np.zeros((len(uniq_rows), voxels.shape[1]), dtype=np.float64)
        cnt = np.zeros(len(uniq_rows), dtype=np.int64)
        raw = np.asarray(voxels[te_trials], dtype=np.float32)
        np.add.at(acc, inverse, raw)
        np.add.at(cnt, inverse, 1)
        averaged = (acc / cnt[:, None]).astype(np.float32)
        test = Subset(
            voxels=averaged,
            coco_ids=np.array([coco_ids[r] for r in uniq_rows], dtype=np.int64),
            image_rows=uniq_rows,
            repeats=cnt,
        )
        log.info(
            "test set: %d trials -> %d unique images (mean %.2f repeats)",
            len(te_trials), len(uniq_rows), cnt.mean(),
        )
    else:
        test = Subset(
            voxels=np.asarray(voxels[te_trials], dtype=np.float32),
            coco_ids=behav_test[:, BEHAV_COCO_IDX].astype(np.int64),
            image_rows=te_imgrows,
            repeats=None,
        )
    return train, test, coco_ids


def normalise_voxels(train: Subset, test: Subset) -> dict[str, np.ndarray]:
    """Z-score per voxel using *training* statistics only (no test leakage)."""
    mu = train.voxels.mean(0, keepdims=True)
    sd = train.voxels.std(0, keepdims=True) + 1e-6
    train.voxels = (train.voxels - mu) / sd
    test.voxels = (test.voxels - mu) / sd
    return {"mean": mu, "std": sd}


def make_loaders(
    train_ds: Dataset,
    test_ds: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
    eval_batch_size: int | None = None,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_seed_fn(seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, generator=g, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size or batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, test_loader
