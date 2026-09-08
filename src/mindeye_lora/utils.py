"""Small shared utilities: logging, seeding, JSON/CSV IO, timers, parameter counting."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

_LOG_CONFIGURED = False


def get_logger(name: str = "mindeye_lora") -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("mindeye_lora")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


log = get_logger()


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed python/numpy/torch. Torch is imported lazily so this module stays light."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def worker_seed_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        seed_everything(base_seed + worker_id)

    return _init


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D102
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, cls=_Encoder))
    tmp.replace(path)  # atomic-ish; protects against Colab dying mid-write
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("Corrupt JSON at %s; ignoring.", path)
        return default


class CSVLogger:
    """Append-only CSV logger that survives kernel restarts."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(",".join(self.fieldnames) + "\n")

    def log(self, **row: Any) -> None:
        vals = []
        for k in self.fieldnames:
            v = row.get(k, "")
            vals.append("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
        with self.path.open("a") as fh:
            fh.write(",".join(vals) + "\n")


# --------------------------------------------------------------------------------------
# Timing / budget
# --------------------------------------------------------------------------------------
class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def reset(self) -> None:
        self.t0 = time.time()


class TimeBudget:
    """Graceful stop before a Colab session is reclaimed."""

    def __init__(self, minutes: float | None):
        self.limit = None if not minutes or minutes <= 0 else minutes * 60.0
        self.t0 = time.time()

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and (time.time() - self.t0) > self.limit

    @property
    def remaining_min(self) -> float:
        if self.limit is None:
            return float("inf")
        return max(0.0, (self.limit - (time.time() - self.t0)) / 60.0)


@contextlib.contextmanager
def timed(label: str):
    t0 = time.time()
    log.info("▶ %s", label)
    yield
    log.info("✔ %s (%.1fs)", label, time.time() - t0)


# --------------------------------------------------------------------------------------
# Model introspection
# --------------------------------------------------------------------------------------
def count_parameters(module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def parameter_report(module, group_depth: int = 1) -> dict[str, dict[str, int]]:
    """Trainable/total parameter counts grouped by top-level submodule name."""
    groups: dict[str, dict[str, int]] = {}
    for name, p in module.named_parameters():
        key = ".".join(name.split(".")[:group_depth]) or "<root>"
        g = groups.setdefault(key, {"total": 0, "trainable": 0})
        g["total"] += p.numel()
        if p.requires_grad:
            g["trainable"] += p.numel()
    return groups


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def disk_free(path: str | Path) -> int:
    return shutil.disk_usage(str(path)).free


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha(repo_dir: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception:
        return None


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}
