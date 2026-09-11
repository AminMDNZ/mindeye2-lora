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
# Crash-durable checkpoints
# --------------------------------------------------------------------------------------
def robust_save(obj, path: str | Path, verify: bool = True, keep_backup: bool = True) -> Path:
    """Save a torch object so that a killed session cannot leave it unusable.

    Google Drive's FUSE mount uploads asynchronously: a file that finished writing
    locally can still be truncated in the cloud if the VM is reclaimed a second later.
    A resume checkpoint that crashes the next run is worse than no checkpoint, so:

    1. write to a temporary file next to the target;
    2. flush and fsync, forcing the mount to commit rather than buffer;
    3. read it straight back to prove it deserialises;
    4. rotate the previous good file to `.bak` before swapping the new one in.

    The `.bak` generation is the part that matters for long runs: if the newest
    checkpoint is corrupt anyway, `robust_load` falls back to the previous one, so the
    worst case is losing one checkpoint interval rather than the whole arm.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())

    if verify:
        try:
            torch.load(tmp, map_location="cpu", weights_only=False)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise IOError(f"checkpoint failed verification, not written: {exc}") from exc

    if keep_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.unlink(missing_ok=True)
            path.replace(backup)
        except OSError:
            pass
    tmp.replace(path)
    return path


def robust_load(path: str | Path, validate=None, map_location="cpu", **load_kwargs):
    """Load a `robust_save` file, falling back to the `.bak` generation.

    Returns None if neither generation is usable, having removed the bad files so the
    caller simply starts fresh instead of failing.

    `validate` is an optional callable raising on a structurally wrong payload — a file
    can deserialise cleanly and still be missing keys if it was truncated at a lucky
    boundary.
    """
    import torch

    path = Path(path)
    for candidate, label in ((path, "checkpoint"), (path.with_suffix(path.suffix + ".bak"), "backup")):
        if not candidate.exists():
            continue
        try:
            # map_location is an explicit parameter, not swept into **load_kwargs:
            # callers naturally pass it, and having it in both places raised
            # "got multiple values for keyword argument 'map_location'" — which the
            # error handling then treated as a corrupt file and deleted, so training
            # resume silently never worked.
            obj = torch.load(candidate, map_location=map_location, weights_only=False,
                             **load_kwargs)
            if validate is not None:
                validate(obj)
            if label == "backup":
                log.warning("primary checkpoint unusable; recovered from %s", candidate.name)
            return obj
        except Exception as exc:
            log.warning("discarding unusable %s %s (%s)", label, candidate.name, exc)
            candidate.unlink(missing_ok=True)
    return None


# --------------------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------------------
def progress(iterable=None, total: int | None = None, desc: str = "", leave: bool = True,
             unit: str = "it", disable: bool = False):
    """tqdm if available, otherwise a no-op wrapper.

    Uses `tqdm.auto` so the bar renders as a widget in notebooks and as text in a
    terminal. Long stages without a bar are indistinguishable from a hang, which costs
    more in wasted waiting than the bar costs in output.
    """
    if disable:
        return iterable if iterable is not None else _NullBar()
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=leave, unit=unit,
                    dynamic_ncols=True, smoothing=0.1)
    except ImportError:  # pragma: no cover
        return iterable if iterable is not None else _NullBar()


class _NullBar:
    """Stand-in with tqdm's interface, for when tqdm is unavailable."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, *a, **k) -> None:
        pass

    def set_description(self, *a, **k) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class RunTracker:
    """Overall progress across a grid of (arm, seed) runs.

    Individual progress bars answer "is this step moving?"; this answers "how much of
    the whole experiment is left?", which is the question that matters when a full sweep
    is 18 runs spread over several Colab sessions.

    Counts runs skipped as already-finished separately, so resuming a partly-complete
    sweep reports an honest ETA based only on work actually done in this session.
    """

    def __init__(self, total: int, label: str = "runs"):
        self.total = total
        self.label = label
        self.done = 0
        self.skipped = 0
        self.t0 = time.time()

    def start(self, name: str) -> None:
        remaining = self.total - self.done - self.skipped
        log.info(
            "╔═ %s %d/%d · %s ═══", self.label, self.done + self.skipped + 1,
            self.total, name,
        )
        if self.done:
            per = (time.time() - self.t0) / self.done
            log.info("║  ~%.0f min left (%.1f min per %s so far, %d remaining)",
                     per * remaining / 60, per / 60, self.label.rstrip("s"), remaining)

    def finish(self, name: str, skipped: bool = False) -> None:
        if skipped:
            self.skipped += 1
        else:
            self.done += 1
        pct = 100 * (self.done + self.skipped) / max(1, self.total)
        log.info("╚═ %s %s · %d/%d complete (%.0f%%)%s", self.label.rstrip("s"), name,
                 self.done + self.skipped, self.total, pct,
                 "  [skipped, already done]" if skipped else "")

    def summary(self) -> str:
        mins = (time.time() - self.t0) / 60
        return (f"{self.done + self.skipped}/{self.total} {self.label} complete "
                f"({self.done} run here, {self.skipped} already done) in {mins:.1f} min")


def eta_string(done: int, total: int, elapsed_s: float) -> str:
    """'12/63 · 4.1 min elapsed · ~8.7 min left' — the line people actually want."""
    if done <= 0:
        return f"0/{total}"
    rate = elapsed_s / done
    remaining = rate * (total - done)
    return (f"{done}/{total} · {elapsed_s/60:.1f} min elapsed · "
            f"~{remaining/60:.1f} min left")


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
