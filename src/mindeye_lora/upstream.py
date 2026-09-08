"""Bind to the official MindEyeV2 implementation at runtime.

Why not reimplement the architecture here?
------------------------------------------
The whole experiment hinges on loading the *published* pretrained shared-subject
checkpoint. A from-memory reimplementation would have to match every parameter name and
shape exactly; a single renamed attribute silently turns "fine-tuning a pretrained
model" into "training from scratch", which would invalidate every number in the report.

So we clone `MedARC-AI/MindEyeV2` into the persistent workspace, put `src/` on
`sys.path`, and import `models.py` / `utils.py` from it. The clone is pinned by commit
SHA in the manifest so a run is reproducible even if upstream moves.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from .utils import git_sha, log

UPSTREAM_URL = "https://github.com/MedARC-AI/MindEyeV2.git"

# dalle2-pytorch is what MindEyeV2's models.py builds its diffusion prior on top of.
# Installing it with its own dependency pins tends to drag in an incompatible torch on
# Colab, so we install --no-deps and supply the modules it actually imports.
DALLE2_NO_DEPS_REQS = [
    "dalle2-pytorch==1.15.6",
]
DALLE2_SUPPORT_REQS = [
    "einops>=0.7",
    "einops-exts>=0.0.4",
    "rotary-embedding-torch>=0.3",
    "ema-pytorch>=0.2",
    "vector-quantize-pytorch>=1.9",
    "x-clip>=0.12",
    "coca-pytorch>=0.1",
    "clip-anytorch>=2.5",
    "resize-right>=0.0.2",
    "kornia>=0.7",
    "torchvision",
    "webdataset",
    "click",
    "pydantic",
    "pytorch-warmup",     # imported by dalle2_pytorch.trainer
    "accelerate",
    "embedding-reader",
]


def _pip(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    log.info("pip %s", " ".join(args[:4]) + (" ..." if len(args) > 4 else ""))
    subprocess.run(cmd, check=True)


def ensure_dalle2(verbose: bool = True) -> ModuleType:
    """Import dalle2_pytorch, installing it in a torch-safe way if needed."""
    try:
        return importlib.import_module("dalle2_pytorch")
    except ImportError:
        pass
    if verbose:
        log.info("Installing dalle2-pytorch (no-deps) + its runtime imports ...")
    _pip(DALLE2_SUPPORT_REQS)
    _pip(["--no-deps", *DALLE2_NO_DEPS_REQS])
    importlib.invalidate_caches()
    try:
        return importlib.import_module("dalle2_pytorch")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "dalle2_pytorch could not be imported after installation. It is required by "
            "MindEyeV2's models.py. Try, in a fresh Colab runtime:\n"
            "  !pip install dalle2-pytorch==1.15.6\n"
            "and restart the runtime."
        ) from exc


def clone_or_update(dest: Path, ref: str = "main", update: bool = False) -> Path:
    """Clone MindEyeV2 into `dest/MindEyeV2` (idempotent)."""
    repo = Path(dest) / "MindEyeV2"
    if repo.exists() and (repo / "src").exists():
        if update:
            subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", ref], check=False)
            subprocess.run(["git", "-C", str(repo), "checkout", "-f", ref], check=False)
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    log.info("Cloning %s -> %s", UPSTREAM_URL, repo)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, UPSTREAM_URL, str(repo)],
        check=True,
    )
    return repo


class Upstream:
    """Handle to the imported upstream modules."""

    def __init__(self, repo: Path, models: ModuleType, utils: ModuleType | None):
        self.repo = repo
        self.models = models
        self.utils = utils
        self.sha = git_sha(repo)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Upstream {self.repo} @ {self.sha}>"


_CACHED: Upstream | None = None


def load_upstream(workspace_upstream_dir: Path, ref: str = "main", update: bool = False) -> Upstream:
    """Clone (if needed) and import MindEyeV2's `models` and `utils` modules."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    repo = clone_or_update(Path(workspace_upstream_dir), ref=ref, update=update)
    src = repo / "src"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — the upstream layout changed.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    ensure_dalle2()

    models = importlib.import_module("models")
    try:
        upstream_utils = importlib.import_module("utils")
    except Exception as exc:  # upstream utils pulls optional deps in some revisions
        log.warning("Upstream utils.py not importable (%s); using local loss implementations.", exc)
        upstream_utils = None

    for required in ("RidgeRegression", "BrainNetwork", "PriorNetwork", "BrainDiffusionPrior"):
        if not hasattr(models, required):
            raise AttributeError(
                f"Upstream models.py has no `{required}`. Pin a known-good revision with "
                f"--upstream_ref=<sha>."
            )

    _CACHED = Upstream(repo, models, upstream_utils)
    log.info("Upstream MindEyeV2 loaded (commit %s)", _CACHED.sha)
    return _CACHED
