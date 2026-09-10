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

# Upstream's models.py imports utils.py, which imports the vendored `generative_models`
# (sgm) at module level -- so these are needed even when no image decoder is used.
UPSTREAM_IMPORT_REQS = [
    "open-clip-torch",     # sgm.modules.encoders.modules imports open_clip directly
    "pytorch-lightning",
    "lightning-utilities",
    "torchmetrics",
    "omegaconf",
    "diffusers",
    "transformers",
    "kornia",
]

# Import name -> pip package, wherever the two differ. Guessing by replacing "_" with
# "-" is right often enough to be tempting and wrong often enough to be dangerous:
# `open_clip` lives in `open-clip-torch`, not `open-clip` (which is an unrelated
# package), so the guess installs the wrong thing and the import still fails.
IMPORT_TO_PKG = {
    "open_clip": "open-clip-torch",
    "pytorch_lightning": "pytorch-lightning", "lightning_utilities": "lightning-utilities",
    "lightning_fabric": "pytorch-lightning", "torchmetrics": "torchmetrics",
    "pytorch_warmup": "pytorch-warmup", "embedding_reader": "embedding-reader",
    "ema_pytorch": "ema-pytorch", "einops_exts": "einops-exts",
    "rotary_embedding_torch": "rotary-embedding-torch", "x_clip": "x-clip",
    "coca_pytorch": "coca-pytorch", "clip": "clip-anytorch",
    "resize_right": "resize-right", "vector_quantize_pytorch": "vector-quantize-pytorch",
    "dalle2_pytorch": "dalle2-pytorch", "taming": "taming-transformers-rom1504",
    "PIL": "pillow", "sklearn": "scikit-learn", "skimage": "scikit-image",
    "yaml": "pyyaml", "cv2": "opencv-python-headless", "omegaconf": "omegaconf",
    "safetensors": "safetensors", "transformers": "transformers",
    "huggingface_hub": "huggingface-hub", "kornia": "kornia",
}


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


def preinstall_upstream_reqs() -> None:
    """Install what upstream's import chain needs, before touching it.

    models.py imports utils.py, which imports the vendored sgm at module level, which
    needs pytorch_lightning. Resolving that one ModuleNotFoundError at a time works but
    is slow and noisy, so install the known set up front and let
    `import_with_autoinstall` mop up whatever is left.
    """
    module_for = {"pytorch-lightning": "pytorch_lightning",
                  "lightning-utilities": "lightning_utilities",
                  "open-clip-torch": "open_clip"}
    missing = []
    for pkg in UPSTREAM_IMPORT_REQS:
        mod = module_for.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.info("installing upstream import requirements: %s", ", ".join(missing))
        for pkg in missing:
            _install(pkg)


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


def _install(pkg: str) -> None:
    # --no-deps everywhere: nothing here may replace Colab's torch build
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", pkg], check=False
    )


def patch_diffusers_vae() -> None:
    """Alias `diffusers.models.vae`, which modern diffusers moved.

    Upstream's models.py does `from diffusers.models.vae import Decoder`. Recent
    diffusers relocated that to `diffusers.models.autoencoders.vae`. Aliasing beats
    pinning an ancient diffusers release, and `Decoder` is only used by the low-level
    blurry-recon branch, which this project leaves disabled.
    """
    if "diffusers.models.vae" in sys.modules:
        return
    try:
        importlib.import_module("diffusers")
    except ModuleNotFoundError:
        _install("diffusers")
    for path in ("diffusers.models.vae", "diffusers.models.autoencoders.vae"):
        try:
            real = importlib.import_module(path)
        except ModuleNotFoundError:
            continue
        if path != "diffusers.models.vae":
            shim = ModuleType("diffusers.models.vae")
            for name in dir(real):
                if not name.startswith("_"):
                    setattr(shim, name, getattr(real, name))
            sys.modules["diffusers.models.vae"] = shim
            log.info("aliased diffusers.models.vae -> %s", path)
        return
    log.warning("Could not locate the diffusers VAE module; blurry_recon will not work.")


def import_with_autoinstall(module: str, max_installs: int = 25) -> ModuleType:
    """Import `module`, installing missing dependencies as they surface.

    Stops if the same module goes missing twice: that means a *moved* import path
    rather than an absent package, which pip cannot fix and which needs a shim.
    """
    seen: set[str] = set()
    for _ in range(max_installs):
        try:
            importlib.invalidate_caches()
            return importlib.import_module(module)
        except ModuleNotFoundError as exc:
            name = exc.name or ""
            if not name or name == module:
                raise
            if name in seen:
                raise ModuleNotFoundError(
                    f"'{name}' is still missing after installing {pkg!r}. Either the "
                    f"import name maps to a differently-named package (add it to "
                    f"IMPORT_TO_PKG in upstream.py), or the module was relocated and "
                    f"needs an alias (see patch_diffusers_vae for that pattern). "
                    f"Running `bash setup/colab_setup.sh` installs the known set.",
                    name=name,
                ) from exc
            seen.add(name)
            pkg = IMPORT_TO_PKG.get(name, name.replace("_", "-"))
            log.info("  %s needs %r -> installing %s", module, name, pkg)
            _install(pkg)
    raise RuntimeError(f"{module} unresolved after {max_installs} installs")


def extract_ridge_regression(repo: Path):
    """Pull `RidgeRegression` out of Train.ipynb.

    Upstream defines it inline in the training notebook rather than in models.py, so
    it cannot simply be imported. Executing the original source keeps the parameter
    names (`linears.N.weight`) byte-identical to the published checkpoint, which a
    reimplementation could silently get wrong.
    """
    import json

    import torch
    import torch.nn as nn

    for nb_name in ("Train.ipynb", "recon_inference.ipynb"):
        nb_path = repo / "src" / nb_name
        if not nb_path.exists():
            continue
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell["source"])
            if "class RidgeRegression" not in source:
                continue
            body, inside = [], False
            for line in source.split("\n"):
                if line.startswith("class RidgeRegression"):
                    inside = True
                elif inside and line and not line[0].isspace():
                    break          # dedent to column 0 ends the class
                if inside:
                    body.append(line)
            namespace = {"torch": torch, "nn": nn}
            exec("\n".join(body), namespace)          # noqa: S102 - upstream's own source
            cls = namespace.get("RidgeRegression")
            if cls is not None:
                log.info("extracted RidgeRegression from %s", nb_name)
                return cls
    raise AttributeError(
        "RidgeRegression not found in models.py or in Train.ipynb/recon_inference.ipynb. "
        "Pin a known-good revision with --upstream_ref=<sha>."
    )


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
    preinstall_upstream_reqs()
    patch_diffusers_vae()

    models = import_with_autoinstall("models")
    try:
        upstream_utils = import_with_autoinstall("utils")
    except Exception as exc:  # upstream utils pulls optional deps in some revisions
        log.warning("Upstream utils.py not importable (%s); using local loss implementations.", exc)
        upstream_utils = None

    # RidgeRegression lives in Train.ipynb rather than models.py, so graft it on.
    if not hasattr(models, "RidgeRegression"):
        models.RidgeRegression = extract_ridge_regression(repo)

    for required in ("RidgeRegression", "BrainNetwork", "PriorNetwork", "BrainDiffusionPrior"):
        if not hasattr(models, required):
            raise AttributeError(
                f"Upstream provides no `{required}`. Pin a known-good revision with "
                f"--upstream_ref=<sha>."
            )

    _CACHED = Upstream(repo, models, upstream_utils)
    log.info("Upstream MindEyeV2 loaded (commit %s)", _CACHED.sha)
    return _CACHED
