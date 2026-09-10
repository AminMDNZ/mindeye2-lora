"""Turn predicted CLIP embeddings into images with MindEye2's SDXL unCLIP decoder.

This stage is **optional and heavy**. The decoder checkpoint is 18 GB, and it is
completely frozen — it is the same weights for every arm, so it contributes no
between-arm variance. Every statistical conclusion in this project is therefore
available with `--decoder none`; reconstructions exist to answer the "show me the
pictures" half of the question and to compute the eight published image metrics.

Storage strategy on Colab: download to *local* disk (`/content`), optionally slim to
fp16 once and keep the slim copy on Drive so later sessions skip the 18 GB pull.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .assets import UNCLIP_CKPT, hf_download
from .utils import eta_string, human_bytes, log, progress, robust_load, robust_save

GENERATIVE_MODELS_URL = "https://github.com/Stability-AI/generative-models.git"


# --------------------------------------------------------------------------------------
# decoder setup
# --------------------------------------------------------------------------------------
def ensure_generative_models(upstream_dir: Path) -> Path:
    """Make both `generative_models.sgm` and bare `sgm` importable.

    Upstream's unclip6.yaml names its targets as `sgm.models.autoencoder.AutoencoderKL`,
    and `instantiate_from_config` imports that string literally. Putting only
    `generative_models`'s parent on the path makes `generative_models.sgm` work while
    bare `sgm` raises ModuleNotFoundError, so the engine fails the moment it tries to
    build its first stage. Add the package directory itself to `sys.path` as well, and
    alias the module both ways so either spelling resolves to the same object.
    """
    candidates = [
        Path(upstream_dir) / "MindEyeV2" / "src" / "generative_models",
        Path(upstream_dir) / "generative-models",
    ]
    package = next((c for c in candidates if c.exists()), None)
    if package is None:
        package = Path(upstream_dir) / "generative-models"
        log.info("Cloning Stability generative-models -> %s", package)
        subprocess.run(["git", "clone", "--depth", "1", GENERATIVE_MODELS_URL, str(package)],
                       check=True)

    for entry in (str(package.parent), str(package)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    # `sgm` lives inside the package directory; importing it top-level now works because
    # that directory is on the path. Alias so both import spellings share one instance.
    import importlib

    try:
        sgm = importlib.import_module("sgm")
        sys.modules.setdefault("generative_models.sgm", sgm)
    except ModuleNotFoundError:
        try:
            sgm = importlib.import_module("generative_models.sgm")
            sys.modules["sgm"] = sgm
            for sub in ("models", "util", "modules"):
                with contextlib.suppress(Exception):
                    sys.modules[f"sgm.{sub}"] = importlib.import_module(
                        f"generative_models.sgm.{sub}")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Neither `sgm` nor `generative_models.sgm` is importable from {package}. "
                "The decoder cannot be built; use --decoder none."
            ) from exc
    log.info("sgm importable from %s", package)
    patch_xformers_shim()
    return package


def patch_xformers_shim() -> bool:
    """Provide an `xformers` stand-in backed by PyTorch's native attention.

    sgm's VAE builds a `MemoryEfficientAttnBlock` when the config asks for
    `vanilla-xformers`, and that block calls `xformers.ops.memory_efficient_attention`
    unconditionally — the "no module 'xformers'. Processing without..." warning only
    covers the paths that check the flag, so decoding still dies with
    `NameError: name 'xformers' is not defined`.

    Installing real xformers would mean matching a build to torch 2.11, which is fragile
    on Colab. `F.scaled_dot_product_attention` computes exactly the same thing and is
    itself memory-efficient on PyTorch >= 2.0, so we register a shim module and bind it
    into the sgm modules that reference the bare name.

    Returns True if a shim was installed, False if real xformers was already present.
    """
    import importlib
    from types import ModuleType

    try:
        importlib.import_module("xformers")
        return False
    except ImportError:
        pass

    def memory_efficient_attention(query, key, value, attn_bias=None, op=None, **kwargs):
        # sgm passes (B, L, C) tensors; SDPA accepts that shape directly.
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias
        )

    xformers = ModuleType("xformers")
    ops = ModuleType("xformers.ops")
    ops.memory_efficient_attention = memory_efficient_attention
    ops.MemoryEfficientAttentionFlashAttentionOp = None
    xformers.ops = ops
    sys.modules.setdefault("xformers", xformers)
    sys.modules.setdefault("xformers.ops", ops)

    # `import xformers` inside a try/except leaves the module-level name unbound, so
    # registering in sys.modules is not enough — bind the name in each module that uses it.
    bound = 0
    for name in ("sgm.modules.diffusionmodules.model", "sgm.modules.attention",
                 "generative_models.sgm.modules.diffusionmodules.model",
                 "generative_models.sgm.modules.attention"):
        module = sys.modules.get(name)
        if module is not None:
            module.xformers = xformers
            if hasattr(module, "XFORMERS_IS_AVAILABLE"):
                module.XFORMERS_IS_AVAILABLE = True
            bound += 1
    log.info("xformers shim installed (PyTorch SDPA) and bound into %d sgm module(s)", bound)
    return True


def find_unclip_config(upstream_dir: Path) -> Path:
    """Locate the `unclip6.yaml` config MindEye2 ships with."""
    for pattern in ("**/configs/unclip6.yaml", "**/unclip6.yaml"):
        for base in (Path(upstream_dir) / "MindEyeV2", Path(upstream_dir)):
            hits = sorted(base.glob(pattern))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "unclip6.yaml not found. It ships inside MindEyeV2/src/generative_models/configs/. "
        "If the upstream layout changed, pass --unclip_config explicitly."
    )


def download_unclip_checkpoint(target_dir: Path) -> Path:
    """Fetch the 18 GB unCLIP checkpoint to local disk (not Drive)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return hf_download(UNCLIP_CKPT, target_dir)


def shard_unclip_checkpoint(src: Path, shard_dir: Path, shard_gb: float = 1.5) -> Path:
    """Split the 18 GB decoder checkpoint into fp16 shards without ever holding it whole.

    `torch.load` on the full file needs ~18 GB of RAM, which kills a standard 12 GB
    Colab VM outright ("your session crashed after using all available RAM"). Loading
    with `mmap=True` keeps the tensors on disk and pages them in on demand, so we can
    walk the state dict, cast each tensor to fp16, and write ~1.5 GB shards. Peak RAM
    stays around one shard.

    Done once; later sessions reuse the shards.
    """
    import torch

    shard_dir = Path(shard_dir)
    index_path = shard_dir / "index.json"
    if index_path.exists():
        log.info("using existing decoder shards in %s", shard_dir)
        return shard_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    log.info("sharding %s (%s) -> fp16 shards", src.name, human_bytes(src.stat().st_size))
    try:
        ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not memory-map {src.name} ({exc}). Without mmap the whole 18 GB "
            "checkpoint must fit in RAM. Switch Colab to a High-RAM runtime "
            "(Runtime > Change runtime type > High-RAM) and retry."
        ) from exc

    sd = ckpt.get("state_dict", ckpt)
    limit = int(shard_gb * 1024**3)
    shards, current, current_bytes, index = [], {}, 0, {}

    def flush():
        nonlocal current, current_bytes
        if not current:
            return
        name = f"shard_{len(shards):03d}.pt"
        torch.save(current, shard_dir / name)
        for key in current:
            index[key] = name
        shards.append(name)
        log.info("  wrote %s (%d tensors, %s)", name, len(current), human_bytes(current_bytes))
        current, current_bytes = {}, 0

    for key, value in sd.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.half() if torch.is_floating_point(value) else value
        current[key] = tensor.clone()          # detach from the mmap before it is closed
        current_bytes += tensor.numel() * tensor.element_size()
        if current_bytes >= limit:
            flush()
    flush()
    del ckpt, sd

    import json

    index_path.write_text(json.dumps({"shards": shards, "keys": index}))
    log.info("decoder sharded into %d files -> %s", len(shards), shard_dir)
    return shard_dir


def load_sharded_state_dict(engine, shard_dir: Path) -> None:
    """Copy sharded weights into an already-constructed engine, one shard at a time."""
    import json

    import torch

    index = json.loads((Path(shard_dir) / "index.json").read_text())
    own = dict(engine.state_dict())
    loaded = missing = skipped_embedder = 0
    bar = progress(total=len(index["shards"]), desc="loading decoder weights",
                   unit="shard", leave=False)
    for name in index["shards"]:
        shard = torch.load(Path(shard_dir) / name, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for key, value in shard.items():
                target = own.get(key)
                if target is None:
                    # embedder 0 is deliberately absent; do not report it as a problem
                    if "conditioner.embedders.0." in key:
                        skipped_embedder += 1
                    else:
                        missing += 1
                    continue
                if tuple(target.shape) != tuple(value.shape):
                    log.warning("shape mismatch for %s: %s vs %s", key,
                                tuple(target.shape), tuple(value.shape))
                    missing += 1
                    continue
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                loaded += 1
        del shard
        bar.update(1)
    bar.close()
    log.info("decoder weights: %d loaded, %d unmatched, %d skipped (dropped embedder)",
             loaded, missing, skipped_embedder)
    if missing:
        log.warning("%d weights had no home in the engine — reconstructions may be wrong",
                    missing)


def build_diffusion_engine(
    ckpt_path: Path, config_path: Path, device: str = "cuda", num_steps: int = 38,
    sharded: bool = True, dtype=None, drop_image_embedder: bool = True,
):
    """Construct the SDXL unCLIP engine small enough to fit a 16 GB card.

    Three things matter here, all learned the hard way on a T4:

    1. **Construct in fp16.** `DiffusionEngine` instantiates and moves its submodules
       during `__init__`, so converting afterwards is too late — the OOM happens inside
       the constructor. Setting the default dtype around construction makes every module
       come out half precision from the start.

    2. **Drop the image embedder.** `FrozenOpenCLIPImageEmbedder` is 1.9B parameters
       (3.6 GB even in fp16) and exists to encode a real image into a CLIP embedding. We
       already *have* the embedding — that is the model's output — so it is pure
       overhead at inference. Removing it from the conditioner config is the single
       biggest saving available.

    3. **Build on CPU, then move.** Keeps a transient GPU copy from colliding with the
       weights being streamed in.
    """
    import importlib

    from omegaconf import OmegaConf

    dtype = dtype if dtype is not None else torch.float16
    try:
        DiffusionEngine = importlib.import_module("sgm.models.diffusion").DiffusionEngine
    except ModuleNotFoundError:
        DiffusionEngine = importlib.import_module(
            "generative_models.sgm.models.diffusion").DiffusionEngine

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    params = cfg["model"]["params"]
    params["sampler_config"]["params"]["num_steps"] = num_steps
    params["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"

    conditioner = params["conditioner_config"]
    dropped = []
    if drop_image_embedder:
        kept = []
        for emb in conditioner["params"]["emb_models"]:
            target = str(emb.get("target", ""))
            if "ImageEmbedder" in target:
                dropped.append(target.rsplit(".", 1)[-1])
                continue
            kept.append(emb)
        conditioner["params"]["emb_models"] = kept
    # Any embedder that names a device should build on CPU; we move the whole engine.
    for emb in conditioner["params"]["emb_models"]:
        if isinstance(emb.get("params"), dict) and "device" in emb["params"]:
            emb["params"]["device"] = "cpu"
    if dropped:
        log.info("decoder: dropped %s (embedding supplied directly, saves ~3.6 GB)",
                 ", ".join(dropped))

    previous_dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        torch.set_default_dtype(torch.float16)
    try:
        engine = DiffusionEngine(
            network_config=params["network_config"],
            denoiser_config=params["denoiser_config"],
            first_stage_config=params["first_stage_config"],
            conditioner_config=conditioner,
            sampler_config=params["sampler_config"],
            scale_factor=params["scale_factor"],
            disable_first_stage_autocast=params["disable_first_stage_autocast"],
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    n_params = sum(p.numel() for p in engine.parameters())
    log.info("decoder built: %.2fB parameters (%s)", n_params / 1e9,
             "fp16" if dtype == torch.float16 else str(dtype))

    if sharded:
        engine = engine.half() if dtype == torch.float16 else engine
        engine = engine.to(device).eval().requires_grad_(False)
        if torch.cuda.is_available():
            log.info("decoder on GPU: %.1f GB allocated, %.1f GB free",
                     torch.cuda.memory_allocated() / 1024**3,
                     torch.cuda.mem_get_info()[0] / 1024**3)
        load_sharded_state_dict(engine, ckpt_path)

        # The VAE decoder runs outside autocast: sgm's `decode_first_stage` honours
        # `disable_first_stage_autocast`, so it receives fp32 latents from the sampler
        # while its own weights are fp16 -- "Input type (float) and bias type
        # (c10::Half) should be the same". It is only 84M parameters (~0.3 GB in fp32),
        # so keeping it at full precision is the cheap and correct fix, and it avoids
        # fp16 artefacts in the final pixels.
        if hasattr(engine, "first_stage_model"):
            engine.first_stage_model = engine.first_stage_model.float()
            log.info("first stage (VAE) kept in fp32 for decoding")
        if torch.cuda.is_available():
            log.info("decoder ready: %.1f GB allocated, %.1f GB free",
                     torch.cuda.memory_allocated() / 1024**3,
                     torch.cuda.mem_get_info()[0] / 1024**3)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        engine.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        engine = engine.to(device).eval().requires_grad_(False)
        del ckpt
    torch.cuda.empty_cache()
    return engine


@torch.no_grad()
def _vector_suffix(engine, device: str, size: int = 768):
    """The size/crop conditioning vector SDXL expects, computed once."""
    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * size,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
    }
    out = engine.conditioner(batch)
    return out["vector"].to(device)


@torch.no_grad()
def unclip_reconstruct(
    engine,
    clip_embeddings: torch.Tensor,
    vector_suffix: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 4,
    upstream_utils=None,
    checkpoint_path=None,
    desc: str = "decode",
) -> torch.Tensor:
    """Decode [N, 256, 1664] CLIP token embeddings into [N, 3, H, W] images in [0, 1].

    Diffusion decoding runs at 3-5 s/image, so a 200-image arm is a 15-minute stage.
    Results are checkpointed every few batches and resumed, because re-decoding from
    scratch after a disconnect is the most expensive restart in the pipeline.
    """
    if upstream_utils is not None and hasattr(upstream_utils, "unclip_recon"):
        outs, start_at = [], 0
        if checkpoint_path is not None:
            def _validate(part):
                if "images" not in part or "done" not in part:
                    raise ValueError("missing keys")
                if int(part["done"]) != int(part["images"].shape[0]):
                    raise ValueError("count disagrees with tensor length")

            part = robust_load(checkpoint_path, validate=_validate)
            if part is not None:
                outs = [part["images"]]
                start_at = int(part["done"])
                log.info("resuming decode from image %d/%d", start_at, len(clip_embeddings))

        import time

        t0 = time.time()
        bar = progress(total=len(clip_embeddings), desc=desc, unit="img")
        if start_at:
            bar.update(start_at)
        for start in range(start_at, len(clip_embeddings), batch_size):
            chunk = clip_embeddings[start : start + batch_size].to(device)
            try:
                samples = upstream_utils.unclip_recon(chunk, engine, vector_suffix,
                                                      num_samples=1)
            except RuntimeError as exc:
                if "should be the same" not in str(exc):
                    raise
                # A dtype boundary we did not anticipate. Promote the first stage to
                # fp32 and retry once rather than losing the whole run.
                log.warning("dtype mismatch during decode (%s); promoting first stage "
                            "to fp32 and retrying", exc)
                engine.first_stage_model = engine.first_stage_model.float()
                samples = upstream_utils.unclip_recon(chunk, engine, vector_suffix,
                                                      num_samples=1)
            outs.append(samples.float().cpu())
            done = min(start + batch_size, len(clip_embeddings))
            bar.update(done - start)
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(eta_string(done - start_at,
                                               len(clip_embeddings) - start_at,
                                               time.time() - t0))
            # Every batch for short runs, every third for long ones: decoding is ~4 s an
            # image, so a lost batch is cheap, but writing a growing tensor every time
            # would dominate a 200-image arm.
            checkpoint_now = (len(clip_embeddings) <= 32
                              or (start // batch_size) % 3 == 0)
            if checkpoint_path is not None and checkpoint_now:
                try:
                    robust_save({"images": torch.cat(outs), "done": done}, checkpoint_path)
                except IOError as exc:
                    log.warning("decode checkpoint failed (%s); continuing", exc)
        bar.close()
        return torch.cat(outs).clamp(0, 1)

    raise RuntimeError(
        "Upstream `utils.unclip_recon` is unavailable, and reimplementing the sgm sampling "
        "loop here would risk diverging from the published decoder. Either make upstream "
        "utils importable, or run with --decoder none (all statistics still work)."
    )


# --------------------------------------------------------------------------------------
# stage entry point
# --------------------------------------------------------------------------------------
def reconstruct_for_run(
    predictions_path: Path,   # directory holding the streamed PredictionStore
    out_path: Path,
    ws,
    n_images: int,
    decoder: str = "sdxl_unclip",
    device: str = "cuda",
    num_steps: int = 38,
    batch_size: int = 4,
    unclip_dir: Path | None = None,
) -> Path | None:
    """Decode the first `n_images` predicted embeddings of one run."""
    if decoder == "none":
        log.info("decoder disabled — skipping reconstruction.")
        return None

    from .evaluate import PredictionStore

    preds = PredictionStore.load(predictions_path)
    # Slice first, then materialise: the store is a memmap and the full array is ~1.7 GB.
    source = PredictionStore.embedding(preds)
    # np.array(...) rather than ascontiguousarray: the memmap slice is read-only and
    # torch.from_numpy warns loudly about non-writable buffers.
    emb = torch.from_numpy(np.array(source[:n_images], dtype=np.float32))
    rows = torch.from_numpy(np.asarray(preds["rows"][:n_images]))

    from .upstream import load_upstream

    up = load_upstream(ws["upstream"])
    ensure_generative_models(ws["upstream"])
    config_path = find_unclip_config(ws["upstream"])

    unclip_dir = Path(unclip_dir or "/content/unclip_cache")
    shard_dir = ws["assets"] / "unclip6_fp16_shards"
    if not (shard_dir / "index.json").exists():
        raw = download_unclip_checkpoint(unclip_dir)
        shard_unclip_checkpoint(raw, shard_dir)
        # The 18 GB original is no longer needed; the fp16 shards are ~9 GB total.
        with contextlib.suppress(OSError):
            raw.unlink()
            log.info("removed the raw 18 GB checkpoint after sharding")

    # The encoder is not needed from here on, and the decoder wants every spare byte.
    from .train import release_cuda

    release_cuda()
    if torch.cuda.is_available():
        log.info("before decoder build: %.1f GB free",
                 torch.cuda.mem_get_info()[0] / 1024**3)

    try:
        engine = build_diffusion_engine(shard_dir, config_path, device=device,
                                        num_steps=num_steps, sharded=True,
                                        drop_image_embedder=True)
        suffix = _vector_suffix(engine, device)
    except Exception as exc:
        # The image embedder is unused when embeddings are supplied directly, but if
        # upstream's conditioner insists on it, fall back rather than fail outright.
        log.warning("decoder build without the image embedder failed (%s); retrying "
                    "with it (needs ~3.6 GB more VRAM)", exc)
        release_cuda(locals().get("engine"))
        engine = build_diffusion_engine(shard_dir, config_path, device=device,
                                        num_steps=num_steps, sharded=True,
                                        drop_image_embedder=False)
        suffix = _vector_suffix(engine, device)
    out_path = Path(out_path)
    partial = out_path.with_name(out_path.stem + ".partial.pt")
    images = unclip_reconstruct(engine, emb, suffix, device=device, batch_size=batch_size,
                                upstream_utils=up.utils, checkpoint_path=partial,
                                desc=f"decode {Path(predictions_path).parent.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    robust_save({"recons": images, "rows": rows}, out_path)
    partial.unlink(missing_ok=True)
    Path(str(partial) + ".bak").unlink(missing_ok=True)
    log.info("saved %d reconstructions -> %s", len(images), out_path.name)

    del engine
    torch.cuda.empty_cache()
    return out_path


def load_ground_truth(images_path: Path, rows: np.ndarray, size: int | None = None) -> torch.Tensor:
    imgs = np.load(images_path, mmap_mode="r")
    gt = torch.from_numpy(np.asarray(imgs[np.asarray(rows, dtype=np.int64)], dtype=np.float32))
    if size and gt.shape[-1] != size:
        gt = torch.nn.functional.interpolate(gt, size=(size, size), mode="bilinear",
                                             align_corners=False, antialias=True)
    return gt.clamp(0, 1)
