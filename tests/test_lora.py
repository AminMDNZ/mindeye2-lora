"""Tests for the LoRA implementation.

The identity-at-init and merge-equivalence tests are the important ones: if either
fails, the LoRA arm and the full fine-tuning arm are no longer starting from the same
model and the whole comparison is invalid.
"""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mindeye_lora.lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    apply_lora,
    find_target_linears,
    load_adapter_state_dict,
    merge_lora,
    set_trainable,
    unmerge_lora,
)


class Toy(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ridge = nn.Linear(64, d)
        self.backbone = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.diffusion_prior = nn.ModuleDict({"net": nn.Linear(d, d)})

    def forward(self, x):
        h = self.backbone(self.ridge(x))
        return self.diffusion_prior["net"](h)


def test_zero_init_is_identity():
    torch.manual_seed(0)
    model, ref = Toy(), Toy()
    ref.load_state_dict(model.state_dict())
    x = torch.randn(8, 64)
    before = ref(x)
    apply_lora(model, LoRAConfig(r=8))
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6), "LoRA must be a no-op at initialisation"


def test_merge_matches_unmerged_forward():
    torch.manual_seed(1)
    model = Toy()
    apply_lora(model, LoRAConfig(r=4, alpha=8))
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)
    x = torch.randn(6, 64)
    unmerged = model(x)
    merge_lora(model)
    merged = model(x)
    assert torch.allclose(unmerged, merged, atol=1e-5)
    unmerge_lora(model)
    assert torch.allclose(model(x), unmerged, atol=1e-5)


def test_targeting_excludes_ridge():
    model = Toy()
    names = find_target_linears(model, LoRAConfig())
    assert not any(n.startswith("ridge") for n in names)
    assert any(n.startswith("backbone") for n in names)
    assert any(n.startswith("diffusion_prior") for n in names)


def test_targeting_raises_when_nothing_matches():
    model = Toy()
    with pytest.raises(ValueError):
        apply_lora(model, LoRAConfig(r=4, target_modules=[r"^nonexistent\."]))


def test_trainable_sets():
    torch.manual_seed(2)
    model = Toy()
    apply_lora(model, LoRAConfig(r=8))
    info = set_trainable(model, "lora")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(".lora_" in n or n.startswith("ridge.") for n in trainable)
    assert info["trainable_params"] < info["total_params"]

    full = set_trainable(Toy(), "full")
    assert full["trainable_fraction"] == 1.0

    frozen_model = Toy()
    frozen = set_trainable(frozen_model, "frozen")
    frozen_names = {n for n, p in frozen_model.named_parameters() if p.requires_grad}
    assert frozen_names == {"ridge.weight", "ridge.bias"}
    assert frozen["trainable_params"] < full["trainable_params"]


def test_adapter_roundtrip_is_small_and_lossless():
    torch.manual_seed(3)
    model = Toy()
    apply_lora(model, LoRAConfig(r=8))
    set_trainable(model, "lora")
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.02)
    x = torch.randn(4, 64)
    expected = model(x)

    sd = adapter_state_dict(model)
    assert all(".lora_" in k or k.startswith("ridge.") for k in sd)
    full_params = sum(p.numel() for p in model.parameters())
    assert sum(v.numel() for v in sd.values()) < full_params

    fresh = Toy()
    fresh.load_state_dict({k: v for k, v in model.state_dict().items()
                           if ".lora_" not in k and not k.startswith("ridge.")},
                          strict=False)
    apply_lora(fresh, LoRAConfig(r=8))
    load_adapter_state_dict(fresh, sd)
    assert torch.allclose(fresh(x), expected, atol=1e-5)


def test_rank_scales_parameter_count():
    counts = []
    for r in (4, 16, 64):
        model = Toy()
        info = apply_lora(model, LoRAConfig(r=r))
        counts.append(info["adapter_params"])
    assert counts[0] < counts[1] < counts[2]
    assert counts[1] == pytest.approx(counts[0] * 4)


def test_dora_variant_runs():
    torch.manual_seed(4)
    model = Toy()
    apply_lora(model, LoRAConfig(r=4, variant="dora"))
    out = model(torch.randn(3, 64))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()
