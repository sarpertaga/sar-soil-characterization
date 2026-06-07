import pytest

torch = pytest.importorskip("torch")  # soilgeo-dl env only

from soilgeo.models.unet import UNet, resolve_device  # noqa: E402


def test_forward_shape_single_head():
    model = UNet(in_channels=13, base_channels=16, depth=3,
                 n_classes=6, regression=False)
    x = torch.randn(2, 13, 64, 64)
    out = model(x)
    assert "class_logits" in out
    assert out["class_logits"].shape == (2, 6, 64, 64)
    assert "regression" not in out


def test_forward_shape_dual_head():
    model = UNet(in_channels=8, base_channels=16, depth=3,
                 n_classes=5, regression=True)
    x = torch.randn(1, 8, 32, 32)
    out = model(x)
    assert out["class_logits"].shape == (1, 5, 32, 32)
    assert out["regression"].shape == (1, 1, 32, 32)


def test_regression_only_head():
    model = UNet(in_channels=4, base_channels=8, depth=2,
                 n_classes=0, regression=True)
    x = torch.randn(1, 4, 32, 32)
    out = model(x)
    assert "class_logits" not in out
    assert out["regression"].shape == (1, 1, 32, 32)


def test_odd_size_preserved_by_padding():
    # non-power-of-two spatial size must round-trip through down/up sampling
    model = UNet(in_channels=3, base_channels=8, depth=3,
                 n_classes=2, regression=False)
    x = torch.randn(1, 3, 50, 70)
    out = model(x)
    assert out["class_logits"].shape == (1, 2, 50, 70)


def test_resolve_device_returns_valid():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"
    auto = resolve_device("auto")
    assert auto.type in ("cuda", "mps", "cpu")
