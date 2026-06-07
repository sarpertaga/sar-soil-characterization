import numpy as np
import pytest

pytest.importorskip("torch")  # soilgeo-dl env only

from soilgeo.models.train import (  # noqa: E402
    aggregate_to_scale,
    mean_iou,
    per_class_f1,
    train_one_epoch,
)


def test_mean_iou_perfect_prediction():
    import torch
    target = torch.tensor([[0, 1], [2, 1]])
    pred = target.clone()
    miou = mean_iou(pred, target, n_classes=3)
    assert abs(miou - 1.0) < 1e-6


def test_mean_iou_partial():
    import torch
    target = torch.tensor([[0, 0], [1, 1]])
    pred = torch.tensor([[0, 0], [0, 1]])  # one of the class-1 pixels wrong
    miou = mean_iou(pred, target, n_classes=2)
    assert 0.0 < miou < 1.0


def test_per_class_f1_perfect():
    import torch
    target = torch.tensor([0, 1, 2, 1, 0])
    pred = target.clone()
    f1 = per_class_f1(pred, target, n_classes=3)
    assert len(f1) == 3
    assert all(abs(v - 1.0) < 1e-6 for v in f1)


def test_aggregate_to_scale_block_mean():
    # 4x4 -> factor 2 -> 2x2 block means
    arr = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ], dtype=np.float32)
    agg = aggregate_to_scale(arr, factor=2, nodata=-9999.0)
    assert agg.shape == (2, 2)
    np.testing.assert_allclose(agg, [[1, 2], [3, 4]])


def test_aggregate_ignores_nodata():
    arr = np.array([[1, -9999.0], [3, 3]], dtype=np.float32)
    agg = aggregate_to_scale(arr, factor=2, nodata=-9999.0)
    # mean of valid {1,3,3} = 2.333...
    assert abs(float(agg[0, 0]) - (7.0 / 3.0)) < 1e-4


def test_train_one_epoch_reduces_loss_on_overfit():
    import torch

    from soilgeo.models.unet import UNet

    torch.manual_seed(0)
    model = UNet(in_channels=3, base_channels=8, depth=2, n_classes=0, regression=True)
    # one tiny fixed batch the model should overfit
    x = torch.randn(2, 3, 16, 16)
    y = torch.randn(2, 1, 16, 16)
    batch = [(x, y)]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = train_one_epoch(model, batch, opt, task="regression", device=torch.device("cpu"))
    for _ in range(15):
        last = train_one_epoch(model, batch, opt, task="regression", device=torch.device("cpu"))
    assert last < first
