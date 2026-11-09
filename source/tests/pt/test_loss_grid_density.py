# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the grid-density loss transform and MAE anchor options."""

import unittest

import torch

from deepmd.pt.loss.charge import (
    GridDensityLoss,
)
from deepmd.pt.utils import (
    env,
)

device = env.DEVICE


class _FakeModel(torch.nn.Module):
    """Return a fixed density prediction regardless of the input dict."""

    def __init__(self, pred: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.pred = pred
        self.mask = mask

    def forward(self, **kwargs) -> dict:
        del kwargs
        out = {"density": self.pred}
        if self.mask is not None:
            out["mask"] = self.mask
        return out


def _run_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor | None = None,
    **loss_kwargs,
) -> tuple[torch.Tensor, dict]:
    loss_fn = GridDensityLoss(
        starter_learning_rate=1.0,
        start_pref_d=1.0,
        limit_pref_d=1.0,
        **loss_kwargs,
    )
    model = _FakeModel(pred, mask)
    _, loss, more_loss = loss_fn({}, model, {"density": label}, 2, 1.0)
    return loss, more_loss


class TestGridDensityLossTransform(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        # one frame, eight grid points spanning four orders of magnitude
        self.label = torch.tensor(
            [[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 200.0]],
            dtype=torch.float64,
            device=device,
        )
        self.pred = (self.label * 1.1).detach().clone().requires_grad_(True)

    def test_default_matches_plain_mse(self) -> None:
        loss, more = _run_loss(self.pred, self.label)
        ref = torch.square(self.label - self.pred).mean()
        torch.testing.assert_close(loss, ref.reshape(()))
        # display metrics stay in raw space
        torch.testing.assert_close(more["rmse_d"], ref.sqrt().reshape(()))
        torch.testing.assert_close(
            more["mae_d"], (self.label - self.pred).abs().mean().reshape(())
        )

    def test_asinh_computes_transformed_mse(self) -> None:
        scale = 0.5
        loss, more = _run_loss(
            self.pred, self.label, transform="asinh", asinh_scale=scale
        )
        ref = torch.square(
            torch.asinh(self.label / scale) - torch.asinh(self.pred / scale)
        ).mean()
        torch.testing.assert_close(loss, ref.reshape(()))
        # displayed metrics are still raw-space
        raw = torch.square(self.label - self.pred).mean()
        torch.testing.assert_close(more["rmse_d"], raw.sqrt().reshape(()))

    def test_asinh_balances_gradient_across_scales(self) -> None:
        # raw MSE: the per-point gradient grows linearly with rho, so the
        # high-density point dominates; asinh: the gradient is damped like
        # 1/rho at high density, so the two points become comparable
        pred = torch.tensor([[100.0, 0.01]], dtype=torch.float64, device=device)
        label = torch.tensor([[90.0, 0.009]], dtype=torch.float64, device=device)

        pred_raw = pred.clone().requires_grad_(True)
        loss_raw, _ = _run_loss(pred_raw, label)
        loss_raw.backward()

        pred_asinh = pred.clone().requires_grad_(True)
        loss_asinh, _ = _run_loss(pred_asinh, label, transform="asinh", asinh_scale=1.0)
        loss_asinh.backward()

        ratio_raw = (pred_raw.grad[0, 0] / pred_raw.grad[0, 1]).item()
        ratio_asinh = (pred_asinh.grad[0, 0] / pred_asinh.grad[0, 1]).item()
        # 100x larger density -> ~100x gradient ratio for raw MSE
        self.assertGreater(ratio_raw, 1000.0)
        # damped to order unity for asinh
        self.assertLess(ratio_asinh, 200.0)

    def test_mae_anchor_adds_raw_mae(self) -> None:
        loss_plain, _ = _run_loss(self.pred, self.label)
        loss_anchor, _ = _run_loss(self.pred, self.label, mae_weight=0.5)
        mae = (self.label - self.pred).abs().mean()
        torch.testing.assert_close(loss_anchor, (loss_plain + 0.5 * mae).reshape(()))

    def test_mask_excludes_points(self) -> None:
        mask = torch.ones(1, 8, dtype=torch.int32, device=device)
        mask[0, -2:] = 0
        loss, _ = _run_loss(self.pred, self.label, mask=mask)
        ref = torch.square((self.label - self.pred)[:, :6]).mean()
        torch.testing.assert_close(loss, ref.reshape(()))

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "transform"):
            GridDensityLoss(transform="log")
        with self.assertRaisesRegex(ValueError, "asinh_scale"):
            GridDensityLoss(transform="asinh", asinh_scale=0.0)
        with self.assertRaisesRegex(ValueError, "mae_weight"):
            GridDensityLoss(mae_weight=-1.0)

    def test_serialization_roundtrip(self) -> None:
        loss_fn = GridDensityLoss(
            start_pref_d=1.0,
            limit_pref_d=1.0,
            transform="asinh",
            asinh_scale=0.3,
            mae_weight=0.2,
        )
        loss2 = GridDensityLoss.deserialize(loss_fn.serialize())
        self.assertEqual(loss2.transform, "asinh")
        self.assertAlmostEqual(loss2.asinh_scale, 0.3)
        self.assertAlmostEqual(loss2.mae_weight, 0.2)
        # legacy serialization without the new keys still loads
        legacy = {
            "@class": "GridDensityLoss",
            "@version": 1,
            "starter_learning_rate": 1.0,
            "start_pref_d": 1.0,
            "limit_pref_d": 1.0,
        }
        loss3 = GridDensityLoss.deserialize(legacy)
        self.assertEqual(loss3.transform, "none")
        self.assertAlmostEqual(loss3.mae_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
