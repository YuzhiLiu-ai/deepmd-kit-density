# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DPA4-LR latent charge/dipole supervision.

NOTE: source/tests/pt/__init__.py sets torch.set_default_device("cuda:9999999")
to catch tests that forget to specify device.  All tensor creations here
must pass device="cpu" explicitly.
"""

import unittest

import torch

from deepmd.pt.loss import (
    EnergyLatentLoss,
)


class _LatentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.charge = torch.nn.Parameter(
            torch.tensor([[[1.0], [2.0], [100.0]]], dtype=torch.float64, device="cpu")
        )
        self.dipole = torch.nn.Parameter(
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [100.0, 100.0, 100.0]]],
                dtype=torch.float64,
                device="cpu",
            )
        )

    def forward(self, **kwargs):
        del kwargs
        return {
            "latent_charge": self.charge,
            "latent_dipole": self.dipole,
            "mask": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64, device="cpu"),
        }


class TestEnergyLatentLoss(unittest.TestCase):
    def test_masked_loss_and_gradients(self) -> None:
        model = _LatentModel()
        loss_fn = EnergyLatentLoss(
            starter_learning_rate=1.0,
            start_pref_lq=2.0,
            limit_pref_lq=2.0,
            start_pref_ld=3.0,
            limit_pref_ld=3.0,
        )
        labels = {
            "atom_charge": torch.zeros(1, 3, 1, dtype=torch.float64, device="cpu"),
            "atom_dipole": torch.zeros(1, 3, 3, dtype=torch.float64, device="cpu"),
            "find_atom_charge": 1.0,
            "find_atom_dipole": 1.0,
        }
        _, loss, more_loss = loss_fn(
            input_dict={},
            model=model,
            label=labels,
            natoms=3,
            learning_rate=1.0,
        )

        # q MSE = (1^2 + 2^2) / 2 = 2.5; p MSE = (1^2 + 2^2) / 6.
        self.assertAlmostEqual(loss.item(), 7.5)
        self.assertAlmostEqual(more_loss["rmse_latent_charge"].item(), 2.5**0.5)
        self.assertAlmostEqual(
            more_loss["rmse_latent_dipole"].item(), (5.0 / 6.0) ** 0.5
        )

        loss.backward()
        torch.testing.assert_close(
            model.charge.grad,
            torch.tensor([[[2.0], [4.0], [0.0]]], dtype=torch.float64, device="cpu"),
        )
        torch.testing.assert_close(
            model.dipole.grad,
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]],
                dtype=torch.float64,
                device="cpu",
            ),
        )

    def test_label_requirements_and_serialization(self) -> None:
        loss_fn = EnergyLatentLoss(
            starter_learning_rate=1.0,
            start_pref_lq=1.0,
            limit_pref_lq=1.0,
            start_pref_ld=1.0,
            limit_pref_ld=1.0,
        )
        requirements = {item.key: item for item in loss_fn.label_requirement}
        self.assertEqual(requirements["atom_charge"].ndof, 1)
        self.assertTrue(requirements["atom_charge"].atomic)
        self.assertEqual(requirements["atom_dipole"].ndof, 3)
        self.assertTrue(requirements["atom_dipole"].atomic)

        restored = EnergyLatentLoss.deserialize(loss_fn.serialize())
        self.assertEqual(restored.start_pref_lq, 1.0)
        self.assertEqual(restored.limit_pref_ld, 1.0)

    def test_zero_prefactors_degrade_to_energy_loss(self) -> None:
        loss_fn = EnergyLatentLoss(starter_learning_rate=1.0)
        self.assertFalse(loss_fn.has_lq)
        self.assertFalse(loss_fn.has_ld)
        requirement_keys = {item.key for item in loss_fn.label_requirement}
        self.assertNotIn("atom_charge", requirement_keys)
        self.assertNotIn("atom_dipole", requirement_keys)

        # Latent outputs and labels are ignored: no latent term enters the
        # loss and no latent metric is reported, exactly like the plain
        # energy loss used before latent supervision existed.
        model = _LatentModel()
        labels = {
            "atom_charge": torch.zeros(1, 3, 1, dtype=torch.float64, device="cpu"),
            "atom_dipole": torch.zeros(1, 3, 3, dtype=torch.float64, device="cpu"),
            "find_atom_charge": 1.0,
            "find_atom_dipole": 1.0,
        }
        _, loss, more_loss = loss_fn(
            input_dict={},
            model=model,
            label=labels,
            natoms=3,
            learning_rate=1.0,
        )
        self.assertNotIn("rmse_latent_charge", more_loss)
        self.assertNotIn("rmse_latent_dipole", more_loss)
        # The parent energy loss finds no energy/force labels here either,
        # so with every prefactor at zero the total loss is exactly zero.
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
