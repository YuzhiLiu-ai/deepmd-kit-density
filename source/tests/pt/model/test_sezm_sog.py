# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the SeZM/DPA4 SOG-LES long-range model (non-periodic)."""

import math
import unittest

import numpy as np
import torch
from packaging.version import parse as parse_version

from deepmd.pt.loss import (
    EnergyLatentLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.sezm_sog_model import (
    E2_PER_ANGSTROM_TO_EV,
    SeZMSOGModel,
    _SogLRKernel,
    _les_potential_and_field,
    _les_radial_coefficients,
)
from deepmd.pt.model.task.sog_energy_fitting import (
    SOGEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.compile_compat import (
    SUPPORTED_COMPILE_TORCH,
)

_SQRT_PI = math.sqrt(math.pi)

_TORCH_VERSION = parse_version(torch.__version__)
_SKIP_OFF_COMPILE_TORCH = (
    _TORCH_VERSION.major,
    _TORCH_VERSION.minor,
) not in SUPPORTED_COMPILE_TORCH
_SKIP_OFF_COMPILE_TORCH_REASON = (
    "SeZM's torch.compile path is only supported on torch "
    + ", ".join(f"{major}.{minor}.x" for major, minor in SUPPORTED_COMPILE_TORCH)
    + f"; current torch is {torch.__version__}."
)


def _les_kernel_value(alpha: float, r: float) -> float:
    return math.erf(alpha * r) / r


def _les_kernel_deriv(alpha: float, r: float) -> float:
    return (2.0 * alpha / _SQRT_PI) * math.exp(-(alpha**2) * r * r) / r - math.erf(
        alpha * r
    ) / (r * r)


class TestSogLRKernelMath(unittest.TestCase):
    """Hand-computed checks of the non-periodic SOG/LES kernels."""

    def setUp(self) -> None:
        self.dtype = torch.float64
        self.device = env.DEVICE

    def _two_atom(self, d: float) -> torch.Tensor:
        return torch.tensor(
            [[[0.0, 0.0, 0.0], [d, 0.0, 0.0]]], dtype=self.dtype, device=self.device
        )

    def test_les_two_body(self) -> None:
        d, alpha = 2.3, 0.7
        q = torch.tensor([[[1.5], [-0.4]]], dtype=self.dtype, device=self.device)
        kernel = _SogLRKernel(lr_kernel="les")
        e = kernel(
            self._two_atom(d),
            q,
            amp=torch.zeros(1, dtype=self.dtype, device=self.device),
            bandwidth=torch.ones(1, dtype=self.dtype, device=self.device),
            les_alpha=torch.tensor(alpha, dtype=self.dtype, device=self.device),
        )
        expected = (
            E2_PER_ANGSTROM_TO_EV * q[0, 0, 0].item() * q[0, 1, 0].item()
        ) * _les_kernel_value(alpha, d)
        self.assertAlmostEqual(e.item(), expected, places=12)

    def test_sog_two_body(self) -> None:
        # ``sog`` mode: SOG Gaussian sum only (no LES term).
        d = 1.9
        amp = torch.tensor([2.0, 3.0], dtype=self.dtype, device=self.device)
        bw = torch.tensor([1.0, 2.0], dtype=self.dtype, device=self.device)
        q = torch.tensor([[[1.2], [0.7]]], dtype=self.dtype, device=self.device)
        q1, q2 = q[0, 0, 0].item(), q[0, 1, 0].item()

        for remove_self, expect_self in ((False, True), (True, False)):
            kernel = _SogLRKernel(lr_kernel="sog", remove_self_interaction=remove_self)
            e = kernel(
                self._two_atom(d),
                q,
                amp=amp,
                bandwidth=bw,
                les_alpha=torch.tensor(0.5, dtype=self.dtype, device=self.device),
            )
            gauss = sum(
                a.item() * math.exp(-0.5 * d * d / (b.item() ** 2))
                for a, b in zip(amp, bw, strict=True)
            )
            expected = q1 * q2 * gauss
            if expect_self:
                expected += 0.5 * (q1 * q1 + q2 * q2) * amp.sum().item()
            expected *= E2_PER_ANGSTROM_TO_EV
            self.assertAlmostEqual(e.item(), expected, places=10)

    def test_les_dipole_axial(self) -> None:
        # Atoms on the x-axis. Dipoles along z: p.r_hat = 0, so only the
        # (K'/r)(p_i.p_j) dipole-dipole part and the self-polarization survive.
        d, alpha, apol = 2.1, 0.6, 1.7
        q = torch.tensor([[[0.8], [-0.3]]], dtype=self.dtype, device=self.device)
        p = torch.tensor(
            [[[0.0, 0.0, 0.5], [0.0, 0.0, -0.2]]],
            dtype=self.dtype,
            device=self.device,
        )
        kernel = _SogLRKernel(lr_kernel="dipole")
        e = kernel(
            self._two_atom(d),
            q,
            amp=torch.zeros(1, dtype=self.dtype, device=self.device),
            bandwidth=torch.ones(1, dtype=self.dtype, device=self.device),
            les_alpha=torch.tensor(alpha, dtype=self.dtype, device=self.device),
            latent_dipoles=p,
            alpha_pol_atom=torch.full(
                (1, 2), apol, dtype=self.dtype, device=self.device
            ),
        )
        kp = _les_kernel_deriv(alpha, d)
        p1_dot_p2 = 0.5 * (-0.2)
        e_pp = -(kp / d) * p1_dot_p2
        e_pol = (0.5**2 + 0.2**2) / (2.0 * apol)
        e_qq = 0.8 * (-0.3) * _les_kernel_value(alpha, d)
        expected = E2_PER_ANGSTROM_TO_EV * (e_qq + e_pp + e_pol)
        self.assertAlmostEqual(e.item(), expected, places=10)

    def test_les_dipole_charge_dipole(self) -> None:
        # p1 along x (the bond direction), p2 = 0: E_qp = q2 * (p1.r_hat) * K'(d).
        d, alpha = 1.7, 0.9
        q = torch.tensor([[[1.1], [0.6]]], dtype=self.dtype, device=self.device)
        p = torch.tensor(
            [[[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=self.dtype,
            device=self.device,
        )
        kernel = _SogLRKernel(lr_kernel="dipole")
        e = kernel(
            self._two_atom(d),
            q,
            amp=torch.zeros(1, dtype=self.dtype, device=self.device),
            bandwidth=torch.ones(1, dtype=self.dtype, device=self.device),
            les_alpha=torch.tensor(alpha, dtype=self.dtype, device=self.device),
            latent_dipoles=p,
        )
        e_qq = 1.1 * 0.6 * _les_kernel_value(alpha, d)
        e_qp = 0.6 * 0.4 * _les_kernel_deriv(alpha, d)
        expected = E2_PER_ANGSTROM_TO_EV * (e_qq + e_qp)
        self.assertAlmostEqual(e.item(), expected, places=10)

    def test_les_dipole_self_pol_flag(self) -> None:
        # Self-polarization is a positive term; turning it off lowers energy.
        d, alpha, apol = 2.1, 0.6, 1.7
        q = torch.tensor([[[0.8], [-0.3]]], dtype=self.dtype, device=self.device)
        p = torch.tensor(
            [[[0.0, 0.0, 0.5], [0.0, 0.0, -0.2]]],
            dtype=self.dtype,
            device=self.device,
        )
        args = {
            "positions": self._two_atom(d),
            "latent_charges": q,
            "amp": torch.zeros(1, dtype=self.dtype, device=self.device),
            "bandwidth": torch.ones(1, dtype=self.dtype, device=self.device),
            "les_alpha": torch.tensor(alpha, dtype=self.dtype, device=self.device),
            "latent_dipoles": p,
        }
        kernel_with = _SogLRKernel(lr_kernel="dipole")
        e_with = kernel_with(
            **args,
            alpha_pol_atom=torch.full(
                (1, 2), apol, dtype=self.dtype, device=self.device
            ),
        )
        kernel_without = _SogLRKernel(lr_kernel="dipole")
        e_without = kernel_without(**args, alpha_pol_atom=None)
        self.assertGreater(e_with.item(), e_without.item())

    def test_les_radial_coefficients_are_finite_at_zero(self) -> None:
        r_sq = torch.tensor(
            [0.0, 1e-24, 1e-12, 1e-4, 1.0],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        alpha = torch.tensor(
            0.8, dtype=torch.float32, device=self.device, requires_grad=True
        )
        coeffs = _les_radial_coefficients(r_sq, alpha)
        loss = sum(value.sum() for value in coeffs)
        grad_r, grad_alpha = torch.autograd.grad(loss, (r_sq, alpha))
        for value in (*coeffs, grad_r, grad_alpha):
            self.assertTrue(torch.isfinite(value).all().item())

    def test_les_dipole_coincident_atoms_float32_is_finite(self) -> None:
        positions = torch.zeros(
            (1, 2, 3), dtype=torch.float32, device=self.device, requires_grad=True
        )
        q = torch.tensor(
            [[[0.4], [-0.3]]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        p = torch.tensor(
            [[[0.2, -0.1, 0.3], [-0.1, 0.4, 0.2]]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        alpha = torch.tensor(
            0.8, dtype=torch.float32, device=self.device, requires_grad=True
        )
        energy = _SogLRKernel(lr_kernel="dipole")(
            positions,
            q,
            amp=torch.zeros(1, dtype=torch.float32, device=self.device),
            bandwidth=torch.ones(1, dtype=torch.float32, device=self.device),
            les_alpha=alpha,
            latent_dipoles=p,
        ).sum()
        gradients = torch.autograd.grad(energy, (positions, q, p, alpha))
        self.assertTrue(torch.isfinite(energy).all().item())
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all().item())

    def test_les_field_matches_multipole_energy_derivatives(self) -> None:
        positions = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.2, -0.3, 0.4], [-0.2, 1.1, 0.7]]],
            dtype=self.dtype,
            device=self.device,
        )
        q = torch.tensor(
            [[[0.3], [-0.2], [0.1]]],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        p = torch.tensor(
            [[[0.1, -0.2, 0.3], [0.2, 0.1, -0.1], [-0.3, 0.1, 0.2]]],
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        alpha = torch.tensor(0.7, dtype=self.dtype, device=self.device)
        kernel = _SogLRKernel(lr_kernel="dipole")
        energy = (
            kernel(
                positions,
                q,
                amp=torch.zeros(1, dtype=self.dtype, device=self.device),
                bandwidth=torch.ones(1, dtype=self.dtype, device=self.device),
                les_alpha=alpha,
                latent_dipoles=p,
            ).sum()
            / E2_PER_ANGSTROM_TO_EV
        )
        grad_q, grad_p = torch.autograd.grad(energy, (q, p))

        nloc = positions.shape[1]
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(2)
        eye = 1.0 - torch.eye(nloc, dtype=self.dtype, device=self.device).unsqueeze(0)
        potential, field = _les_potential_and_field(r_ij, eye, q, p, alpha)
        np.testing.assert_allclose(
            potential.detach().cpu().numpy(),
            grad_q.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            field.detach().cpu().numpy(),
            grad_p.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )


class TestSeZMSOGModel(unittest.TestCase):
    """End-to-end tests of SeZMSOGModel on tiny non-periodic clusters."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)
        # Two frames, seven atoms, no box (non-periodic).
        self.coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.3, 0.0],
                    [0.2, 1.5, 0.4],
                    [1.7, 1.2, 0.2],
                    [2.3, 0.1, 1.0],
                    [0.8, 2.2, 1.1],
                    [2.6, 1.8, 1.5],
                ],
                [
                    [0.3, 0.2, 0.1],
                    [1.0, 0.5, 0.2],
                    [0.4, 1.3, 0.6],
                    [1.9, 1.0, 0.1],
                    [2.1, 0.4, 1.2],
                    [0.6, 2.4, 0.9],
                    [2.4, 1.6, 1.7],
                ],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1]],
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _randomize_params(model: torch.nn.Module, seed: int = 1234) -> None:
        """Fill all parameters with small random values.

        The DPA4/SeZM descriptor zero-initializes its output projections, so
        a freshly built model sits at a degenerate point where the descriptor
        is coordinate-independent; randomization makes finite-difference
        force checks meaningful.
        """
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
            # bandwidth / les_alpha are positive by contract (squared on use).
            fitting = getattr(model.atomic_model, "fitting_net", None)
            if isinstance(fitting, SOGEnergyFittingNet):
                fitting.bandwidth.abs_()
                fitting.les_alpha.abs_()

    def _model_params(self, fitting_extra: dict) -> dict:
        return {
            "type": "dpa4_lr",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "dpa4",
                "sel": 6,
                "rcut": 4.0,
                "channels": 8,
                "n_radial": 4,
                "lmax": 2,
                "mmax": 1,
                "n_blocks": 2,
                "so2_layers": 1,
                "n_atten_head": 1,
                "use_amp": False,
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "type": "sog_energy",
                "neuron": [16],
                "neuron_lr": [8],
                "precision": "float64",
                "seed": 7,
                **fitting_extra,
            },
        }

    def _fd_force(self, model: torch.nn.Module, eps: float = 1e-6) -> torch.Tensor:
        """Central-difference force from the model energy."""
        nf, nloc, _ = self.coord.shape
        fd = torch.zeros_like(self.coord)
        model.eval()
        for i in range(nloc):
            for k in range(3):
                dc = torch.zeros_like(self.coord)
                dc[:, i, k] = eps
                e_plus = model.forward(self.coord + dc, self.atype)["energy"]
                e_minus = model.forward(self.coord - dc, self.atype)["energy"]
                fd[:, i, k] = -(e_plus - e_minus).squeeze(-1) / (2.0 * eps)
        return fd

    def _check_force_virial(self, fitting_extra: dict, tag: str) -> None:
        model = get_model(self._model_params(fitting_extra))
        assert isinstance(model, SeZMSOGModel)
        self._randomize_params(model)
        model.eval()
        ret = model.forward(self.coord, self.atype)
        force = ret["force"]
        virial = ret["virial"]
        self.assertIn("latent_charge", ret)
        self.assertEqual(ret["latent_charge"].shape, (*self.atype.shape, 1))
        if "dipole" in tag:
            self.assertIn("latent_dipole", ret)
            self.assertEqual(ret["latent_dipole"].shape, (*self.atype.shape, 3))

        # Finite-difference force check (covers the dedicated coord-autograd
        # branch of the long-range correction).
        fd = self._fd_force(model)
        np.testing.assert_allclose(
            force.detach().cpu().numpy(),
            fd.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"[{tag}] force mismatch",
        )

        # Virial consistency: for a conservative, translationally invariant
        # potential, W == sum_i F_i (x) r_i.  This pins the sign convention of
        # the long-range virial to the edge-path virial.
        vir_from_force = torch.einsum("fni,fnj->fij", force, self.coord).reshape(-1, 9)
        np.testing.assert_allclose(
            virial.detach().cpu().numpy(),
            vir_from_force.detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"[{tag}] virial mismatch",
        )

    def test_forward_les(self) -> None:
        self._check_force_virial({"lr_kernel": "les"}, "les")

    def test_forward_sog(self) -> None:
        self._check_force_virial({"lr_kernel": "sog"}, "sog")

    def test_forward_sog_keep_self(self) -> None:
        self._check_force_virial(
            {"lr_kernel": "sog", "remove_self_interaction": False},
            "sog+self",
        )

    def test_forward_les_dipole(self) -> None:
        self._check_force_virial(
            {
                "lr_kernel": "dipole",
                "alpha_pol": [1.0, 0.5],
            },
            "dipole",
        )

    def test_forward_iterative_les_dipole(self) -> None:
        self._check_force_virial(
            {
                "lr_kernel": "dipole",
                "use_self_pol": False,
                "use_charge_constraint": True,
                "lr_update_steps": 2,
                "lr_update_neuron": [8, 8],
                "lr_update_damping": 0.2,
            },
            "iterative-dipole",
        )

    def test_force_loss_backward_reaches_lr_params(self) -> None:
        model = get_model(
            self._model_params({"lr_kernel": "dipole", "alpha_pol": [1.0, 0.5]})
        )
        self._randomize_params(model)
        model.train()
        ret = model.forward(self.coord, self.atype)
        loss = ret["energy"].sum() + 0.1 * (ret["force"] ** 2).sum()
        loss.backward()
        fitting = model.atomic_model.fitting_net
        assert isinstance(fitting, SOGEnergyFittingNet)
        self.assertIsNotNone(fitting.les_alpha.grad)
        self.assertIsNotNone(fitting.alpha_pol.grad)
        dipole_grads = [
            p.grad for p in fitting.dipole_head.parameters() if p.grad is not None
        ]
        self.assertTrue(len(dipole_grads) > 0)
        # LR branch network receives gradients through both q and p.
        lr_weight_grads = [
            p.grad for p in fitting.filter_layers_lr.parameters() if p.grad is not None
        ]
        self.assertTrue(len(lr_weight_grads) > 0)

    def test_force_loss_backward_reaches_polar_update(self) -> None:
        model = get_model(
            self._model_params(
                {
                    "lr_kernel": "dipole",
                    "use_self_pol": False,
                    "use_charge_constraint": True,
                    "lr_update_steps": 2,
                    "lr_update_neuron": [8],
                }
            )
        )
        self._randomize_params(model)
        model.train()
        ret = model.forward(self.coord, self.atype)
        loss = ret["energy"].sum() + 0.1 * (ret["force"] ** 2).sum()
        loss.backward()
        fitting = model.atomic_model.fitting_net
        assert isinstance(fitting, SOGEnergyFittingNet)
        assert fitting.polar_update_net is not None
        update_grads = [
            param.grad
            for param in fitting.polar_update_net.parameters()
            if param.grad is not None
        ]
        self.assertTrue(update_grads)
        self.assertTrue(all(torch.isfinite(grad).all().item() for grad in update_grads))

    def test_latent_loss_backward_reaches_electrostatic_heads(self) -> None:
        model = get_model(
            self._model_params(
                {
                    "lr_kernel": "dipole",
                    "use_charge_constraint": True,
                    "lr_update_steps": 2,
                    "lr_update_neuron": [8],
                }
            )
        )
        self._randomize_params(model)
        model.train()
        loss_fn = EnergyLatentLoss(
            starter_learning_rate=1.0,
            start_pref_lq=1.0,
            limit_pref_lq=1.0,
            start_pref_ld=1.0,
            limit_pref_ld=1.0,
        )
        labels = {
            "atom_charge": torch.zeros(
                (*self.atype.shape, 1), dtype=self.coord.dtype, device=self.device
            ),
            "atom_dipole": torch.zeros(
                (*self.atype.shape, 3), dtype=self.coord.dtype, device=self.device
            ),
            "find_atom_charge": 1.0,
            "find_atom_dipole": 1.0,
        }
        _, loss, _ = loss_fn(
            input_dict={"coord": self.coord, "atype": self.atype},
            model=model,
            label=labels,
            natoms=self.atype.shape[1],
            learning_rate=1.0,
        )
        loss.backward()

        fitting = model.atomic_model.fitting_net
        assert isinstance(fitting, SOGEnergyFittingNet)
        assert fitting.dipole_head is not None
        assert fitting.polar_update_net is not None
        for module in (
            fitting.filter_layers_lr,
            fitting.dipole_head,
            fitting.polar_update_net,
        ):
            grads = [
                param.grad for param in module.parameters() if param.grad is not None
            ]
            self.assertTrue(grads)
            self.assertTrue(
                all(torch.isfinite(grad).all().item() for grad in grads)
            )

    def test_serialize_deserialize_fitting(self) -> None:
        model = get_model(
            self._model_params({"lr_kernel": "dipole", "alpha_pol": [1.0, 0.5]})
        )
        self._randomize_params(model)
        model.eval()
        e1 = model.forward(self.coord, self.atype)["energy"]
        fitting = model.atomic_model.fitting_net
        data = fitting.serialize()
        fitting2 = SOGEnergyFittingNet.deserialize(data)
        model.atomic_model.fitting_net = fitting2
        e2 = model.forward(self.coord, self.atype)["energy"]
        np.testing.assert_allclose(
            e1.detach().cpu().numpy(),
            e2.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_self_pol_flag(self) -> None:
        # With use_self_pol=False the self-polarization term is dropped, so the
        # total energy should be lower (the term is positive).
        params = self._model_params(
            {"lr_kernel": "dipole", "alpha_pol": 0.5, "use_self_pol": True}
        )
        model_on = get_model(params)
        self._randomize_params(model_on, seed=2025)
        model_on.eval()
        e_on = model_on.forward(self.coord, self.atype)["energy"]

        params["fitting_net"]["use_self_pol"] = False
        model_off = get_model(params)
        self._randomize_params(model_off, seed=2025)
        model_off.eval()
        e_off = model_off.forward(self.coord, self.atype)["energy"]

        self.assertTrue(
            (e_off < e_on).all().item(),
            msg="turning off self-polarization should lower the energy",
        )

    def test_charge_constraint(self) -> None:
        fitting = SOGEnergyFittingNet(
            ntypes=2,
            dim_descrpt=8,
            neuron=[8],
            dim_out_lr=1,
            neuron_lr=[8],
            numb_fparam=1,
            use_charge_constraint=True,
            lr_kernel="les",
            precision="float64",
            seed=3,
        )
        nf, nloc = 2, 5
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=self.device)
        atype = torch.tensor(
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.long, device=self.device
        )
        target_q = torch.tensor([0.5, -1.0], dtype=torch.float64, device=self.device)
        out = fitting(descriptor, atype, fparam=target_q.view(nf, 1))
        q_sum = out["latent_charge"].sum(dim=1).squeeze(-1)
        np.testing.assert_allclose(
            q_sum.detach().cpu().numpy(),
            target_q.cpu().numpy(),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_charge_constraint_via_charge_spin(self) -> None:
        fitting = SOGEnergyFittingNet(
            ntypes=2,
            dim_descrpt=8,
            neuron=[8],
            dim_out_lr=1,
            neuron_lr=[8],
            use_charge_constraint=True,
            lr_kernel="les",
            precision="float64",
            seed=3,
        )
        nf, nloc = 2, 5
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=self.device)
        atype = torch.tensor(
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.long, device=self.device
        )
        charge_spin = torch.tensor(
            [[0.5, 1.0], [-1.0, 2.0]], dtype=torch.float64, device=self.device
        )
        out = fitting(descriptor, atype, charge_spin=charge_spin)
        q_sum = out["latent_charge"].sum(dim=1).squeeze(-1)
        np.testing.assert_allclose(
            q_sum.detach().cpu().numpy(),
            charge_spin[:, 0].cpu().numpy(),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_polar_update_charge_constraint_and_equivariance(self) -> None:
        fitting = SOGEnergyFittingNet(
            ntypes=2,
            dim_descrpt=8,
            neuron=[8],
            neuron_lr=[8],
            use_charge_constraint=True,
            use_self_pol=False,
            lr_kernel="dipole",
            dipole_lmax=1,
            dipole_channels=2,
            lr_update_steps=2,
            lr_update_neuron=[8],
            precision="float64",
            seed=3,
        )
        assert fitting.polar_update_net is not None
        with torch.no_grad():
            for param in fitting.polar_update_net.parameters():
                param.copy_(0.1 * torch.randn_like(param))
        nf, nloc = 2, 4
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=self.device)
        q = torch.randn(nf, nloc, 1, dtype=torch.float64, device=self.device)
        p = torch.randn(nf, nloc, 3, dtype=torch.float64, device=self.device)
        p0 = torch.randn_like(p)
        potential = torch.randn_like(q)
        field = torch.randn_like(p)
        mask = torch.ones(nf, nloc, dtype=torch.bool, device=self.device)
        target = torch.tensor(
            [[[0.5]], [[-1.0]]], dtype=torch.float64, device=self.device
        )
        q1, p1 = fitting.update_polar_state(
            descriptor, q, p, p0, potential, field, mask, target
        )
        np.testing.assert_allclose(
            q1.sum(dim=1, keepdim=True).detach().cpu().numpy(),
            target.cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )

        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=self.device,
        )

        def rotate(vector: torch.Tensor) -> torch.Tensor:
            return torch.einsum("ij,fnj->fni", rotation, vector)

        q2, p2 = fitting.update_polar_state(
            descriptor,
            q,
            rotate(p),
            rotate(p0),
            potential,
            rotate(field),
            mask,
            target,
        )
        np.testing.assert_allclose(
            q2.detach().cpu().numpy(),
            q1.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            p2.detach().cpu().numpy(),
            rotate(p1).detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_polar_update_keeps_fitting_precision(self) -> None:
        fitting = SOGEnergyFittingNet(
            ntypes=2,
            dim_descrpt=8,
            neuron=[8],
            neuron_lr=[8],
            use_self_pol=False,
            lr_kernel="dipole",
            dipole_lmax=1,
            dipole_channels=2,
            lr_update_steps=2,
            lr_update_neuron=[8],
            precision="float32",
            seed=3,
        )
        nf, nloc = 1, 4
        descriptor = torch.randn(nf, nloc, 8, dtype=torch.float64, device=self.device)
        q = torch.randn(nf, nloc, 1, dtype=torch.float64, device=self.device)
        p = torch.randn(nf, nloc, 3, dtype=torch.float64, device=self.device)
        mask = torch.ones(nf, nloc, dtype=torch.bool, device=self.device)
        q_new, p_new = fitting.update_polar_state(
            descriptor,
            q,
            p,
            p,
            torch.randn_like(q),
            torch.randn_like(p),
            mask,
            q.sum(dim=1, keepdim=True),
        )
        self.assertEqual(q_new.dtype, torch.float32)
        self.assertEqual(p_new.dtype, torch.float32)

    def test_polar_update_blockwise_glu_parity(self) -> None:
        fitting = SOGEnergyFittingNet(
            ntypes=2,
            dim_descrpt=8,
            neuron=[8],
            neuron_lr=[8],
            use_self_pol=False,
            lr_kernel="dipole",
            dipole_lmax=1,
            dipole_channels=2,
            lr_update_steps=2,
            lr_update_neuron=[8, 8],
            precision="float64",
            seed=3,
        )
        update_net = fitting.polar_update_net
        assert update_net is not None
        with torch.no_grad():
            for parameter in update_net.parameters():
                parameter.copy_(0.1 * torch.randn_like(parameter))
        blocks = [
            torch.randn(2, 4, 8, dtype=torch.float64, device=self.device),
            *[
                torch.randn(2, 4, 1, dtype=torch.float64, device=self.device)
                for _ in range(5)
            ],
        ]
        expected = update_net(torch.cat(blocks, dim=-1))
        actual = update_net.forward_from_blocks(blocks)
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_serialize_deserialize_iterative_fitting(self) -> None:
        extra = {
            "lr_kernel": "dipole",
            "use_self_pol": False,
            "use_charge_constraint": True,
            "lr_update_steps": 2,
            "lr_update_neuron": [8],
        }
        model = get_model(self._model_params(extra))
        self._randomize_params(model)
        model.eval()
        e1 = model.forward(self.coord, self.atype)["energy"]
        fitting = model.atomic_model.fitting_net
        assert isinstance(fitting, SOGEnergyFittingNet)
        fitting2 = SOGEnergyFittingNet.deserialize(fitting.serialize())
        model.atomic_model.fitting_net = fitting2
        e2 = model.forward(self.coord, self.atype)["energy"]
        np.testing.assert_allclose(
            e1.detach().cpu().numpy(),
            e2.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_error_periodic_box(self) -> None:
        model = get_model(self._model_params({"lr_kernel": "les"}))
        box = torch.eye(3, dtype=torch.float64, device=self.device).reshape(1, 9) * 10
        box = box.repeat(2, 1)
        with self.assertRaises(NotImplementedError):
            model.forward(self.coord, self.atype, box=box)

    def test_error_model_type_without_sog_fitting(self) -> None:
        params = self._model_params({"lr_kernel": "les"})
        params["fitting_net"]["type"] = "dpa4_ener"
        with self.assertRaises(ValueError):
            get_model(params)

    def test_error_les_requires_scalar_charge(self) -> None:
        with self.assertRaises(ValueError):
            SOGEnergyFittingNet(
                ntypes=2,
                dim_descrpt=8,
                dim_out_lr=2,
                lr_kernel="les",
            )

    def test_error_dipole_requires_lmax(self) -> None:
        with self.assertRaises(ValueError):
            SOGEnergyFittingNet(
                ntypes=2,
                dim_descrpt=8,
                lr_kernel="dipole",
            )

    def test_error_unknown_lr_kernel(self) -> None:
        with self.assertRaises(ValueError):
            SOGEnergyFittingNet(
                ntypes=2,
                dim_descrpt=8,
                lr_kernel="sog_les",
            )

    def test_error_polar_update_requires_dipole_kernel(self) -> None:
        with self.assertRaises(ValueError):
            SOGEnergyFittingNet(
                ntypes=2,
                dim_descrpt=8,
                lr_kernel="les",
                lr_update_steps=1,
            )


class TestSeZMSOGModelCompile(unittest.TestCase):
    """Compile-vs-eager parity for the SOG long-range branch.

    The traced ``core_compute`` contains two nested ``autograd.grad`` calls
    (the shared edge-vector backward and the explicit-coordinate long-range
    branch).  This class pins both the forward outputs and the parameter
    gradients of a force-loss step between the two paths.
    """

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)
        self.coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.3, 0.0],
                    [0.2, 1.5, 0.4],
                    [1.7, 1.2, 0.2],
                    [2.3, 0.1, 1.0],
                    [0.8, 2.2, 1.1],
                    [2.6, 1.8, 1.5],
                ]
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1]], dtype=torch.long, device=self.device
        )

    def _model_params(self, fitting_extra: dict, use_compile: bool) -> dict:
        return {
            "type": "dpa4_lr",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "dpa4",
                "sel": 6,
                "rcut": 4.0,
                "channels": 8,
                "n_radial": 4,
                "lmax": 2,
                "mmax": 1,
                "n_blocks": 2,
                "so2_layers": 1,
                "n_atten_head": 1,
                "use_amp": False,
                "precision": "float64",
                "seed": 7,
            },
            "fitting_net": {
                "type": "sog_energy",
                "neuron": [16],
                "neuron_lr": [8],
                "precision": "float64",
                "seed": 7,
                **fitting_extra,
            },
            "use_compile": use_compile,
        }

    def _make_model(self, fitting_extra: dict, use_compile: bool) -> SeZMSOGModel:
        model = get_model(self._model_params(fitting_extra, use_compile))
        assert isinstance(model, SeZMSOGModel)
        TestSeZMSOGModel._randomize_params(model)
        return model

    def _check_compile_parity(self, extra: dict, tag: str) -> SeZMSOGModel:
        m_eager = self._make_model(extra, False)
        m_comp = self._make_model(extra, True)

        # --- eval forward parity ---
        m_eager.eval()
        m_comp.eval()
        r1 = m_eager.forward(self.coord, self.atype)
        r2 = m_comp.forward(self.coord, self.atype)
        for key in ("energy", "force", "virial"):
            np.testing.assert_allclose(
                r1[key].detach().cpu().numpy(),
                r2[key].detach().cpu().numpy(),
                rtol=1e-9,
                atol=1e-11,
                err_msg=f"[{tag}] eval {key}",
            )

        # --- training parity: force loss -> parameter grads ---
        for m in (m_eager, m_comp):
            m.train()
            m.zero_grad()
        losses = []
        for m in (m_eager, m_comp):
            ret = m.forward(self.coord, self.atype)
            loss = ret["energy"].sum() + 0.1 * (ret["force"] ** 2).sum()
            loss.backward()
            losses.append(loss)
        np.testing.assert_allclose(
            losses[0].detach().cpu().numpy(),
            losses[1].detach().cpu().numpy(),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"[{tag}] train loss",
        )
        g1 = {k: v.grad for k, v in m_eager.named_parameters() if v.grad is not None}
        g2 = {k: v.grad for k, v in m_comp.named_parameters() if v.grad is not None}
        self.assertEqual(
            set(g1),
            set(g2),
            msg=f"[{tag}] parameter-grad key mismatch",
        )
        for k in g1:
            np.testing.assert_allclose(
                g1[k].detach().cpu().numpy(),
                g2[k].detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"[{tag}] grad of {k}",
            )
        return m_comp

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_eager_parity_les(self) -> None:
        self._check_compile_parity({"lr_kernel": "les"}, "les")

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_eager_parity_sog(self) -> None:
        self._check_compile_parity({"lr_kernel": "sog"}, "sog")

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_eager_parity_les_dipole(self) -> None:
        self._check_compile_parity(
            {"lr_kernel": "dipole", "alpha_pol": [1.0, 0.5]},
            "dipole",
        )

    @unittest.skipIf(_SKIP_OFF_COMPILE_TORCH, _SKIP_OFF_COMPILE_TORCH_REASON)
    def test_compile_eager_parity_iterative_dipole(self) -> None:
        model = self._check_compile_parity(
            {
                "lr_kernel": "dipole",
                "use_self_pol": False,
                "use_charge_constraint": True,
                "lr_update_steps": 2,
                "lr_update_neuron": [8],
                "precision": "float32",
            },
            "iterative-dipole",
        )
        # Reuse the same dynamic compiled graph with a different nloc.  Polar
        # update views must not retain the first batch's leading strides in the
        # AOTAutograd force-loss backward.
        model.zero_grad()
        ret = model.forward(self.coord[:, :5, :], self.atype[:, :5])
        loss = ret["energy"].sum() + 0.1 * (ret["force"] ** 2).sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()
