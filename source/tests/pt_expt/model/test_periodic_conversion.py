# SPDX-License-Identifier: LGPL-3.0-or-later
"""Non-periodic -> periodic conversion of long-range plugin checkpoints.

The conversion flips only the summation topology (direct all-pairs ->
reciprocal space); the converted model must remain bitwise-compatible with
the original on non-periodic (box=None) frames and converge to it for a
cluster in a large cell.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.long_range.migration import (
    convert_long_range_config_to_periodic,
)
from deepmd.entrypoints.convert_lr_periodic import (
    convert_checkpoint_to_periodic,
)
from deepmd.pt_expt.model.get_model import get_model
from deepmd.pt_expt.utils import env

from .test_long_range import (
    _batch,
    _charge_config,
    _multipole_config,
)


def _model(config):
    return get_model(config).to(env.DEVICE).eval()


class TestConvertLongRangeConfig:
    def test_single_model_config(self) -> None:
        cfg = _multipole_config(steps=1)
        assert not cfg["long_range"].get("periodic", False)
        out = convert_long_range_config_to_periodic(cfg)
        lr = out["long_range"]
        assert lr["periodic"] is True
        assert lr["k_grid_mode"] == "kgrid"
        assert lr["pme_mesh"] is None
        # The input is not mutated, and every other field is untouched.
        assert not cfg["long_range"].get("periodic", False)
        for key, value in cfg["long_range"].items():
            if key in ("periodic", "k_grid_mode", "pme_mesh", "pme_mesh_spacing"):
                continue
            assert out["long_range"][key] == value, key

    def test_multitask_model_dict(self) -> None:
        cfg = {"model_dict": {"A": _multipole_config(steps=1)}}
        out = convert_long_range_config_to_periodic(cfg)
        assert out["model_dict"]["A"]["long_range"]["periodic"] is True

    def test_already_periodic_rejected(self) -> None:
        cfg = _multipole_config(steps=1)
        cfg["long_range"]["periodic"] = True
        with pytest.raises(ValueError, match="already evaluates periodic"):
            convert_long_range_config_to_periodic(cfg)

    def test_no_long_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="no `long_range` section"):
            convert_long_range_config_to_periodic({"descriptor": {}})

    def test_bad_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="k_grid_mode"):
            convert_long_range_config_to_periodic(
                _multipole_config(steps=1), k_grid_mode="ewald"
            )


class TestConvertedModelDispatch:
    """A converted model dispatches on the cell: direct sum without a box,
    reciprocal space with one.
    """

    @staticmethod
    def _pair():
        config = _multipole_config(steps=1)
        original = _model(config)
        converted = _model(convert_long_range_config_to_periodic(config))
        converted.load_state_dict(original.state_dict())
        return original, converted

    def test_box_none_matches_original_bitwise(self) -> None:
        """On non-periodic frames the converted model takes the direct path
        (full pair graph, empty k grid) and reproduces the original model.
        """
        original, converted = self._pair()
        coord, atype = _batch((4, 3), seed=61)
        out_o = original(coord, atype)
        out_c = converted(coord, atype)
        torch.testing.assert_close(
            out_c["energy"], out_o["energy"], rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            out_c["force"], out_o["force"], rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            out_c["latent_charge"], out_o["latent_charge"], rtol=1e-12, atol=1e-12
        )

    def test_large_cell_converges_to_cluster(self) -> None:
        """A cluster centered in a large cell: the periodic (tin-foil) sum
        approaches the direct cluster sum; the residual is the tin-foil
        dipole term ~ 2 pi |M|^2 / (3V), here below 0.02 eV.

        Uses the charge-only plugin: for latent_multipole the direct and
        reciprocal charge-dipole sign conventions differ (a pre-existing
        inconsistency of the direct path), which this test is not meant to
        absorb.
        """
        config = _charge_config()
        original = _model(config)
        converted = _model(convert_long_range_config_to_periodic(config))
        converted.load_state_dict(original.state_dict())
        coord, atype = _batch((5,), seed=63)
        L = 20.0
        box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        e_direct = original(coord, atype)["energy"].flatten()[0]
        e_periodic = converted(coord, atype, box=box)["energy"].flatten()[0]
        assert abs(e_periodic.item() - e_direct.item()) < 0.02, (
            f"cluster {e_direct.item():.6f} vs periodic {e_periodic.item():.6f}"
        )

    def test_extensivity_dimer_crystal(self) -> None:
        """A neutral-dimer 'crystal': doubling the cell must double the
        energy and leave per-atom forces unchanged (k-space self-consistency).
        """
        _, converted = self._pair()
        L = 8.0
        dimer = torch.tensor(
            [[1.0, 1.0, 1.0], [1.8, 1.2, 1.1]], dtype=torch.float64, device=env.DEVICE
        )
        atype1 = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
        box1 = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * L)[None]
        out1 = converted(dimer[None], atype1, box=box1)
        # 2x1x1 supercell: same content translated by one cell vector.
        coord2 = torch.cat(
            [
                dimer,
                dimer
                + torch.tensor([L, 0.0, 0.0], dtype=torch.float64, device=env.DEVICE),
            ]
        )[None]
        atype2 = torch.cat([atype1, atype1], dim=1)
        box2 = box1.clone()
        box2[0, 0, 0] *= 2
        out2 = converted(coord2, atype2, box=box2)
        e1 = out1["energy"].flatten()[0]
        e2 = out2["energy"].flatten()[0]
        torch.testing.assert_close(e2, 2 * e1, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            out2["force"][0, :2], out1["force"][0], rtol=1e-4, atol=1e-6
        )


class TestCheckpointConversion:
    def test_round_trip(self, tmp_path) -> None:
        config = _multipole_config(steps=1)
        model = _model(config)
        ckpt = {
            "model": {
                **model.state_dict(),
                "_extra_state": {"model_params": config},
            }
        }
        src = tmp_path / "in.pt"
        dst = tmp_path / "out.pt"
        torch.save(ckpt, src)
        convert_checkpoint_to_periodic(str(src), str(dst))
        out = torch.load(dst, map_location="cpu", weights_only=True)
        mp = out["model"]["_extra_state"]["model_params"]
        assert mp["long_range"]["periodic"] is True
        assert mp["long_range"]["k_grid_mode"] == "kgrid"
        # Weights are untouched, and the converted checkpoint rebuilds and
        # reproduces the original model on non-periodic frames.
        rebuilt = _model(mp)
        rebuilt.load_state_dict(
            {k: v for k, v in out["model"].items() if k != "_extra_state"}
        )
        coord, atype = _batch((4,), seed=67)
        np.testing.assert_allclose(
            rebuilt(coord, atype)["energy"].detach().cpu().numpy(),
            model(coord, atype)["energy"].detach().cpu().numpy(),
            rtol=1e-12,
        )

    def test_missing_extra_state_rejected(self, tmp_path) -> None:
        src = tmp_path / "bad.pt"
        torch.save({"model": {}}, src)
        with pytest.raises(ValueError, match="model_params"):
            convert_checkpoint_to_periodic(str(src), str(tmp_path / "out.pt"))
