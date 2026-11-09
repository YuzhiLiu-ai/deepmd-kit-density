# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of models predicting charge density on grid points."""

from typing import (
    Any,
)

import numpy as np

from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    _write_per_frame_details,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.eval_metrics import (
    mae,
    rmse,
)

__all__ = ["DensityTester"]


class DensityTester(ModelTester):
    """Test a model of charge density on grid points."""

    report = (
        ("mae_density", "DENSITY MAE             : {} units"),
        ("rmse_density", "DENSITY RMSE            : {} units"),
    )

    def __init__(
        self,
        dp: Any,
        *,
        atomic: bool,
        grid_sample_ratio: float = 0.1,
        rand_seed: int | None = None,
    ) -> None:
        super().__init__(dp, atomic=atomic)
        if not 0.0 < grid_sample_ratio <= 1.0:
            raise ValueError("grid_sample_ratio must be in the interval (0, 1]")
        self.grid_sample_ratio = grid_sample_ratio
        # A dedicated generator makes grid sampling reproducible without
        # changing frame shuffling or other users of the global random state.
        self.grid_rng = np.random.default_rng(0 if rand_seed is None else rand_seed)

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels a density test consumes."""
        dp = self.dp
        # The grid and the density are defined on grid points rather than on
        # atoms, and their extent (ngrid) is not known until the data is
        # loaded. They are declared frame-major so the loader keeps the
        # (nframes, ngrid, ndof) layout without reshaping to natoms.
        data.add(
            "grid",
            3,
            atomic=False,
            must=True,
            high_prec=True,
            special_shape="frame_major",
        )
        data.add(
            "density",
            1,
            atomic=False,
            must=True,
            high_prec=True,
            special_shape="frame_major",
        )
        if dp.get_dim_fparam() > 0:
            data.add(
                "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
            )
        if dp.get_dim_aparam() > 0:
            data.add(
                "aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False
            )

    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one chunk of a density test."""
        dp = self.dp
        mixed_type = data.mixed_type
        nframes = test_data["box"].shape[0]

        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"] if data.pbc else None
        if mixed_type:
            atype = test_data["type"].reshape([nframes, -1])
        else:
            atype = test_data["type"][0]
        fparam = test_data["fparam"] if dp.get_dim_fparam() > 0 else None
        aparam = test_data["aparam"] if dp.get_dim_aparam() > 0 else None
        grid = test_data["grid"].reshape(nframes, -1, 3)
        label = test_data["density"].reshape(nframes, -1)
        if self.grid_sample_ratio < 1.0:
            ngrid = grid.shape[1]
            nsample = min(ngrid, max(1, int(np.ceil(ngrid * self.grid_sample_ratio))))
            sampled_grid = []
            sampled_label = []
            for ii in range(nframes):
                idx = np.sort(self.grid_rng.choice(ngrid, nsample, replace=False))
                sampled_grid.append(grid[ii, idx])
                sampled_label.append(label[ii, idx])
            grid = np.stack(sampled_grid)
            label = np.stack(sampled_label)

        prediction = dp.eval(
            coord,
            box,
            atype,
            grid,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
        ).reshape(nframes, -1)

        diff = prediction - label
        errors: dict[str, tuple[float, float]] = {
            "mae_density": (mae(diff), diff.size),
            "rmse_density": (rmse(diff), diff.size),
        }

        if context.detail_path is not None:
            _write_per_frame_details(
                context,
                suffix="density",
                reference=label,
                prediction=prediction,
            )

        return errors
