"""Ghia et al. (1982) lid-driven cavity benchmark data.

Reference:
Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method. Journal of Computational Physics, 48(3), 387-411.

This is the standard benchmark for validating CFD codes.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


class GhiaCavityData:
    """Benchmark data for lid-driven cavity at various Reynolds numbers.

    Data extracted from Ghia et al. (1982) paper.
    Provides u-velocity along vertical centerline and v-velocity along
    horizontal centerline.
    """

    # Reynolds number 100 data
    # u-velocity along vertical centerline (x=0.5)
    RE100_Y = jnp.array(
        [
            1.0000,
            0.9766,
            0.9688,
            0.9609,
            0.9531,
            0.8516,
            0.7344,
            0.6172,
            0.5000,
            0.4531,
            0.2813,
            0.1719,
            0.1016,
            0.0703,
            0.0625,
            0.0547,
            0.0000,
        ]
    )

    RE100_U_CENTERLINE = jnp.array(
        [
            1.00000,
            0.84123,
            0.78871,
            0.73722,
            0.68717,
            0.23151,
            0.00332,
            -0.13641,
            -0.20581,
            -0.21090,
            -0.15662,
            -0.10150,
            -0.06434,
            -0.04775,
            -0.04192,
            -0.03717,
            0.00000,
        ]
    )

    # v-velocity along horizontal centerline (y=0.5)
    RE100_X = jnp.array(
        [
            1.0000,
            0.9688,
            0.9609,
            0.9531,
            0.9453,
            0.9063,
            0.8594,
            0.8047,
            0.5000,
            0.2344,
            0.2266,
            0.1563,
            0.0938,
            0.0781,
            0.0703,
            0.0625,
            0.0000,
        ]
    )

    RE100_V_CENTERLINE = jnp.array(
        [
            0.00000,
            -0.05906,
            -0.07391,
            -0.08864,
            -0.10313,
            -0.16914,
            -0.22445,
            -0.24533,
            0.05454,
            0.17527,
            0.17507,
            0.16077,
            0.12317,
            0.10890,
            0.10091,
            0.09233,
            0.00000,
        ]
    )

    # Reynolds number 400 data
    RE400_Y = jnp.array(
        [
            1.0000,
            0.9766,
            0.9688,
            0.9609,
            0.9531,
            0.8516,
            0.7344,
            0.6172,
            0.5000,
            0.4531,
            0.2813,
            0.1719,
            0.1016,
            0.0703,
            0.0625,
            0.0547,
            0.0000,
        ]
    )

    RE400_U_CENTERLINE = jnp.array(
        [
            1.00000,
            0.75837,
            0.68439,
            0.61756,
            0.55892,
            0.29093,
            0.16256,
            0.02135,
            -0.11477,
            -0.17119,
            -0.32726,
            -0.24299,
            -0.14612,
            -0.10338,
            -0.09266,
            -0.08186,
            0.00000,
        ]
    )

    RE400_X = jnp.array(
        [
            1.0000,
            0.9688,
            0.9609,
            0.9531,
            0.9453,
            0.9063,
            0.8594,
            0.8047,
            0.5000,
            0.2344,
            0.2266,
            0.1563,
            0.0938,
            0.0781,
            0.0703,
            0.0625,
            0.0000,
        ]
    )

    RE400_V_CENTERLINE = jnp.array(
        [
            0.00000,
            -0.12146,
            -0.15663,
            -0.19254,
            -0.22847,
            -0.23827,
            -0.44993,
            -0.38598,
            0.05188,
            0.30174,
            0.30203,
            0.28124,
            0.22965,
            0.20920,
            0.19713,
            0.18360,
            0.00000,
        ]
    )

    # Reynolds number 1000 data
    RE1000_Y = jnp.array(
        [
            1.0000,
            0.9766,
            0.9688,
            0.9609,
            0.9531,
            0.8516,
            0.7344,
            0.6172,
            0.5000,
            0.4531,
            0.2813,
            0.1719,
            0.1016,
            0.0703,
            0.0625,
            0.0547,
            0.0000,
        ]
    )

    RE1000_U_CENTERLINE = jnp.array(
        [
            1.00000,
            0.65928,
            0.57492,
            0.51117,
            0.46604,
            0.33304,
            0.18719,
            0.05702,
            -0.06080,
            -0.10648,
            -0.27805,
            -0.38289,
            -0.29730,
            -0.22220,
            -0.20196,
            -0.18109,
            0.00000,
        ]
    )

    RE1000_X = jnp.array(
        [
            1.0000,
            0.9688,
            0.9609,
            0.9531,
            0.9453,
            0.9063,
            0.8594,
            0.8047,
            0.5000,
            0.2344,
            0.2266,
            0.1563,
            0.0938,
            0.0781,
            0.0703,
            0.0625,
            0.0000,
        ]
    )

    RE1000_V_CENTERLINE = jnp.array(
        [
            0.00000,
            -0.21388,
            -0.27669,
            -0.33714,
            -0.39188,
            -0.51550,
            -0.42665,
            -0.31966,
            0.02526,
            0.32235,
            0.33075,
            0.37095,
            0.32627,
            0.30353,
            0.29012,
            0.27485,
            0.00000,
        ]
    )

    @classmethod
    def get_data(cls, reynolds_number: int) -> Dict[str, jnp.ndarray]:
        """Get benchmark data for specified Reynolds number.

        Args:
            reynolds_number: Reynolds number (100, 400, or 1000)

        Returns:
            Dictionary with keys:
                - y_coords: y-coordinates for u-velocity
                - u_velocity: u-velocity at vertical centerline
                - x_coords: x-coordinates for v-velocity
                - v_velocity: v-velocity at horizontal centerline

        Raises:
            ValueError: If Reynolds number not available
        """
        if reynolds_number == 100:
            return {
                "y_coords": cls.RE100_Y,
                "u_velocity": cls.RE100_U_CENTERLINE,
                "x_coords": cls.RE100_X,
                "v_velocity": cls.RE100_V_CENTERLINE,
            }
        elif reynolds_number == 400:
            return {
                "y_coords": cls.RE400_Y,
                "u_velocity": cls.RE400_U_CENTERLINE,
                "x_coords": cls.RE400_X,
                "v_velocity": cls.RE400_V_CENTERLINE,
            }
        elif reynolds_number == 1000:
            return {
                "y_coords": cls.RE1000_Y,
                "u_velocity": cls.RE1000_U_CENTERLINE,
                "x_coords": cls.RE1000_X,
                "v_velocity": cls.RE1000_V_CENTERLINE,
            }
        else:
            raise ValueError(
                f"Reynolds number {reynolds_number} not available. " "Available: 100, 400, 1000"
            )

    @classmethod
    def compare_with_pinn(
        cls,
        pinn_state,
        reynolds_number: int,
    ) -> Tuple[float, float, Dict[str, jnp.ndarray]]:
        """Compare PINN results with Ghia benchmark data.

        Args:
            pinn_state: Trained PINN model state
            reynolds_number: Reynolds number

        Returns:
            u_error: Mean absolute error for u-velocity
            v_error: Mean absolute error for v-velocity
            predictions: Dictionary with PINN predictions at benchmark points
        """
        data = cls.get_data(reynolds_number)

        # Predict u-velocity at vertical centerline (x=0.5)
        x_u = jnp.ones_like(data["y_coords"]) * 0.5
        y_u = data["y_coords"]
        t_u = jnp.zeros_like(y_u)

        uvp_u = jax.vmap(pinn_state.apply_fn, in_axes=(None, 0, 0, 0))(
            pinn_state.params,
            x_u[:, None],
            y_u[:, None],
            t_u[:, None],
        )
        u_pred = uvp_u[:, 0]

        # Predict v-velocity at horizontal centerline (y=0.5)
        x_v = data["x_coords"]
        y_v = jnp.ones_like(x_v) * 0.5
        t_v = jnp.zeros_like(x_v)

        uvp_v = jax.vmap(pinn_state.apply_fn, in_axes=(None, 0, 0, 0))(
            pinn_state.params,
            x_v[:, None],
            y_v[:, None],
            t_v[:, None],
        )
        v_pred = uvp_v[:, 1]

        # Compute errors
        u_error = jnp.mean(jnp.abs(u_pred - data["u_velocity"]))
        v_error = jnp.mean(jnp.abs(v_pred - data["v_velocity"]))

        predictions = {
            "u_pred": u_pred,
            "v_pred": v_pred,
            "u_benchmark": data["u_velocity"],
            "v_benchmark": data["v_velocity"],
            "y_coords": data["y_coords"],
            "x_coords": data["x_coords"],
        }

        return u_error, v_error, predictions
