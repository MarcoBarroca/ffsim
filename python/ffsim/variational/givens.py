# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Givens rotation ansatz."""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.optimize
from scipy.linalg.lapack import zrot

from ffsim import linalg, protocols
from ffsim.gates import apply_orbital_rotation


@dataclass(frozen=True)
class GivensAnsatzOp(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    """A Givens rotation ansatz operator.

    The Givens rotation ansatz consists of a sequence of `Givens rotations`_ followed
    by a layer of single-orbital phase gates.

    Note that this ansatz does not implement any interactions between spin alpha and
    spin beta orbitals.

    .. _Givens rotations: ffsim.html#ffsim.apply_givens_rotation
    """

    norb: int
    """The number of spatial orbitals."""
    interaction_pairs: list[tuple[int, int]]
    """The orbital pairs to apply the Givens rotations to."""
    thetas: np.ndarray
    """The angles for the Givens rotations."""
    phis: np.ndarray | None
    """The optional phase angles for the Givens rotations."""
    phase_angles: np.ndarray | None
    """The optional phase angles for the layer of single-orbital phase gates."""

    def __post_init__(self):
        if len(self.thetas) != len(self.interaction_pairs):
            raise ValueError(
                "The number of thetas must equal the number of interaction pairs. "
                f"Got {len(self.thetas)} and {len(self.interaction_pairs)}."
            )
        if self.phis is not None and len(self.phis) != len(self.interaction_pairs):
            raise ValueError(
                "The number of phis must equal the number of interaction pairs. "
                f"Got {len(self.phis)} and {len(self.interaction_pairs)}."
            )
        if self.phase_angles is not None and len(self.phase_angles) != self.norb:
            raise ValueError(
                "The number of phase angles must equal the number of orbitals. "
                f"Got {len(self.phase_angles)} and {self.norb}."
            )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        return apply_orbital_rotation(
            vec, self.to_orbital_rotation(), norb=norb, nelec=nelec, copy=copy
        )

    @staticmethod
    def n_params(
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_phis: bool = True,
        with_phase_angles: bool = True,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
            with_phis: Whether to include complex phases for the Givens rotations.
            with_phase_angles: Whether to include a layer of single-orbital phase gates.
        """
        return (1 + with_phis) * len(interaction_pairs) + with_phase_angles * norb

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector."""
        if self.phis is not None and self.phase_angles is not None:
            return np.concatenate([self.thetas, self.phis, self.phase_angles])
        if self.phis is not None:
            return np.concatenate([self.thetas, self.phis])
        if self.phase_angles is not None:
            return np.concatenate([self.thetas, self.phase_angles])
        return self.thetas

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_phis: bool = True,
        with_phase_angles: bool = True,
    ) -> GivensAnsatzOp:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
            with_phis: Whether to include complex phases for the Givens rotations.
            with_phase_angles: Whether to include a layer of single-orbital phase gates.
        """
        n_params = (1 + with_phis) * len(interaction_pairs) + with_phase_angles * norb
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        thetas = params[: len(interaction_pairs)]
        phis = None
        phase_angles = None
        if with_phis and with_phase_angles:
            phis = params[len(interaction_pairs) : 2 * len(interaction_pairs)]
            phase_angles = params[2 * len(interaction_pairs) :]
        elif with_phis:
            phis = params[len(interaction_pairs) :]
            phase_angles = None
        elif with_phase_angles:
            phis = None
            phase_angles = params[len(interaction_pairs) :]
        return GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=thetas,
            phis=phis,
            phase_angles=phase_angles,
        )

    @staticmethod
    def from_orbital_rotation(
        orbital_rotation: np.ndarray,
        *,
        n_layers: int | None = None,
        tol: float = 1e-12,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback=None,
        options: dict | None = None,
        return_optimize_result: bool = False,
    ) -> GivensAnsatzOp | tuple[GivensAnsatzOp, scipy.optimize.OptimizeResult]:
        """Initialize the operator from an orbital rotation.

        Args:
            orbital_rotation: The orbital rotation.
            n_layers: The number of brickwork layers of Givens rotations to use.
                If not specified, the full ``norb`` layers are used. If fewer than
                ``norb`` layers are specified, then the returned operator is generally
                an approximation of the orbital rotation.
            tol: Tolerance for the Givens decomposition of the orbital rotation.
                Matrix entries smaller than this value will be treated as equal to zero.
            optimize: Whether to optimize the compressed Givens ansatz parameters to
                maximize the Hilbert-Schmidt overlap with the orbital rotation.
                This argument is ignored when ``n_layers`` is not specified.
            method: The optimization method. See the documentation of
                `scipy.optimize.minimize`_ for possible values.
                This argument is ignored if ``optimize`` is set to ``False``.
            callback: Callback function for the optimization. See the documentation of
                `scipy.optimize.minimize`_ for usage.
                This argument is ignored if ``optimize`` is set to ``False``.
            options: Options for the optimization. See the documentation of
                `scipy.optimize.minimize`_ for usage.
                This argument is ignored if ``optimize`` is set to ``False``.
            return_optimize_result: Whether to also return the `OptimizeResult`_
                returned by `scipy.optimize.minimize`_.

        Raises:
            ValueError: ``orbital_rotation`` was not a square matrix.
            ValueError: ``n_layers`` was negative or larger than ``norb``.
            ValueError: ``return_optimize_result`` was set to ``True`` but
                ``optimize`` was set to ``False``.

        .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        """
        if (
            orbital_rotation.ndim != 2
            or orbital_rotation.shape[0] != orbital_rotation.shape[1]
        ):
            raise ValueError("orbital_rotation must be a square matrix.")
        norb, _ = orbital_rotation.shape
        if n_layers is None:
            n_layers = norb
            optimize = False
        if n_layers < 0 or n_layers > norb:
            raise ValueError(
                f"n_layers must be between 0 and norb={norb}. Got {n_layers}."
            )
        if return_optimize_result and not optimize:
            raise ValueError(
                "return_optimize_result can only be True if optimize is True."
            )

        givens_rotations, phases = linalg.givens_decomposition(
            orbital_rotation, tol=tol
        )
        interaction_pairs = []
        thetas = []
        phis = []
        for c, s, i, j in givens_rotations:
            interaction_pairs.append((i, j))
            r, phi = cmath.polar(s)
            thetas.append(math.atan2(r, c))
            phis.append(phi)
        interaction_pairs, thetas, phis = _brickwork_givens_rotations(
            interaction_pairs, thetas, phis, norb=norb
        )
        n_givens = len(_brickwork_layer_interaction_pairs(norb, n_layers))
        operator = GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs[:n_givens],
            thetas=np.array(thetas[:n_givens]),
            phis=np.array(phis[:n_givens]),
            phase_angles=np.angle(phases),
        )
        if n_layers != norb:
            operator = _best_initial_givens_ansatz(
                orbital_rotation, operator, norb=norb, n_layers=n_layers
            )
        if optimize:
            result = _optimize_givens_ansatz(
                orbital_rotation,
                operator,
                method=method,
                callback=callback,
                options=options,
            )
            operator = GivensAnsatzOp.from_parameters(
                result.x,
                norb=operator.norb,
                interaction_pairs=operator.interaction_pairs,
                with_phis=operator.phis is not None,
                with_phase_angles=operator.phase_angles is not None,
            )
            if return_optimize_result:
                return operator, result
        return operator

    def to_orbital_rotation(self) -> np.ndarray:
        """Convert the Givens ansatz operator to an orbital rotation."""
        phis = self.phis
        phase_angles = self.phase_angles
        if phis is None:
            phis = np.zeros(len(self.interaction_pairs))
        if phase_angles is None:
            phase_angles = np.zeros(self.norb)
        orbital_rotation = np.diag(np.exp(1j * phase_angles))
        for (i, j), theta, phi in zip(
            self.interaction_pairs[::-1], self.thetas[::-1], phis[::-1]
        ):
            orbital_rotation[:, j], orbital_rotation[:, i] = zrot(
                orbital_rotation[:, j],
                orbital_rotation[:, i],
                math.cos(theta),
                cmath.rect(math.sin(theta), -phi),
            )
        return orbital_rotation

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, GivensAnsatzOp):
            if self.norb != other.norb:
                return False
            if self.interaction_pairs != other.interaction_pairs:
                return False
            if (self.phis is None) != (other.phis is None):
                return False
            if (self.phase_angles is None) != (other.phase_angles is None):
                return False
            if not np.allclose(self.thetas, other.thetas, rtol=rtol, atol=atol):
                return False
            if self.phis is not None and not np.allclose(
                cast(np.ndarray, self.phis),
                cast(np.ndarray, other.phis),
                rtol=rtol,
                atol=atol,
            ):
                return False
            if self.phase_angles is not None and not np.allclose(
                cast(np.ndarray, self.phase_angles),
                cast(np.ndarray, other.phase_angles),
                rtol=rtol,
                atol=atol,
            ):
                return False
            return True
        return NotImplemented


def _brickwork_givens_rotations(
    interaction_pairs: list[tuple[int, int]],
    thetas: list[float],
    phis: list[float],
    norb: int,
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Expand a sparse Givens rotation decomposition to a full brickwork pattern."""
    # Construct a brickwork pattern of Givens rotations with angles set to zero
    q, r = divmod(norb, 2)
    even_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(0, norb - 1, 2)] for _ in range(q + r)
    ]
    odd_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(1, norb - 1, 2)] for _ in range(q)
    ]
    # even_layer_index[i] is the index of the last even layer acting on orbital i
    even_layer_index = [-1] * norb
    # odd_layer_index[i] is the index of the last odd layer acting on orbital i
    odd_layer_index = [-1] * norb
    for (i, j), theta, phi in zip(interaction_pairs, thetas, phis):
        if i > j:
            # Enforce i < j
            i, j = j, i
            theta = -theta
            phi = -phi
        if i % 2 == 0:
            # Even layer
            # Get the index of the even layer this Givens rotation should go in
            index = (
                max(
                    even_layer_index[i],
                    even_layer_index[j],
                    odd_layer_index[i],
                    odd_layer_index[j],
                )
                + 1
            )
            # Add the Givens rotation in the appropriate place
            even_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the even layer index
            even_layer_index[i] = index
            even_layer_index[j] = index
        else:
            # Odd layer
            # Get the index of the odd layer this Givens rotation should go in
            index = max(
                odd_layer_index[i] + 1,
                odd_layer_index[j] + 1,
                even_layer_index[i],
                even_layer_index[j],
            )
            # Add the Givens rotation in the appropriate place
            odd_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the odd layer index
            odd_layer_index[i] = index
            odd_layer_index[j] = index
    # Construct the new Givens rotation decomposition and return
    new_interaction_pairs = []
    new_thetas = []
    new_phis = []
    for even_layer, odd_layer in itertools.zip_longest(
        even_layers, odd_layers, fillvalue=()
    ):
        for layer in [even_layer, odd_layer]:
            for pair, theta, phi in layer:
                new_interaction_pairs.append(pair)
                new_thetas.append(theta)
                new_phis.append(phi)
    return new_interaction_pairs, new_thetas, new_phis


def _brickwork_layer_interaction_pairs(
    norb: int, n_layers: int
) -> list[tuple[int, int]]:
    """Return interaction pairs for the requested number of brickwork layers."""
    return [
        (i, i + 1) for layer in range(n_layers) for i in range(layer % 2, norb - 1, 2)
    ]


def _best_initial_givens_ansatz(
    orbital_rotation: np.ndarray,
    operator: GivensAnsatzOp,
    *,
    norb: int,
    n_layers: int,
) -> GivensAnsatzOp:
    """Choose the better of truncated-decomposition and identity initial guesses."""
    interaction_pairs = _brickwork_layer_interaction_pairs(norb, n_layers)
    identity_operator = GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=np.zeros(len(interaction_pairs)),
        phis=np.zeros(len(interaction_pairs)),
        phase_angles=np.angle(np.diag(orbital_rotation)),
    )
    if _givens_ansatz_error(identity_operator, orbital_rotation) < _givens_ansatz_error(
        operator, orbital_rotation
    ):
        return identity_operator
    return operator


def _optimize_givens_ansatz(
    orbital_rotation: np.ndarray,
    operator: GivensAnsatzOp,
    *,
    method: str,
    callback,
    options: dict | None,
) -> scipy.optimize.OptimizeResult:
    """Optimize a Givens ansatz to approximate an orbital rotation."""
    return scipy.optimize.minimize(
        lambda x: _givens_ansatz_error(
            GivensAnsatzOp.from_parameters(
                x,
                norb=operator.norb,
                interaction_pairs=operator.interaction_pairs,
                with_phis=operator.phis is not None,
                with_phase_angles=operator.phase_angles is not None,
            ),
            orbital_rotation,
        ),
        operator.to_parameters(),
        method=method,
        callback=callback,
        options=options,
    )


def _givens_ansatz_error(
    operator: GivensAnsatzOp, orbital_rotation: np.ndarray
) -> float:
    """Return Hilbert-Schmidt distance from an orbital rotation, up to global phase."""
    if operator.norb == 0:
        return 0.0
    overlap = np.trace(orbital_rotation.T.conj() @ operator.to_orbital_rotation())
    return 1 - abs(overlap) / operator.norb
