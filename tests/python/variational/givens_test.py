# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Givens rotation ansatz."""

from __future__ import annotations

import itertools

import numpy as np
import pyscf
import pytest
import scipy.linalg

import ffsim
import ffsim.linalg.givens

RNG = np.random.default_rng(161874205665885421362924793664282585535)


def test_givens_parameters_roundtrip():
    """Test converting to and back from parameters gives consistent results."""
    norb = 5

    interaction_pairs = list(itertools.combinations(range(norb), 2))
    thetas = RNG.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = RNG.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = RNG.uniform(-np.pi, np.pi, size=norb)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=None,
        phase_angles=None,
    )
    assert (
        operator.n_params(
            norb, interaction_pairs, with_phis=False, with_phase_angles=False
        )
        == len(operator.to_parameters())
        == norb * (norb - 1) // 2
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phis=False,
        with_phase_angles=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=phis,
        phase_angles=None,
    )
    assert (
        operator.n_params(norb, interaction_pairs, with_phase_angles=False)
        == len(operator.to_parameters())
        == norb * (norb - 1)
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phase_angles=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=None,
        phase_angles=phase_angles,
    )
    assert (
        operator.n_params(norb, interaction_pairs, with_phis=False)
        == len(operator.to_parameters())
        == norb * (norb - 1) // 2 + norb
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phis=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=phis,
        phase_angles=phase_angles,
    )
    assert (
        operator.n_params(norb, interaction_pairs)
        == len(operator.to_parameters())
        == norb**2
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("norb", range(5))
def test_givens_orbital_rotation_roundtrip(norb: int):
    """Test round-tripping orbital rotation."""
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    roundtripped = operator.to_orbital_rotation()
    np.testing.assert_allclose(roundtripped, orbital_rotation)


def _brickwork_layer_interaction_pairs(
    norb: int, n_layers: int
) -> list[tuple[int, int]]:
    """Return interaction pairs for a fixed number of brickwork layers."""
    return _brickwork_layers_interaction_pairs(norb, range(n_layers))


def _brickwork_layers_interaction_pairs(norb: int, layers) -> list[tuple[int, int]]:
    """Return interaction pairs for fixed brickwork layers."""
    return [(i, i + 1) for layer in layers for i in range(layer % 2, norb - 1, 2)]


def _brickwork_layer_givens_indices(norb: int, layers) -> list[int]:
    """Return flattened Givens rotation indices for fixed brickwork layers."""
    layer_set = set(layers)
    indices = []
    offset = 0
    for layer in range(norb):
        n_givens = len(range(layer % 2, norb - 1, 2))
        if layer in layer_set:
            indices.extend(range(offset, offset + n_givens))
        offset += n_givens
    return indices


def _givens_overlap_error(target: np.ndarray, actual: np.ndarray) -> float:
    """Return Hilbert-Schmidt distance between matrices, up to global phase."""
    if target.shape[0] == 0:
        return 0.0
    return 1 - abs(np.trace(target.T.conj() @ actual)) / target.shape[0]


def test_givens_orbital_rotation_compressed_layers_roundtrip():
    """Test round-tripping an orbital rotation with fewer Givens layers."""
    norb = 6
    n_layers = 3
    interaction_pairs = _brickwork_layer_interaction_pairs(norb, n_layers)
    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=RNG.uniform(-0.2, 0.2, size=len(interaction_pairs)),
        phis=RNG.uniform(-np.pi, np.pi, size=len(interaction_pairs)),
        phase_angles=RNG.uniform(-np.pi, np.pi, size=norb),
    )

    orbital_rotation = operator.to_orbital_rotation()
    compressed = ffsim.GivensAnsatzOp.from_orbital_rotation(
        orbital_rotation, n_layers=n_layers
    )

    assert compressed.interaction_pairs == interaction_pairs
    assert len(compressed.interaction_pairs) < norb * (norb - 1) // 2
    np.testing.assert_allclose(
        compressed.to_orbital_rotation(), orbital_rotation, atol=1e-12
    )


def test_givens_orbital_rotation_compressed_layers_optimize():
    """Test optimizing a compressed Givens ansatz."""
    norb = 6
    n_layers = 2
    generator = 0.02j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)

    initial = ffsim.GivensAnsatzOp.from_orbital_rotation(
        orbital_rotation, n_layers=n_layers
    )
    optimized, result = ffsim.GivensAnsatzOp.from_orbital_rotation(
        orbital_rotation,
        n_layers=n_layers,
        optimize=True,
        options={"maxiter": 300},
        return_optimize_result=True,
    )

    initial_error = _givens_overlap_error(
        orbital_rotation, initial.to_orbital_rotation()
    )
    optimized_error = _givens_overlap_error(
        orbital_rotation, optimized.to_orbital_rotation()
    )
    assert result.fun == pytest.approx(optimized_error)
    assert optimized_error < initial_error


def test_givens_orbital_rotation_drop_layers():
    """Test dropping explicit Givens brickwork layers."""
    norb = 6
    drop_layers = (1, 4)
    layers = tuple(layer for layer in range(norb) if layer not in drop_layers)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    full = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    indices = _brickwork_layer_givens_indices(norb, layers)
    expected = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=[full.interaction_pairs[i] for i in indices],
        thetas=full.thetas[indices],
        phis=None if full.phis is None else full.phis[indices],
        phase_angles=full.phase_angles,
    )
    compressed = ffsim.GivensAnsatzOp.from_orbital_rotation(
        orbital_rotation, drop_layers=drop_layers
    )

    assert ffsim.approx_eq(compressed, expected)


def test_givens_orbital_rotation_compressed_layers_validation():
    """Test validation of compressed Givens layer count."""
    orbital_rotation = np.eye(4)
    with pytest.raises(ValueError, match="n_layers"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation, n_layers=-1)
    with pytest.raises(ValueError, match="n_layers"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation, n_layers=5)
    with pytest.raises(ValueError, match="return_optimize_result"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(
            orbital_rotation, n_layers=2, return_optimize_result=True
        )
    with pytest.raises(ValueError, match="cannot both"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(
            orbital_rotation, n_layers=2, drop_layers=(1,)
        )
    with pytest.raises(ValueError, match="drop_layers"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(
            orbital_rotation, drop_layers=(-1,)
        )
    with pytest.raises(ValueError, match="duplicate"):
        _ = ffsim.GivensAnsatzOp.from_orbital_rotation(
            orbital_rotation, drop_layers=(1, 1)
        )


def test_givens_orbital_rotation_t1_roundtrip():
    """Test round-tripping orbital rotation from t1 amplitudes."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)

    nocc, _, _, _ = ccsd.t2.shape
    orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
    orbital_rotation_generator[:nocc, nocc:] = ccsd.t1
    orbital_rotation_generator[nocc:, :nocc] = -ccsd.t1.T
    orbital_rotation = scipy.linalg.expm(orbital_rotation_generator)
    assert ffsim.linalg.is_unitary(orbital_rotation)

    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    roundtripped = operator.to_orbital_rotation()
    np.testing.assert_allclose(roundtripped, orbital_rotation, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_givens_orbital_rotation_unitary(norb: int, nelec: tuple[int, int]):
    """Test initialization from orbital rotation."""
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_orbital_rotation_t_amplitudes():
    """Test initialization from orbital rotation gives fully parametrized ansatz."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)

    nocc, _, _, _ = ccsd.t2.shape
    orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
    orbital_rotation_generator[:nocc, nocc:] = ccsd.t1
    orbital_rotation_generator[nocc:, :nocc] = -ccsd.t1.T
    orbital_rotation = scipy.linalg.expm(orbital_rotation_generator)
    assert ffsim.linalg.is_unitary(orbital_rotation)

    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    assert len(operator.interaction_pairs) == norb * (norb - 1) // 2
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_orbital_rotation_sparse_roundtrip():
    """Test round-tripping a sparse orbital rotation."""
    orbital_rotation = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        dtype=complex,
    )
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    roundtripped = operator.to_orbital_rotation()
    np.testing.assert_allclose(roundtripped, orbital_rotation, atol=1e-12)


def test_givens_orbital_rotation_fully_parameterized():
    """Test initialization from orbital rotation gives fully parametrized ansatz."""
    norb = 8
    nelec = 4
    orbital_rotation = scipy.linalg.block_diag(
        ffsim.random.random_unitary(norb // 2, seed=RNG),
        ffsim.random.random_unitary(norb // 2, seed=RNG),
    )
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    assert len(operator.interaction_pairs) == norb * (norb - 1) // 2
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_incorrect_num_params():
    """Test that passing incorrect number of parameters throws an error."""
    norb = 5
    interaction_pairs = [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs) + 1),
            phis=np.zeros(len(interaction_pairs)),
            phase_angles=np.zeros(norb),
        )
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs)),
            phis=np.zeros(len(interaction_pairs) + 1),
            phase_angles=np.zeros(norb),
        )
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs)),
            phis=np.zeros(len(interaction_pairs)),
            phase_angles=np.zeros(norb + 1),
        )
