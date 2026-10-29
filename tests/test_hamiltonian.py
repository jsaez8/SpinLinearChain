from spin_linear_chain.hamiltonians import build_spin_chain
from spin_linear_chain.operators import build_spin_operators
from spin_linear_chain.parameters import SpinChainParameters

import numpy as np

import scipy.constants
from scipy.constants import physical_constants, eV


MU_B = physical_constants["Bohr magneton"][0] / eV
PLANCK = scipy.constants.h / eV

def test_two_site_xy_hamiltonian():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="XY",
    )
    operators = build_spin_operators(params.n_sites)

    actual = build_spin_chain(params, operators)
    expected = (
        operators.sx[0] * operators.sx[1]
        + operators.sy[0] * operators.sy[1]
    )

    assert (actual - expected).norm() < 1e-12

def test_two_site_heisenberg_hamiltonian():
    params = SpinChainParameters(
        n_sites=2,
        J=4.0,
        delta=0.0,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    operators = build_spin_operators(params.n_sites)

    actual = build_spin_chain(params, operators)
    expected = (
        operators.sx[0] * operators.sx[1]
        + operators.sy[0] * operators.sy[1]
        + operators.sz[0] * operators.sz[1]
    )

    assert (actual - expected).norm() < 1e-12

def test_magnetic_field_term():
    params = SpinChainParameters(
        n_sites=2,
        J=0.0,
        delta=0.0,
        magnetic_field=0.01,
        g_factors=np.array([0.1, 0.3]),
        model="Heisenberg",
    )
    energy_scale = MU_B / PLANCK * 1e-6

    operators = build_spin_operators(params.n_sites)
    actual = build_spin_chain(params, operators)
    
    expected =  -0.5* energy_scale * 0.01 * (
                params.g_factors[0] * operators.sz[0] +
                params.g_factors[1] * operators.sz[1]
               )
    assert (actual - expected).norm() < 1e-12

def test_dimerization_term():
    params = SpinChainParameters(
        n_sites=4,
        J=4.0,
        delta=0.5,
        magnetic_field=0.0,
        model="Heisenberg",
    )
    operators = build_spin_operators(params.n_sites)

    actual = build_spin_chain(params, operators)
    expected = (
        0.25 * 6.0 * (operators.sx[0] * operators.sx[1] + operators.sy[0] * operators.sy[1] + operators.sz[0] * operators.sz[1])
        + 0.25 * 2.0 * (operators.sx[1] * operators.sx[2] + operators.sy[1] * operators.sy[2] + operators.sz[1] * operators.sz[2])
        + 0.25 * 6.0 * (operators.sx[2] * operators.sx[3] + operators.sy[2] * operators.sy[3] + operators.sz[2] * operators.sz[3])
    )

    assert (actual - expected).norm() < 1e-12
