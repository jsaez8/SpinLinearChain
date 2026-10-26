from spin_linear_chain.hamiltonians import build_spin_chain
from spin_linear_chain.operators import build_spin_operators
from spin_linear_chain.parameters import SpinChainParameters

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

def test_two_site_Heisenberg_hamiltonian():
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
        model="Heisenberg",
    )
    g_factor = 0.2
    energy_scale = MU_B / PLANCK * 1e-6

    operators = build_spin_operators(params.n_sites)

    actual = build_spin_chain(params, operators)
    expected =  -0.5 * g_factor * energy_scale * 0.01 *  (operators.sz[0] + operators.sz[1])

    assert (actual - expected).norm() < 1e-12