import numpy as np

from spin_linear_chain.parameters import default_spin_chain_params


def test_default_params():
    
    params = default_spin_chain_params()

    assert params.n_sites == 6
    assert params.J == 30
    assert params.delta == 0.0
    assert np.all(params.g_factors == np.full(params.n_sites, 0.2))

def test_default_params_with_custom_values():
    params = default_spin_chain_params(
        n_sites=8,
        J=30,
        delta=0.7,
        magnetic_field=0.03,
        delta_STmin=0.3,
        model="Heisenberg",
    )

    assert params.n_sites == 8
    assert params.J == 30
    assert params.delta == 0.7
    assert params.magnetic_field == 0.03
    assert params.delta_STmin == 0.3
    assert params.model == "Heisenberg"