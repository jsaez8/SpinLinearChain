import numpy as np

from spin_linear_chain.parameters import SpinChainParameters
from spin_linear_chain.sweeps import run_delta_sweep, run_magnetic_field_sweep


def make_small_chain_params():
	return SpinChainParameters(
		n_sites=2,
		J=4.0,
		delta=0.0,
		magnetic_field=0.0,
		model="Heisenberg",
	)


def test_delta_sweep_returns_arrays_with_sweep_points_first():
	results = run_delta_sweep(
		params=make_small_chain_params(),
		deltas=np.array([-1.0, 0.0, 1.0]),
		eigvals=0,
	)

	assert results.parameter_name == "delta"
	np.testing.assert_array_equal(results.parameter_values, [-1.0, 0.0, 1.0])
	assert results.eigenvalues.shape == (3, 4)
	assert results.total_magnetizations.shape == (3,)
	assert results.local_magnetizations.shape == (3, 2)


def test_magnetic_field_sweep_returns_arrays_with_sweep_points_first():
	results = run_magnetic_field_sweep(
		params=make_small_chain_params(),
		magnetic_fields=np.array([-0.5, 0.0, 0.5]),
		eigvals=0,
	)

	assert results.parameter_name == "magnetic_field"
	np.testing.assert_array_equal(results.parameter_values, [-0.5, 0.0, 0.5])
	assert results.eigenvalues.shape == (3, 4)
	assert results.total_magnetizations.shape == (3,)
	assert results.local_magnetizations.shape == (3, 2)
