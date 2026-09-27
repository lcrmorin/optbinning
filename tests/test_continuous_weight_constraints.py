"""Weight-mass boundaries must not depend on an integer quantization grid."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from optbinning import ContinuousOptimalBinning


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
@pytest.mark.parametrize('bound', ['min_bin_size', 'max_bin_size', 'both'])
def test_equal_thirds_at_size_boundary(scale, bound):
    params = dict(user_splits=[.5, 1.5], min_n_bins=3, max_n_bins=3,
                  monotonic_trend=None)
    if bound in ('min_bin_size', 'both'):
        params['min_bin_size'] = 1 / 3
    if bound in ('max_bin_size', 'both'):
        params['max_bin_size'] = 1 / 3
    model = ContinuousOptimalBinning(**params)
    model.fit([0., 1., 2.], [1., 3., 7.], np.full(3, scale / 3))
    assert model.status == 'OPTIMAL'
    assert_allclose(model.binning_table.n_records[:3], scale / 3)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
@pytest.mark.parametrize('bound', ['min_bin_size', 'max_bin_size'])
def test_near_boundary_violation_is_not_rounded_away(scale, bound):
    # Both weights used to round to 500000 units, erasing the violation.
    weights = scale * np.array([.5 - 1e-8, .5 + 1e-8])
    model = ContinuousOptimalBinning(user_splits=[.5], min_n_bins=2,
                                     max_n_bins=2, monotonic_trend=None,
                                     **{bound: .5})
    model.fit([0., 1.], [1., 3.], weights)
    assert model.status == 'INFEASIBLE'
