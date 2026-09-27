import numpy as np
import pytest
from numpy.testing import assert_allclose
from optbinning import ContinuousOptimalBinning


@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('special', [[-1], {'special': [-1], 'empty': [-2]}])
def test_integer_weights_match_repeated_observations(categorical, special):
    x = np.array([0, 0, 1, 1, 2, 2, -1, -1, np.nan, np.nan])
    y = np.array([0, 4, 2, 9, 7, 12, 0, 8, 0, 6.])
    weights = np.array([2, 3, 1, 4, 3, 2, 2, 3, 4, 1])
    params = dict(dtype='categorical' if categorical else 'numerical',
                  special_codes=special, monotonic_trend=None)
    weighted = ContinuousOptimalBinning(**params).fit(x, y, weights)
    repeated = ContinuousOptimalBinning(**params).fit(
        np.repeat(x, weights), np.repeat(y, weights))
    for attr in ['n_records', 'sums', 'stds', 'min_target', 'max_target', 'n_zeros']:
        assert_allclose(getattr(weighted.binning_table, attr),
                        getattr(repeated.binning_table, attr), equal_nan=True)
    assert_allclose(weighted.transform(x, metric_missing='empirical',
                                      metric_special='empirical'),
                    repeated.transform(x, metric_missing='empirical',
                                       metric_special='empirical'))


@pytest.mark.parametrize("user_splits", [None, []])
def test_fractional_single_bin_statistics(user_splits):
    x = np.zeros(4)
    y = np.array([0., 4., 8., 1000.])
    weights = np.array([.25, .5, .75, 0.])
    table = ContinuousOptimalBinning(user_splits=user_splits).fit(
        x, y, weights).binning_table
    mean = np.average(y[:3], weights=weights[:3])
    assert_allclose(table.n_records[0], 1.5)
    assert_allclose(table.sums[0], 8.)
    assert_allclose(table.stds[0], np.sqrt(np.average(
        (y[:3] - mean)**2, weights=weights[:3])))
    assert table.min_target[0] == 0
    assert table.max_target[0] == 8
    assert table.n_zeros[0] == .25


@pytest.mark.parametrize('scale', [1., .125, 10.])
def test_weighted_bin_size_constraints(scale):
    x = np.arange(10.)
    y = np.array([7.74725, 6.52756, 5.95177, 5.47397, 5.63503,
                  5.17733, 5.24266, 4.68119, 4.92113, 4.69843])
    model = ContinuousOptimalBinning(min_bin_size=.3, max_bin_size=.7)
    model.fit(x, y, np.full(10, scale))
    assert model.status == 'OPTIMAL'
    counts = model.binning_table.n_records[:-2]
    assert np.all(counts >= 3 * scale - 1e-6)
    assert np.all(counts <= 7 * scale + 1e-6)
    repeated = ContinuousOptimalBinning(min_bin_size=.3, max_bin_size=.7)
    repeated.fit(np.repeat(x, 10), np.repeat(y, 10))
    assert_allclose(model.transform(x), repeated.transform(x))


@pytest.mark.parametrize("user_splits", [None, [["a"], ["b"]]])
def test_categorical_others_weighted_statistics(user_splits):
    x = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    y = np.array([0., 10., 5., 7., 0., 3.])
    weights = np.array([10, 1, 5, 5, 1, 1])
    params = dict(dtype='categorical', cat_cutoff=.15, monotonic_trend=None,
                  user_splits=user_splits)
    a = ContinuousOptimalBinning(**params).fit(x, y, weights)
    b = ContinuousOptimalBinning(**params).fit(
        np.repeat(x, weights), np.repeat(y, weights))
    for attr in ['n_records', 'sums', 'stds', 'min_target', 'max_target', 'n_zeros']:
        assert_allclose(getattr(a.binning_table, attr),
                        getattr(b.binning_table, attr), equal_nan=True)


@pytest.mark.parametrize('weights', [[0, 0], [-1, 2], [1, np.nan], [1], [[1], [2]]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        ContinuousOptimalBinning().fit([0., 1.], [2., 3.], weights)


def test_issue_324_original_example():
    x = np.arange(1, 11.)
    y = np.array([.7739560485559633, .4388784397520523, .8585979199113825,
                  .6973680290593639, .09417734788764953, .9756223516367559,
                  .761139701990353, .7860643052769538, .12811363267554587,
                  .45038593789556713])
    weights = np.array([5, 4, 3, 7, 6, 6, 5, 7, 5, 5])
    model = ContinuousOptimalBinning().fit(x, y, weights)
    indices = model.transform(x, metric='indices')
    table = model.binning_table
    for i in np.unique(indices):
        mask = indices == i
        expanded = np.repeat(y[mask], weights[mask])
        assert_allclose(table.stds[i], expanded.std(), atol=1e-7)
        assert_allclose(table.min_target[i], expanded.min())
        assert_allclose(table.max_target[i], expanded.max())
