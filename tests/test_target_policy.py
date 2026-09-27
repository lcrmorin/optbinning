import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from optbinning import BinningProcess, Scorecard
from optbinning.scorecard import ScorecardMonitoring
from optbinning.binning.target import resolve_target_dtype


@pytest.mark.parametrize('values,expected', [
    ([0, 1, 2], 'continuous'), ([10., 20., 30.], 'continuous'),
    ([1.2, 2.8], 'continuous'), ([1, 2], 'continuous'),
    ([0, 1], 'binary'), ([False, True], 'binary'),
    (['no', 'yes'], 'binary'), (['a', 'b', 'c'], 'multiclass'),
    (pd.Series([1, 2, 3], dtype='Int64'), 'continuous'),
    (pd.Series([1., 2., 3.], dtype='Float64'), 'continuous'),
    (pd.Series([False, True], dtype='boolean'), 'binary'),
    (np.array(['a','b'], dtype=np.dtypes.StringDType()), 'binary'),
    (pd.Series(pd.Categorical([1, 2, 3])), 'continuous')])
def test_inference(values, expected):
    assert resolve_target_dtype(values) == expected
    assert resolve_target_dtype([0, 1, 2], 'multiclass') == 'multiclass'


@pytest.mark.parametrize('override', [None, 'binary', 'continuous', 'multiclass'])
@pytest.mark.parametrize('values', [[0, 1, np.nan], [0, 1, np.inf],
                                  pd.Series([0, 1, pd.NA], dtype='Int64'),
                                  [[0], [1]], [1, 'a'], [1j, 2j]])
def test_validation_with_overrides(values, override):
    with pytest.raises((ValueError, TypeError)):
        resolve_target_dtype(values, override)


def sample():
    x = np.repeat([0., 1., 2.], 60)
    y = np.concatenate([np.r_[np.zeros(60-n), np.ones(n)]
                        for n in [10, 30, 50]])
    return pd.DataFrame({'x': x}), y


def test_string_binary_scorecard_and_monitoring():
    X, y = sample()
    labels = pd.Series(np.where(y, 'yes', 'no'), dtype='string')
    sc = Scorecard(BinningProcess(), LogisticRegression()).fit(X, labels)
    assert set(sc.predict(X)) == {'no', 'yes'}
    assert list(sc.binning_process_.classes_) == ['no', 'yes']
    baseline = BinningProcess().fit(X, y)
    np.testing.assert_allclose(sc.binning_process_.transform(X),
                               baseline.transform(X))
    monitoring = ScorecardMonitoring(sc)
    monitoring.fit(X, labels, X, labels)
    with pytest.raises(ValueError):
        monitoring.fit(X, np.repeat('other', len(X)), X, labels)


def test_string_class_weight_mapping():
    X, y = sample()
    labels = np.where(y, 'yes', 'no')
    params = {'x': {'class_weight': {'no': 1, 'yes': 2}}}
    process = BinningProcess(binning_fit_params=params).fit(X, labels)
    baseline = BinningProcess(binning_fit_params={
        'x': {'class_weight': {0: 1, 1: 2}}}).fit(X, y)
    np.testing.assert_allclose(process.transform(X), baseline.transform(X))
    assert params['x']['class_weight'] == {'no': 1, 'yes': 2}


def test_multiclass_strings_and_explicit_numbers():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({'x': rng.normal(size=300)})
    labels = rng.choice(['a', 'b', 'c'], 300)
    process = BinningProcess().fit(X, labels)
    numeric = BinningProcess(target_dtype='multiclass').fit(
        X, np.searchsorted(['a', 'b', 'c'], labels))
    np.testing.assert_allclose(process.transform(X), numeric.transform(X))
    assert process._target_dtype == 'multiclass'


def test_numeric_regression_and_nested_override():
    X, y = sample()
    sc = Scorecard(BinningProcess(), LinearRegression()).fit(X, y + 10)
    assert sc._target_dtype == 'continuous'
    sc = Scorecard(BinningProcess(target_dtype='binary'),
                   LogisticRegression()).fit(X, y + 10)
    assert sc._target_dtype == 'binary'
    assert set(sc.predict(X)) == {10, 11}
