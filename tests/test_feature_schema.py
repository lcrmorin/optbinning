import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from optbinning import BinningProcess


@pytest.fixture
def sample():
    x = np.repeat([0., 1., 2.], 60)
    y = np.concatenate([np.r_[np.zeros(60-n), np.ones(n)]
                        for n in [10, 30, 50]])
    return pd.DataFrame({'a': x, 'b': x + 1}), y


@pytest.mark.parametrize('names', [['a', 'a'], [0, 1]])
def test_invalid_dataframe_names(sample, names):
    X, y = sample
    X.columns = names
    with pytest.raises((ValueError, TypeError), match='names'):
        BinningProcess().fit(X, y)


def test_duplicate_transform_and_explicit_names(sample):
    X, y = sample
    process = BinningProcess().fit(X, y)
    with pytest.raises(ValueError, match='duplicate'):
        process.transform(X.rename(columns={'b': 'a'}))
    with pytest.raises(ValueError, match='match'):
        BinningProcess(['a', 'c']).fit(X, y)
    with pytest.raises(ValueError, match='duplicate'):
        BinningProcess(['a', 'a']).fit(X.to_numpy(), y)


def test_refit_clears_old_features_and_failed_fit(sample):
    X, y = sample
    process = BinningProcess().fit(X, y)
    process.fit(X[['a']].rename(columns={'a': 'c'}), y)
    assert list(process.summary()['name']) == ['c']
    assert set(process._binned_variables) == {'c'}
    with pytest.raises(ValueError):
        process.fit(X, y[:-1])
    with pytest.raises(NotFittedError):
        check_is_fitted(process)


def test_declared_names_are_copied_and_output_order(sample):
    X, y = sample
    names = ['b', 'a']
    process = BinningProcess(names).fit(X, y)
    names[0] = 'changed'
    assert list(process.get_feature_names_out(X.columns)) == ['b', 'a']
    assert list(process.transform(X).columns) == ['b', 'a']


def test_selected_subset_and_reordering(sample):
    X, y = sample
    process = BinningProcess(selection_criteria={
        'iv': {'strategy': 'highest', 'top': 1}}).fit(X, y)
    selected = list(process.get_feature_names_out())
    pd.testing.assert_frame_equal(process.transform(X),
                                  process.transform(X[selected]))
    pd.testing.assert_frame_equal(process.transform(X),
                                  process.transform(X[['b', 'a']]))


def test_shapes_even_without_selected_features(sample):
    X, y = sample
    process = BinningProcess(selection_criteria={'iv': {'min': 100}}).fit(X, y)
    for bad in [np.zeros((2, 3)), np.zeros(3), np.zeros((0, 2))]:
        with pytest.raises(ValueError):
            process.transform(bad)
    for bad, target in [(np.zeros((0, 2)), []), (np.zeros((3, 0)), [0, 1, 0])]:
        with pytest.raises(ValueError):
            BinningProcess().fit(bad, target)


def test_complex_features_rejected(sample):
    X, y = sample
    with pytest.raises(ValueError, match="Complex data"):
        BinningProcess().fit(X.to_numpy().astype(complex) + 1j, y)
