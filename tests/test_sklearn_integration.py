"""Sklearn protocols on the integration branch."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from optbinning import BinningProcess


@pytest.fixture
def sample():
    x = np.repeat([0., 1., 2.], 100)
    y = np.concatenate([np.r_[np.zeros(100-n), np.ones(n)]
                        for n in [10, 50, 90]])
    return pd.DataFrame({'a': x, 'b': x + 1}, index=np.arange(300) * 2), y


def test_feature_names_and_fitted_state(sample):
    X, y = sample
    process = BinningProcess()
    with pytest.raises(NotFittedError):
        check_is_fitted(process)
    with pytest.raises(NotFittedError):
        process.get_feature_names_out()
    process.fit(X, y)
    check_is_fitted(process)
    assert process.n_features_in_ == 2
    np.testing.assert_array_equal(process.feature_names_in_, X.columns)
    np.testing.assert_array_equal(process.get_feature_names_out(), X.columns)
    with pytest.raises(ValueError, match='input_features'):
        process.get_feature_names_out(['b', 'a'])
    with pytest.raises(NotFittedError):
        check_is_fitted(clone(process))
    process.fit(X.to_numpy(), y)
    assert not hasattr(process, 'feature_names_in_')
    assert list(process.get_feature_names_out()) == ['x0', 'x1']


def test_sklearn_output_and_pipeline(sample):
    X, y = sample
    process = BinningProcess(selection_criteria={
        'iv': {'strategy': 'highest', 'top': 1}}).set_output(transform='pandas')
    result = process.fit_transform(X.to_numpy(), y)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(process.get_feature_names_out())
    pipeline = Pipeline([('binning', BinningProcess())])
    pipeline.set_output(transform='pandas')
    result = pipeline.fit_transform(X, y)
    pd.testing.assert_index_equal(result.index, X.index)
    assert list(pipeline.get_feature_names_out()) == ['a', 'b']
    np.testing.assert_allclose(result, pipeline.transform(X))


def test_output_no_selected_features(sample):
    X, y = sample
    process = BinningProcess(selection_criteria={'iv': {'min': 100}})
    process.set_output(transform='pandas')
    with pytest.warns(UserWarning, match='No variables'):
        result = process.fit_transform(X, y)
    assert result.shape == (len(X), 0)
    pd.testing.assert_index_equal(result.index, X.index)


def test_name_based_transform_and_array_validation(sample):
    X, y = sample
    process = BinningProcess().fit(X, y)
    pd.testing.assert_frame_equal(process.transform(X),
                                  process.transform(X[['b', 'a']]))
    with pytest.raises(ValueError):
        process.transform(np.zeros((3, 3)))
