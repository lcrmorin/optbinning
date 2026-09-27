"""Integration coverage for sklearn's inherited metadata routing."""
import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from optbinning import BinningProcess


@pytest.fixture
def weighted_data():
    x = np.repeat([0., 1., 2.], 60)
    y = np.concatenate([np.r_[np.zeros(60-n), np.ones(n)]
                        for n in [10, 30, 50]])
    weights = np.where(y == 1, 3., 1.)
    return pd.DataFrame({'x': x}), y, weights


@pytest.mark.parametrize('route_to_binning', [True, False])
def test_weighted_pipeline_matches_explicit_fit(weighted_data, route_to_binning):
    X, y, weights = weighted_data
    with config_context(enable_metadata_routing=True):
        process = BinningProcess().set_fit_request(sample_weight=route_to_binning)
        model = LogisticRegression().set_fit_request(sample_weight=True)
        pipe = clone(Pipeline([('bins', process), ('model', model)]))
        pipe.fit(X, y, sample_weight=weights)
        direct = BinningProcess().fit(
            X, y, sample_weight=weights if route_to_binning else None)
        expected = LogisticRegression().fit(
            direct.transform(X), y, sample_weight=weights)
        np.testing.assert_allclose(pipe.predict_proba(X),
                                   expected.predict_proba(direct.transform(X)))
        table = pipe['bins'].get_binned_variable('x').binning_table.build()
        expected_total = weights.sum() if route_to_binning else len(X)
        assert table.loc['Totals', 'Count'] == expected_total


def test_legacy_step_weight_parameters(weighted_data):
    X, y, weights = weighted_data
    with config_context(enable_metadata_routing=False):
        pipe = Pipeline([('bins', BinningProcess()),
                         ('model', LogisticRegression())])
        pipe.fit(X, y, bins__sample_weight=weights, model__sample_weight=weights)
        table = pipe['bins'].get_binned_variable('x').binning_table.build()
        assert table.loc['Totals', 'Count'] == weights.sum()


def test_alias_and_column_transformer(weighted_data):
    X, y, weights = weighted_data
    with config_context(enable_metadata_routing=True):
        process = BinningProcess().set_fit_request(sample_weight='weights')
        transformer = ColumnTransformer([('bins', process, ['x'])])
        actual = transformer.fit_transform(X, y, weights=weights)
        expected = BinningProcess().fit_transform(X, y, sample_weight=weights)
        np.testing.assert_allclose(actual, expected)
        table = transformer.named_transformers_['bins'].get_binned_variable(
            'x').binning_table.build()
        assert table.loc['Totals', 'Count'] == weights.sum()


def test_unrequested_metadata_is_not_silently_dropped(weighted_data):
    X, y, weights = weighted_data
    with config_context(enable_metadata_routing=True):
        pipe = Pipeline([('bins', BinningProcess()),
                         ('model', LogisticRegression().set_fit_request(
                             sample_weight=True))])
        with pytest.raises(UnsetMetadataPassedError):
            pipe.fit(X, y, sample_weight=weights)


def test_fit_transform_routes_transform_options(weighted_data):
    X, y, weights = weighted_data
    with config_context(enable_metadata_routing=True):
        process = (BinningProcess().set_fit_request(sample_weight=True)
                   .set_transform_request(metric=True))
        pipe = Pipeline([('bins', process)])
        actual = pipe.fit_transform(X, y, sample_weight=weights,
                                    metric='event_rate')
        expected = BinningProcess().fit_transform(
            X, y, sample_weight=weights, metric='event_rate')
        np.testing.assert_allclose(actual, expected)
