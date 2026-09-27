import numpy as np
import pytest
from numpy.testing import assert_allclose
from optbinning import BinningProcess, ContinuousOptimalBinning


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_process_forwards_continuous_weights(n_jobs):
    x = np.repeat([0., 1., 2.], 5)
    y = np.arange(15.)
    weights = np.linspace(.5, 2, 15)
    process = BinningProcess(n_jobs=n_jobs).fit(x[:, None], y, weights)
    direct = ContinuousOptimalBinning().fit(x, y, weights)
    assert_allclose(process.transform(x[:, None])[:, 0], direct.transform(x))
    assert_allclose(process.get_binned_variable('x0').binning_table.n_records,
                    direct.binning_table.n_records)


def test_regression_metadata_routing():
    from sklearn import config_context
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    x = np.arange(30.)[:, None]
    y = np.sqrt(x[:, 0] + 1)
    weights = np.linspace(.5, 2., 30)
    with config_context(enable_metadata_routing=True):
        process = BinningProcess().set_fit_request(sample_weight=True)
        estimator = LinearRegression().set_fit_request(sample_weight=True)
        pipeline = Pipeline([('binning', process), ('regression', estimator)])
        pipeline.fit(x, y, sample_weight=weights)
    direct = BinningProcess().fit(x, y, sample_weight=weights)
    expected = LinearRegression().fit(direct.transform(x), y,
                                      sample_weight=weights)
    assert_allclose(pipeline.predict(x), expected.predict(direct.transform(x)))
