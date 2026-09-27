"""Exercise sklearn's public estimator contract on every supported CI version."""
from sklearn.utils.estimator_checks import parametrize_with_checks

from optbinning import BinningProcess


@parametrize_with_checks([BinningProcess(max_n_prebins=5)])
def test_sklearn_estimator_contract(estimator, check):
    check(estimator)
