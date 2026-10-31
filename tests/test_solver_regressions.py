"""Small counterexamples to invalid binning model constraints."""
import numpy as np
import pytest

from optbinning.binning.cp import BinningCP
from optbinning.binning.mip import BinningMIP


@pytest.mark.parametrize("solver", ["cp", "mip"])
@pytest.mark.parametrize("trend,turn", [
    ("ascending", None), ("descending", None),
    ("peak_heuristic", 0), ("valley_heuristic", 6)])
def test_monotonic_pruning_preserves_valid_partition(solver, trend, turn):
    ne = np.array([3, 11, 16, 4, 10, 3, 6])
    e = np.array([18, 4, 16, 6, 17, 15, 1])
    if trend == "ascending":
        ne, e = ne[::-1], e[::-1]
    args = (trend, 1, 4, 10, 100, 3, 60, 3, 60, 0,
            None, "consecutive", 0, None)
    model = (BinningCP(*args, 5) if solver == "cp" else
             BinningMIP(*args, "cbc", 5))
    model.build_model("iv", ne, e, turn)
    status, cuts = model.solve()
    assert status == "OPTIMAL"
    # These feasible boundaries were excluded by monotonic preprocessing.
    expected = ([5, 7] if trend == "ascending" else
                [1, 2, 7] if trend == "valley_heuristic" else [2, 7])
    assert (np.flatnonzero(cuts) + 1).tolist() == expected
    groups = np.split(np.arange(7), np.flatnonzero(cuts)[:-1] + 1)
    rates = []
    for ids in groups:
        a, b = ne[ids].sum(), e[ids].sum()
        assert 3 <= a <= 60 and 3 <= b <= 60 and 10 <= a + b <= 100
        rates.append(b / (a + b))
    differences = np.diff(rates)
    if trend == "ascending":
        assert (differences >= 0).all()
    elif trend == "valley_heuristic":
        assert differences[0] <= 0 <= differences[1]
    else:
        assert (differences <= 0).all()
