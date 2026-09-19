"""Composition of existing binning plots without changing their appearance."""

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import fixture, raises
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from optbinning import BinningProcess


@fixture(params=["binary", "continuous", "multiclass"])
def process(request):
    data = load_iris()
    X = pd.DataFrame(data.data[:, :3], columns=data.feature_names[:3])
    if request.param == "binary":
        y = (data.target == 0).astype(int)
    elif request.param == "continuous":
        y = data.data[:, 3] + np.linspace(0, 0.01, len(X))
    else:
        y = data.target
    return BinningProcess(list(X.columns), max_n_bins=4).fit(X, y)


def test_grid(process, tmp_path):
    with patch.object(plt, "show") as show:
        fig, axes = process.plot()
        show.assert_not_called()
    assert axes.shape == (2, 2)
    names = process.get_support(names=True)
    assert [ax.get_title() for ax in axes.flat if ax.get_visible()] == list(names)
    assert not axes[1, 1].get_visible()
    assert len(fig.axes) == 7  # Four primary axes and three metric axes.
    fig.canvas.draw()
    assert sum(ax.get_legend() is not None for ax in fig.axes) == 3
    fig.savefig(tmp_path / "grid.png")
    plt.close(fig)


def test_single_variable_and_order(process):
    names = list(process.get_support(names=True))
    fig, axes = process.plot(variable_names=[names[-1]])
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_title() == names[-1]
    plt.close(fig)
    fig, axes = process.plot(variable_names=names[::-1], ncols=3,
                             add_special=False, add_missing=False)
    assert [ax.get_title() for ax in axes.flat] == names[::-1]
    plt.close(fig)


def test_supplied_axes_ownership(process, tmp_path):
    table = process.get_binned_variable(process.variable_names[0]).binning_table
    table.build()
    fig, ax = plt.subplots()
    unrelated, _ = plt.subplots()  # Must not draw into the current figure.
    with patch.object(plt, "show") as show, patch.object(plt, "close") as close:
        assert table.plot(ax=ax, savefig=str(tmp_path / "panel.png")) is ax
        show.assert_not_called()
        close.assert_not_called()
    assert len(fig.axes) == 2
    assert len(unrelated.axes) == 1
    assert ax.get_title() == table.name
    assert len(ax.patches) > 0
    assert fig.axes[1].get_legend() is not None
    plt.close(fig)
    plt.close(unrelated)
    with patch.object(plt, "show") as show:
        assert table.plot() is None
        show.assert_called_once()
    plt.close("all")


def test_validation(process):
    figures = plt.get_fignums()
    for ncols in [0, -1, 1.5, True]:
        with raises(ValueError):
            process.plot(ncols=ncols)
    for names in [[], [process.variable_names[0]] * 2, ["not_a_feature"]]:
        with raises(ValueError):
            process.plot(variable_names=names)
    with raises(TypeError):
        process.plot(variable_names="not_a_list")
    with raises(TypeError):
        process.plot(add_missing="yes")
    assert plt.get_fignums() == figures


def test_not_fitted():
    with raises(NotFittedError):
        BinningProcess(["x"]).plot()


def test_selection_and_large_grid():
    values = np.repeat([0., 1., 2.], 60)
    y = np.concatenate([np.r_[np.zeros(60-n), np.ones(n)]
                        for n in [10, 30, 50]])
    names = [f"x{i}" for i in range(26)]
    X = pd.DataFrame({name: values + i for i, name in enumerate(names)})
    process = BinningProcess(
        names, selection_criteria={"iv": {"strategy": "highest", "top": 2}}
    ).fit(X, y)
    fig, axes = process.plot()
    assert sum(ax.get_visible() for ax in axes.flat) == 2
    plt.close(fig)
    fig, axes = process.plot(variable_names=names, ncols=5)
    assert axes.shape == (6, 5)
    assert sum(ax.get_visible() for ax in axes.flat) == 26
    assert axes.flat[25].get_title() == "x25"
    plt.close(fig)
