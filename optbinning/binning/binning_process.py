"""
Binning process.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numbers
import pickle
import time

from collections.abc import Mapping

from typing import Any
from warnings import warn

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.preprocessing import LabelEncoder
from .target import resolve_target_dtype

from ..logging import Logger
from .base import Base, BaseOptimalBinning
from .binning import OptimalBinning
from .binning_process_information import print_binning_process_information
from .continuous_binning import ContinuousOptimalBinning
from .multiclass_binning import MulticlassOptimalBinning
from .piecewise.binning import OptimalPWBinning
from .piecewise.continuous_binning import ContinuousOptimalPWBinning
from .preprocessing import _check_variable_dtype


logger = Logger(__name__).logger


_METRICS = {
    "binary": {
        "metrics": ["iv", "js", "gini", "quality_score"],
        "iv": {"min": 0, "max": np.inf},
        "gini": {"min": 0, "max": 1},
        "js": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    },
    "multiclass": {
        "metrics": ["js", "quality_score"],
        "js": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    },
    "continuous": {
        "metrics": ["woe", "quality_score"],
        "woe": {"min": 0, "max": np.inf},
        "quality_score": {"min": 0, "max": 1}
    }
}


_OPTB_TYPES = (OptimalBinning, ContinuousOptimalBinning,
               MulticlassOptimalBinning)


_OPTBPW_TYPES = (OptimalPWBinning, ContinuousOptimalPWBinning)


def _check_feature_schema(X, fitting=False):
    if issparse(X):
        raise TypeError("Sparse input is not supported; use a dense array "
                        "or DataFrame.")
    if isinstance(X, Mapping):
        raise TypeError("X must be a two-dimensional array-like or DataFrame.")
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        X = check_array(X, dtype=None, ensure_all_finite=False,
                        ensure_min_samples=0, ensure_min_features=0)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional. Reshape your data.")
    if X.shape[0] == 0:
        raise ValueError("Found array with 0 sample(s); at least 1 is required.")
    if fitting and X.shape[1] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={}) while a "
                         "minimum of 1 is required.".format(X.shape))
    if np.iscomplexobj(X):
        raise ValueError("Complex data not supported.")
    if isinstance(X, pd.DataFrame):
        # Keep pandas dtypes and index while giving binners string names.
        if all(isinstance(name, numbers.Real) for name in X.columns):
            X = X.rename(columns=str)
        _check_feature_names(list(X.columns))
    return X


def _check_feature_names(names):
    if any(not isinstance(name, str) for name in names):
        raise TypeError("Feature names must be strings; rename DataFrame columns.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique; duplicate names found.")


def _read_column(input_path, extension, column, **kwargs):
    if extension == "csv":
        x = pd.read_csv(input_path, engine='c', usecols=[column],
                        low_memory=False, memory_map=True, **kwargs)
    elif extension == "parquet":
        x = pd.read_parquet(input_path, columns=[column], **kwargs)

    return x.iloc[:, 0].values


def _fit_variable(x, y, name, target_dtype, categorical_variables,
                  binning_fit_params, max_n_prebins, min_prebin_size,
                  min_n_bins, max_n_bins, min_bin_size, max_pvalue,
                  max_pvalue_policy, special_codes, split_digits,
                  sample_weight=None):
    params = {}
    dtype = _check_variable_dtype(x)

    if categorical_variables is not None:
        if name in categorical_variables:
            dtype = "categorical"

    if binning_fit_params is not None:
        params = binning_fit_params.get(name, {})

    if target_dtype == "binary":
        optb = OptimalBinning(
            name=name, dtype=dtype, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)
    elif target_dtype == "continuous":
        optb = ContinuousOptimalBinning(
            name=name, dtype=dtype, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)
    else:
        if dtype == "categorical":
            raise ValueError("MulticlassOptimalBinning does not support "
                             "categorical variables.")
        optb = MulticlassOptimalBinning(
            name=name, max_n_prebins=max_n_prebins,
            min_prebin_size=min_prebin_size,
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            min_bin_size=min_bin_size, max_pvalue=max_pvalue,
            max_pvalue_policy=max_pvalue_policy,
            special_codes=special_codes,
            split_digits=split_digits)

    optb.set_params(**params)

    if target_dtype in ("binary", "continuous"):
        optb.fit(x, y, sample_weight)
    else:
        optb.fit(x, y)

    return dtype, optb


def _fit_block(X, y, names, target_dtype, categorical_variables,
               binning_fit_params, max_n_prebins, min_prebin_size,
               min_n_bins, max_n_bins, min_bin_size, max_pvalue,
               max_pvalue_policy, special_codes, split_digits,
               sample_weight=None):

    variable_dtypes = {}
    binned_variables = {}

    for i, name in enumerate(names):
        if isinstance(X, np.ndarray):
            dtype, optb = _fit_variable(
                X[:, i], y, name, target_dtype, categorical_variables,
                binning_fit_params, max_n_prebins, min_prebin_size, min_n_bins,
                max_n_bins, min_bin_size, max_pvalue, max_pvalue_policy,
                special_codes, split_digits, sample_weight)
        else:
            dtype, optb = _fit_variable(
                X[name], y, name, target_dtype, categorical_variables,
                binning_fit_params, max_n_prebins, min_prebin_size, min_n_bins,
                max_n_bins, min_bin_size, max_pvalue, max_pvalue_policy,
                special_codes, split_digits, sample_weight)

        variable_dtypes[name] = dtype
        binned_variables[name] = optb

    return variable_dtypes, binned_variables


def _check_selection_criteria(selection_criteria, target_dtype):
    default_metrics_info = _METRICS[target_dtype]
    default_metrics = default_metrics_info["metrics"]

    if not all(m in default_metrics for m in selection_criteria.keys()):
        raise ValueError("metric for {} target must be in {}."
                         .format(target_dtype, default_metrics))

    for metric, info in selection_criteria.items():
        if not isinstance(info, dict):
            raise TypeError("metric {} info is not a dict.".format(metric))

        for key, value in info.items():
            if key == "min":
                min_ref = default_metrics_info[metric][key]
                if value < min_ref:
                    raise ValueError("metric {} min value {} < {}."
                                     .format(metric, value, min_ref))
            elif key == "max":
                max_ref = default_metrics_info[metric][key]
                if value > max_ref:
                    raise ValueError("metric {} max value {} > {}."
                                     .format(metric, value, max_ref))
            elif key == "strategy":
                if value not in ("highest", "lowest"):
                    raise ValueError('strategy value for metric {} must be '
                                     '"highest" or "lowest"; got {}.'
                                     .format(value, metric))
            elif key == "top":
                if isinstance(value, numbers.Integral):
                    if value < 1:
                        raise ValueError("top value must be at least 1 or "
                                         "in (0, 1); got {}.".format(value))
                else:
                    if not 0. < value < 1.:
                        raise ValueError("top value must be at least 1 or "
                                         "in (0, 1); got {}.".format(value))
            else:
                raise KeyError(key)


def _check_parameters(variable_names, max_n_prebins, min_prebin_size,
                      min_n_bins, max_n_bins, min_bin_size, max_bin_size,
                      max_pvalue, max_pvalue_policy, selection_criteria,
                      fixed_variables, categorical_variables, special_codes,
                      split_digits, binning_fit_params,
                      binning_transform_params, target_dtype, n_jobs,
                      verbose):

    if (variable_names is not None
            and not isinstance(variable_names, (np.ndarray, list))):
        raise TypeError("variable_names must be a list, numpy.ndarray or "
                        "None.")

    if not isinstance(max_n_prebins, numbers.Integral) or max_n_prebins <= 1:
        raise ValueError("max_prebins must be an integer greater than 1; "
                         "got {}.".format(max_n_prebins))

    if not 0. < min_prebin_size <= 0.5:
        raise ValueError("min_prebin_size must be in (0, 0.5]; got {}."
                         .format(min_prebin_size))

    if min_n_bins is not None:
        if not isinstance(min_n_bins, numbers.Integral) or min_n_bins <= 0:
            raise ValueError("min_n_bins must be a positive integer; got {}."
                             .format(min_n_bins))

    if max_n_bins is not None:
        if not isinstance(max_n_bins, numbers.Integral) or max_n_bins <= 0:
            raise ValueError("max_n_bins must be a positive integer; got {}."
                             .format(max_n_bins))

    if min_n_bins is not None and max_n_bins is not None:
        if min_n_bins > max_n_bins:
            raise ValueError("min_n_bins must be <= max_n_bins; got {} <= {}."
                             .format(min_n_bins, max_n_bins))

    if min_bin_size is not None:
        if (not isinstance(min_bin_size, numbers.Number) or
                not 0. < min_bin_size <= 0.5):
            raise ValueError("min_bin_size must be in (0, 0.5]; got {}."
                             .format(min_bin_size))

    if max_bin_size is not None:
        if (not isinstance(max_bin_size, numbers.Number) or
                not 0. < max_bin_size <= 1.0):
            raise ValueError("max_bin_size must be in (0, 1.0]; got {}."
                             .format(max_bin_size))

    if min_bin_size is not None and max_bin_size is not None:
        if min_bin_size > max_bin_size:
            raise ValueError("min_bin_size must be <= max_bin_size; "
                             "got {} <= {}.".format(min_bin_size,
                                                    max_bin_size))

    if max_pvalue is not None:
        if (not isinstance(max_pvalue, numbers.Number) or
                not 0. < max_pvalue <= 1.0):
            raise ValueError("max_pvalue must be in (0, 1.0]; got {}."
                             .format(max_pvalue))

    if max_pvalue_policy not in ("all", "consecutive"):
        raise ValueError('Invalid value for max_pvalue_policy. Allowed string '
                         'values are "all" and "consecutive".')

    if selection_criteria is not None:
        if not isinstance(selection_criteria, dict):
            raise TypeError("selection_criteria must be a dict.")

    if fixed_variables is not None:
        if not isinstance(fixed_variables, (np.ndarray, list)):
            raise TypeError("fixed_variables must be a list or numpy.ndarray.")

    if categorical_variables is not None:
        if not isinstance(categorical_variables, (np.ndarray, list)):
            raise TypeError("categorical_variables must be a list or "
                            "numpy.ndarray.")

        if not all(isinstance(c, str) for c in categorical_variables):
            raise TypeError("variables in categorical_variables must be "
                            "strings.")

    if special_codes is not None:
        if not isinstance(special_codes, (np.ndarray, list, dict)):
            raise TypeError("special_codes must be a dit, list or "
                            "numpy.ndarray.")

        if isinstance(special_codes, dict) and not len(special_codes):
            raise ValueError("special_codes empty. special_codes dict must "
                             "contain at least one special.")

    if split_digits is not None:
        if (not isinstance(split_digits, numbers.Integral) or
                split_digits > 8):
            raise ValueError("split_digits must be an integer <= 8; "
                             "got {}.".format(split_digits))

    if binning_fit_params is not None:
        if not isinstance(binning_fit_params, dict):
            raise TypeError("binning_fit_params must be a dict.")

    if binning_transform_params is not None:
        if not isinstance(binning_transform_params, dict):
            raise TypeError("binning_transform_params must be a dict.")

    if target_dtype is not None:
        if target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError('target_dtype must be "binary", "continuous", '
                             '"multiclass" or None; got {}.'
                             .format(target_dtype))

    if n_jobs is not None:
        if not isinstance(n_jobs, numbers.Integral):
            raise ValueError("n_jobs must be an integer or None; got {}."
                             .format(n_jobs))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


class BaseBinningProcess:
    @classmethod
    def load(cls, path: str) -> "BaseBinningProcess":
        """Load binning process from pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.

        Example
        -------
        >>> from optbinning import BinningProcess
        >>> binning_process = BinningProcess.load("my_binning_process.pkl")
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        """Save binning process to pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _support_selection_criteria(self) -> None:
        self._support = np.full(self._n_variables, True, dtype=bool)

        if self.selection_criteria is None:
            return

        default_metrics_info = _METRICS[self._target_dtype]
        criteria_metrics = self.selection_criteria.keys()

        binning_metrics = pd.DataFrame.from_dict(self._variable_stats).T

        for metric in default_metrics_info["metrics"]:
            if metric in criteria_metrics:
                metric_info = self.selection_criteria[metric]
                metric_values = binning_metrics[metric].values

                if "min" in metric_info:
                    self._support &= metric_values >= metric_info["min"]
                if "max" in metric_info:
                    self._support &= metric_values <= metric_info["max"]
                if all(m in metric_info for m in ("strategy", "top")):
                    indices_valid = np.where(self._support)[0]
                    metric_values = metric_values[indices_valid]
                    n_valid = len(metric_values)

                    # Auxiliary support
                    support = np.full(self._n_variables, False, dtype=bool)

                    top = metric_info["top"]
                    if not isinstance(top, numbers.Integral):
                        top = int(np.ceil(n_valid * top))
                    n_selected = min(n_valid, top)

                    if metric_info["strategy"] == "highest":
                        mask = np.argsort(-metric_values)[:n_selected]
                    elif metric_info["strategy"] == "lowest":
                        mask = np.argsort(metric_values)[:n_selected]

                    support[indices_valid[mask]] = True
                    self._support &= support

        # Fixed variables
        if self.fixed_variables is not None:
            for fv in self.fixed_variables:
                idfv = list(self._variable_names).index(fv)
                self._support[idfv] = True

    def _binning_selection_criteria(self) -> None:
        for i, name in enumerate(self._variable_names):
            optb = self._binned_variables[name]
            optb.binning_table.build()

            dtype = ("numerical" if isinstance(optb, _OPTBPW_TYPES)
                     else getattr(optb, "_dtype", optb.dtype))
            n_bins = len(optb.splits)
            if dtype == "numerical":
                n_bins += 1

            info = {"dtype": dtype,
                    "status": optb.status,
                    "n_bins": n_bins}

            optb.binning_table.analysis(print_output=False)

            if self._target_dtype == "binary":
                metrics = {
                    "iv": optb.binning_table.iv,
                    "gini": optb.binning_table.gini,
                    "js": optb.binning_table.js,
                    "quality_score": optb.binning_table.quality_score}
            elif self._target_dtype == "multiclass":
                metrics = {
                    "js": optb.binning_table.js,
                    "quality_score": optb.binning_table.quality_score}
            elif self._target_dtype == "continuous":
                metrics = {
                    "woe": optb.binning_table.woe,
                    "quality_score": optb.binning_table.quality_score}

            info = {**info, **metrics}
            self._variable_stats[name] = info

        self._support_selection_criteria()


class BinningProcess(Base, TransformerMixin, BaseEstimator,
                     BaseBinningProcess):
    """Binning process to compute optimal binning of variables in a dataset,
    given a binary, continuous or multiclass target dtype.

    Parameters
    ----------
    variable_names : array-like or None, optional (default=None)
        List of variable names. If None, names are inferred from ``X`` at
        fit time: column names for a ``pandas.DataFrame``, or
        ``"x0", "x1", ...`` for a ``numpy.ndarray``. Required (cannot be
        None) when using ``fit_disk``.

        .. versionchanged:: 1.1.0
           ``variable_names`` is now optional.

    max_n_prebins : int (default=20)
        The maximum number of bins after pre-binning (prebins).

    min_prebin_size : float (default=0.05)
        The fraction of mininum number of records for each prebin
        (including missing and ``special_code`` groups).

    min_n_bins : int or None, optional (default=None)
        The minimum number of bins. If None, then ``min_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    max_n_bins : int or None, optional (default=None)
        The maximum number of bins. If None, then ``max_n_bins`` is
        a value in ``[0, max_n_prebins]``.

    min_bin_size : float or None, optional (default=None)
        The fraction of minimum number of records for each bin
        (including missing and ``special_code`` groups). If None,
        ``min_bin_size = min_prebin_size``.

    max_bin_size : float or None, optional (default=None)
        The fraction of maximum number of records for each bin
        (including missing and ``special_code`` groups). If None,
        ``max_bin_size = 1.0``.

    max_pvalue : float or None, optional (default=None)
        The maximum p-value among bins.

    max_pvalue_policy : str, optional (default="consecutive")
        The method to determine bins not satisfying the p-value constraint.
        Supported methods are "consecutive" to compare consecutive bins and
        "all" to compare all bins.

    selection_criteria : dict or None (default=None)
        Variable selection criteria. See notes.

        .. versionadded:: 0.6.0

    fixed_variables : array-like or None
        List of variables to be fixed. The binning process will retain these
        variables if the selection criteria is not satisfied.

        .. versionadded:: 0.12.1

    categorical_variables : array-like or None, optional (default=None)
        List of variables numerical variables to be considered categorical.
        These are nominal variables. Not applicable when target type is
        multiclass.

    special_codes : array-like or None, optional (default=None)
        List of special codes. Use special codes to specify the data values
        that must be treated separately.

    split_digits : int or None, optional (default=None)
        The significant digits of the split points. If ``split_digits`` is set
        to 0, the split points are integers. Negative values round to the
        left of the decimal point (e.g., -2 rounds to the nearest 100). If
        None, then all significant digits in the split points are considered.

    binning_fit_params : dict or None, optional (default=None)
        Dictionary with optimal binning fitting options for specific variables.
        Example: ``{"variable_1": {"max_n_bins": 4}}``.

    binning_transform_params : dict or None, optional (default=None)
        Dictionary with optimal binning transform options for specific
        variables. Example ``{"variable_1": {"metric": "event_rate"}}``.

    target_dtype : str or None, optional (default=None)
        The target type, one of "binary", "continuous" or "multiclass".
        By default, numeric targets use continuous binning, except numeric
        0/1 and booleans, which use binary binning. String targets use binary
        or multiclass binning according to their class count. Numeric class
        labels require an explicit override. Missing targets are rejected.

        .. versionadded:: 1.1.0

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while binning variables.
        ``None`` means 1 core. ``-1`` means using all processors.

        .. versionadded:: 0.7.1

    verbose : bool (default=False)
        Enable verbose output.

    Notes
    -----
    Parameter ``selection_criteria`` allows to specify criteria for
    variable selection. The input is a dictionary as follows

    .. code::

        selection_criteria = {
            "metric_1":
                {
                    "min": 0, "max": 1, "strategy": "highest", "top": 0.25
                },
            "metric_2":
                {
                    "min": 0.02
                }
        }

    where several metrics can be combined. For example, above dictionary
    indicates that top 25% variables with "metric_1" in [0, 1] and "metric_2"
    greater or equal than 0.02 are selected. Supported key values are:

    * keys ``min`` and ``max`` support numerical values.
    * key ``strategy`` supports options "highest" and "lowest".
    * key ``top`` supports an integer or decimal (percentage).


    .. warning::

        If the binning process instance is going to be saved, do not pass the
        option ``"solver": "mip"`` via the ``binning_fit_params`` parameter.

    """
    def __init__(
        self,
        variable_names: npt.ArrayLike | list[str] | None = None,
        max_n_prebins: int = 20,
        min_prebin_size: float = 0.05,
        min_n_bins: int | None = None,
        max_n_bins: int | None = None,
        min_bin_size: float | None = None,
        max_bin_size: float | None = None,
        max_pvalue: float | None = None,
        max_pvalue_policy: str = "consecutive",
        selection_criteria: dict[str, Any] | None = None,
        fixed_variables: npt.ArrayLike | list[str] | None = None,
        categorical_variables: npt.ArrayLike | list[str] | None = None,
        special_codes: npt.ArrayLike | None = None,
        split_digits: int | None = None,
        binning_fit_params: dict[str, Any] | None = None,
        binning_transform_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
        verbose: bool = False,
        target_dtype: str | None = None,
    ):
        self.variable_names = variable_names

        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy

        self.selection_criteria = selection_criteria
        self.fixed_variables = fixed_variables

        self.binning_fit_params = binning_fit_params
        self.binning_transform_params = binning_transform_params

        self.special_codes = special_codes
        self.split_digits = split_digits
        self.categorical_variables = categorical_variables
        self.target_dtype = target_dtype
        self.n_jobs = n_jobs
        self.verbose = verbose

        # auxiliary
        self._n_samples = None
        self._n_variables = None
        self._variable_names = None
        self._target_dtype = None
        self._n_numerical = None
        self._n_categorical = None
        self._n_selected = None
        self._binned_variables = {}
        self._variable_dtypes = {}
        self._variable_stats = {}

        self._support = None

        # timing
        self._time_total = None

        self._is_updated = False
        self._is_fitted = False

    def fit(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
        check_input: bool = False
    ) -> Self:
        """Fit the binning process. Fit the optimal binning to all variables
        according to the given training data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

            .. versionchanged:: 0.4.0
            X supports ``numpy.ndarray`` and ``pandas.DataFrame``.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Weights apply to bin statistics and CART prebinning. This option
            is available for binary and continuous targets.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit(X, y, sample_weight, check_input)

    def fit_disk(self, input_path: str, target: str, **kwargs) -> Self:
        """Fit the binning process according to the given training data on
        disk.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv or .parquet.

        target : str
            Target column.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv`` or
            ``pandas.read_parquet``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit_disk(input_path, target, **kwargs)

    def fit_from_dict(self, dict_optb: dict[str, object]) -> Self:
        """Fit the binning process from a dict of OptimalBinning objects
        already fitted.

        Parameters
        ----------
        dict_optb : dict
            Dictionary with OptimalBinning objects for binary, continuous
            or multiclass target. All objects must share the same class.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self._fit_from_dict(dict_optb)

    def fit_transform(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
        metric: str | None = None,
        metric_special: float | str = 0,
        metric_missing: float | str = 0,
        show_digits: int = 2,
        check_input: bool = False
    ) -> np.ndarray:
        """Fit the binning process according to the given training data, then
        transform it.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Weights apply to bin statistics and CART prebinning. This option
            is available for binary and continuous targets.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        if (metric is None and metric_special == 0 and metric_missing == 0
                and show_digits == 2 and not check_input):
            return super().fit_transform(X, y, sample_weight=sample_weight)

        # Preserve legacy per-call transformation options; TransformerMixin
        # forwards keyword arguments to fit only, not to transform.
        return self.fit(X, y, sample_weight, check_input).transform(
            X, metric, metric_special, metric_missing, show_digits,
            check_input)

    def fit_transform_disk(
        self,
        input_path: str,
        output_path: str,
        target: str,
        chunksize: int,
        metric: str | None = None,
        metric_special: float | str = 0,
        metric_missing: float | str = 0,
        show_digits: int = 2,
        **kwargs
    ) -> Self:
        """Fit the binning process according to the given training data on
        disk, then transform it and save to comma-separated values (csv) file.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv.

        output_path : str
            Any valid string path to a file with extension .csv.

        target : str
            Target column.

        chunksize : int
            Rows to read, transform and write at a time.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        return self.fit_disk(input_path, target, **kwargs).transform_disk(
            input_path, output_path, chunksize, metric, metric_special,
            metric_missing, show_digits, **kwargs)

    def transform(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        metric: str | None = None,
        metric_special: float | str = 0,
        metric_missing: float | str = 0,
        show_digits: int = 2,
        check_input: bool = False
    ) -> np.ndarray:
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array or pandas.DataFrame, shape = (n_samples,
        n_features_new)
            Transformed array.
        """
        self._check_is_fitted()

        return self._transform(X, metric, metric_special, metric_missing,
                               show_digits, check_input)

    def transform_disk(
        self,
        input_path: str,
        output_path: str,
        chunksize: int,
        metric: str | None = None,
        metric_special: float | str = 0,
        metric_missing: float | str = 0,
        show_digits: int = 2,
        **kwargs
    ) -> Self:
        """Transform given data on disk to metric using bins from each fitted
        optimal binning. Save to comma-separated values (csv) file.

        Parameters
        ----------
        input_path : str
            Any valid string path to a file with extension .csv.

        output_path : str
            Any valid string path to a file with extension .csv.

        chunksize : int
            Rows to read, transform and write at a time.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        **kwargs : keyword arguments
            Keyword arguments for ``pandas.read_csv``.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        self._check_is_fitted()

        return self._transform_disk(input_path, output_path, chunksize, metric,
                                    metric_special, metric_missing,
                                    show_digits, **kwargs)

    def information(self, print_level: int = 1) -> None:
        """Print overview information about the options settings and
        statistics.

        Parameters
        ----------
        print_level : int (default=1)
            Level of details.
        """
        self._check_is_fitted()

        if not isinstance(print_level, numbers.Integral) or print_level < 0:
            raise ValueError("print_level must be an integer >= 0; got {}."
                             .format(print_level))

        n_numerical = list(self._variable_dtypes.values()).count("numerical")
        n_categorical = self._n_variables - n_numerical

        self._n_selected = np.count_nonzero(self._support)

        dict_user_options = self.get_params()

        print_binning_process_information(
            print_level, self._n_samples, self._n_variables,
            self._target_dtype, n_numerical, n_categorical,
            self._n_selected, self._time_total, dict_user_options)

    def summary(self) -> pd.DataFrame:
        """Binning process summary with main statistics for all binned
        variables.

        Parameters
        ----------
        df_summary : pandas.DataFrame
            Binning process summary.
        """
        self._check_is_fitted()

        if self._is_updated:
            self._binning_selection_criteria()
            self._is_updated = False

        df_summary = pd.DataFrame.from_dict(self._variable_stats).T
        df_summary.reset_index(inplace=True)
        df_summary.rename(columns={"index": "name"}, inplace=True)
        df_summary["selected"] = self._support

        columns = ["name", "dtype", "status", "selected", "n_bins"]
        columns += _METRICS[self._target_dtype]["metrics"]

        return df_summary[columns]

    def get_binned_variable(self, name: str) -> BaseOptimalBinning:
        """Return optimal binning object for a given variable name.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        optb : BaseOptimalBinning
            Optimal binning class (binary or continuous).
        """
        self._check_is_fitted()

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if name in self._variable_names:
            return self._binned_variables[name]
        else:
            raise ValueError("name {} does not match a binned variable."
                             .format(name))

    def update_binned_variable(
        self,
        name: str,
        optb: BaseOptimalBinning
    ) -> None:
        """Update optimal binning object for a given variable.

        Parameters
        ----------
        name : str
            The variable name.

        optb : BaseOptimalBinning
            The optimal binning object already fitted.
        """
        self._check_is_fitted()

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if name not in self._variable_names:
            raise ValueError("name {} does not match a binned variable."
                             .format(name))

        optb_types = _OPTB_TYPES + _OPTBPW_TYPES

        if not isinstance(optb, optb_types):
            raise TypeError("Object {} must be of type ({}); got {}"
                            .format(name, optb_types, type(optb)))

        # Check current class
        if self._target_dtype == "binary":
            optb_binary = (OptimalBinning, OptimalPWBinning)
            if not isinstance(optb, optb_binary):
                raise TypeError("target is binary and Object {} must be of "
                                "type {}.".format(optb, optb_binary))
        elif self._target_dtype == "continuous":
            optb_continuous = (ContinuousOptimalBinning,
                               ContinuousOptimalPWBinning)
            if not isinstance(optb, optb_continuous):
                raise TypeError("target is continuous and Object {} must be "
                                "of type {}.".format(optb, optb_continuous))
        elif self._target_dtype == "multiclass":
            if not isinstance(optb, MulticlassOptimalBinning):
                raise TypeError("target is multiclass and Object {} must be "
                                "of type {}.".format(
                                    optb, MulticlassOptimalBinning))

        optb_old = self._binned_variables[name]
        if optb_old.name and optb_old.name != optb.name:
            raise ValueError("Update object name must match old object name; "
                             "{} != {}.".format(optb_old.name, optb.name))

        if optb.name and name != optb.name:
            raise ValueError("name and object name must coincide.")

        self._binned_variables[name] = optb
        self._is_updated = True

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        check_is_fitted(self)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.transformer_tags.preserves_dtype = []
        return tags

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray:
        """Return selected feature names, in transformation order.

        Parameters
        ----------
        input_features : array-like of str or None (default=None)
            If supplied, must match the fitted variable names in order.

        Returns
        -------
        feature_names_out : ndarray of str
            Names of the retained variables.
        """
        check_is_fitted(self)
        names = np.asarray(self._variable_names, dtype=object)
        expected = getattr(self, "feature_names_in_", names)
        if input_features is not None:
            supplied = np.asarray(input_features, dtype=object)
            if supplied.ndim != 1 or not np.array_equal(supplied, expected):
                raise ValueError("input_features must match fitted variable names.")
        return names[self.get_support()].copy()

    def get_support(
        self,
        indices: bool = False,
        names: bool = False
    ) -> np.ndarray:
        """Get a mask, or integer index, or names of the variables selected.

        Parameters
        ----------
        indices : boolean (default=False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        names : boolean (default=False)
            If True, the return value will be an array of strings, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector. If `names` is
            True, this is an string array of sahpe [# output features], whose
            values are names of the selected features.
        """
        self._check_is_fitted()

        if indices and names:
            raise ValueError("Only indices or names can be True.")

        mask = self._support
        if indices:
            return np.where(mask)[0]
        elif names:
            return np.asarray(self._variable_names)[mask]
        else:
            return mask

    def _reset_fit_state(self):
        self._is_fitted = False
        self._is_updated = False
        self._binned_variables = {}
        self._variable_dtypes = {}
        self._variable_stats = {}
        self._support = None
        for attr in ("classes_", "n_features_in_", "feature_names_in_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _target_fit_params(self):
        params = self.binning_fit_params
        if self._target_dtype != "binary" or params is None:
            return params
        result = {}
        for name, options in params.items():
            options = options.copy()
            weights = options.get("class_weight")
            if isinstance(weights, dict):
                options["class_weight"] = {
                    i: weights[label] for i, label in enumerate(self.classes_)
                    if label in weights}
            result[name] = options
        return result

    def _fit(self, X, y, sample_weight, check_input):
        self._reset_fit_state()
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # check X dtype
        X = _check_feature_schema(X, fitting=True)
        check_consistent_length(X, y)

        # check target dtype
        self._target_dtype = resolve_target_dtype(y, self.target_dtype)
        if self._target_dtype == "binary":
            encoder = LabelEncoder().fit(np.asarray(y))
            self.classes_ = encoder.classes_
            y = encoder.transform(np.asarray(y))
        else:
            if hasattr(self, "classes_"):
                del self.classes_
            if self._target_dtype == "continuous":
                y = np.asarray(y, dtype=float)
            else:
                y = np.asarray(y)
                self.classes_ = np.unique(y)

        binning_fit_params = self._target_fit_params()

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError(
                "Target type {} is not supported. If auto-detection is "
                "incorrect for your target (e.g. a continuous target with "
                "integer values), pass target_dtype explicitly."
                .format(self._target_dtype))

        # check sample weight
        if sample_weight is not None and self._target_dtype == "multiclass":
            raise ValueError("Target type {} does not support sample weight."
                             .format(self._target_dtype))

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        # Register sklearn feature metadata without coercing pandas dtypes.
        validate_data(self, X, reset=True, skip_check_array=True)
        input_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # check X and y data
        if check_input:
            check_array(X, ensure_2d=False, dtype=None,
                            ensure_all_finite='allow-nan')

            y = check_array(y, ensure_2d=False, dtype=None,
                            ensure_all_finite=True)

            check_consistent_length(X, y)

        self._n_samples, self._n_variables = X.shape

        if self.variable_names is None:
            if input_names is not None:
                self._variable_names = input_names
            else:
                self._variable_names = ["x{}".format(i)
                                        for i in range(self._n_variables)]
        else:
            self._variable_names = list(self.variable_names)

        _check_feature_names(self._variable_names)
        if (input_names is not None and
                set(input_names) != set(self._variable_names)):
            raise ValueError("variable_names must match the DataFrame columns.")
        if self._n_variables != len(self._variable_names):
            raise ValueError("The number of columns must be equal to the"
                             "length of variable_names.")

        if self.verbose:
            logger.info("Dataset: number of samples: {}."
                        .format(self._n_samples))

            logger.info("Dataset: number of variables: {}."
                        .format(self._n_variables))

        # Number of jobs
        n_jobs = effective_n_jobs(self.n_jobs)

        if self.verbose:
            logger.info("Options: number of jobs (cores): {}."
                        .format(n_jobs))

        if n_jobs == 1:
            for i, name in enumerate(self._variable_names):
                if self.verbose:
                    logger.info("Binning variable ({} / {}): {}."
                                .format(i, self._n_variables, name))

                if isinstance(X, np.ndarray):
                    dtype, optb = _fit_variable(
                        X[:, i], y, name, self._target_dtype,
                        self.categorical_variables, binning_fit_params,
                        self.max_n_prebins, self.min_prebin_size,
                        self.min_n_bins, self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)
                else:
                    dtype, optb = _fit_variable(
                        X[name], y, name, self._target_dtype,
                        self.categorical_variables, binning_fit_params,
                        self.max_n_prebins, self.min_prebin_size,
                        self.min_n_bins, self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)

                self._variable_dtypes[name] = dtype
                self._binned_variables[name] = optb
        else:
            ids = np.arange(len(self._variable_names))
            id_blocks = np.array_split(ids, n_jobs)
            names = np.asarray(self._variable_names)

            if isinstance(X, np.ndarray):
                blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fit_block)(
                        X[:, id_block], y, names[id_block],
                        self._target_dtype, self.categorical_variables,
                        binning_fit_params, self.max_n_prebins,
                        self.min_prebin_size, self.min_n_bins,
                        self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)
                    for id_block in id_blocks)

            else:
                blocks = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_fit_block)(
                        X[names[id_block]], y, names[id_block],
                        self._target_dtype, self.categorical_variables,
                        binning_fit_params, self.max_n_prebins,
                        self.min_prebin_size, self.min_n_bins,
                        self.max_n_bins, self.min_bin_size,
                        self.max_pvalue, self.max_pvalue_policy,
                        self.special_codes, self.split_digits, sample_weight)
                    for id_block in id_blocks)

            for b in blocks:
                vt, bv = b
                self._variable_dtypes.update(vt)
                self._binned_variables.update(bv)

        if self.verbose:
            logger.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_disk(self, input_path, target, **kwargs):
        self._reset_fit_state()
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        # variable_names cannot be inferred from a file path; unlike fit(),
        # fit_disk() requires it explicitly.
        if self.variable_names is None:
            raise ValueError("variable_names cannot be None when using "
                             "fit_disk.")
        self._variable_names = list(self.variable_names)
        _check_feature_names(self._variable_names)
        self.n_features_in_ = len(self._variable_names)
        self.feature_names_in_ = np.asarray(self._variable_names, dtype=object)

        # Input file extension
        extension = input_path.split(".")[1]

        # Check extension
        if extension not in ("csv", "parquet"):
            raise ValueError("input_path extension must be csv or parquet; "
                             "got {}.".format(extension))

        # Check target
        if not isinstance(target, str):
            raise TypeError("target must be a string.")

        # Retrieve target and check dtype
        y = _read_column(input_path, extension, target, **kwargs)
        self._target_dtype = resolve_target_dtype(y, self.target_dtype)
        if self._target_dtype == "binary":
            encoder = LabelEncoder().fit(np.asarray(y))
            self.classes_ = encoder.classes_
            y = encoder.transform(np.asarray(y))
        else:
            if hasattr(self, "classes_"):
                del self.classes_
            if self._target_dtype == "continuous":
                y = np.asarray(y, dtype=float)
            else:
                y = np.asarray(y)
                self.classes_ = np.unique(y)

        binning_fit_params = self._target_fit_params()

        if self._target_dtype not in ("binary", "continuous", "multiclass"):
            raise ValueError(
                "Target type {} is not supported. If auto-detection is "
                "incorrect for your target (e.g. a continuous target with "
                "integer values), pass target_dtype explicitly."
                .format(self._target_dtype))

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        if self.fixed_variables is not None:
            for fv in self.fixed_variables:
                if fv not in self._variable_names:
                    raise ValueError("Variable {} to be fixed is not a valid "
                                     "variable name.".format(fv))

        self._n_samples = len(y)
        self._n_variables = len(self._variable_names)

        if self.verbose:
            logger.info("Dataset: number of samples: {}."
                        .format(self._n_samples))

            logger.info("Dataset: number of variables: {}."
                        .format(self._n_variables))

        for name in self._variable_names:
            x = _read_column(input_path, extension, name, **kwargs)

            dtype, optb = _fit_variable(
                x, y, name, self._target_dtype, self.categorical_variables,
                binning_fit_params, self.max_n_prebins,
                self.min_prebin_size, self.min_n_bins, self.max_n_bins,
                self.min_bin_size, self.max_pvalue, self.max_pvalue_policy,
                self.special_codes, self.split_digits)

            self._variable_dtypes[name] = dtype
            self._binned_variables[name] = optb

        if self.verbose:
            logger.info("Binning process variable selection...")

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _fit_from_dict(self, dict_optb):
        self._reset_fit_state()
        time_init = time.perf_counter()

        if self.verbose:
            logger.info("Binning process started.")
            logger.info("Options: check parameters.")

        _check_parameters(**self.get_params())

        if not isinstance(dict_optb, dict):
            raise TypeError("dict_optb must be a dict.")

        # variable_names cannot be inferred here; unlike fit(), this
        # method requires it explicitly.
        if self.variable_names is None:
            raise ValueError("variable_names cannot be None when using "
                             "_fit_from_dict.")
        self._variable_names = list(self.variable_names)
        _check_feature_names(self._variable_names)
        self.n_features_in_ = len(self._variable_names)
        self.feature_names_in_ = np.asarray(self._variable_names, dtype=object)

        # Check variable names
        if set(dict_optb.keys()) != set(self._variable_names):
            raise ValueError("dict_optb keys and variable names must "
                             "coincide.")

        # Check objects class
        optb_types = _OPTB_TYPES
        types = set()
        for name, optb in dict_optb.items():
            if not isinstance(name, str):
                raise TypeError("Object key must be a string.")

            if not isinstance(optb, optb_types):
                raise TypeError("Object {} must be of type ({}); got {}"
                                .format(name, optb_types, type(optb)))

            types.add(type(optb).__name__)
            if len(types) > 1:
                raise TypeError("All binning objects must be of the same "
                                "class.")

            # Check if fitted
            if not optb._is_fitted:
                raise NotFittedError("Object with key={} is not fitted yet. "
                                     "Call 'fit' for this object before "
                                     "passing to a binning process."
                                     .format(name))

            # Check if name was provided and matches dict_optb key.
            if optb.name and optb.name != name:
                raise ValueError("Object with key={} has attribute name={}. "
                                 "If object has a name those must coincide."
                                 .format(name, optb.name))

        obj_class = types.pop()
        if obj_class == "OptimalBinning":
            self._target_dtype = "binary"
        elif obj_class == "ContinuousOptimalBinning":
            self._target_dtype = "continuous"
        elif obj_class == "MulticlassOptimalBinning":
            self._target_dtype = "multiclass"

        if self.selection_criteria is not None:
            _check_selection_criteria(self.selection_criteria,
                                      self._target_dtype)

        self._n_samples = 0
        self._n_variables = len(self._variable_names)

        for name, optb in dict_optb.items():
            self._variable_dtypes[name] = getattr(optb, "_dtype", optb.dtype)
            self._binned_variables[name] = optb

        # Compute binning statistics and decide whether a variable is selected
        self._binning_selection_criteria()

        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Binning process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def _transform(self, X, metric, metric_special, metric_missing,
                   show_digits, check_input):

        # Check X dtype
        X = _check_feature_schema(X)

        n_samples, n_variables = X.shape

        if isinstance(X, np.ndarray):
            validate_data(self, X, reset=False, skip_check_array=True)

        mask = self.get_support()
        if not mask.any():
            warn("No variables were selected: either the data is"
                 " too noisy or the selection_criteria too strict.",
                 UserWarning)
            return np.empty(0).reshape((n_samples, 0))

        if isinstance(X, pd.DataFrame):
            selected_variables = self.get_support(names=True)
            for name in selected_variables:
                if name not in X.columns:
                    raise ValueError("Selected variable {} must be a column "
                                     "in the input dataframe.".format(name))

        # Check metric
        if metric in ("indices", "bins"):
            if any(isinstance(optb, _OPTBPW_TYPES)
                   for optb in self._binned_variables.values()):
                raise TypeError("metric {} not supported for piecewise "
                                "optimal binning objects.".format(metric))

        indices_selected_variables = self.get_support(indices=True)
        n_selected_variables = len(indices_selected_variables)

        # Check if specific binning transform metrics were supplied, and
        # whether these are compatible. Default base metric is the binning
        # process transform metric.
        base_metric = metric

        if self.binning_transform_params is not None:
            metrics = set()

            if metric is not None:
                metrics.add(metric)

            for idx in indices_selected_variables:
                name = self._variable_names[idx]
                params = self.binning_transform_params.get(name, {})
                metrics.add(params.get("metric", metric))

            if len(metrics) > 1:
                # indices and default transform metrics are numeric. If bins
                # metrics is present the dtypes are incompatible.
                if "bins" in metrics:
                    raise ValueError(
                        "metric 'bins' cannot be mixed with numeric metrics.")
            else:
                base_metric = metrics.pop()

        if base_metric == "indices":
            X_transform = np.full(
                (n_samples, n_selected_variables), -1, dtype=int)
        elif base_metric == "bins":
            X_transform = np.full(
                (n_samples, n_selected_variables), "", dtype=object)
        else:
            X_transform = np.zeros((n_samples, n_selected_variables))

        for i, idx in enumerate(indices_selected_variables):
            name = self._variable_names[idx]
            optb = self._binned_variables[name]

            if isinstance(X, np.ndarray):
                x = X[:, idx]
            else:
                x = X[name]

            params = {}
            if self.binning_transform_params is not None:
                params = self.binning_transform_params.get(name, {})

            # Resolve into new locals, not into metric/metric_special/
            # metric_missing themselves -- overwriting those leaks this
            # variable's override into every later variable (GH #355).
            var_metric = params.get("metric", metric)
            var_metric_missing = params.get("metric_missing", metric_missing)
            var_metric_special = params.get("metric_special", metric_special)

            tparams = {
                "x": x,
                "metric": var_metric,
                "metric_special": var_metric_special,
                "metric_missing": var_metric_missing,
                "check_input": check_input,
                "show_digits": show_digits
                }

            if isinstance(optb, _OPTBPW_TYPES):
                tparams.pop("show_digits")

            if var_metric is None:
                tparams.pop("metric")

            X_transform[:, i] = optb.transform(**tparams)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_transform, columns=selected_variables, index=X.index)

        return X_transform

    def _transform_disk(self, input_path, output_path, chunksize, metric,
                        metric_special, metric_missing, show_digits, **kwargs):

        # check input_path and output_path extensions
        input_extension = input_path.split(".")[1]
        output_extension = output_path.split(".")[1]

        if input_extension != "csv" or output_extension != "csv":
            raise ValueError("input_path and output_path must be csv files.")

        # check chunksize
        if not isinstance(chunksize, numbers.Integral) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer; got {}."
                             .format(chunksize))

        # Check metric
        if metric in ("indices", "bins"):
            if any(isinstance(optb, _OPTBPW_TYPES)
                   for optb in self._binned_variables.values()):
                raise TypeError("metric {} not supported for piecewise "
                                "optimal binning objects.".format(metric))

        selected_variables = self.get_support(names=True)
        n_selected_variables = len(selected_variables)

        # Check if specific binning transform metrics were supplied, and
        # whether these are compatible. Default base metric is the binning
        # process transform metric.
        base_metric = metric

        if self.binning_transform_params is not None:
            metrics = set()

            if metric is not None:
                metrics.add(metric)

            for name in selected_variables:
                params = self.binning_transform_params.get(name, {})
                metrics.add(params.get("metric", metric))

            if len(metrics) > 1:
                # indices and default transform metrics are numeric. If bins
                # metrics is present the dtypes are incompatible.
                if "bins" in metrics:
                    raise ValueError(
                        "metric 'bins' cannot be mixed with numeric metrics.")
            else:
                base_metric = metrics.pop()

        chunks = pd.read_csv(input_path, engine='c', chunksize=chunksize,
                             usecols=selected_variables, **kwargs)

        for k, chunk in enumerate(chunks):
            n_samples, n_variables = chunk.shape

            if base_metric == "indices":
                X_transform = np.full(
                    (n_samples, n_selected_variables), -1, dtype=int)
            elif base_metric == "bins":
                X_transform = np.full(
                    (n_samples, n_selected_variables), "", dtype=object)
            else:
                X_transform = np.zeros((n_samples, n_selected_variables))

            for i, name in enumerate(selected_variables):
                optb = self._binned_variables[name]

                params = {}
                if self.binning_transform_params is not None:
                    params = self.binning_transform_params.get(name, {})

                # Same fix as _transform() (GH #355) -- resolve into new
                # locals so an override can't leak into the next variable
                # or chunk.
                var_metric = params.get("metric", metric)
                var_metric_missing = params.get(
                    "metric_missing", metric_missing)
                var_metric_special = params.get(
                    "metric_special", metric_special)

                tparams = {
                    "x": chunk[name],
                    "metric": var_metric,
                    "metric_special": var_metric_special,
                    "metric_missing": var_metric_missing,
                    "show_digits": show_digits
                    }

                if isinstance(optb, _OPTBPW_TYPES):
                    tparams.pop("show_digits")

                if var_metric is None:
                    tparams.pop("metric")

                X_transform[:, i] = optb.transform(**tparams)

            df = pd.DataFrame(X_transform, columns=selected_variables)
            df.to_csv(output_path, mode='a', index=False, header=(k == 0))

        return self
