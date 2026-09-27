"""Target validation and inference for binning processes and scorecards."""
import numbers

import numpy as np
import numpy.typing as npt
import pandas as pd


def resolve_target_dtype(
    y: npt.ArrayLike, target_dtype: str | None = None
) -> str:
    """Infer regression for numbers, classification for strings or booleans.

    Numeric 0/1 retains the binary event convention. Other numeric class
    labels require an explicit binary or multiclass override. Overrides do
    not bypass target validation.
    """
    if target_dtype not in (None, "binary", "continuous", "multiclass"):
        raise ValueError("Invalid target_dtype.")
    values = np.asarray(y, dtype=object)
    if values.ndim != 1 or not values.size:
        raise ValueError("y must be a non-empty one-dimensional target.")
    if pd.isna(values).any():
        raise ValueError("y contains missing values.")
    numeric = all(isinstance(v, (numbers.Real, np.bool_)) for v in values)
    strings = all(isinstance(v, str) for v in values)
    if not numeric and not strings:
        raise TypeError("y must contain only real numbers or only strings.")
    if numeric and not np.isfinite(values.astype(float)).all():
        raise ValueError("y must contain only finite values.")
    unique = np.unique(values)
    if target_dtype is None:
        if strings:
            target_dtype = "binary" if len(unique) == 2 else "multiclass"
        elif set(unique).issubset({0, 1}):
            target_dtype = "binary"
        else:
            target_dtype = "continuous"
    if target_dtype == "continuous" and not numeric:
        raise TypeError("A continuous target must contain real numbers.")
    if target_dtype == "binary" and len(unique) != 2:
        raise ValueError("A binary target must contain exactly two classes.")
    if target_dtype == "multiclass" and len(unique) < 2:
        raise ValueError("A multiclass target must contain at least two classes.")
    return target_dtype


def encode_binary_target(y: npt.ArrayLike, classes: npt.ArrayLike) -> np.ndarray:
    """Encode using training classes, also for one-class monitoring samples."""
    from sklearn.preprocessing import LabelEncoder
    values = np.asarray(y, dtype=object)
    if values.ndim != 1 or not values.size or pd.isna(values).any():
        raise ValueError("y must be one-dimensional, non-empty and non-missing.")
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(classes)
    return encoder.transform(values)
