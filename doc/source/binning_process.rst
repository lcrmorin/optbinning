Binning process
===============

.. autoclass:: optbinning.BinningProcess
   :members:
   :inherited-members:
   :show-inheritance:


Scikit-learn integration
-----------------------

``BinningProcess`` implements ``TransformerMixin`` and can infer variable names
from a DataFrame (or generate ``x0``, ``x1``, ... for an array)::

    process = BinningProcess().set_output(transform="pandas")
    transformed = process.fit_transform(X, y)
    names = process.get_feature_names_out()

Output names reflect feature selection. Standard ``Pipeline`` and
``ColumnTransformer`` output configuration uses scikit-learn's inherited
``set_output`` implementation. With the default output configuration, existing
behavior is retained: DataFrame inputs return DataFrames and array inputs
return arrays. The estimator exposes ``n_features_in_`` and, when fitted on a
DataFrame with string column names, ``feature_names_in_``.

Ordinary ``fit_transform`` calls delegate to ``TransformerMixin``. The public
wrapper remains for existing calls that supply transformation options, such as
``fit_transform(X, y, metric="event_rate")``, because scikit-learn's default
implementation forwards extra arguments only to ``fit``. This also preserves
Scorecard's existing call interface.

``get_feature_names_out(input_features=...)`` validates the supplied names
against the fitted variable names. DataFrame transformation continues to
select columns by name, so columns may be reordered and unselected columns
may be omitted. This is intentionally more permissive than strict positional
feature-name validation. Array inputs must have the fitted number of columns.


Target interpretation on the sklearn integration branch
-------------------------------------------------------

Numeric targets default to regression, including integer-valued measurements.
Booleans and numeric 0/1 targets retain the binary event convention. String
labels select binary binning for two classes and multiclass binning for more.
Numeric categories also default to regression; specify
``target_dtype="multiclass"`` for numeric class labels, or
``target_dtype="binary"`` for two numeric classes other than 0/1.

Explicit overrides never bypass validation: targets must be one-dimensional,
non-empty, non-missing, and contain either real finite numbers or strings.
Continuous targets must be numeric. This policy intentionally differs from
scikit-learn's general-purpose ``type_of_target`` inference.

Binary labels are encoded using scikit-learn's ``LabelEncoder``. The fitted
``classes_`` records the mapping: the first sorted class is the non-event
(code 0), the second is the event (code 1). To choose another event meaning,
encode the target as 0/1 before fitting. Per-variable class-weight dictionaries
use the original labels. Multiclass tables retain their original class labels.

Scorecard uses its own explicit override first, then the binning process's
override, then this inference policy. Its estimator retains the original
labels for predictions, and monitoring uses the same fitted class mapping.
