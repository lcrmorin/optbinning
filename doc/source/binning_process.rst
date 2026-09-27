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
