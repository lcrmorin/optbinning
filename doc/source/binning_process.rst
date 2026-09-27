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


Routing sample weights in pipelines
----------------------------------

Metadata routing is inherited from scikit-learn. Enable it and explicitly
request weights on every step that should consume them::

    from sklearn import config_context
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    with config_context(enable_metadata_routing=True):
        process = BinningProcess().set_fit_request(sample_weight=True)
        estimator = LogisticRegression().set_fit_request(sample_weight=True)
        pipeline = Pipeline([("binning", process), ("model", estimator)])
        pipeline.fit(X, y, sample_weight=weights)

This classifier example requires a binary target. Weighted fitting in
``BinningProcess`` supports binary and continuous targets. Each step
receives the original sample weights. Requesting them at the estimator alone
does not make the binning step weighted.

Use ``set_fit_request(sample_weight=False)`` on the binning process to fit
unweighted bins while routing weights to the estimator. The default request
is unset, so scikit-learn raises an error if weights are supplied without an
explicit routing choice, rather than silently discarding them. Requests also
support aliases, for example ``set_fit_request(sample_weight="weights")``.
Cloning and ``ColumnTransformer`` preserve these inherited routing rules.

Without metadata routing, the existing step-prefixed arguments still work::

    pipeline = Pipeline([("binning", BinningProcess()),
                         ("model", LogisticRegression())])
    pipeline.fit(X, y, binning__sample_weight=weights,
                 model__sample_weight=weights)

The second example assumes scikit-learn's default configuration with metadata
routing disabled. For per-call transformation options in a routing-enabled
pipeline, use ``set_transform_request(metric=True)`` before passing
``metric="event_rate"`` to ``pipeline.fit_transform``. The compatibility
wrapper forwards these options to transformation; ordinary calls continue
to use the inherited ``TransformerMixin.fit_transform`` implementation.


Feature schema and refitting
----------------------------

Inputs must be a two-dimensional array-like or pandas DataFrame. Lists and
objects exposing the NumPy array protocol are converted using sklearn
validation. DataFrames retain their column dtypes and index. Sparse
inputs and complex-valued arrays are unsupported. Fitting requires at least
one sample and one feature. DataFrame column names must be unique strings
or all numeric. Numeric
column labels are converted to strings without modifying the input, for
example ``0`` becomes ``"0"``. Mixed numeric/string labels are rejected.
Explicit ``variable_names`` must be unique strings matching these normalized
column names, although their order may differ.

DataFrame transformation matches selected features by name, so columns may
be reordered and unselected columns may be omitted. NumPy transformation
requires the original number of features in their original order, even if
selection leaves no output features. ``get_feature_names_out`` reports
selected features in output order; its optional ``input_features`` argument
must match the normalized input names and their original order.

Refitting replaces the previous fitted state, including feature metadata and
selected variables. A failed refit leaves the process unfitted rather than
exposing bins from an earlier fit.

The transformer supports sklearn cloning, pipelines, output containers and
metadata routing. Sparse inputs and mixed-type column names remain unsupported.

Continuous sample weights
-------------------------

Continuous targets accept finite, non-negative sample weights with positive
total mass. Zero-weight observations are excluded. Counts (including zero
counts), sums, means and population standard deviations are weighted; minima
and maxima use original target values with positive weight. Fractional counts
are retained. Categorical ordering and rare-category grouping also use weights.
Integer weights represent repeated observations.

CART prebinning uses weights; the other prebinning methods retain their own
unweighted split-generation rules. Bin-size fractions refer to weight mass.
CP-SAT bin-size constraints quantize fractional prebin masses to one million
units of the total clean mass; reported statistics retain the original weights.
Multiclass targets still do not accept sample weights. Data-driven outlier
filters remain unweighted, so they need not match filtering repeated rows.
