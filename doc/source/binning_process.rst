Binning process
===============

.. autoclass:: optbinning.BinningProcess
   :members:
   :inherited-members:
   :show-inheritance:


Plotting several variables
--------------------------

Plot selected variables using their existing binning-table charts::

    fig, axes = binning_process.plot(ncols=2)
    fig.savefig("binning_overview.png")

Choose a subset and its order with ``variable_names=["age", "income"]``.
The grid supports standard binary, continuous and multiclass binning tables,
uses their default metrics, and builds tables if needed. It returns a figure
and a two-dimensional array of primary axes without showing or closing them.
Unused panels are hidden. Piecewise and two-dimensional binning are outside
this interface.

To compose individual plots yourself, build a table and pass existing axes::

    fig, ax = plt.subplots()
    table = binning_process.get_binned_variable("age").binning_table
    table.build()
    table.plot(ax=ax)

Import ``matplotlib.pyplot as plt`` for the second example. The existing
standalone ``table.plot()`` behavior is unchanged.


Example: numerical and categorical features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This synthetic example plots a continuous-valued numerical feature (Age)
and a categorical feature (Channel) together. The target is binary. Declare
categorical columns explicitly with ``categorical_variables``.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from optbinning import BinningProcess

    rng = np.random.default_rng(315)
    age = rng.uniform(18, 75, 2000)
    channel = rng.choice(['Branch', 'Online', 'Partner'], size=age.size)
    channel_effect = pd.Series(channel).map(
        {'Branch': -0.8, 'Online': 0.2, 'Partner': 1.0}).to_numpy()
    probability = 1 / (1 + np.exp(-((age - 45) / 15 + channel_effect)))
    y = rng.binomial(1, probability)
    X = pd.DataFrame({'Age': age, 'Channel': channel})
    process = BinningProcess(
        variable_names=['Age', 'Channel'], categorical_variables=['Channel'],
        max_n_bins=4)
    process.fit(X, y)
    fig, axes = process.plot(
        ncols=2, figsize=(12, 5), add_special=False, add_missing=False,
        show_bin_labels=False)
    # Use the returned axes to display plain category names.
    groups = process.get_binned_variable('Channel').splits
    axes[0, 1].set_xticks(
        np.arange(len(groups)), [', '.join(map(str, group)) for group in groups])
    axes[0, 1].set_xlabel('Category')
    fig.savefig('binning_mixed_features.png', dpi=140)
    plt.close(fig)

.. figure:: _images/binning_process_mixed_features.png
   :alt: Side-by-side binning plots for numerical Age and categorical Channel, with stacked event counts and WoE curves.
   :align: center

   Age is shown by bin ID; Channel uses category names set through the returned
   axes. Stacked bars show event/non-event counts; black curves show WoE on
   each panel's right axis. Special and missing bins are hidden because this
   synthetic dataset contains neither.

No feature-selection criteria are configured here, so ``plot()`` includes
both fitted variables. To include all fitted variables even when selection
criteria are configured, pass ``variable_names=process.variable_names``.
For an interactive display, replace ``plt.close(fig)`` with ``plt.show()``.
