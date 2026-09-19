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
