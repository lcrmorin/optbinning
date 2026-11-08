# CP construction and complete-fit benchmark

This benchmark compares two **clean Git checkouts** using the same interpreter,
installed dependencies, inputs and benchmark source. It needs only OptBinning's
normal dependencies; the small harness checks additionally need pytest.
No private data or downloads are used. Run on an otherwise idle machine.

```sh
python benchmarks/cp/bench_cp.py \
  --reference /path/to/correctness-checkout \
  --candidate /path/to/construction-checkout \
  --repeats 7 --require-equivalent --output /tmp/cp-comparison.json
python -m pytest benchmarks/cp/test_bench_cp.py
```

Create detached worktrees at the revisions to compare. For an optimization,
use the immediately preceding **corrected** formulation as reference. For a
correctness change, omit `--require-equivalent`: changed models remain in the
output but do not receive speedup ratios. Correctness must be established by
the regression tests, not by reproducing the old model.

## What is measured

* **Construction:** 112 deterministic aggregate cases covering binary,
  continuous, multiclass and the shared two-dimensional CP builder. The
  numerical cases use 8/20 prebins, seven shapes and regularization on/off;
  multiclass cases use 5/12 prebins and minimum event-rate gaps. The small 2D
  grid covers both target types, both directions and regularization.
  Timing includes coefficient preparation inside `build_model`, excludes
  constructor/import/hash computation and does not include solving.
* **Complete fitting:** eight cases from scikit-learn's bundled breast-cancer
  and diabetes datasets (first feature), numerical and derived categorical
  predictors, with native automatic trend selection or no trend constraint.
  Timing covers the full public `fit` call, including prebinning, model
  construction, optimization and post-processing. Estimator instantiation,
  data preparation and transforms used for equivalence are outside timing.
  These small examples are runtime controls, not predictive benchmarks.

Each repeat launches one process per revision, in seeded randomized order,
without concurrent timing jobs. Both get identical warm-ups. BLAS/OpenMP
threads are restricted to one; library CP solver settings are not modified.
The binary/continuous solvers use their existing single-worker settings and
five-second solve limits. Each fit case receives a separate warm-up. The
construction family gets one binary build warm-up per process.

The JSON retains every measurement, status, input/model hash, dependency
version, Git revision, benchmark hash and execution order. `complete` stays
false if collection or equivalence validation fails. Preserve the JSON when
reporting results. Speedup is reference time / candidate time, computed for
paired repeats; each case reports the median and observed min/max (not a
confidence interval). Build and complete-fit ratios must be reported separately.

## Equivalence and limits

Construction checks compare complete model protobufs, normalizing only full
linear-constraint sign reversal (including infinite bounds). No constraint,
variable, objective or solver setting is dropped. This is deliberately stricter
than equality of the optimal objective. Complete-fit checks require identical
statuses and transformed predictions on the input rows. Only optimal fit pairs
with equal predictions receive speedups; infeasible and time-limited outcomes
are retained without rankings. Ties with different optimal partitions can
therefore fail the conservative equivalence gate and require manual review.

This is finite coverage, not a proof for all parameter combinations. It does
not time large 2D models, scenario fits or all public estimators. Fixed splits,
p-values and other interactions additionally rely on the library test suite.
Do not extrapolate a construction speedup into a whole-fit or general dataset
speedup. Regression tests and benchmark-harness checks are ordinary tests;
there are no timing thresholds in CI.
