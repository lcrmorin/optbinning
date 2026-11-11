CP benchmark updates
====================

The fixed reference is OptBinning **1.0.0**, commit
``271c906d5250f17633396f88d246d64b4c449380``. Every comparison reruns both
revisions together on the same runner. These are CP construction and small
complete-fit controls, not predictive-performance measurements.

Construction uses 112 seeded synthetic aggregate cases. Complete fitting uses
8 controls from scikit-learn's bundled breast-cancer and diabetes datasets,
using their first feature (numerical or quantile-derived categorical), with
no shape constraint or native automatic shape selection. No data download is needed.

Speedup = reference time / candidate time; above 1 is faster. Each case first
uses the median of 7 paired repetitions. Panel speedups and p5–p95 summarize
those case medians, only for cases equivalent and eligible in **all** repeats.
Percentiles describe case variation, not confidence intervals. Timings are
median case times over all cases, including changed cases; their quotient
need not equal the restricted speedup. Changed models receive no equivalent-work
speedup. Fit eligibility requires OPTIMAL status and identical predictions.
A changed model is not itself a failure: correctness fixes need regression tests.

Runs from different dates or protocols are not a controlled head-to-head
comparison. Use the measured preceding-update comparison for incremental gains.
Hosted runners are noisy; no timing threshold gates a merge.

One compact record and compressed raw measurement file per measured revision
are retained here. Rerunning that revision replaces its record. This is not a
collection of old notebook packages. Every code update on an enabled branch
is added automatically; documentation-only changes reuse the last code record.

Develop before CP changes (d31da39)
-----------------------------------

Measured 2026-11-11T05:37:19+00:00. Protocol ``0dd83a46837c``. Runner: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39.

`Raw paired measurements <data/d31da39481a6bc7fd187ec1d864948fbed3632e1.json.gz>`__. 

**Against OptBinning 1.0.0 (271c906)**

.. list-table::
   :header-rows: 1

   * - Stage / cases
     - Reference ms
     - Update ms
     - Speedup (p5–p95)
     - Eligible
     - Changed
   * - construction / binary
     - 7.219
     - 7.124
     - 1.00× (0.99–1.01)
     - 28/28
     - 0
   * - construction / continuous
     - 6.475
     - 6.402
     - 1.00× (0.99–1.01)
     - 28/28
     - 0
   * - construction / multiclass
     - 3.906
     - 3.919
     - 1.00× (0.99–1.01)
     - 20/20
     - 0
   * - construction / 2d
     - 0.491
     - 0.496
     - 0.99× (0.94–1.02)
     - 36/36
     - 0
   * - fit / binary / numerical
     - 6.044
     - 6.152
     - 0.99× (0.98–1.00)
     - 2/2
     - 0
   * - fit / binary / categorical
     - 8.306
     - 8.323
     - 1.00× (0.99–1.00)
     - 2/2
     - 0
   * - fit / continuous / numerical
     - 15.079
     - 15.088
     - 1.00× (0.99–1.00)
     - 2/2
     - 0
   * - fit / continuous / categorical
     - 12.473
     - 12.466
     - 1.00× (1.00–1.00)
     - 2/2
     - 0

CP/MIP formulation corrections (2999797)
----------------------------------------

Measured 2026-11-11T05:43:15+00:00. Protocol ``0dd83a46837c``. Runner: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39.

`Raw paired measurements <data/2999797f9e0535beea14da3d3d4f1e70af91536a.json.gz>`__. 

**Against OptBinning 1.0.0 (271c906)**

.. list-table::
   :header-rows: 1

   * - Stage / cases
     - Reference ms
     - Update ms
     - Speedup (p5–p95)
     - Eligible
     - Changed
   * - construction / binary
     - 7.199
     - 7.160
     - 1.00× (1.00–1.02)
     - 20/28
     - 8
   * - construction / continuous
     - 6.419
     - 6.382
     - 1.00× (1.00–1.01)
     - 20/28
     - 8
   * - construction / multiclass
     - 3.915
     - 3.887
     - 1.01× (1.00–1.01)
     - 16/20
     - 4
   * - construction / 2d
     - 0.494
     - 0.496
     - 1.00× (0.98–1.02)
     - 36/36
     - 0
   * - fit / binary / numerical
     - 6.048
     - 7.201
     - 0.88× (0.77–0.99)
     - 2/2
     - 0
   * - fit / binary / categorical
     - 8.294
     - 8.343
     - 0.99× (0.98–1.00)
     - 2/2
     - 0
   * - fit / continuous / numerical
     - 15.051
     - 15.056
     - 1.00× (1.00–1.00)
     - 2/2
     - 0
   * - fit / continuous / categorical
     - 12.474
     - 12.397
     - 1.00× (1.00–1.01)
     - 2/2
     - 0

**Against preceding update (d31da39)**

.. list-table::
   :header-rows: 1

   * - Stage / cases
     - Reference ms
     - Update ms
     - Speedup (p5–p95)
     - Eligible
     - Changed
   * - construction / binary
     - 7.152
     - 7.177
     - 1.00× (0.99–1.01)
     - 20/28
     - 8
   * - construction / continuous
     - 6.400
     - 6.420
     - 1.00× (0.98–1.00)
     - 20/28
     - 8
   * - construction / multiclass
     - 3.918
     - 3.917
     - 1.00× (0.99–1.00)
     - 16/20
     - 4
   * - construction / 2d
     - 0.497
     - 0.499
     - 1.00× (0.97–1.01)
     - 36/36
     - 0
   * - fit / binary / numerical
     - 6.061
     - 7.253
     - 0.86× (0.76–0.96)
     - 2/2
     - 0
   * - fit / binary / categorical
     - 8.293
     - 8.365
     - 0.99× (0.99–0.99)
     - 2/2
     - 0
   * - fit / continuous / numerical
     - 15.092
     - 15.091
     - 1.00× (0.99–1.00)
     - 2/2
     - 0
   * - fit / continuous / categorical
     - 12.424
     - 12.446
     - 0.99× (0.99–1.00)
     - 2/2
     - 0

Native CP expression construction (cebfbcc)
-------------------------------------------

Measured 2026-11-11T05:48:48+00:00. Protocol ``0dd83a46837c``. Runner: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39.

`Raw paired measurements <data/cebfbcce3a5390c827155c4d0c6c8861fe275e9e.json.gz>`__. 

**Against OptBinning 1.0.0 (271c906)**

.. list-table::
   :header-rows: 1

   * - Stage / cases
     - Reference ms
     - Update ms
     - Speedup (p5–p95)
     - Eligible
     - Changed
   * - construction / binary
     - 7.194
     - 4.014
     - 1.84× (1.21–3.36)
     - 20/28
     - 8
   * - construction / continuous
     - 6.422
     - 3.221
     - 1.99× (1.15–3.42)
     - 20/28
     - 8
   * - construction / multiclass
     - 3.959
     - 3.121
     - 1.24× (1.09–2.02)
     - 16/20
     - 4
   * - construction / 2d
     - 0.502
     - 0.462
     - 1.12× (1.05–1.20)
     - 36/36
     - 0
   * - fit / binary / numerical
     - 6.146
     - 6.366
     - 0.97× (0.92–1.03)
     - 2/2
     - 0
   * - fit / binary / categorical
     - 8.432
     - 7.366
     - 1.12× (1.05–1.20)
     - 2/2
     - 0
   * - fit / continuous / numerical
     - 15.127
     - 12.128
     - 1.20× (1.08–1.32)
     - 2/2
     - 0
   * - fit / continuous / categorical
     - 12.502
     - 10.112
     - 1.21× (1.07–1.34)
     - 2/2
     - 0

**Against preceding update (2999797)**

.. list-table::
   :header-rows: 1

   * - Stage / cases
     - Reference ms
     - Update ms
     - Speedup (p5–p95)
     - Eligible
     - Changed
   * - construction / binary
     - 7.371
     - 4.242
     - 1.79× (1.23–3.32)
     - 28/28
     - 0
   * - construction / continuous
     - 6.564
     - 3.435
     - 1.94× (1.23–3.41)
     - 28/28
     - 0
   * - construction / multiclass
     - 4.083
     - 3.227
     - 1.28× (1.11–1.98)
     - 20/20
     - 0
   * - construction / 2d
     - 0.519
     - 0.474
     - 1.14× (1.08–1.22)
     - 36/36
     - 0
   * - fit / binary / numerical
     - 7.773
     - 6.620
     - 1.15× (1.06–1.24)
     - 2/2
     - 0
   * - fit / binary / categorical
     - 9.446
     - 7.902
     - 1.20× (1.16–1.23)
     - 2/2
     - 0
   * - fit / continuous / numerical
     - 16.245
     - 12.734
     - 1.23× (1.12–1.34)
     - 2/2
     - 0
   * - fit / continuous / categorical
     - 13.605
     - 10.883
     - 1.23× (1.13–1.33)
     - 2/2
     - 0

