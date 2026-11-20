"""
Rangamati Land-Cover Change, 1993-2023
scripts/python/13_spatial_cv_mcnemar.py  (Script 13) - PRODUCTION

Added for the ICCIT 2026 camera-ready revision (Reviewer #4, comment 5).

Summarises the per-sample predictions exported by GEE Script 12:
  * rep = 0      : original seed-42 spatial-block holdout (n = 214).
                   Must reproduce Script 04 (RF 77.10 / SVM 78.04 / CART 72.90 %).
  * rep = <seed> : 5-fold spatial-block CV, one repetition per seed.

Reports
  1. Overall accuracy and Cohen's kappa per classifier on the original holdout.
  2. Repeated spatial-block CV: pooled OA/kappa per repetition (every point
     tested exactly once per repetition), mean +/- SD across repetitions,
     and mean +/- SD across all 25 individual folds.
  3. McNemar's paired test (RF vs SVM, RF vs CART, SVM vs CART), with
     continuity-corrected chi-square and exact binomial p-values, on the
     original holdout and on each CV repetition.

Input : data/Script12_SpatialCV_Predictions.csv (exported by Script 12; pass
        a different path as the first argument to override).
Output: printed tables + spatial_cv_summary.csv + mcnemar_results.csv
Depends: pandas, numpy (no scipy needed).
"""

import os
import sys
from math import comb, erfc, sqrt

import numpy as np
import pandas as pd

CLASSIFIERS = {'RF': 'p_rf', 'SVM': 'p_svm', 'CART': 'p_cart'}
PAIRS = [('RF', 'SVM'), ('RF', 'CART'), ('SVM', 'CART')]

# Default: the CSV committed alongside this script's GEE counterpart, in
# data/, regardless of the current working directory the script is run from.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'Script12_SpatialCV_Predictions.csv')


def overall_accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def cohen_kappa(y, p):
    y, p = np.asarray(y), np.asarray(p)
    labels = np.union1d(y, p)
    po = np.mean(y == p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def mcnemar(y, pa, pb):
    """b = A right & B wrong; c = A wrong & B right."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    a_ok, b_ok = pa == y, pb == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n            # continuity-corrected
    p_chi2 = erfc(sqrt(chi2 / 2))               # chi-square sf, 1 df
    k = min(b, c)
    p_exact = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, chi2, p_chi2, p_exact


def main(path):
    df = pd.read_csv(path)
    for col in ['rep', 'fold', 'landcover', 'p_rf', 'p_svm', 'p_cart']:
        df[col] = pd.to_numeric(df[col]).astype(int)

    # ------------------------------------------------------------------ 1
    hold = df[df.rep == 0]
    print(f'=== 1. Original spatial-block holdout (n = {len(hold)}) ===')
    for name, col in CLASSIFIERS.items():
        print(f'  {name:<5} OA = {overall_accuracy(hold.landcover, hold[col])*100:6.2f}%   '
              f'kappa = {cohen_kappa(hold.landcover, hold[col]):.3f}')
    print('  (sanity check: should equal Script 04 -> RF 77.10, SVM 78.04, CART 72.90)\n')

    # ------------------------------------------------------------------ 2
    cv = df[df.rep != 0]
    reps = sorted(cv.rep.unique())
    print(f'=== 2. Repeated 5-fold spatial-block CV ({len(reps)} repetitions) ===')
    rows = []
    for name, col in CLASSIFIERS.items():
        rep_oa = [overall_accuracy(g.landcover, g[col]) for _, g in cv.groupby('rep')]
        rep_k = [cohen_kappa(g.landcover, g[col]) for _, g in cv.groupby('rep')]
        fold_oa = [overall_accuracy(g.landcover, g[col])
                   for _, g in cv.groupby(['rep', 'fold'])]
        rows.append({
            'classifier': name,
            'OA_mean_pct': np.mean(rep_oa) * 100,
            'OA_sd_across_reps_pct': np.std(rep_oa, ddof=1) * 100,
            'kappa_mean': np.mean(rep_k),
            'kappa_sd': np.std(rep_k, ddof=1),
            'fold_OA_mean_pct': np.mean(fold_oa) * 100,
            'fold_OA_sd_pct': np.std(fold_oa, ddof=1) * 100,
            'fold_OA_min_pct': np.min(fold_oa) * 100,
            'fold_OA_max_pct': np.max(fold_oa) * 100,
            'n_folds': len(fold_oa),
        })
    summary = pd.DataFrame(rows)
    with pd.option_context('display.float_format', '{:.2f}'.format, 'display.width', 160):
        print(summary.to_string(index=False))
    print()

    # fold sizes (blocks are uneven, so folds are uneven)
    sizes = cv.groupby(['rep', 'fold']).size()
    print(f'  Fold sizes: min {sizes.min()}, max {sizes.max()}, '
          f'median {int(sizes.median())} test points\n')

    # ------------------------------------------------------------------ 3
    print('=== 3. McNemar paired tests ===')
    mrows = []
    for label, g in [('holdout (rep 0)', hold)] + [(f'CV seed {r}', cv[cv.rep == r]) for r in reps]:
        for a, b_ in PAIRS:
            b, c, chi2, p1, p2 = mcnemar(g.landcover, g[CLASSIFIERS[a]], g[CLASSIFIERS[b_]])
            mrows.append({'evaluation': label, 'pair': f'{a} vs {b_}', 'n': len(g),
                          'b_A_only_correct': b, 'c_B_only_correct': c,
                          'chi2_cc': chi2, 'p_chi2': p1, 'p_exact': p2})
    mres = pd.DataFrame(mrows)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 160):
        print(mres.to_string(index=False))
    print('\n  p < 0.05 -> the two classifiers differ significantly on these points.')

    summary.to_csv('spatial_cv_summary.csv', index=False)
    mres.to_csv('mcnemar_results.csv', index=False)
    print('\nSaved: spatial_cv_summary.csv, mcnemar_results.csv')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
