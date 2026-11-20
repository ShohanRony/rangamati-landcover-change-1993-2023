"""
Rangamati Land-Cover Change, 1993-2023
scripts/python/14_collapsed_class_accuracy.py  (Script 14) - PRODUCTION

Added for the ICCIT 2026 camera-ready revision (Reviewer #4 comment 3;
Reviewer #2 comments 3-4).

Re-expresses the 2023 area-adjusted accuracy assessment (Olofsson et al.
2014) at coarser levels of the class hierarchy, using the SAME 200
reference points, the SAME map strata and the SAME mapped-area weights as
olofsson_accuracy.py. Only the labels are merged after the fact; the
stratified design (strata = the five original map classes) is unchanged.

Schemes reported
  * 5-class   : original scheme (must reproduce 73.17 % area-adjusted OA)
  * 3-class   : Forest (Dense + Degraded/Jhum) / Water / Other non-forest
  * 4-class   : Dense / Degraded-Jhum / Water / Other non-forest

For each scheme: area-adjusted OA with analytic SE (stratified estimator),
user's and producer's accuracy, bias-adjusted class areas with analytic
95 % CI (Olofsson eq. 10), and 95 % bootstrap CIs for UA/PA (points
resampled with replacement within each map stratum, 5,000 replicates).

Inputs (same files as olofsson_accuracy.py; not committed to this repo -
run Script 07 / Script 06b in GEE and download their exports next to this
script, same as for olofsson_accuracy.py):
  RF_MappedArea_PerClass_2023.csv          (class, area_ha)   - Script 07
  ValidationPoints_200_Reclassified_v2.csv (rf_class_new, manual_class) - Script 06b
Depends: pandas, numpy.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 5000
STRATA = [1, 2, 3, 4, 5]

SCHEMES = {
    '5-class (original)': {1: 'Dense Forest', 2: 'Degraded/Jhum', 3: 'Water',
                           4: 'Agri/Settlement', 5: 'Bare Land'},
    '3-class': {1: 'Forest', 2: 'Forest', 3: 'Water', 4: 'Other', 5: 'Other'},
    '4-class': {1: 'Dense Forest', 2: 'Degraded/Jhum', 3: 'Water',
                4: 'Other', 5: 'Other'},
}


def estimate(map_cls, ref_cls, W, mapping):
    """Area-adjusted estimates; strata are always the original map classes."""
    groups = list(dict.fromkeys(mapping.values()))
    m = np.array([mapping[c] for c in map_cls])
    r = np.array([mapping[c] for c in ref_cls])
    s = np.asarray(map_cls)
    # p[g_map, g_ref] = sum over strata in g_map of W_i * n_i,g_ref / n_i
    P = pd.DataFrame(0.0, index=groups, columns=groups)
    for i in STRATA:
        sel = s == i
        n_i = sel.sum()
        if n_i == 0:
            continue
        for g in groups:
            P.loc[mapping[i], g] += W[i] * np.sum(r[sel] == g) / n_i
    oa = float(np.trace(P.values))
    ua = {g: P.loc[g, g] / P.loc[g].sum() for g in groups}
    pa = {g: P.loc[g, g] / P[g].sum() for g in groups}
    prop = {g: P[g].sum() for g in groups}
    return oa, ua, pa, prop, groups, m, r, s


def analytic_se(map_cls, ref_cls, W, mapping, groups):
    s = np.asarray(map_cls)
    r = np.array([mapping[c] for c in ref_cls])
    m = np.array([mapping[c] for c in map_cls])
    var_oa = 0.0
    var_prop = {g: 0.0 for g in groups}
    for i in STRATA:
        sel = s == i
        n_i = sel.sum()
        if n_i < 2:
            continue
        q = np.mean(r[sel] == m[sel])                  # correct in stratum i
        var_oa += W[i] ** 2 * q * (1 - q) / (n_i - 1)
        for g in groups:
            f = np.mean(r[sel] == g)
            var_prop[g] += W[i] ** 2 * f * (1 - f) / (n_i - 1)
    return np.sqrt(var_oa), {g: np.sqrt(v) for g, v in var_prop.items()}


def main():
    area = pd.read_csv('RF_MappedArea_PerClass_2023.csv')
    A = dict(zip(area['class'].astype(int), area['area_ha'].astype(float)))
    total = sum(A[c] for c in STRATA)
    W = {c: A[c] / total for c in STRATA}

    df = pd.read_csv('ValidationPoints_200_Reclassified_v2.csv')
    mc = df['rf_class_new'].astype(int).values
    rc = df['manual_class'].astype(int).values

    out = []
    for name, mapping in SCHEMES.items():
        oa, ua, pa, prop, groups, *_ = estimate(mc, rc, W, mapping)
        se_oa, se_prop = analytic_se(mc, rc, W, mapping, groups)

        # stratified bootstrap for UA / PA
        boot_ua = {g: [] for g in groups}
        boot_pa = {g: [] for g in groups}
        idx_by_stratum = {i: np.where(mc == i)[0] for i in STRATA}
        for _ in range(N_BOOT):
            idx = np.concatenate([RNG.choice(v, size=len(v), replace=True)
                                  for v in idx_by_stratum.values() if len(v)])
            _, bua, bpa, *_ = estimate(mc[idx], rc[idx], W, mapping)
            for g in groups:
                boot_ua[g].append(bua[g]); boot_pa[g].append(bpa[g])

        unweighted = np.mean(np.array([mapping[c] for c in mc]) ==
                             np.array([mapping[c] for c in rc]))
        print(f'\n=== {name} ===')
        print(f'Area-adjusted OA: {oa*100:.2f}%  (SE {se_oa*100:.2f} pp; '
              f'95% CI {100*(oa-1.96*se_oa):.2f}-{100*(oa+1.96*se_oa):.2f}%)   '
              f'unweighted OA: {unweighted*100:.2f}%')
        print(f"{'Class':<18}{'UA %':>8}{'UA 95% CI':>16}{'PA %':>8}{'PA 95% CI':>16}"
              f"{'Mapped ha':>12}{'Adjusted ha':>13}{'+/-95% ha':>11}{'Diff %':>8}")
        for g in groups:
            mapped = sum(A[i] for i in STRATA if mapping[i] == g)
            adj = prop[g] * total
            ci = 1.96 * se_prop[g] * total
            ul, uh = np.nanpercentile(boot_ua[g], [2.5, 97.5]) * 100
            pl, ph = np.nanpercentile(boot_pa[g], [2.5, 97.5]) * 100
            print(f'{g:<18}{ua[g]*100:>8.2f}{f"{ul:.1f}-{uh:.1f}":>16}{pa[g]*100:>8.2f}'
                  f'{f"{pl:.1f}-{ph:.1f}":>16}{mapped:>12.0f}{adj:>13.0f}{ci:>11.0f}'
                  f'{(mapped-adj)/adj*100:>+8.1f}')
            out.append({'scheme': name, 'class': g, 'OA_pct': oa * 100,
                        'OA_SE_pp': se_oa * 100, 'UA_pct': ua[g] * 100,
                        'UA_CI_low': ul, 'UA_CI_high': uh, 'PA_pct': pa[g] * 100,
                        'PA_CI_low': pl, 'PA_CI_high': ph, 'mapped_ha': mapped,
                        'adjusted_ha': adj, 'adjusted_CI95_ha': ci,
                        'mapped_minus_adjusted_pct': (mapped - adj) / adj * 100})
    pd.DataFrame(out).to_csv('collapsed_class_accuracy.csv', index=False)
    print('\nSaved: collapsed_class_accuracy.csv')


if __name__ == '__main__':
    main()
