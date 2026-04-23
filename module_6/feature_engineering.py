"""
feature_engineering.py — Merged Advanced Feature Engineering (v2 + Deep FE)

Purpose:
  1. Variance filtering on spectral bands (removes low-variance bands)
  2. Correlation filtering (keeps bands correlated with targets)
  3. Window averaging (spectral smoothing for flavonol and chlorophyll)
  4. Derivative features (spectral slopes - green shoulder, red edge, etc.)
  5. Literature indices (NDVI, GNDVI, ARI, RVSI, ZTM, FRI, SFI, ANTH, REIP, etc.)
  6. ARI-based composite features (flavonol-specific combinations)
  7. Diagnostic analysis (why is flavonol hard to predict?)
  8. Saves to dataset_output_fe/

Usage:
  python feature_engineering.py            # FE + save
  python feature_engineering.py --plot     # + correlation plots
  python feature_engineering.py --report   # + detailed report
  python feature_engineering.py --diagnose # + flavonol diagnostics
"""

import numpy as np
import os
import sys
import argparse

# ── PATH ────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

DATASET_DIR = os.path.join(_THIS_DIR, "dataset_output")
OUTPUT_DIR = os.path.join(_THIS_DIR, "dataset_output_fe")
PLOT_DIR = os.path.join(_THIS_DIR, "model_outputs", "feature_engineering")

_INDEX_NAMES_ORIG = ["NDVI", "GNDVI", "ARI", "RVSI", "ZTM"]
eps = 1e-6


# ── HELPERS ─────────────────────────────────────────────────────────────────
def _band_idx(feature_names: list, target_nm: float) -> int:
    """Find closest band index to target wavelength."""
    band_wl = []
    for i, name in enumerate(feature_names):
        if name.startswith("band_"):
            try:
                band_wl.append((i, float(name.replace("band_", ""))))
            except ValueError:
                pass
    if not band_wl:
        raise ValueError("No 'band_XXX' format names found.")
    best_i, best_nm = min(band_wl, key=lambda x: abs(x[1] - target_nm))
    return best_i


def _band_range_indices(feature_names: list, lo_nm: float, hi_nm: float) -> list:
    """Get all band indices in wavelength range [lo_nm, hi_nm]."""
    indices = []
    for i, name in enumerate(feature_names):
        if name.startswith("band_"):
            try:
                wl = float(name.replace("band_", ""))
                if lo_nm <= wl <= hi_nm:
                    indices.append(i)
            except ValueError:
                pass
    return indices


def _window_mean(X_bands, feature_names, lo_nm, hi_nm):
    """Compute pixel-wise mean of bands in wavelength range."""
    idxs = _band_range_indices(feature_names, lo_nm, hi_nm)
    if not idxs:
        raise ValueError(f"No bands in range {lo_nm}–{hi_nm} nm.")
    return X_bands[:, idxs].mean(axis=1), idxs


def _window_deriv(X_bands, feature_names, lo_nm, hi_nm):
    """Compute finite difference slope over wavelength range."""
    idxs = _band_range_indices(feature_names, lo_nm, hi_nm)
    if len(idxs) < 2:
        raise ValueError(
            f"Not enough bands in {lo_nm}–{hi_nm} nm for derivative."
        )
    wls = np.array([float(feature_names[i].replace("band_", "")) for i in idxs])
    refs = X_bands[:, idxs]
    slope = (refs[:, -1] - refs[:, 0]) / (wls[-1] - wls[0] + eps)
    return slope


def _range_mean(X_bands, feat, lo, hi):
    """Mean reflectance in wavelength range."""
    idxs = [
        i
        for i, f in enumerate(feat)
        if f.startswith("band_") and lo <= float(f.replace("band_", "")) <= hi
    ]
    return X_bands[:, idxs].mean(axis=1) if idxs else np.zeros(X_bands.shape[0])


# ── MAIN FE FUNCTION ────────────────────────────────────────────────────────
def build_features(
    dataset_dir: str = DATASET_DIR,
    corr_threshold: float = 0.05,
    var_threshold: float = 1e-5,
    log_transform_flav: bool = True,
    report: bool = False,
) -> tuple:
    """
    Advanced feature engineering: variance → correlation → window → deriv →
    literature indices → ARI composites.

    Returns
    -------
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress
    """
    print("\n" + "=" * 70)
    print("ADVANCED FEATURE ENGINEERING (v2 + Deep FE merged)")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────────────────
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    y_chl = np.load(os.path.join(dataset_dir, "y_chl.npy"))
    y_flav = np.load(os.path.join(dataset_dir, "y_flav.npy"))
    y_stress = np.load(os.path.join(dataset_dir, "y_stress.npy"))
    feat = np.load(
        os.path.join(dataset_dir, "feature_names.npy"), allow_pickle=True
    ).tolist()

    print(f"  Raw data       : X={X.shape}, features={len(feat)}")
    n_bands_orig = len(feat) - len(_INDEX_NAMES_ORIG)
    X_bands = X[:, :n_bands_orig]
    X_idx = X[:, n_bands_orig:]

    # ── 2. Variance filter ──────────────────────────────────────────────────
    band_vars = X_bands.var(axis=0)
    var_mask = band_vars >= var_threshold
    X_bands_f = X_bands[:, var_mask]
    feat_bands_f = [feat[i] for i in range(n_bands_orig) if var_mask[i]]
    n_removed_var = n_bands_orig - var_mask.sum()
    print(
        f"\n  [Variance Filter] Removed: {n_removed_var} bands "
        f"(threshold={var_threshold:.0e})"
    )

    # ── 3. Correlation filter ──────────────────────────────────────────────
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    r_flav = np.array(
        [
            abs(np.corrcoef(X_bands_f[valid, i], y_flav[valid])[0, 1])
            for i in range(X_bands_f.shape[1])
        ]
    )
    r_chl = np.array(
        [
            abs(np.corrcoef(X_bands_f[valid, i], y_chl[valid])[0, 1])
            for i in range(X_bands_f.shape[1])
        ]
    )
    corr_mask = (r_flav >= corr_threshold) | (r_chl >= corr_threshold)
    X_bands_sel = X_bands_f[:, corr_mask]
    feat_bands_sel = [feat_bands_f[i] for i in range(len(feat_bands_f))
                      if corr_mask[i]]
    n_removed_corr = corr_mask.shape[0] - corr_mask.sum()
    print(
        f"  [Correlation Filter] Removed: {n_removed_corr} bands "
        f"(|r|<{corr_threshold})"
    )
    print(f"  Remaining raw bands: {X_bands_sel.shape[1]}")

    if report:
        removed_corr = [
            feat_bands_f[i] for i in range(len(feat_bands_f))
            if not corr_mask[i]
        ]
        print(f"\n  Removed bands ({len(removed_corr)}):")
        for b in removed_corr:
            print(f"    - {b}")

    # ── 4. Window averaging (spectral smoothing) ───────────────────────────
    print("\n  [Window Averaging — Spectral Smoothing]")
    window_features = {}

    flav_windows = [
        ("W_400_420", 400, 420),
        ("W_440_460", 440, 460),
        ("W_520_540", 520, 540),
        ("W_555_575", 555, 575),
        ("W_575_595", 575, 595),
        ("W_640_660", 640, 660),
    ]
    chl_windows = [
        ("W_700_720", 700, 720),
        ("W_720_740", 720, 740),
        ("W_740_760", 740, 760),
        ("W_780_800", 780, 800),
    ]

    for name, lo, hi in flav_windows + chl_windows:
        try:
            mean_vals, idxs = _window_mean(X_bands, feat, lo, hi)
            window_features[name] = mean_vals
            print(f"    {name}: {lo}–{hi} nm  ({len(idxs)} bands)")
        except ValueError as e:
            print(f"    [WARNING] {name}: {e}")

    # ── 5. Derivative features (spectral slopes) ──────────────────────────
    print("\n  [Derivative Features — Spectral Slopes]")
    deriv_features = {}

    deriv_regions = [
        ("Deriv_520_580", 520, 580),
        ("Deriv_680_740", 680, 740),
        ("Deriv_400_500", 400, 500),
        ("Deriv_740_800", 740, 800),
    ]

    for name, lo, hi in deriv_regions:
        try:
            slope = _window_deriv(X_bands, feat, lo, hi)
            deriv_features[name] = slope
            print(f"    {name}: {lo}–{hi} nm slope")
        except ValueError as e:
            print(f"    [WARNING] {name}: {e}")

    # ── 6. Literature indices ──────────────────────────────────────────────
    print("\n  [Literature Indices]")

    def col(nm):
        return X_bands[:, _band_idx(feat, nm)]

    lit_features = {}

    # Flavonol-specific
    lit_features["FRI"] = col(690) / (col(600) + eps)
    lit_features["SFI"] = col(440) / (col(690) + eps)
    lit_features["ANTH"] = (1.0 / (col(550) + eps) - 1.0 / (col(700) + eps)) * col(
        800
    )
    lit_features["FlavGitelson"] = col(800) / (col(550) + eps) - 1

    r700 = col(700)
    r740 = col(740)
    r670 = col(670)
    r780 = col(780)
    reip_num = (r670 + r780) / 2.0 - r700
    reip_den = r740 - r700 + eps
    lit_features["REIP"] = 700 + 40 * (reip_num / reip_den)

    r531 = col(531)
    r570 = col(570)
    lit_features["PRI"] = (r531 - r570) / (r531 + r570 + eps)

    # Chlorophyll
    lit_features["CIRedEdge"] = col(750) / (col(705) + eps) - 1
    lit_features["RENDVI"] = (col(800) - col(715)) / (col(800) + col(715) + eps)
    lit_features["GreenRatio"] = col(550) / (col(670) + eps)

    r705 = col(705)
    r550 = col(550)
    r670_v = col(670)
    mcari_a = r705 - r670_v
    mcari_b = r705 - r550
    lit_features["MCARI"] = (mcari_a - 0.2 * mcari_b) * (r705 / (r670_v + eps))

    for k in lit_features:
        print(f"    + {k}")

    # ── 7. Polynomial interaction terms ─────────────────────────────────────
    print("\n  [Polynomial Interaction Terms]")
    poly_features = {}

    ari_orig = X_idx[:, _INDEX_NAMES_ORIG.index("ARI")]
    poly_features["ARI_x_FRI"] = ari_orig * lit_features["FRI"]
    poly_features["ARI_x_FlavGitel"] = ari_orig * lit_features["FlavGitelson"]
    poly_features["PRI_x_SFI"] = lit_features["PRI"] * lit_features["SFI"]
    poly_features["Deriv520_x_FRI"] = (
        deriv_features.get("Deriv_520_580", np.zeros(X.shape[0]))
        * lit_features["FRI"]
    )
    poly_features["W575_x_ARI"] = (
        window_features.get("W_575_595", np.zeros(X.shape[0])) * ari_orig
    )

    for k in poly_features:
        print(f"    + {k}")

    # ── 8. ARI-based composite features (flavonol-specific) ────────────────
    print("\n  [ARI-based Composite Features (Flavonol-specific)]")
    ari_composites = _build_ari_composites(X, feat, X_bands, X_idx)
    for k in ari_composites:
        print(f"    + {k}")

    # ── 9. Combine all features ────────────────────────────────────────────
    all_new = {
        **window_features,
        **deriv_features,
        **lit_features,
        **poly_features,
        **ari_composites,
    }
    new_cols = np.column_stack(list(all_new.values()))
    new_names = list(all_new.keys())

    X_final = np.hstack([X_bands_sel, X_idx, new_cols])
    feat_final = feat_bands_sel + _INDEX_NAMES_ORIG + new_names

    print(f"\n  {'─' * 60}")
    print(f"  Initial features         : {X.shape[1]}")
    print(f"  Removed (var+corr)       : {n_removed_var + n_removed_corr}")
    print(f"  Added engineered features: {len(all_new)}")
    print(f"  TOTAL (final)            : {X_final.shape[1]}")
    print(f"  {'─' * 60}")

    # ── 10. Clean NaN/Inf ──────────────────────────────────────────────────
    bad_cols = np.where(~np.isfinite(X_final).all(axis=0))[0]
    if len(bad_cols) > 0:
        print(f"\n  [WARNING] {len(bad_cols)} features with NaN/Inf, cleaning...")
        X_final = np.where(np.isfinite(X_final), X_final, 0.0)

    # ── 11. Log transform ──────────────────────────────────────────────────
    y_flav_log = np.log1p(y_flav) if log_transform_flav else None

    if report:
        _print_corr_report(X_final, feat_final, y_flav, y_chl, valid)

    return X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress


def _build_ari_composites(X, feat, X_bands, X_idx):
    """ARI-based composite features for flavonol prediction."""
    n_bands = len(feat) - len(_INDEX_NAMES_ORIG)
    ari = X_idx[:, _INDEX_NAMES_ORIG.index("ARI")]
    gndvi = X_idx[:, _INDEX_NAMES_ORIG.index("GNDVI")]
    ndvi = X_idx[:, _INDEX_NAMES_ORIG.index("NDVI")]
    rvsi = X_idx[:, _INDEX_NAMES_ORIG.index("RVSI")]

    new = {}

    # Window averages (critical flavonol regions)
    w575 = _range_mean(X_bands, feat, 570, 590)
    w555 = _range_mean(X_bands, feat, 548, 565)
    w520 = _range_mean(X_bands, feat, 515, 535)
    w800 = _range_mean(X_bands, feat, 795, 810)

    # ARI composites
    new["ARI_W575_normNDVI"] = (ari * w575) / (ndvi + eps)
    new["ARI2_x_W575"] = (ari**2) * w575
    new["ARI_x_GreenSlope"] = ari * (w575 - w555)
    new["ARI_x_520_575_ratio"] = ari * (w520 / (w575 + eps))

    chl_norm_green = w555 / (w800 + eps)
    new["ARI_ChlFree"] = ari / (chl_norm_green + eps)
    new["ARI_minus_GNDVI"] = ari - gndvi

    # Green shoulder slope
    new["GreenShoulder_slope"] = (w575 - w520) / (55 + eps)
    new["W575_W440_ratio"] = w575 / (_range_mean(X_bands, feat, 435, 455) + eps)

    w400 = _range_mean(X_bands, feat, 395, 415)
    w440 = _range_mean(X_bands, feat, 435, 455)
    new["UV_Blue_slope"] = (w440 - w400) / (40 + eps)

    # Triple combination
    new["ARI_GS_UV"] = ari * new["GreenShoulder_slope"] * new["UV_Blue_slope"]

    # RVSI combinations
    new["RVSI_x_ARI"] = rvsi * ari
    new["RVSI_x_W575"] = rvsi * w575

    # Stress proxy (inverted GNDVI)
    new["InvGNDVI"] = 1.0 / (gndvi + eps)
    new["ARI_x_InvGNDVI"] = ari * new["InvGNDVI"]

    return new


def _print_corr_report(X, feat, y_flav, y_chl, valid):
    """Print top features by correlation."""
    r_flav = np.array(
        [np.corrcoef(X[valid, i], y_flav[valid])[0, 1] for i in range(X.shape[1])]
    )
    r_chl = np.array(
        [np.corrcoef(X[valid, i], y_chl[valid])[0, 1] for i in range(X.shape[1])]
    )
    print("\n  [REPORT] Top features by |r| with Flavonol:")
    top_flav = np.argsort(np.abs(r_flav))[::-1][:15]
    for i in top_flav:
        print(
            f"    {feat[i]:<30s}  r_flav={r_flav[i]:+.3f}  r_chl={r_chl[i]:+.3f}"
        )


def _diagnose_flavonol(X, feat, y_flav, y_chl, dataset_dir):
    """Diagnostic analysis: why is flavonol hard to predict?"""
    print("\n" + "━" * 70)
    print("DIAGNOSTIC: Why is Flavonol Hard to Predict?")
    print("━" * 70)

    valid = ~np.isnan(y_flav) & ~np.isnan(y_chl)
    yf = y_flav[valid]

    print(f"\n[1] DISTRIBUTION ANALYSIS")
    print(f"    n={len(yf)}, mean={yf.mean():.3f}, std={yf.std():.3f}")
    print(f"    min={yf.min():.3f}, max={yf.max():.3f}")
    cv = yf.std() / yf.mean() if yf.mean() != 0 else 0
    print(
        f"    CV (std/mean) = {cv:.3f}  "
        f"{'← LOW: narrow target range' if cv < 0.2 else ''}"
    )

    q25, q50, q75 = np.percentile(yf, [25, 50, 75])
    iqr = q75 - q25
    print(f"    Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}, IQR={iqr:.3f}")

    near_threshold = np.sum(np.abs(yf - 3.5) < 0.3)
    print(
        f"\n    Ph.Eur. threshold (3.5) ±0.3: {near_threshold} "
        f"({100 * near_threshold / len(yf):.1f}%)"
    )
    if near_threshold / len(yf) > 0.25:
        print("    ⚠ >25% samples near threshold → classification harder")

    r_cf = np.corrcoef(y_chl[valid], yf)[0, 1]
    print(f"\n[2] CHLOROPHYLL-FLAVONOL CORRELATION")
    print(f"    r = {r_cf:.3f}")
    if abs(r_cf) < 0.3:
        print("    ⚠ Weak correlation → flavonol independent from chlorophyll")

    n_bands = len(feat) - len(_INDEX_NAMES_ORIG)
    X_b = X[:, :n_bands][valid]
    r_bands = np.array(
        [abs(np.corrcoef(X_b[:, i], yf)[0, 1]) for i in range(n_bands)]
    )
    r_max = r_bands.max()
    r_max_nm = float(feat[np.argmax(r_bands)].replace("band_", ""))
    print(f"\n[3] STRONGEST BAND-FLAVONOL CORRELATION")
    print(f"    Max |r| = {r_max:.3f}  @ {r_max_nm:.1f} nm")
    if r_max < 0.5:
        print("    ⚠ Max |r|<0.5 → inherently weak spectral signal for flavonol")
        print(
            "      This is a known challenge in hyperspectral flavonol estimation"
        )

    print(f"\n[4] RECOMMENDATIONS")
    print(f"    a) R²~0.35 may be a reasonable ceiling for this dataset")
    print(f"    b) Consider classification (Ph.Eur. binary) instead")
    print(f"    c) Add stress level (y_stress) as feature")
    print(f"    d) Try variety-specific models")
    print("━" * 70)


def save_features(
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress, out_dir=OUTPUT_DIR
):
    """Save engineered features."""
    import shutil

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"), X_final)
    np.save(
        os.path.join(out_dir, "feature_names.npy"),
        np.array(feat_final, dtype=object),
    )
    np.save(os.path.join(out_dir, "y_chl.npy"), y_chl)
    np.save(os.path.join(out_dir, "y_flav.npy"), y_flav)
    np.save(os.path.join(out_dir, "y_stress.npy"), y_stress)
    if y_flav_log is not None:
        np.save(os.path.join(out_dir, "y_flav_log.npy"), y_flav_log)

    src_csv = os.path.join(DATASET_DIR, "dataset_full.csv")
    dst_csv = os.path.join(out_dir, "dataset_full.csv")
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        shutil.copy(src_csv, dst_csv)

    print(f"\n  Saved → {out_dir}")
    print(f"    X.npy           : {X_final.shape}")
    print(f"    feature_names   : {len(feat_final)} features")
    print(f"    samples         : {len(y_chl)}")


def plot_correlation(X_final, feat_final, y_flav, y_chl, save_dir=PLOT_DIR):
    """Plot feature correlations."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)

    r_flav = np.array(
        [
            abs(np.corrcoef(X_final[valid, i], y_flav[valid])[0, 1])
            for i in range(X_final.shape[1])
        ]
    )
    r_chl = np.array(
        [
            abs(np.corrcoef(X_final[valid, i], y_chl[valid])[0, 1])
            for i in range(X_final.shape[1])
        ]
    )

    new_start = len([f for f in feat_final if f.startswith("band_")])
    new_names = feat_final[new_start:]
    r_flav_new = r_flav[new_start:]
    r_chl_new = r_chl[new_start:]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, r_vals, title in [
        (axes[0], r_flav_new, "Engineered Features — |r| with Flavonol"),
        (axes[1], r_chl_new, "Engineered Features — |r| with Chlorophyll"),
    ]:
        order = np.argsort(r_vals)
        sorted_names = [new_names[i] for i in order]
        sorted_vals = r_vals[order]
        colors_bar = [
            "#e74c3c" if v >= 0.5 else "#3498db" if v >= 0.3 else "#95a5a6"
            for v in sorted_vals
        ]
        y_pos = range(len(sorted_names))
        ax.barh(
            list(y_pos), sorted_vals, color=colors_bar, edgecolor="k",
            linewidth=0.5
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel("|Pearson r|", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.7, label="|r|=0.5")
        ax.axvline(0.3, color="orange", linestyle=":", alpha=0.7, label="|r|=0.3")
        ax.set_xlim(0, 1)
        for i, (bar_y, v) in enumerate(zip(y_pos, sorted_vals)):
            ax.text(v + 0.01, bar_y, f"{v:.3f}", va="center", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle(
        "Feature Engineering — Correlation Analysis",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "fe_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def main(plot=False, report=False, diagnose=False):
    """Main pipeline."""
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress = build_features(
        report=report
    )

    save_features(X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress)

    if plot:
        print("\n  Plotting correlation analysis...")
        plot_correlation(X_final, feat_final, y_flav, y_chl)

    if diagnose:
        _diagnose_flavonol(
            np.load(os.path.join(DATASET_DIR, "X.npy")),
            np.load(
                os.path.join(DATASET_DIR, "feature_names.npy"), allow_pickle=True
            ).tolist(),
            y_flav,
            y_chl,
            DATASET_DIR,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save correlation plots",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print correlation report",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Flavonol prediction diagnostic",
    )

    args = parser.parse_args()
    main(plot=args.plot, report=args.report, diagnose=args.diagnose)
