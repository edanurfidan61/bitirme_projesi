"""
feature_engineering_v2.py — Gelişmiş Özellik Mühendisliği (Flavonol Odaklı)
Konum: modul_6/ kökünde (feature_engineering.py ile aynı seviye)

Değişiklikler (v1'e göre):
  1. Komşu bant pencere ortalamaları  → multikollineariteyi kırar
  2. Flavonol-spesifik türev özellikleri  → yeşil omuz eğimi (520–580 nm)
  3. Genişletilmiş literatür indeksleri  → FRI, SFI, ANTH, REIP
  4. Varyans / korelasyon bazlı otomatik bant seçimi  → düşük bilgili bantları atar
  5. Polinom etkileşim terimleri (seçili indeksler arası)

Çalıştırma:
  python feature_engineering.py            # FE + kaydet
  python feature_engineering.py --plot     # + korelasyon grafikleri
  python feature_engineering.py --report   # + konsol raporu (hangi bantlar atıldı)
"""

import numpy as np
import os
import sys
import argparse

# ── PATH ────────────────────────────────────────────────────────────────────
_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
_MODUL6_DIR    = _THIS_DIR
sys.path.insert(0, _MODUL6_DIR)

DATASET_DIR    = os.path.join(_MODUL6_DIR, "dataset_output")
OUTPUT_DIR_FE  = os.path.join(_MODUL6_DIR, "dataset_output_fe_v2")
PLOT_DIR       = os.path.join(_MODUL6_DIR, "model_outputs", "feature_engineering_v2")

_INDEX_NAMES_ORIG = ["NDVI", "GNDVI", "ARI", "RVSI", "ZTM"]

eps = 1e-6


# ── YARDIMCILAR ─────────────────────────────────────────────────────────────
def _band_idx(feature_names: list, target_nm: float) -> int:
    """Hedef dalga boyuna en yakın bant sütun indeksini döndürür."""
    band_wl = []
    for i, name in enumerate(feature_names):
        if name.startswith("band_"):
            try:
                band_wl.append((i, float(name.replace("band_", ""))))
            except ValueError:
                pass
    if not band_wl:
        raise ValueError("'band_XXX' formatında isim bulunamadı.")
    best_i, best_nm = min(band_wl, key=lambda x: abs(x[1] - target_nm))
    return best_i


def _band_range_indices(feature_names: list, lo_nm: float, hi_nm: float) -> list:
    """lo_nm – hi_nm aralığındaki tüm bant sütun indekslerini döndürür."""
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
    """Belirtilen aralıktaki bantların piksel bazlı ortalamasını alır."""
    idxs = _band_range_indices(feature_names, lo_nm, hi_nm)
    if not idxs:
        raise ValueError(f"{lo_nm}–{hi_nm} nm aralığında bant bulunamadı.")
    return X_bands[:, idxs].mean(axis=1), idxs


def _window_deriv(X_bands, feature_names, lo_nm, hi_nm):
    """
    Belirtilen aralıktaki bantların merkezi fark türevini (birinci türev) alır.
    Eğim = Δreflektans / Δdalga_boyu
    """
    idxs = _band_range_indices(feature_names, lo_nm, hi_nm)
    if len(idxs) < 2:
        raise ValueError(f"{lo_nm}–{hi_nm} nm aralığında türev için yeterli bant yok.")
    wls = np.array([float(feature_names[i].replace("band_", "")) for i in idxs])
    refs = X_bands[:, idxs]
    # Sonlu fark: (son - ilk) / (son_wl - ilk_wl)
    slope = (refs[:, -1] - refs[:, 0]) / (wls[-1] - wls[0] + eps)
    return slope


# ── ANA FONKSİYON ───────────────────────────────────────────────────────────
def build_features_v2(
    dataset_dir: str = DATASET_DIR,
    corr_threshold: float = 0.05,   # |r| < bu değer olan bantları at
    var_threshold: float  = 1e-5,   # varyansı bu değerin altında olan bantları at
    log_transform_flav: bool = True,
    report: bool = False,
) -> tuple:
    """
    Gelişmiş özellik mühendisliği.

    Döndürür
    --------
    X_sel, feat_sel, y_chl, y_flav, y_flav_log, y_stress
    """
    print("\n" + "=" * 65)
    print("ÖZELLİK MÜHENDİSLİĞİ v2  (Flavonol Odaklı)")
    print("=" * 65)

    # ── 1. Veri yükle ────────────────────────────────────────────────────────
    X        = np.load(os.path.join(dataset_dir, "X.npy"))
    y_chl    = np.load(os.path.join(dataset_dir, "y_chl.npy"))
    y_flav   = np.load(os.path.join(dataset_dir, "y_flav.npy"))
    y_stress = np.load(os.path.join(dataset_dir, "y_stress.npy"))
    feat     = np.load(os.path.join(dataset_dir, "feature_names.npy"),
                       allow_pickle=True).tolist()

    print(f"  Ham veri       : X={X.shape}, özellik={len(feat)}")
    n_bands_orig = len(feat) - len(_INDEX_NAMES_ORIG)
    X_bands = X[:, :n_bands_orig]
    X_idx   = X[:, n_bands_orig:]

    # ── 2. Ham bantlar için varyans filtresi ─────────────────────────────────
    band_vars = X_bands.var(axis=0)
    var_mask  = band_vars >= var_threshold
    X_bands_f = X_bands[:, var_mask]
    feat_bands_f = [feat[i] for i in range(n_bands_orig) if var_mask[i]]
    n_removed_var = n_bands_orig - var_mask.sum()
    print(f"\n  [Varyans filtresi] Kaldırılan bant: {n_removed_var}  "
          f"(eşik={var_threshold:.0e})")

    # ── 3. Korelasyon filtresi (y_flav + y_chl birlikte) ────────────────────
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    r_flav = np.array([
        abs(np.corrcoef(X_bands_f[valid, i], y_flav[valid])[0, 1])
        for i in range(X_bands_f.shape[1])
    ])
    r_chl = np.array([
        abs(np.corrcoef(X_bands_f[valid, i], y_chl[valid])[0, 1])
        for i in range(X_bands_f.shape[1])
    ])
    # Her iki hedeften en az birinde anlamlı korelasyon varsa tut
    corr_mask = (r_flav >= corr_threshold) | (r_chl >= corr_threshold)
    X_bands_sel  = X_bands_f[:, corr_mask]
    feat_bands_sel = [feat_bands_f[i] for i in range(len(feat_bands_f)) if corr_mask[i]]
    n_removed_corr = corr_mask.shape[0] - corr_mask.sum()
    print(f"  [Korelasyon filtresi] Kaldırılan bant: {n_removed_corr}  "
          f"(|r|<{corr_threshold} her iki hedef için)")
    print(f"  Kalan ham bant : {X_bands_sel.shape[1]}")

    if report:
        removed_corr = [feat_bands_f[i] for i in range(len(feat_bands_f)) if not corr_mask[i]]
        print(f"\n  Korelasyon filtresiyle atılan bantlar ({len(removed_corr)}):")
        for b in removed_corr:
            print(f"    - {b}")

    # ── 4. PENCERE ORTALAMA özellikleri ──────────────────────────────────────
    # Komşu bantları grupla → gürültüyü azalt, multikollineariteyi kır
    print("\n  [Pencere Ortalamaları]")
    window_features = {}

    # Flavonol için kritik bölgeler
    flav_windows = [
        ("W_400_420",  400,  420),   # UV-A flavonol absorpsiyonu
        ("W_440_460",  440,  460),   # Mavi bant
        ("W_520_540",  520,  540),   # Yeşil omuz başlangıcı
        ("W_555_575",  555,  575),   # Yeşil tepe (flavonol duyarlı)
        ("W_575_595",  575,  595),   # Sarı-yeşil geçiş (feature importance'ta çok önemli)
        ("W_640_660",  640,  660),   # Kırmızı kenar öncesi
    ]
    # Klorofil için
    chl_windows = [
        ("W_700_720",  700,  720),   # Red-edge dip
        ("W_720_740",  720,  740),   # Red-edge yükselen
        ("W_740_760",  740,  760),   # Red-edge tepe
        ("W_780_800",  780,  800),   # NIR plato
    ]

    all_windows = flav_windows + chl_windows
    used_band_sets = []  # hangi bantların gruplandığını takip et

    for name, lo, hi in all_windows:
        try:
            mean_vals, idxs = _window_mean(X_bands, feat, lo, hi)
            window_features[name] = mean_vals
            used_band_sets.append((name, lo, hi, len(idxs)))
            print(f"    {name}: {lo}–{hi} nm  ({len(idxs)} bant)")
        except ValueError as e:
            print(f"    [UYARI] {name}: {e}")

    # ── 5. TÜREV özellikleri ─────────────────────────────────────────────────
    print("\n  [Türev Özellikleri — Spektral Eğimler]")
    deriv_features = {}

    deriv_regions = [
        ("Deriv_520_580",  520,  580),   # Yeşil omuz eğimi → Flavonol proxy
        ("Deriv_680_740",  680,  740),   # Red-edge eğimi → Klorofil proxy
        ("Deriv_400_500",  400,  500),   # UV eğimi → UV-absorbent pigmentler
        ("Deriv_740_800",  740,  800),   # NIR eğimi → yapı
    ]

    for name, lo, hi in deriv_regions:
        try:
            slope = _window_deriv(X_bands, feat, lo, hi)
            deriv_features[name] = slope
            print(f"    {name}: {lo}–{hi} nm")
        except ValueError as e:
            print(f"    [UYARI] {name}: {e}")

    # ── 6. LİTERATÜR İNDEKSLERİ ─────────────────────────────────────────────
    print("\n  [Literatür İndeksleri]")

    def col(nm):
        return X_bands[:, _band_idx(feat, nm)]

    lit_features = {}

    # --- Flavonol Spesifik ---
    # FRI (Flavonoid Reflectance Index): R690/R600
    # Gitelson et al. 2006 — doğrudan flavonoid içeriğiyle ilişkili
    lit_features["FRI"] = col(690) / (col(600) + eps)

    # SFI (Simple Flavonol Index): R440/R690
    # Mavi absorpsiyon / kırmızı absorpsiyon oranı
    lit_features["SFI"] = col(440) / (col(690) + eps)

    # ANTH (Anthocyanin proxy, flavonol ile korelasyonlu):
    # (1/R550 - 1/R700) * R800  [Gitelson 2001]
    lit_features["ANTH"] = (1.0 / (col(550) + eps) - 1.0 / (col(700) + eps)) * col(800)

    # Flav indeksi (Gitelson 2006): R800/R550 - 1
    lit_features["FlavGitelson"] = col(800) / (col(550) + eps) - 1

    # REIP (Red-Edge Inflection Point proxy): 
    # Lineer interpolasyon: 700 + 40 * (R670+R780)/2 - R700) / (R740 - R700)
    # Flavonol ile red-edge shift arasında güçlü bağ
    r700 = col(700); r740 = col(740); r670 = col(670); r780 = col(780)
    reip_num = (r670 + r780) / 2.0 - r700
    reip_den = r740 - r700 + eps
    lit_features["REIP"] = 700 + 40 * (reip_num / reip_den)

    # PRI (Photochemical Reflectance Index): (R531 - R570) / (R531 + R570)
    # Karotenoid/klorofil oranıyla ilişkili, stres indikatörü
    r531 = col(531); r570 = col(570)
    lit_features["PRI"] = (r531 - r570) / (r531 + r570 + eps)

    # --- Klorofil (mevcut FE'dekileri koru, iyileştir) ---
    # CIRed-Edge: (R750/R705) - 1
    lit_features["CIRedEdge"] = col(750) / (col(705) + eps) - 1

    # RENDVI: (R800 - R715) / (R800 + R715)
    lit_features["RENDVI"] = (col(800) - col(715)) / (col(800) + col(715) + eps)

    # GreenRatio: R550/R670
    lit_features["GreenRatio"] = col(550) / (col(670) + eps)

    # MCARI: [(R700-R670) - 0.2*(R700-R550)] * (R700/R670)
    r705 = col(705); r550 = col(550); r670_v = col(670)
    mcari_a = r705 - r670_v
    mcari_b = r705 - r550
    lit_features["MCARI"] = (mcari_a - 0.2 * mcari_b) * (r705 / (r670_v + eps))

    for k in lit_features:
        print(f"    + {k}")

    # ── 7. POLİNOM ETKİLEŞİM TERİMLERİ ─────────────────────────────────────
    # Sadece flavonol için anlamlı kombinasyonlar
    print("\n  [Polinom Etkileşim Terimleri]")
    poly_features = {}

    # ARI × FRI  (antosyanin × flavonoid indeksi etkileşimi)
    ari_orig = X_idx[:, _INDEX_NAMES_ORIG.index("ARI")]
    poly_features["ARI_x_FRI"]       = ari_orig * lit_features["FRI"]
    poly_features["ARI_x_FlavGitel"] = ari_orig * lit_features["FlavGitelson"]
    poly_features["PRI_x_SFI"]       = lit_features["PRI"] * lit_features["SFI"]
    poly_features["Deriv520_x_FRI"]  = deriv_features.get("Deriv_520_580", np.zeros(X.shape[0])) * lit_features["FRI"]
    poly_features["W575_x_ARI"]      = window_features.get("W_575_595", np.zeros(X.shape[0])) * ari_orig

    for k in poly_features:
        print(f"    + {k}")

    # ── 8. HEPSİNİ BİRLEŞTİR ────────────────────────────────────────────────
    # Seçilmiş ham bantlar + orijinal indeksler + yeni özellikler
    all_new = {**window_features, **deriv_features, **lit_features, **poly_features}
    new_cols  = np.column_stack(list(all_new.values()))
    new_names = list(all_new.keys())

    # Matris: seçilmiş bantlar | orijinal indeksler | yeni özellikler
    X_final   = np.hstack([X_bands_sel, X_idx, new_cols])
    feat_final = feat_bands_sel + _INDEX_NAMES_ORIG + new_names

    print(f"\n  {'─'*55}")
    print(f"  Başlangıç özellik sayısı : {X.shape[1]}")
    print(f"  Varyans/korelasyon ile azaltılan bant : "
          f"{n_removed_var + n_removed_corr}")
    print(f"  Eklenen yeni özellik     : {len(all_new)}")
    print(f"  TOPLAM (final)           : {X_final.shape[1]}")
    print(f"  {'─'*55}")

    # ── 9. NaN/Inf kontrolü ──────────────────────────────────────────────────
    bad_cols = np.where(~np.isfinite(X_final).all(axis=0))[0]
    if len(bad_cols) > 0:
        print(f"\n  [UYARI] {len(bad_cols)} özellikte NaN/Inf var, temizleniyor...")
        X_final = np.where(np.isfinite(X_final), X_final, 0.0)

    # ── 10. Log transform ────────────────────────────────────────────────────
    y_flav_log = np.log1p(y_flav) if log_transform_flav else None

    if report:
        _print_corr_report(X_final, feat_final, y_flav, y_chl, valid)

    return X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress


# ── RAPOR ────────────────────────────────────────────────────────────────────
def _print_corr_report(X, feat, y_flav, y_chl, valid):
    """Tüm özelliklerin korelasyonunu raporlar."""
    r_flav = np.array([
        np.corrcoef(X[valid, i], y_flav[valid])[0, 1]
        for i in range(X.shape[1])
    ])
    r_chl = np.array([
        np.corrcoef(X[valid, i], y_chl[valid])[0, 1]
        for i in range(X.shape[1])
    ])
    print("\n  [RAPOR] Flavonol için en yüksek |r| özellikler:")
    top_flav = np.argsort(np.abs(r_flav))[::-1][:15]
    for i in top_flav:
        print(f"    {feat[i]:<30s}  r_flav={r_flav[i]:+.3f}  r_chl={r_chl[i]:+.3f}")


# ── KAYDET ───────────────────────────────────────────────────────────────────
def save_features_v2(X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress,
                     out_dir=OUTPUT_DIR_FE):
    import shutil
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"),             X_final)
    np.save(os.path.join(out_dir, "feature_names.npy"), np.array(feat_final, dtype=object))
    np.save(os.path.join(out_dir, "y_chl.npy"),         y_chl)
    np.save(os.path.join(out_dir, "y_flav.npy"),        y_flav)
    np.save(os.path.join(out_dir, "y_stress.npy"),      y_stress)
    if y_flav_log is not None:
        np.save(os.path.join(out_dir, "y_flav_log.npy"), y_flav_log)

    src_csv = os.path.join(DATASET_DIR, "dataset_full.csv")
    dst_csv = os.path.join(out_dir, "dataset_full.csv")
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        shutil.copy(src_csv, dst_csv)

    print(f"\n  Kaydedildi → {out_dir}")
    print(f"    X.npy          : {X_final.shape}")
    print(f"    feature_names  : {len(feat_final)} özellik")
    print(f"    Örnek sayısı   : {len(y_chl)}")


# ── KORELASyon GRAFİĞİ ───────────────────────────────────────────────────────
def plot_v2_correlation(X_final, feat_final, y_flav, y_chl, save_dir=PLOT_DIR):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)

    r_flav = np.array([
        abs(np.corrcoef(X_final[valid, i], y_flav[valid])[0, 1])
        for i in range(X_final.shape[1])
    ])
    r_chl = np.array([
        abs(np.corrcoef(X_final[valid, i], y_chl[valid])[0, 1])
        for i in range(X_final.shape[1])
    ])

    # Sadece bant olmayan (yeni) özellikleri göster
    new_start = len([f for f in feat_final if f.startswith("band_")])
    new_names = feat_final[new_start:]
    r_flav_new = r_flav[new_start:]
    r_chl_new  = r_chl[new_start:]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, r_vals, color, title in [
        (axes[0], r_flav_new, "darkorange", "Yeni Özellikler — |r| ile Flavonol"),
        (axes[1], r_chl_new,  "forestgreen","Yeni Özellikler — |r| ile Klorofil"),
    ]:
        order = np.argsort(r_vals)
        sorted_names = [new_names[i] for i in order]
        sorted_vals  = r_vals[order]
        colors_bar   = ["#e74c3c" if v >= 0.5 else "#3498db" if v >= 0.3 else "#95a5a6"
                        for v in sorted_vals]
        y_pos = range(len(sorted_names))
        ax.barh(list(y_pos), sorted_vals, color=colors_bar, edgecolor='k', linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel("|Pearson r|", fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.axvline(0.5, color='red',    linestyle=':', alpha=0.7, label='|r|=0.5 (iyi)')
        ax.axvline(0.3, color='orange', linestyle=':', alpha=0.7, label='|r|=0.3 (orta)')
        ax.set_xlim(0, 1)
        for i, (bar_y, v) in enumerate(zip(y_pos, sorted_vals)):
            ax.text(v + 0.01, bar_y, f"{v:.3f}", va='center', fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle("FE v2 — Yeni Özellikler Korelasyon Analizi\n"
                 "(Kırmızı: |r|≥0.5 • Mavi: |r|≥0.3 • Gri: zayıf)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "fe_v2_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"  Grafik kaydedildi: {path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(plot=False, report=False):
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress = build_features_v2(
        dataset_dir=DATASET_DIR,
        corr_threshold=0.05,
        var_threshold=1e-5,
        log_transform_flav=True,
        report=report,
    )
    save_features_v2(X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress)

    if plot:
        plot_v2_correlation(X_final, feat_final, y_flav, y_chl)

    print("\n" + "=" * 65)
    print("ÖZELLİK MÜHENDİSLİĞİ v2 TAMAMLANDI")
    print("=" * 65)
    print(f"  Çıktı klasörü : dataset_output_fe_v2/")
    print(f"  Modelleri şöyle çalıştır:")
    print(f"    python pipeline_models.py --fe_dir dataset_output_fe_v2 --binary")
    print()
    print("  Eğer pipeline_models.py --fe_dir parametresini desteklemiyorsa,")
    print("  dataset_output_fe_v2/ klasörünü dataset_output_fe/ ile takas et:")
    print("    [Windows] rename dataset_output_fe dataset_output_fe_v1_backup")
    print("    [Windows] rename dataset_output_fe_v2 dataset_output_fe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Özellik Mühendisliği v2 (Flavonol Odaklı)")
    parser.add_argument("--plot",   action="store_true", help="Korelasyon grafikleri çiz")
    parser.add_argument("--report", action="store_true", help="Konsol korelasyon raporu")
    args = parser.parse_args()
    main(plot=args.plot, report=args.report)