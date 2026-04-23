"""
flav_deep_fe.py — Flavonol İçin Derinlemesine Özellik Mühendisliği
Konum: modul_6/ kökünde

Ne yapar:
  1. Korelasyon grafiğinden çıkan bulguya dayanarak ARI-merkezli
     kompozit özellikler üretir
  2. y_flav dağılımını, outlier'ları ve sınıf dengesizliğini analiz eder
     (neden bu kadar zor tahmin ediliyor — temel neden tespiti)
  3. dataset_output_fe_v3/ klasörüne kaydeder
     (sadece flavonol için optimize edilmiş, klorofile dokunulmaz)

Çalıştırma:
  python flav_deep_fe.py            # FE + kaydet
  python flav_deep_fe.py --diagnose # + neden zor? analizi
  python flav_deep_fe.py --plot     # + korelasyon grafikleri
"""

import numpy as np
import os, sys, argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

DATASET_DIR = os.path.join(_THIS_DIR, "dataset_output")
OUTPUT_DIR_V3 = os.path.join(_THIS_DIR, "dataset_output_fe_v3")
PLOT_DIR = os.path.join(_THIS_DIR, "model_outputs", "flav_deep_fe")

_INDEX_NAMES_ORIG = ["NDVI", "GNDVI", "ARI", "RVSI", "ZTM"]
eps = 1e-6


# ── YARDIMCILAR ─────────────────────────────────────────────────────────────
def _bidx(feat, nm):
    best = min(
        [
            (i, abs(float(f.replace("band_", "")) - nm))
            for i, f in enumerate(feat)
            if f.startswith("band_")
        ],
        key=lambda x: x[1],
    )
    return best[0]


def _col(X_bands, feat, nm):
    return X_bands[:, _bidx(feat, nm)]


def _range_mean(X_bands, feat, lo, hi):
    idxs = [
        i
        for i, f in enumerate(feat)
        if f.startswith("band_") and lo <= float(f.replace("band_", "")) <= hi
    ]
    return X_bands[:, idxs].mean(axis=1) if idxs else np.zeros(X_bands.shape[0])


# ── TANISAL ANALİZ ──────────────────────────────────────────────────────────
def diagnose_flav(X, feat, y_flav, y_chl):
    """
    Flavonol tahmininin neden zor olduğunu gösteren tanısal analiz.
    Konsola yazdırır, problemi net ortaya koyar.
    """
    print("\n" + "━" * 60)
    print("TANIM: FLAVONOL NEDEN ZOR TAHMİN EDİLİYOR?")
    print("━" * 60)

    valid = ~np.isnan(y_flav) & ~np.isnan(y_chl)
    yf = y_flav[valid]

    # 1. Dağılım analizi
    print(f"\n[1] DAĞILIM ANALİZİ")
    print(f"    n={len(yf)}, ort={yf.mean():.3f}, std={yf.std():.3f}")
    print(f"    min={yf.min():.3f}, max={yf.max():.3f}")
    print(
        f"    CV (std/mean) = {yf.std()/yf.mean():.3f}  "
        f"{'← DÜŞÜK: hedef dar aralıkta, tahmin zor' if yf.std()/yf.mean() < 0.2 else ''}"
    )

    # Çeyrekler
    q25, q50, q75 = np.percentile(yf, [25, 50, 75])
    iqr = q75 - q25
    print(f"    Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}, IQR={iqr:.3f}")

    # Eşik etrafında yoğunlaşma?
    near_threshold = np.sum(np.abs(yf - 3.5) < 0.3)
    print(
        f"\n    Ph.Eur. eşiği (3.5) ±0.3 içindeki örnek: {near_threshold} "
        f"({100*near_threshold/len(yf):.1f}%)"
    )
    if near_threshold / len(yf) > 0.25:
        print("    ⚠ Örneklerin >%25'i eşik yakınında → sınıflandırma zor")

    # 2. Klorofil ile korelasyon
    r_cf = np.corrcoef(y_chl[valid], yf)[0, 1]
    print(f"\n[2] KLOROFİL-FLAVONOL KORELASYONU")
    print(f"    r = {r_cf:.3f}")
    if abs(r_cf) < 0.3:
        print("    ⚠ Zayıf korelasyon → flavonol klorofilden bağımsız")
        print("      Klorofil sinyali iyi yakalanıyor ama flavonol farklı mekanizma")

    # 3. Bant korelasyonları — max nedir?
    n_bands = len(feat) - len(_INDEX_NAMES_ORIG)
    X_b = X[:, :n_bands][valid]
    r_bands = np.array([abs(np.corrcoef(X_b[:, i], yf)[0, 1]) for i in range(n_bands)])
    r_max = r_bands.max()
    r_max_nm = float(feat[np.argmax(r_bands)].replace("band_", ""))
    print(f"\n[3] EN GÜÇLÜ BANT-FLAVONOL KORELASYONU")
    print(f"    Maksimum |r| = {r_max:.3f}  @ {r_max_nm:.1f} nm")
    if r_max < 0.5:
        print("    ⚠ Ham bantlarda max |r|<0.5 → spektral sinyal inherently zayıf")
        print(
            "      Bu, hiperspektral data ile flavonol tahmininin genel bir sorunudur"
        )
        print("      Çözüm: Daha güçlü kompozit indeksler veya sınıflandırmaya odaklan")

    # 4. Çeşit/semptom bazlı varyans — var mı?
    csv_path = os.path.join(DATASET_DIR, "dataset_full.csv")
    if os.path.exists(csv_path):
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "variety" in df.columns:
            print(f"\n[4] ÇEŞİT BAZLI FLAVONOL VARYANSI")
            for v, grp in df.groupby("variety")["flav"] if "flav" in df.columns else []:
                pass
            var_stats = (
                df.groupby("variety")["flav"].agg(["mean", "std", "count"])
                if "flav" in df.columns
                else None
            )
            if var_stats is not None:
                print(var_stats.to_string())
                between_var = var_stats["mean"].std()
                within_var = var_stats["std"].mean()
                print(f"\n    Çeşitler arası std: {between_var:.3f}")
                print(f"    Çeşit içi ort std : {within_var:.3f}")
                if within_var > between_var:
                    print("    ⚠ Çeşit içi varyans > çeşitler arası varyans")
                    print("      → Flavonol çeşide göre değil, büyük olasılıkla")
                    print("        çevresel/stres faktörlerine göre değişiyor")
                    print(
                        "      → Stres bilgisi (y_stress) ek özellik olarak eklenmeli!"
                    )

    # 5. Öneriler
    print(f"\n[5] ÖNERİLER")
    print(f"    a) R²~0.35 bu veri seti için makul bir tavan olabilir")
    print(f"       (Literatürde hiperspektral flavonol R²: 0.40–0.65 arası)")
    print(f"    b) Regresyon yerine SINIFLANDIRMAYA odaklan (Ph.Eur. binary)")
    print(f"       SVM %78.4 — bu daha güvenilir ve pratik kullanım için yeterli")
    print(f"    c) Stres seviyesini (y_stress) özellik olarak ekle")
    print(f"    d) Çeşit bazlı ayrı model (multi-output) dene")
    print("━" * 60)


# ── ARI KOMPOZİT İNDEKSLER ─────────────────────────────────────────────────
def build_ari_composite(X, feat):
    """
    Korelasyon grafiğinden: ARI_x_FlavGitel, ARI, W575_x_ARI
    hepsi ~0.51 — ARI bu veri setinin flavonol anahtarı.

    Yeni strateji:
      - ARI'yi farklı bantlarla çarparak non-lineer kombinasyonlar üret
      - 575 nm bölgesini ARI ile daha çeşitli şekillerde birleştir
      - Klorofil bilgisini kısmen "söküp" flavonol-spesifik sinyal elde et
    """
    n_bands = len(feat) - len(_INDEX_NAMES_ORIG)
    X_b = X[:, :n_bands]
    X_idx = X[:, n_bands:]
    ari = X_idx[:, _INDEX_NAMES_ORIG.index("ARI")]
    gndvi = X_idx[:, _INDEX_NAMES_ORIG.index("GNDVI")]
    ndvi = X_idx[:, _INDEX_NAMES_ORIG.index("NDVI")]
    rvsi = X_idx[:, _INDEX_NAMES_ORIG.index("RVSI")]

    new = {}

    # ── Pencere ortalamaları (flavonol kritik bölgeler) ──────────────────────
    w575 = _range_mean(X_b, feat, 570, 590)  # en güçlü bölge
    w555 = _range_mean(X_b, feat, 548, 565)
    w520 = _range_mean(X_b, feat, 515, 535)
    w440 = _range_mean(X_b, feat, 435, 455)
    w400 = _range_mean(X_b, feat, 395, 415)
    w700 = _range_mean(X_b, feat, 695, 715)
    w750 = _range_mean(X_b, feat, 745, 760)
    w800 = _range_mean(X_b, feat, 795, 810)

    # ── 1. ARI tabanlı kompozitler ───────────────────────────────────────────
    # Grafikteki en güçlü 3'ü (0.51) → daha da güçlendirmeye çalış

    # ARI × W575 — grafikteki W575_x_ARI'yi zaten FE v2 üretiyordu ama
    # şimdi bunu normalize ediyoruz: NDVI'ye böl → klorofil etkisini söküyoruz
    new["ARI_W575_normNDVI"] = (ari * w575) / (ndvi + eps)

    # ARI² × W575 — quadratic ARI terimi
    new["ARI2_x_W575"] = (ari**2) * w575

    # ARI × (W575 - W555) → yeşil omuz farkı ile çarpım
    new["ARI_x_GreenSlope"] = ari * (w575 - w555)

    # ARI × W520/W575 oranı → yeşil absorpsiyon oranı
    new["ARI_x_520_575_ratio"] = ari * (w520 / (w575 + eps))

    # ── 2. Flavonol-spesifik "arındırılmış" indeksler ────────────────────────
    # Fikir: Flavonol hem yeşil hem mor bölgede absorpsiyon yapar.
    # Klorofil ağırlıklı olarak kırmızı (670) ve mavi (450) absorpsiyon yapar.
    # Fark alarak klorofil etkisini azalt:

    # Yeşil/NIR normalize → klorofil etkisi büyük, flavonolde küçük
    # Klorofil-düzeltilmiş yeşil: W555 / W800  (NIR'a normalize et)
    chl_norm_green = w555 / (w800 + eps)

    # ARI bu oran ile zayıf korelasyondaysa flavonol-spesifik sinyal demek
    new["ARI_ChlFree"] = ari / (chl_norm_green + eps)

    # GNDVI zaten klorofil proxy → ARI'den GNDVI etkisini çıkar
    new["ARI_minus_GNDVI"] = ari - gndvi

    # ── 3. 575 nm bölgesi türev ve oran özellikleri ──────────────────────────
    # Yeşil omuz eğimi: (W575 - W520) / (575 - 520)
    new["GreenShoulder_slope"] = (w575 - w520) / (55 + eps)

    # W575 / W440 — sarı-yeşil / mavi oranı (flavonol mavi absorbe eder)
    new["W575_W440_ratio"] = w575 / (w440 + eps)

    # UV-mavi eğim: (W440 - W400) / (440 - 400)
    new["UV_Blue_slope"] = (w440 - w400) / (40 + eps)

    # ── 4. Üçlü kombinasyon ─────────────────────────────────────────────────
    # ARI × GreenShoulder × UV sinyal
    new["ARI_GS_UV"] = ari * new["GreenShoulder_slope"] * new["UV_Blue_slope"]

    # ── 5. RVSI tabanlı ─────────────────────────────────────────────────────
    # RVSI (Red-edge Vegetation Stress Index) flavonol ile ilişkili olabilir
    new["RVSI_x_ARI"] = rvsi * ari
    new["RVSI_x_W575"] = rvsi * w575

    # ── 6. Stres proxy (klorofil kaybı oranı) ───────────────────────────────
    # Düşük klorofil → yüksek stres → yüksek flavonol birikimi teorisi
    # Klorofil ters proxy: 1 / GNDVI
    new["InvGNDVI"] = 1.0 / (gndvi + eps)
    new["ARI_x_InvGNDVI"] = ari * new["InvGNDVI"]

    print(f"  ARI kompozit özellikler ({len(new)}):")
    for k in new:
        print(f"    + {k}")

    return new


# ── ANA FONKSİYON ───────────────────────────────────────────────────────────
def build_flav_v3(dataset_dir=DATASET_DIR, diagnose=False):
    print("\n" + "=" * 65)
    print("FLAV DEEP FE v3 — ARI Kompozit + Tanısal")
    print("=" * 65)

    X = np.load(os.path.join(dataset_dir, "X.npy"))
    y_chl = np.load(os.path.join(dataset_dir, "y_chl.npy"))
    y_flav = np.load(os.path.join(dataset_dir, "y_flav.npy"))
    y_stress = np.load(os.path.join(dataset_dir, "y_stress.npy"))
    feat = np.load(
        os.path.join(dataset_dir, "feature_names.npy"), allow_pickle=True
    ).tolist()

    print(f"  Ham veri: X={X.shape}")

    if diagnose:
        diagnose_flav(X, feat, y_flav, y_chl)

    # FE v2 zaten dataset_output_fe_v2'deydi, ama rename çalışmadı.
    # Bu script ham dataset_output'tan başlıyor ve tüm FE'yi kendi üretiyor.

    n_bands = len(feat) - len(_INDEX_NAMES_ORIG)
    X_b = X[:, :n_bands]
    X_idx = X[:, n_bands:]

    # ── Mevcut FE v2 özelliklerini de ekle (inline, dosyaya bağımlı değil) ──
    from feature_engineering_v2 import build_features_v2

    print("\n  [FE v2 tabanlı özellikler yükleniyor...]")
    X_v2, feat_v2, _, _, _, _ = build_features_v2(
        dataset_dir=dataset_dir,
        corr_threshold=0.05,
        var_threshold=1e-5,
        log_transform_flav=False,
        report=False,
    )

    # ── ARI kompozit özellikler ──────────────────────────────────────────────
    print("\n  [ARI Kompozit Özellikler]")
    ari_feats = build_ari_composite(X, feat)

    # ── Birleştir ────────────────────────────────────────────────────────────
    ari_cols = np.column_stack(list(ari_feats.values()))
    ari_names = list(ari_feats.keys())

    X_final = np.hstack([X_v2, ari_cols])
    feat_final = feat_v2 + ari_names

    # NaN/Inf temizle
    bad = ~np.isfinite(X_final)
    if bad.any():
        print(f"  [UYARI] {bad.sum()} NaN/Inf → 0 yapılıyor")
        X_final = np.where(np.isfinite(X_final), X_final, 0.0)

    print(f"\n  {'─'*55}")
    print(f"  FE v2 özellik sayısı   : {X_v2.shape[1]}")
    print(f"  Eklenen ARI kompozit   : {len(ari_feats)}")
    print(f"  TOPLAM (v3)            : {X_final.shape[1]}")
    print(f"  {'─'*55}")

    y_flav_log = np.log1p(y_flav)
    return X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress


# ── KAYDET ───────────────────────────────────────────────────────────────────
def save_v3(
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress, out_dir=OUTPUT_DIR_V3
):
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X_final)
    np.save(
        os.path.join(out_dir, "feature_names.npy"), np.array(feat_final, dtype=object)
    )
    np.save(os.path.join(out_dir, "y_chl.npy"), y_chl)
    np.save(os.path.join(out_dir, "y_flav.npy"), y_flav)
    np.save(os.path.join(out_dir, "y_stress.npy"), y_stress)
    np.save(os.path.join(out_dir, "y_flav_log.npy"), y_flav_log)
    src = os.path.join(DATASET_DIR, "dataset_full.csv")
    dst = os.path.join(out_dir, "dataset_full.csv")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
    print(f"\n  Kaydedildi → {out_dir}")
    print(f"    X: {X_final.shape}, özellik: {len(feat_final)}")


# ── KORELASyon GRAFİĞİ ───────────────────────────────────────────────────────
def plot_v3(X_final, feat_final, y_flav, y_chl, save_dir=PLOT_DIR):
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    valid = ~np.isnan(y_flav) & ~np.isnan(y_chl)

    # Sadece yeni (non-band) özellikler
    new_start = len([f for f in feat_final if f.startswith("band_")])
    new_names = feat_final[new_start:]
    r_flav = np.array(
        [
            abs(np.corrcoef(X_final[valid, new_start + i], y_flav[valid])[0, 1])
            for i in range(len(new_names))
        ]
    )

    order = np.argsort(r_flav)[-30:]  # top 30
    fig, ax = plt.subplots(figsize=(10, 10))
    colors_bar = [
        "#e74c3c" if r_flav[i] >= 0.5 else "#3498db" if r_flav[i] >= 0.3 else "#95a5a6"
        for i in order
    ]
    ax.barh(range(len(order)), r_flav[order], color=colors_bar, edgecolor="k", lw=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([new_names[i] for i in order], fontsize=9)
    ax.axvline(0.5, color="red", ls=":", alpha=0.7, label="|r|=0.5")
    ax.axvline(0.3, color="orange", ls=":", alpha=0.7, label="|r|=0.3")
    ax.set_xlabel("|Pearson r|", fontsize=11)
    ax.set_title("FE v3 — Top 30 Özellik (Flavonol)", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "fe_v3_flav_top30.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Grafik: {path}")

    # En yüksek |r| özellikleri konsola yaz
    top10 = np.argsort(r_flav)[::-1][:10]
    print("\n  [v3] Flavonol — Top 10 özellik:")
    for i in top10:
        print(f"    {new_names[i]:<30s}  |r|={r_flav[i]:.3f}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(diagnose=False, plot=False):
    X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress = build_flav_v3(
        dataset_dir=DATASET_DIR,
        diagnose=diagnose,
    )
    save_v3(X_final, feat_final, y_chl, y_flav, y_flav_log, y_stress)

    if plot:
        plot_v3(X_final, feat_final, y_flav, y_chl)

    print("\n" + "=" * 65)
    print("TAMAMLANDI — Çalıştırmak için:")
    print()
    print("  PowerShell'de modul_6/ klasöründen:")
    print()
    print("  # Takas et")
    print("  Rename-Item dataset_output_fe dataset_output_fe_v1_backup")
    print("  Rename-Item dataset_output_fe_v3 dataset_output_fe")
    print()
    print("  # Modelleri çalıştır")
    print("  python pipeline_models.py --fe --binary")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Flavonol neden zor tahmin ediliyor — analiz",
    )
    parser.add_argument("--plot", action="store_true", help="Korelasyon grafikleri")
    args = parser.parse_args()
    main(diagnose=args.diagnose, plot=args.plot)
