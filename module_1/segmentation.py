"""
segmentation.py
===============
Hiperspektral görüntüden yaprak piksellerini arka plandan ayırmak için
segmentasyon yöntemleri.

v4 Değişiklikler:
  - Convex hull kaldırıldı (yaprak lobları arası boşlukları dolduruyordu,
    flavonol/indeks tahminlerini bozuyordu)
  - NDVI eşiği adaptif yapıldı (Otsu yöntemi)
  - Morfolojik temizlik: 3×3, closing→fill_holes→opening
  - fill_holes sadece tamamen kapalı delikleri doldurur
    (loblar arası açık boşluklara dokunmaz)
  - PCA+K-means geri getirildi ama yaprak kümesi seçiminde
    birden fazla küme dahil edilebilir (arka plan hariç tutma yaklaşımı)

Yöntemler:
  1. NDVI tabanlı eşikleme (adaptif Otsu)
  2. K-means kümeleme
  3. PCA tabanlı segmentasyon
  4. Hibrit v4: PCA ile arka plan eleme → NDVI ile disk eleme
               → yumuşak morfolojik temizlik
"""

import numpy as np
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_fill_holes,
    binary_dilation,
)
from scipy.ndimage import label as ndimage_label
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_MODUL0_DIR = os.path.join(_PROJECT_DIR, "modul_0")
if _MODUL0_DIR not in sys.path:
    sys.path.insert(0, _MODUL0_DIR)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================


def _find_band(wavelengths, target_nm):
    wl = np.array(wavelengths, dtype=np.float64)
    return int(np.argmin(np.abs(wl - target_nm)))


def _otsu_threshold(values):
    """
    1D array için Otsu eşik değeri hesaplar.
    Bimodal dağılımlarda (yaprak vs arka plan) optimal eşiği bulur.
    """
    # NaN ve Inf temizle
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return 0.3  # fallback

    # Histogram
    n_bins = 256
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.3

    # Otsu: sınıflar-arası varyansı maksimize et
    best_threshold = bin_centers[0]
    best_variance = 0

    cumsum_w = np.cumsum(hist)
    cumsum_wm = np.cumsum(hist * bin_centers)
    total_mean = cumsum_wm[-1]

    for i in range(1, n_bins):
        w0 = cumsum_w[i]
        w1 = total - w0

        if w0 == 0 or w1 == 0:
            continue

        mean0 = cumsum_wm[i] / w0
        mean1 = (total_mean - cumsum_wm[i]) / w1

        variance = w0 * w1 * (mean0 - mean1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]

    return best_threshold


def _morphological_cleanup(mask, struct_size=3):
    """
    Yumuşak morfolojik temizlik: closing → fill_holes → opening.
    struct_size=3 ile ince yaprak kenarları korunur.
    fill_holes sadece tamamen kapalı delikleri doldurur.
    """
    structure = np.ones((struct_size, struct_size), dtype=bool)

    closed = binary_closing(mask, structure=structure, iterations=2)
    filled = binary_fill_holes(closed)
    cleaned = binary_opening(filled, structure=structure, iterations=1)

    return cleaned.astype(bool)


def _remove_small_regions(mask, min_area_ratio=0.005):
    """Çok küçük bölgeleri temizle."""
    labeled, n_labels = ndimage_label(mask)
    if n_labels == 0:
        return mask

    min_area = mask.size * min_area_ratio
    cleaned = np.zeros_like(mask)

    for i in range(1, n_labels + 1):
        region = labeled == i
        if np.sum(region) >= min_area:
            cleaned[region] = True

    return cleaned


def _detect_white_reference(data, wavelengths):
    """
    Beyaz referans diski tespit et.
    Kriterler: yüksek yansıma + düşük NDVI + düz spektral profil.
    """
    mean_ref = np.mean(data, axis=2)
    high_ref = mean_ref > 0.60

    idx_800 = _find_band(wavelengths, 800.0)
    idx_670 = _find_band(wavelengths, 670.0)
    r800 = data[:, :, idx_800].astype(np.float64)
    r670 = data[:, :, idx_670].astype(np.float64)
    denom = r800 + r670
    ndvi = np.where(denom > 0.01, (r800 - r670) / denom, 0.0)
    low_ndvi = ndvi < 0.4

    spectral_std = np.std(data, axis=2)
    relative_std = np.where(mean_ref > 0.01, spectral_std / mean_ref, 1.0)
    flat_spectrum = relative_std < 0.20

    disk_mask = high_ref & low_ndvi & flat_spectrum

    if np.sum(disk_mask) > 0:
        structure = np.ones((3, 3), dtype=bool)
        disk_mask = binary_closing(disk_mask, structure=structure)
        disk_mask = binary_fill_holes(disk_mask)
        disk_mask = binary_dilation(disk_mask, structure=structure, iterations=2)

    n = np.sum(disk_mask)
    print(f"  [Disk tespiti] {n} piksel ({n / disk_mask.size * 100:.1f}%)")
    return disk_mask.astype(bool)


# =============================================================================
# YÖNTEM 1: NDVI — ADAPTİF OTSU EŞİĞİ
# =============================================================================


def segment_ndvi(data, wavelengths, threshold=None):
    """
    NDVI tabanlı segmentasyon.
    threshold=None → Otsu ile adaptif eşik hesaplanır.
    """
    idx_800 = _find_band(wavelengths, 800.0)
    idx_670 = _find_band(wavelengths, 670.0)

    r800 = data[:, :, idx_800].astype(np.float64)
    r670 = data[:, :, idx_670].astype(np.float64)

    denom = r800 + r670
    ndvi = np.where(denom > 0.02, (r800 - r670) / denom, 0.0)

    if threshold is None:
        threshold = _otsu_threshold(ndvi.ravel())
        # Otsu bazen çok düşük verebilir — alt sınır koy
        threshold = max(threshold, 0.15)
        print(f"[NDVI segmentasyon] Otsu eşik: {threshold:.3f}")
    else:
        print(f"[NDVI segmentasyon] Sabit eşik: {threshold:.3f}")

    raw_mask = ndvi > threshold
    mask = _morphological_cleanup(raw_mask)

    leaf_ratio = np.sum(mask) / mask.size * 100
    print(f"  Yaprak oranı: {leaf_ratio:.1f}%")
    return mask


# =============================================================================
# YÖNTEM 2: K-MEANS
# =============================================================================


def segment_kmeans(data, wavelengths, n_clusters=3, max_iter=50):
    from sklearn.cluster import MiniBatchKMeans

    lines, samples, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float32)
    print(f"[K-means segmentasyon] {n_clusters} küme...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=1024,
        random_state=42,
        n_init=3,
    )
    labels = kmeans.fit_predict(pixels)
    labels_2d = labels.reshape(lines, samples)

    idx_800 = _find_band(wavelengths, 800.0)
    idx_670 = _find_band(wavelengths, 670.0)
    r800 = data[:, :, idx_800].astype(np.float64)
    r670 = data[:, :, idx_670].astype(np.float64)
    denom = r800 + r670
    ndvi = np.where(denom != 0, (r800 - r670) / denom, 0.0)

    best_cluster, best_ndvi = -1, -999
    for c in range(n_clusters):
        cm = labels_2d == c
        if np.sum(cm) == 0:
            continue
        cn = ndvi[cm].mean()
        cr = np.sum(cm) / cm.size * 100
        print(f"  Küme {c}: NDVI={cn:.3f}, oran={cr:.1f}%")
        if cn > best_ndvi:
            best_ndvi, best_cluster = cn, c

    mask = _morphological_cleanup(labels_2d == best_cluster)
    print(f"  → Küme {best_cluster}, yaprak: {np.sum(mask)/mask.size*100:.1f}%")
    return mask


# =============================================================================
# YÖNTEM 3: PCA — ARKA PLAN HARİÇ TUTMA YAKLAŞIMI
# =============================================================================


def segment_pca(data, wavelengths, n_components=5, n_clusters=3):
    """
    PCA+K-means segmentasyon.

    v4: Tek "en iyi küme" seçmek yerine, arka plan kümesini hariç tutarak
    geri kalan TÜM kümeleri yaprak olarak kabul eder. Böylece koyu
    yaprak bölgeleri ayrı bir kümeye düşse bile kaybedilmez.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans

    lines, samples, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float32)

    print(f"[PCA segmentasyon] {bands}→{n_components} bileşen...")
    pca = PCA(n_components=n_components, random_state=42)
    pixels_pca = pca.fit_transform(pixels)
    print(f"  Varyans: %{np.sum(pca.explained_variance_ratio_)*100:.1f}")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, max_iter=50, batch_size=1024, random_state=42, n_init=3
    )
    labels = kmeans.fit_predict(pixels_pca)
    labels_2d = labels.reshape(lines, samples)

    # NIR yansıması ile arka plan kümesini bul
    idx_800 = _find_band(wavelengths, 800.0)
    r800 = data[:, :, idx_800].astype(np.float64)

    # En düşük NIR ortalamasına sahip küme = arka plan
    cluster_nir = {}
    for c in range(n_clusters):
        cm = labels_2d == c
        if np.sum(cm) == 0:
            continue
        cn = r800[cm].mean()
        cr = np.sum(cm) / cm.size * 100
        cluster_nir[c] = cn
        print(f"  Küme {c}: NIR ort={cn:.3f}, oran={cr:.1f}%")

    # Arka plan: en düşük NIR ortalamasına sahip küme
    bg_cluster = min(cluster_nir, key=cluster_nir.get)
    print(f"  → Arka plan kümesi: {bg_cluster} (NIR={cluster_nir[bg_cluster]:.3f})")

    # Arka plan HARİÇ tümü yaprak
    raw_mask = labels_2d != bg_cluster
    mask = _morphological_cleanup(raw_mask)

    leaf_ratio = np.sum(mask) / mask.size * 100
    print(f"  → Yaprak: {leaf_ratio:.1f}%")
    return mask


# =============================================================================
# YÖNTEM 4: HİBRİT v4
# =============================================================================


def segment_hybrid(data, wavelengths):
    """
    Hibrit Segmentasyon v5 — PCA'sız, NIR tabanlı.

    Neden PCA kaldırıldı:
      PCA+K-means kırmızı/sarı yaprak bölgelerini arka plana atabiliyordu.
      3 kümeden koyu yaprak ayrı kümeye düşüyor, "arka plan hariç tut"
      yaklaşımı ise disk+yaprak kümelerini birleştiriyordu.

    Yeni yaklaşım — sade ve güvenilir:
      1. NIR yansıma filtresi (adaptif Otsu eşiği)
         → Yaprak dokusu (yeşil/kırmızı/sarı) NIR'de güçlü yansıtır
         → Arka plan NIR'de çok düşüktür
         → Renk bağımsız ayrım sağlar
      2. Beyaz disk eleme (3 kriter: yüksek yansıma + düşük NDVI + düz profil)
      3. Sap/klips eleme (küçük bölge filtresi + NDVI < 0 kontrolü)
      4. Yumuşak morfolojik temizlik (3×3)

    Convex hull YOK — flavonol/indeks tahminlerini bozmaması için.
    """
    print("[Hibrit Segmentasyon v5 — NIR tabanlı]")

    # =========================================================================
    # Aşama 1: NIR yansıma filtresi (ana kriter)
    # =========================================================================
    print("  Aşama 1: NIR yansıma filtresi...")
    idx_800 = _find_band(wavelengths, 800.0)
    idx_750 = _find_band(wavelengths, 750.0)
    r800 = data[:, :, idx_800].astype(np.float64)
    r750 = data[:, :, idx_750].astype(np.float64)
    nir_mean = (r800 + r750) / 2.0

    # Otsu ile adaptif eşik — NIR histogramında yaprak vs arka plan ayrımı
    nir_threshold = _otsu_threshold(nir_mean.ravel())
    # Güvenlik sınırları
    nir_threshold = np.clip(nir_threshold, 0.05, 0.30)
    print(f"    Otsu NIR eşik: {nir_threshold:.3f}")

    nir_mask = nir_mean > nir_threshold
    nir_ratio = np.sum(nir_mask) / nir_mask.size * 100
    print(f"    NIR maske: {nir_ratio:.1f}%")

    # =========================================================================
    # Aşama 2: Beyaz disk eleme
    # =========================================================================
    print("  Aşama 2: Beyaz disk eleme...")
    disk_mask = _detect_white_reference(data, wavelengths)
    no_disk = nir_mask & (~disk_mask)

    # =========================================================================
    # Aşama 3: Sap / klips temizliği
    # Sap ve klips NIR'de yansıtabilir ama NDVI ≈ 0 veya negatiftir.
    # Bunları çıkarmak için: NDVI çok düşük VE bölge küçükse sil.
    # Ama dikkat: kırmızı yaprak bölgeleri de düşük NDVI verir,
    # bu yüzden sadece çok küçük bağlantısız parçaları sileriz.
    # =========================================================================
    print("  Aşama 3: Sap/klips temizliği...")

    # Morfolojik temizlik — ince sap parçalarını koparmak için
    cleaned = _morphological_cleanup(no_disk, struct_size=3)

    # Küçük bölgeleri sil (sap, klips kalıntıları)
    cleaned = _remove_small_regions(cleaned, min_area_ratio=0.005)

    # =========================================================================
    # Aşama 4: Son fill_holes
    # Sadece tamamen kapalı delikleri doldurur (loblar arası boşluklara dokunmaz)
    # =========================================================================
    final_mask = binary_fill_holes(cleaned).astype(bool)

    leaf_ratio = np.sum(final_mask) / final_mask.size * 100
    print(f"  → Nihai yaprak oranı: %{leaf_ratio:.1f}")
    return final_mask


# =============================================================================
# EN İYİ MASKE
# =============================================================================


def best_mask(data, wavelengths, method="hybrid"):
    method = method.lower()
    if method == "ndvi":
        return segment_ndvi(data, wavelengths)
    elif method == "kmeans":
        return segment_kmeans(data, wavelengths)
    elif method == "pca":
        return segment_pca(data, wavelengths)
    elif method == "hybrid":
        return segment_hybrid(data, wavelengths)
    else:
        raise ValueError(f"Bilinmeyen yöntem: '{method}'")


# =============================================================================
# KARŞILAŞTIRMA
# =============================================================================


def compare_methods(data, wavelengths, save_dir=None):
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("SEGMENTASYON KARŞILAŞTIRMASI")
    print("=" * 60)

    masks = {}
    print("\n--- NDVI (Otsu) ---")
    masks["ndvi"] = segment_ndvi(data, wavelengths)
    print("\n--- K-means ---")
    masks["kmeans"] = segment_kmeans(data, wavelengths)
    print("\n--- PCA (arka plan hariç) ---")
    masks["pca"] = segment_pca(data, wavelengths)
    print("\n--- Hibrit v4 ---")
    masks["hybrid"] = segment_hybrid(data, wavelengths)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, key, title in zip(
        axes,
        ["ndvi", "kmeans", "pca", "hybrid"],
        ["NDVI (Otsu)", "K-means", "PCA", "Hibrit v4"],
    ):
        ax.imshow(masks[key], cmap="gray")
        r = np.sum(masks[key]) / masks[key].size * 100
        ax.set_title(f"{title}\n({r:.1f}%)", fontsize=11)
        ax.axis("off")

    plt.suptitle("Segmentasyon Karşılaştırması", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, "segmentation_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()
    plt.close(fig)
    return masks


if __name__ == "__main__":
    print("=" * 60)
    print("SEGMENTASYON TESTİ v4")
    print("=" * 60)

    HDR_PATH = r"C:\Users\kedin\OneDrive\Masaüstü\BITIRME_PROJESI\BİTKİ\bitirme_projesi\Dataset\Data\2020-09-10_033\results\REFLECTANCE_2020-09-10_033.hdr"

    if not os.path.exists(HDR_PATH):
        print("Sahte veriyle test...\n")
        np.random.seed(42)
        fake_wl = np.linspace(397, 1004, 204).tolist()
        fake_data = (np.random.rand(80, 80, 204) * 0.03).astype(np.float32)

        idx800 = _find_band(fake_wl, 800.0)
        idx670 = _find_band(fake_wl, 670.0)
        idx750 = _find_band(fake_wl, 750.0)

        fake_data[15:65, 20:60, idx800] += 0.6
        fake_data[15:65, 20:60, idx750] += 0.55
        fake_data[15:65, 20:60, idx670] += 0.08
        fake_data[5:12, 5:12, :] = 0.90

        m = segment_hybrid(fake_data, fake_wl)
        print(f"\nYaprak: {np.sum(m[15:65,20:60])}/{50*40}")
        print(f"Disk: {np.sum(m[5:12,5:12])} (0 olmalı)")
    else:
        from load_envi import load_envi

        data, meta = load_envi(HDR_PATH)
        compare_methods(data, meta["wavelengths"])
