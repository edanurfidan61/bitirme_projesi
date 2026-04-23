"""
features.py
===========
Her yaprak için tek bir özellik vektörü çıkarmak amacıyla bu modül oluşturulmuştur.

Özellik vektörü iki ana bileşenden oluşturulmuştur:

  1. Spektral ortalama vektörü (204 boyut)
     Yaprak maskesi içindeki piksellerin her bant için ortalaması
     alınarak tek bir yansıma profili elde edilmiştir.
     (ÖNEMLİ: Bu aşamada ön işleme yöntemlerinin uygulanmasına olanak tanınmıştır.)

  2. İndeks ortalamaları (5 boyut)
     NDVI, GNDVI, ARI, RVSI, ZTM indekslerinin yaprak maskesi
     üzerindeki ortalamaları hesaplanmıştır.
     (ÖNEMLİ: İndeksler, ön işlemden geçmemiş ham veri üzerinden hesaplanmıştır.)

  Toplam: 204 + 5 = 209 boyutlu özellik vektörü oluşturulmuştur.
"""

import numpy as np
import os
import sys

# =============================================================================
# DİZİN VE YOL (PATH) AYARLAMALARI
# Diğer modüllere erişim sağlanabilmesi için proje dizinleri sistem yoluna eklenmiştir.
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

# Modül dizinlerini sistem yoluna ekle (varsa)
for _mod in ["modul_0", "modul_1", "modul_2", "modul_3", "modul_4"]:
    _mod_dir = os.path.join(_PROJECT_DIR, _mod)
    if os.path.isdir(_mod_dir) and _mod_dir not in sys.path:
        sys.path.insert(0, _mod_dir)

# Geçerli dizin de eklenir (aynı dizindeki modüllere erişim için)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# İlgili fonksiyonlar diğer modüllerden içe aktarılmıştır.
from indices import calc_all_indices

# Ön işleme modülünün varlığı kontrol edilmiş ve modül içe aktarılmıştır.
try:
    from preprocessing import apply_pipeline

    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False
    print("UYARI: preprocessing.py bulunamadı. Ön işleme uygulanamayacaktır.")

    def apply_pipeline(spectra, steps):
        print("  UYARI: Ön işleme modülü yok, veri değiştirilmeden döndürüldü.")
        return spectra


# =============================================================================
# İNDEKS İSİMLERİ KÜMESİ
# =============================================================================
INDEX_NAMES = ["NDVI", "GNDVI", "ARI", "RVSI", "ZTM"]


# =============================================================================
# FONKSİYON 1: extract_spectral_means
# =============================================================================
def extract_spectral_means(data, mask, prep_steps=None):
    """
    Yaprak maskesi içindeki piksellerin bant bazında ortalaması hesaplanmıştır.

    Eğer 'prep_steps' parametresi verilmişse, ortalama alma işleminden önce
    yaprak piksellerine bu ön işleme adımları uygulanmıştır.

    Parametreler:
        data (ndarray): Görüntü küpü (lines, samples, bands).
        mask (ndarray): Yaprak piksellerini belirten boolean maske.
        prep_steps (list): Ön işleme adımları (örn: ["sg", "snv"]).

    Döndürür:
        ndarray: Spektral ortalamaları içeren 1D vektör (örn: 204 boyutlu).
    """
    leaf_count = np.sum(mask)

    if leaf_count == 0:
        bands = data.shape[2]
        print(
            "UYARI: Yaprak maskesinde hiç piksel bulunamamıştır! Sıfır vektörü döndürülmektedir."
        )
        return np.zeros(bands, dtype=np.float64)

    # Sadece yaprak piksellerini seç → (N, bands) matris
    leaf_pixels = data[mask].astype(np.float64)

    # Ön işleme uygula (verilmişse)
    if prep_steps and HAS_PREPROCESSING:
        print(
            f"  Spektral verilere şu ön işleme adımları uygulanmaktadır: {prep_steps}"
        )
        leaf_pixels = apply_pipeline(leaf_pixels, prep_steps)

    # Bant bazında ortalama
    spectral_means = np.mean(leaf_pixels, axis=0).astype(np.float64)

    print(
        f"  Spektral ortalama: {leaf_count} piksel → ({spectral_means.shape[0]},) vektör"
    )
    return spectral_means


# =============================================================================
# FONKSİYON 2: extract_index_means
# =============================================================================
def extract_index_means(data, wavelengths, mask):
    """
    Belirlenen spektral indekslerin, yaprak maskesi üzerindeki ortalamaları
    hesaplanmıştır.

    DİKKAT: İndeks değerlerinin bozulmaması adına, hesaplamalar her zaman
    ön işlemden GEÇMEMİŞ (ham) veri üzerinden gerçekleştirilmiştir.

    Parametreler:
        data (ndarray): Ham görüntü küpü.
        wavelengths (list): Dalga boyları listesi.
        mask (ndarray): Yaprak piksellerini belirten boolean maske.

    Döndürür:
        ndarray: İndeks ortalamalarını içeren 1D vektör (5 boyutlu).
    """
    if np.sum(mask) == 0:
        return np.zeros(len(INDEX_NAMES), dtype=np.float64)

    # Tüm indeksler tüm görüntü üzerinden hesaplanmıştır
    all_indices = calc_all_indices(data, wavelengths)

    index_means = np.zeros(len(INDEX_NAMES), dtype=np.float64)

    for i, name in enumerate(INDEX_NAMES):
        index_map = all_indices[name]
        leaf_values = index_map[mask]
        index_means[i] = np.mean(leaf_values)

    print("  İndeks ortalamaları (ham veri üzerinden):")
    for i, name in enumerate(INDEX_NAMES):
        print(f"    {name:6s} → {index_means[i]:.6f}")

    return index_means


# =============================================================================
# FONKSİYON 3: extract_features
# =============================================================================
def extract_features(data, wavelengths, mask, prep_steps=None):
    """
    Spektral ortalamalar ve indeks ortalamaları hesaplanıp tek bir özellik
    vektöründe birleştirilmiştir (toplam 209 boyutlu).

    Parametreler:
        data (ndarray): Görüntü küpü.
        wavelengths (list): Dalga boyları.
        mask (ndarray): Boolean yaprak maskesi.
        prep_steps (list): Sadece spektral ortalamalara uygulanacak ön işleme zinciri.

    Döndürür:
        ndarray: Birleştirilmiş özellik vektörü (209 boyut).
    """
    print("-" * 40)
    print("Özellik çıkarım süreci başlatılmıştır...")

    # 1. Spektral ortalamalar (ön işleme uygulanabilir)
    spectral_means = extract_spectral_means(data, mask, prep_steps=prep_steps)

    # 2. İndeks ortalamaları (her zaman ham veri üzerinden)
    index_means = extract_index_means(data, wavelengths, mask)

    # 3. Birleştir
    features = np.concatenate([spectral_means, index_means])

    print(f"  Toplam özellik vektörü: ({features.shape[0]},)")
    print("-" * 40)

    return features


# =============================================================================
# FONKSİYON 4: get_feature_names
# =============================================================================
def get_feature_names(wavelengths):
    """
    Özellik vektöründeki her bir sütunun ismi üretilmiştir.
    """
    band_names = [f"band_{wl:.2f}" for wl in wavelengths]
    all_names = band_names + INDEX_NAMES
    return all_names


# =============================================================================
# TEST BLOĞU
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("features.py TESTİ (ÖN İŞLEME DESTEKLİ)")
    print("=" * 60)

    HDR_PATH = r"yaprak.hdr"

    if not os.path.exists(HDR_PATH):
        print(
            f"'{HDR_PATH}' dosyası bulunamamıştır. Sahte veriyle test edilmektedir...\n"
        )

        fake_wl = np.linspace(397, 1004, 204).tolist()
        np.random.seed(42)
        fake_data = (np.random.rand(20, 20, 204) * 0.55 + 0.05).astype(np.float32)

        fake_mask = np.zeros((20, 20), dtype=bool)
        fake_mask[5:15, 5:15] = True

        print("\nTEST 1: Ön İşlemesiz")
        feat_raw = extract_features(fake_data, fake_wl, fake_mask)
        print(f"Çıktı boyutu: {feat_raw.shape}")

        if HAS_PREPROCESSING:
            print("\nTEST 2: Ön İşlemeli (SG + SNV)")
            feat_prep = extract_features(
                fake_data, fake_wl, fake_mask, prep_steps=["sg", "snv"]
            )
            print(f"Çıktı boyutu: {feat_prep.shape}")

    else:
        from load_envi import load_envi

        try:
            from segmentation import best_mask
        except ImportError:
            from visualize import make_leaf_mask as best_mask

        print("Gerçek veri yüklenmektedir...")
        data, meta = load_envi(HDR_PATH)
        wl = meta["wavelengths"]

        print("\nYaprak maskesi oluşturulmaktadır...")
        mask = best_mask(data, wl)

        print("\nÖzellikler SG + SNV ile çıkarılmaktadır...")
        feat = extract_features(data, wl, mask, prep_steps=["sg", "snv"])

        print("\nTüm testler tamamlanmıştır.")
