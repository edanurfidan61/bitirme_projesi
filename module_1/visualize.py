"""
visualize.py
============
Hiperspektral görüntü küpünden görsel çıktılar üretilmesi ve analiz sonuçlarının
görselleştirilmesi amacıyla bu yardımcı modül oluşturulmuştur.

GÜNCELLEME: Segmentasyon (maskeleme) işlemleri daha gelişmiş olan `segmentation.py`
modülüne taşındığı için, bu dosya içerisindeki `make_leaf_mask` fonksiyonu kod
tekrarını önlemek adına doğrudan o modüldeki algoritmaları çağıracak şekilde güncellenmiştir.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =============================================================================
# DİZİN VE YOL (PATH) AYARLAMALARI
# İlgili modüllerin içe aktarılabilmesi için yollar sisteme eklenmiştir.
# =============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_MODUL0_DIR = os.path.join(_PROJECT_DIR, "modul_0")

if _MODUL0_DIR not in sys.path:
    sys.path.insert(0, _MODUL0_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# =============================================================================
# SEGMENTASYON MODÜLÜ ENTEGRASYONU
# Maskeleme işlemlerinin dışarıdan sağlanıp sağlanamadığı kontrol edilmiştir.
# =============================================================================
try:
    from segmentation import best_mask

    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False


def find_band(wavelengths, target_nm):
    """
    Hedef alınan dalga boyuna (nm) en yakın olan spektral bantın indeksi
    hesaplanmış ve döndürülmüştür.
    """
    wl_array = np.array(wavelengths, dtype=np.float64)
    distances = np.abs(wl_array - target_nm)
    return int(np.argmin(distances))


def make_rgb(data, wavelengths):
    """
    Hiperspektral veri küpü kullanılarak gerçek renkli (true-color) bir RGB
    görüntüsü oluşturulmuştur.
    """
    # Kırmızı, Yeşil ve Mavi renklere denk gelen bantların indeksleri bulunmuştur.
    idx_red = find_band(wavelengths, 670.0)
    idx_green = find_band(wavelengths, 550.0)
    idx_blue = find_band(wavelengths, 450.0)

    # İlgili bantlardaki yansıma verileri ayrı ayrı matrislere kopyalanmıştır.
    red = data[:, :, idx_red].astype(np.float64)
    green = data[:, :, idx_green].astype(np.float64)
    blue = data[:, :, idx_blue].astype(np.float64)

    # Her bir kanalın 0-1 aralığına normalize edilmesi için içsel bir fonksiyon tanımlanmıştır.
    def normalize_channel(ch):
        c_min, c_max = ch.min(), ch.max()
        if c_max - c_min == 0:
            return np.zeros_like(ch)
        return (ch - c_min) / (c_max - c_min)

    # Normalize edilen kanallar üst üste bindirilerek tek bir RGB matrisi oluşturulmuştur.
    rgb = np.stack(
        [normalize_channel(red), normalize_channel(green), normalize_channel(blue)],
        axis=2,
    )

    # Matris verisi 8-bitlik (0-255) görüntü formatına dönüştürülmüştür.
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8


def make_leaf_mask(data, wavelengths, method="ndvi"):
    """
    Yaprak maskesinin oluşturulması işlemi `segmentation.py` modülüne devredilmiştir.
    Eğer bahsi geçen modüle ulaşılamazsa, sistemin çökmemesi için basit bir NDVI
    maskeleme algoritması yedek (fallback) olarak çalıştırılmıştır.
    """
    if HAS_SEGMENTATION:
        print(
            f"[visualize.py] Maskeleme işlemi 'segmentation.py' modülüne devredilmiş ve '{method}' yöntemi ile uygulanmıştır."
        )
        return best_mask(data, wavelengths, method=method)

    # Fallback (Yedek) Algoritma
    print(
        "[visualize.py] UYARI: 'segmentation.py' dosyasına ulaşılamadığı için temel NDVI maskeleme algoritması uygulanmıştır!"
    )

    idx_800 = find_band(wavelengths, 800.0)
    idx_670 = find_band(wavelengths, 670.0)

    r800 = data[:, :, idx_800].astype(np.float64)
    r670 = data[:, :, idx_670].astype(np.float64)

    denom = r800 + r670
    ndvi = np.where(denom != 0, (r800 - r670) / denom, 0.0)

    # Sadece 0.3 değerinden yüksek olan pikseller yaprak olarak kabul edilmiştir.
    return ndvi > 0.3


def plot_rgb(rgb, title="True-Color RGB", save_path=None):
    """
    Oluşturulan RGB görüntüsü ekrana çizdirilmiş ve isteğe bağlı olarak diske kaydedilmiştir.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"RGB görüntüsü belirtilen yola kaydedilmiştir: {save_path}")

    plt.show()
    plt.close(fig)


def plot_index_map(
    index_array, mask, title="İndeks Haritası", cmap="RdYlGn", save_path=None
):
    """
    Spektral indeks değerleri sahte renkli (false-color) bir harita olarak görselleştirilmiştir.
    Maske dışındaki (yaprak olmayan) bölgeler haritadan temizlenmiştir.
    """
    display = index_array.astype(np.float64).copy()

    # Maske dışında kalan alanlara NaN değeri atanarak görsellerden gizlenmiştir.
    display[~mask] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(display, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Görüntünün yanına değer aralığını gösteren bir renk skalası eklenmiştir.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("İndeks Değeri", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"İndeks haritası belirtilen yola kaydedilmiştir: {save_path}")

    plt.show()
    plt.close(fig)


def plot_spectral_profile(
    data, mask, wavelengths, title="Ortalama Spektral Profil", save_path=None
):
    """
    Yaprak maskesi içindeki tüm piksellerin spektral değerleri ortalanarak,
    yaprağın genel yansıma profili bir çizgi grafiği olarak çizdirilmiştir.
    """
    leaf_pixels = data[mask]

    # Ortalama ve standart sapma değerleri hesaplanmıştır.
    mean_spectrum = np.mean(leaf_pixels, axis=0)
    std_spectrum = np.std(leaf_pixels, axis=0)
    wl = np.array(wavelengths)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Ortalama çizgi grafiğe eklenmiştir.
    ax.plot(
        wl, mean_spectrum, color="darkblue", linewidth=1.5, label="Ortalama Yansıma"
    )

    # Standart sapma değerleri, ortalama çizginin etrafında gölgeli alan olarak gösterilmiştir.
    ax.fill_between(
        wl,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        alpha=0.3,
        color="gray",
        label="± 1 Std Sapma",
    )

    ax.set_xlabel("Dalga Boyu (nm)", fontsize=11)
    ax.set_ylabel("Yansıma (Reflectance)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Spektral profil belirtilen yola kaydedilmiştir: {save_path}")

    plt.show()
    plt.close(fig)


def plot_mask_overlay(rgb, mask, title="Yaprak Maskesi Overlay", save_path=None):
    """
    Segmentasyon işleminin doğruluğunu kontrol edebilmek amacıyla, oluşturulan
    yaprak maskesi orijinal RGB görüntüsünün üzerine yarı saydam olarak bindirilmiştir.
    """
    overlay = rgb.copy()

    # Maske dışında kalan (arka plan) pikseller koyulaştırılarak kırmızı tona dönüştürülmüştür.
    overlay[~mask, 0] = np.clip(overlay[~mask, 0].astype(int) + 100, 0, 255).astype(
        np.uint8
    )
    overlay[~mask, 1] = (overlay[~mask, 1] * 0.3).astype(np.uint8)
    overlay[~mask, 2] = (overlay[~mask, 2] * 0.3).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Orijinal RGB Görüntüsü", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(title, fontsize=11)
    axes[1].axis("off")

    plt.suptitle(
        "Yaprak Maskesinin Doğruluğu Kontrol Edilmiştir", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Maske bindirme görseli belirtilen yola kaydedilmiştir: {save_path}")

    plt.show()
    plt.close(fig)


# =============================================================================
# TEST BLOĞU
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MODÜL 1 — visualize.py TESTİ (Tüm Fonksiyonlar Aktif)")
    print("=" * 60)

    HDR_PATH = r"yaprak.hdr"

    # Görüntü dosyasının sistemde bulunup bulunmadığı kontrol edilmiştir.
    if not os.path.exists(HDR_PATH):
        print(
            f"'{HDR_PATH}' dosyası bulunamamıştır. Modül sahte veriler kullanılarak test edilmektedir...\n"
        )

        # Test işlemleri için rastgele veri üretilmiştir.
        np.random.seed(42)
        fake_data = np.random.rand(20, 20, 10).astype(np.float32)
        fake_data[5:15, 5:15, :] += 0.5
        fake_wl = np.linspace(400, 900, 10).tolist()

        print("TEST: make_leaf_mask fonksiyonu çalıştırılmaktadır...")
        mask = make_leaf_mask(fake_data, fake_wl, method="hybrid")
        print(f"Oluşturulan maskenin boyutları: {mask.shape}")

        print("\nTest işlemleri sahte veri ile başarıyla tamamlanmıştır.")

    else:
        # Eğer dosya mevcutsa, modülün tüm fonksiyonları gerçek verilerle test edilmiştir.
        from load_envi import load_envi

        print("Görüntü küpü disk üzerinden belleğe yüklenmektedir...")
        data, meta = load_envi(HDR_PATH)
        wl = meta["wavelengths"]

        print("Gerçek renkli RGB görüntüsü oluşturulmaktadır...")
        rgb = make_rgb(data, wl)
        plot_rgb(rgb, title="Gerçek Renkli (True Color) RGB")

        print("Maskeleme algoritması uygulanmaktadır...")
        mask = make_leaf_mask(data, wl)

        print(
            "Oluşturulan maske RGB üzerine bindirilerek doğruluğu kontrol edilmektedir..."
        )
        plot_mask_overlay(rgb, mask)

        print("Yaprağa ait spektral profil çizdirilmektedir...")
        plot_spectral_profile(data, mask, wl)

        print("\nTüm test işlemleri gerçek veri kullanılarak başarıyla tamamlanmıştır.")
