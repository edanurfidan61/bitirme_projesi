"""
indices.py
==========
Hiperspektral yansıma verisinden bitki fizyolojisini yansıtan spektral
indeksler hesaplamak için yardımcı modül.

5 spektral indeks tanımlanmıştır:

  İndeks   Formül                        Amaç
  ──────   ──────────────────────────    ─────────────────────────
  NDVI     (R800−R670)/(R800+R670)       Genel bitki sağlığı
  GNDVI    (R800−R550)/(R800+R550)       Klorofil hassasiyeti
  ARI      1/R550 − 1/R700              Flavonol / antosiyanin
  RVSI     (R714+R752)/2 − R733         Kırmızı kenar stresi
  ZTM      R750 / R710                  Klorofil içeriği

Her indeks, belirli dalga boylarındaki yansıma değerlerinin matematiksel
kombinasyonundan oluşturulmuştur. get_band() yardımcı fonksiyonu ile
formüldeki sabit nm değerleri veri setindeki en yakın gerçek dalga
boyuyla eşleştirilmiştir.

Kullanım örneği:
    import sys
    sys.path.append(r'..\\modul_0')
    from load_envi import load_envi
    from indices import calc_all_indices

    data, meta = load_envi("yaprak.hdr")
    indices = calc_all_indices(data, meta['wavelengths'])
    print(indices['ARI'].shape)   # (512, 512)

Literatür:
    - Rouse et al. (1974)           → NDVI
    - Gitelson et al. (1996)        → GNDVI
    - Gitelson et al. (2001)        → ARI
    - Merton & Huntington (1999)    → RVSI
    - Zarco-Tejada et al. (2001)    → ZTM
"""

import numpy as np
import os
import sys

# =============================================================================
# Modül 0 ve Modül 1'i import edebilmek için üst dizin yolları eklendi
# Proje yapısı: BITIRME_PROJESI/modul_2/indices.py
#               BITIRME_PROJESI/modul_0/load_envi.py
#               BITIRME_PROJESI/modul_1/visualize.py
# =============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

_MODUL0_DIR = os.path.join(_PROJECT_DIR, "modul_0")
_MODUL1_DIR = os.path.join(_PROJECT_DIR, "modul_1")

for _d in [_MODUL0_DIR, _MODUL1_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# =============================================================================
# YARDIMCI FONKSİYON: get_band
# Görev: İstenen dalga boyuna en yakın bant indeksini bul
# =============================================================================


def get_band(wavelengths, target_nm):
    """
    Dalga boyu listesinde, hedef nm değerine en yakın bantın indeksi
    hesaplanmıştır.

    Bu fonksiyon tüm indeks hesaplamalarında kullanılmaktadır.
    Formüldeki "R800" gibi sabit değerler, veri setindeki gerçek
    dalga boylarıyla eşleştirilmiştir (örn: R800 → 799.61 nm, bant 134).

    Parametreler:
        wavelengths (list/array) : Dalga boyu listesi (nm cinsinden)
        target_nm (float)        : Hedef dalga boyu (nm)

    Döndürür:
        int : En yakın bantın indeksi (0-tabanlı)

    Örnek:
        >>> wl = [397.32, 400.20, ..., 1003.58]
        >>> get_band(wl, 800.0)
        134   # (gerçek değer: 799.61 nm)
    """

    wl_array = np.array(wavelengths, dtype=np.float64)
    distances = np.abs(wl_array - target_nm)
    best_idx = int(np.argmin(distances))

    return best_idx


# =============================================================================
# İNDEKS 1: NDVI — Normalized Difference Vegetation Index
# Kaynak: Rouse et al. (1974)
# =============================================================================


def calc_ndvi(data, wavelengths):
    """
    NDVI (Normalized Difference Vegetation Index) hesaplanmıştır.

    Formül:
        NDVI = (R800 − R670) / (R800 + R670)

    Değer aralığı: [-1, +1]
      +0.7 ~ +0.9 → sağlıklı, yoğun vejetasyon
      +0.2 ~ +0.5 → seyrek vejetasyon veya hafif stres
      < 0.2        → çıplak toprak, su veya ölü doku

    R800: Yakın kızılötesi (NIR) bölgesi — yaprak hücre yapısından
          güçlü yansıma alınmaktadır
    R670: Kırmızı bölge — klorofil tarafından emilmektedir

    Sağlıklı yapraklarda klorofil R670'i güçlü emer → R670 düşük,
    hücre yapısı R800'ü güçlü yansıtır → R800 yüksek → NDVI yüksek.

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        numpy.ndarray : (lines, samples) NDVI haritası

    Kaynak: Rouse, J.W. et al. (1974). Monitoring vegetation systems
            in the Great Plains with ERTS.
    """

    # İlgili bantların indekslerini bul
    idx_800 = get_band(wavelengths, 800.0)
    idx_670 = get_band(wavelengths, 670.0)

    # Bant değerlerini float64 olarak çıkar (hassas bölme için)
    r800 = data[:, :, idx_800].astype(np.float64)
    r670 = data[:, :, idx_670].astype(np.float64)

    # Pay ve payda hesapla
    numerator = r800 - r670
    denominator = r800 + r670

    # Sıfıra bölme koruması: payda sıfırsa sonuç 0.0 olarak atandı
    ndvi = np.where(denominator != 0, numerator / denominator, 0.0)

    return ndvi


# =============================================================================
# İNDEKS 2: GNDVI — Green Normalized Difference Vegetation Index
# Kaynak: Gitelson et al. (1996)
# =============================================================================


def calc_gndvi(data, wavelengths):
    """
    GNDVI (Green NDVI) hesaplanmıştır.

    Formül:
        GNDVI = (R800 − R550) / (R800 + R550)

    Değer aralığı: [-1, +1]

    Klasik NDVI'dan farkı: R670 (kırmızı) yerine R550 (yeşil) bant
    kullanılmıştır. Klorofil konsantrasyonundaki ince değişimlere
    daha hassas olduğu bildirilmiştir.

    R550: Yeşil bölge — klorofilin en az emdiği görünür dalga boyu.
          Yaprakların yeşil görünmesinin sebebi bu banttaki yüksek
          yansımadır. Klorofil azaldığında R550 artar.

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        numpy.ndarray : (lines, samples) GNDVI haritası

    Kaynak: Gitelson, A.A. et al. (1996). Use of a green channel in
            remote sensing of global vegetation from EOS-MODIS.
    """

    idx_800 = get_band(wavelengths, 800.0)
    idx_550 = get_band(wavelengths, 550.0)

    r800 = data[:, :, idx_800].astype(np.float64)
    r550 = data[:, :, idx_550].astype(np.float64)

    numerator = r800 - r550
    denominator = r800 + r550

    gndvi = np.where(denominator != 0, numerator / denominator, 0.0)

    return gndvi


# =============================================================================
# İNDEKS 3: ARI — Anthocyanin Reflectance Index
# Kaynak: Gitelson et al. (2001)
# =============================================================================


def calc_ari(data, wavelengths):
    """
    ARI (Anthocyanin Reflectance Index) hesaplanmıştır.

    Formül:
        ARI = 1/R550 − 1/R700

    Değer aralığı: genellikle [0, ~0.1] (birimler: 1/reflectance)

    Antosiyanin ve flavonol pigmentlerinin birikimini yansıtmaktadır.
    Bu pigmentler UV koruma ve antioksidan savunma görevi görmektedir.

    R550: Antosiyanin emiliminden etkilenmektedir
    R700: Kırmızı kenar başlangıcı — referans bant olarak kullanılmıştır

    Yüksek ARI:
      → UV/ışık stresi altında flavonol birikimi (savunma yanıtı)
      → Sağlıklı yapraklarda yeterli pigment üretimi

    Düşük ARI:
      → Şiddetli biyotik stres (Flavescence dorée, külleme)
        metabolik çöküş nedeniyle flavonol sentezi baskılanmıştır
      → Ph. Eur. eşiğinin altında flavonol → ilaç kalitesi FAIL

    DİKKAT: Sıfıra bölme koruması uygulanmıştır.
    R550 veya R700 = 0 olduğunda o piksel için sonuç 0.0 atanmıştır.

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        numpy.ndarray : (lines, samples) ARI haritası

    Kaynaklar:
        Gitelson, A.A. et al. (2001). Optical properties and
            nondestructive estimation of anthocyanin content.
        AL-Saddik, H. et al. (2017). Development of spectral
            disease indices for 'Flavescence dorée' grapevine disease.
    """

    idx_550 = get_band(wavelengths, 550.0)
    idx_700 = get_band(wavelengths, 700.0)

    r550 = data[:, :, idx_550].astype(np.float64)
    r700 = data[:, :, idx_700].astype(np.float64)

    # 1/R550 hesapla — sıfır olan piksellerde 0.0 ata
    inv_r550 = np.where(r550 != 0, 1.0 / r550, 0.0)

    # 1/R700 hesapla — sıfır olan piksellerde 0.0 ata
    inv_r700 = np.where(r700 != 0, 1.0 / r700, 0.0)

    ari = inv_r550 - inv_r700

    return ari


# =============================================================================
# İNDEKS 4: RVSI — Red-edge Vegetation Stress Index
# Kaynak: Merton & Huntington (1999)
# =============================================================================


def calc_rvsi(data, wavelengths):
    """
    RVSI (Red-edge Vegetation Stress Index) hesaplanmıştır.

    Formül:
        RVSI = (R714 + R752) / 2 − R733

    Kırmızı kenar bölgesindeki (700–750 nm) yansıma eğrisinin
    eğrilik değişimini (concavity) ölçmektedir.

    Sağlıklı yapraklarda kırmızı kenar geçişi keskin ve diktir
    → RVSI düşük (düz eğri).
    Stresli yapraklarda geçiş yumuşar ve kayar
    → RVSI yükselir (eğri bükülür).

    R714 ve R752: Kırmızı kenar bölgesinin iki ucundaki bantlar
    R733: Ortadaki referans bant — eğriliğin ölçüm noktası

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        numpy.ndarray : (lines, samples) RVSI haritası

    Kaynak: Merton, R.N. & Huntington, J.F. (1999). Early simulation
            results of the ARIES-1 satellite sensor.
    """

    idx_714 = get_band(wavelengths, 714.0)
    idx_752 = get_band(wavelengths, 752.0)
    idx_733 = get_band(wavelengths, 733.0)

    r714 = data[:, :, idx_714].astype(np.float64)
    r752 = data[:, :, idx_752].astype(np.float64)
    r733 = data[:, :, idx_733].astype(np.float64)

    # İki uç noktanın ortalaması − orta noktanın değeri
    rvsi = (r714 + r752) / 2.0 - r733

    return rvsi


# =============================================================================
# İNDEKS 5: ZTM — Zarco-Tejada & Miller Index
# Kaynak: Zarco-Tejada et al. (2001)
# =============================================================================


def calc_ztm(data, wavelengths):
    """
    ZTM (Zarco-Tejada & Miller) indeksi hesaplanmıştır.

    Formül:
        ZTM = R750 / R710

    Klorofil içeriğiyle güçlü korelasyon gösteren basit bir oran
    indeksidir.

    R750: Kırmızı kenar platosu — klorofilden bağımsız yansıma
    R710: Klorofil emilim kenarı — klorofil arttıkça R710 düşer

    Sağlıklı yaprak → yüksek klorofil → R710 düşük → ZTM yüksek
    Stresli yaprak  → düşük klorofil  → R710 yükselir → ZTM düşer

    DİKKAT: Sıfıra bölme koruması uygulanmıştır.
    R710 = 0 olduğunda sonuç 0.0 olarak atanmıştır.

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        numpy.ndarray : (lines, samples) ZTM haritası

    Kaynak: Zarco-Tejada, P.J. et al. (2001). Scaling-up and model
            inversion methods with narrowband optical indices for
            chlorophyll content estimation.
    """

    idx_750 = get_band(wavelengths, 750.0)
    idx_710 = get_band(wavelengths, 710.0)

    r750 = data[:, :, idx_750].astype(np.float64)
    r710 = data[:, :, idx_710].astype(np.float64)

    ztm = np.where(r710 != 0, r750 / r710, 0.0)

    return ztm


# =============================================================================
# TOPLU HESAPLAMA: calc_all_indices
# Görev: 5 indeksi tek seferde hesapla ve sözlük olarak döndür
# =============================================================================


def calc_all_indices(data, wavelengths):
    """
    Tüm spektral indeksler tek seferde hesaplanmıştır.

    features.py modülü bu fonksiyonu kullanarak indeks ortalamalarını
    hesaplayacaktır.

    Parametreler:
        data (numpy.ndarray)     : (lines, samples, bands) 3D array
        wavelengths (list/array) : Dalga boyu listesi (nm)

    Döndürür:
        dict : İndeks adı → (lines, samples) array eşlemesi
               {"NDVI": ..., "GNDVI": ..., "ARI": ..., "RVSI": ..., "ZTM": ...}

    Kullanım:
        indices = calc_all_indices(data, wl)
        ari_map = indices['ARI']
        ndvi_mean = indices['NDVI'][mask].mean()
    """

    print("Tüm spektral indeksler hesaplanıyor...")

    indices = {
        "NDVI": calc_ndvi(data, wavelengths),
        "GNDVI": calc_gndvi(data, wavelengths),
        "ARI": calc_ari(data, wavelengths),
        "RVSI": calc_rvsi(data, wavelengths),
        "ZTM": calc_ztm(data, wavelengths),
    }

    # Her indeks için özet istatistik yazdır
    for name, arr in indices.items():
        print(
            f"  {name:6s} → min: {arr.min():.4f}  max: {arr.max():.4f}  "
            f"mean: {arr.mean():.4f}"
        )

    print("Tüm indeksler hesaplandı.")

    return indices


# =============================================================================
# TEST BLOĞU
# Bu dosya doğrudan çalıştırıldığında (python indices.py) test kodu
# devreye girer. Başka bir dosyadan import edildiğinde çalışmaz.
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODÜL 2 — indices.py TESTİ")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Gerçek veri ile test
    # Kendi dosya yollarını buraya yaz:
    # -----------------------------------------------------------------
    HDR_PATH = r"yaprak.hdr"  # ← KENDİ DOSYA YOLUNU YAZ

    if not os.path.exists(HDR_PATH):
        # -----------------------------------------------------------------
        # Gerçek dosya yoksa → sahte veriyle test
        # -----------------------------------------------------------------
        print(f"'{HDR_PATH}' bulunamadı. Sahte veriyle test yapılıyor...\n")

        # Sahte dalga boyları: 397–1004 nm, 204 bant (gerçek veri setini taklit)
        fake_wl = np.linspace(397, 1004, 204).tolist()

        # Sahte hyperspektral küp: 10×10 piksel, 204 bant
        # Yansıma değerleri 0.05–0.60 aralığında rastgele
        np.random.seed(42)
        fake_data = (np.random.rand(10, 10, 204) * 0.55 + 0.05).astype(np.float32)

        # --- TEST 1: get_band ---
        print("TEST 1: get_band")
        test_targets = [550.0, 670.0, 700.0, 800.0, 750.0]
        for t in test_targets:
            idx = get_band(fake_wl, t)
            print(f"  {t:.0f} nm → bant {idx} ({fake_wl[idx]:.2f} nm)")
        print()

        # --- TEST 2: Tek tek indeks hesaplama ---
        print("TEST 2: Tek tek indeks hesaplama")

        ndvi = calc_ndvi(fake_data, fake_wl)
        print(
            f"  NDVI  → şekil: {ndvi.shape}, "
            f"min: {ndvi.min():.4f}, max: {ndvi.max():.4f}"
        )

        gndvi = calc_gndvi(fake_data, fake_wl)
        print(
            f"  GNDVI → şekil: {gndvi.shape}, "
            f"min: {gndvi.min():.4f}, max: {gndvi.max():.4f}"
        )

        ari = calc_ari(fake_data, fake_wl)
        print(
            f"  ARI   → şekil: {ari.shape}, "
            f"min: {ari.min():.4f}, max: {ari.max():.4f}"
        )

        rvsi = calc_rvsi(fake_data, fake_wl)
        print(
            f"  RVSI  → şekil: {rvsi.shape}, "
            f"min: {rvsi.min():.4f}, max: {rvsi.max():.4f}"
        )

        ztm = calc_ztm(fake_data, fake_wl)
        print(
            f"  ZTM   → şekil: {ztm.shape}, "
            f"min: {ztm.min():.4f}, max: {ztm.max():.4f}"
        )
        print()

        # --- TEST 3: Toplu hesaplama ---
        print("TEST 3: calc_all_indices")
        all_idx = calc_all_indices(fake_data, fake_wl)
        print(f"  Döndürülen indeks sayısı: {len(all_idx)}")
        print(f"  Anahtarlar: {list(all_idx.keys())}")
        print()

        # --- TEST 4: Sıfıra bölme koruması ---
        print("TEST 4: Sıfıra bölme koruması")
        zero_data = np.zeros((3, 3, 204), dtype=np.float32)
        ari_zero = calc_ari(zero_data, fake_wl)
        ztm_zero = calc_ztm(zero_data, fake_wl)
        print(f"  Sıfır veride ARI: tüm 0 mu? → {np.all(ari_zero == 0)}")
        print(f"  Sıfır veride ZTM: tüm 0 mu? → {np.all(ztm_zero == 0)}")

        print("\nTüm testler tamamlandı (sahte veri ile).")

    else:
        # -----------------------------------------------------------------
        # Gerçek dosya varsa → gerçek veriyle test
        # -----------------------------------------------------------------
        from load_envi import load_envi
        from visualize import make_leaf_mask, plot_index_map

        print("Gerçek veri yükleniyor...")
        data, meta = load_envi(HDR_PATH)
        wl = meta["wavelengths"]

        # Tüm indeksleri hesapla
        print()
        indices = calc_all_indices(data, wl)

        # Yaprak maskesi oluştur (görselleştirme için)
        mask = make_leaf_mask(data, wl)

        # ARI haritasını görselleştir
        print()
        print("ARI false-color haritası çiziliyor...")
        plot_index_map(
            indices["ARI"],
            mask,
            title="ARI — Antosiyanin Yansıma İndeksi",
            cmap="RdYlGn",
        )

        # NDVI haritasını görselleştir
        print("NDVI false-color haritası çiziliyor...")
        plot_index_map(
            indices["NDVI"],
            mask,
            title="NDVI — Normalize Fark Vejetasyon İndeksi",
            cmap="RdYlGn",
        )

        print("\nTüm testler tamamlandı (gerçek veri ile).")
