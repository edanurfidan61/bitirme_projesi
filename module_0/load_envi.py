"""
load_envi.py
============
ENVI formatındaki hyperspektral görüntü dosyalarını okumak için yardımcı modül.

ENVI formatı iki dosyadan oluşur:
  - .hdr  : Metin tabanlı header (başlık) dosyası — görüntü hakkında metadata içerir
  - .dat  : Binary veri dosyası — piksel değerlerini ham sayılar olarak tutar

Güncelleme notu:
  Ryckewaert veri setindeki bazı yaprakların (2020-09-10_* serisi) .hdr
  dosyalarında lines/samples değerleri yanlış yazılmıştır (hepsi 512×512
  diyor ama .dat dosyaları farklı boyutlardadır). Bu durumda .dat dosya
  boyutundan gerçek boyutlar hesaplanmaktadır.

Kullanım örneği:
    from load_envi import parse_hdr, load_dat

    meta = parse_hdr("yaprak.hdr")
    data = load_dat("yaprak.dat", meta)

    print(data.shape)   # (512, 512, 204) veya farklı boyut
"""

import numpy as np
import re
import os

# =============================================================================
# FONKSİYON 1: parse_hdr
# Görev: .hdr dosyasını okuyup içindeki bilgileri bir sözlüğe (dict) dönüştür
# =============================================================================


def parse_hdr(hdr_path):
    """
    ENVI .hdr dosyası ayrıştırılmıştır ve metadata sözlüğü döndürülmüştür.

    Parametreler:
        hdr_path (str): .hdr dosyasının yolu (örn: "yaprak.hdr")

    Döndürür:
        dict: Aşağıdaki anahtarları içeren sözlük:
            - "lines"       : Görüntüdeki satır sayısı (int)
            - "samples"     : Görüntüdeki sütun (piksel) sayısı (int)
            - "bands"       : Spektral bant sayısı (int)
            - "interleave"  : Veri düzeni formatı, örn: "bil", "bsq", "bip" (str)
            - "data_type"   : Piksel değerlerinin tipi, örn: 4 = float32 (int)
            - "wavelengths" : Her banda karşılık gelen dalga boyu listesi (list of float)
    """

    # Dosyanın var olup olmadığını kontrol et
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f".hdr dosyası bulunamadı: {hdr_path}")

    # Sonuçları tutacağımız boş sözlük
    metadata = {}

    # .hdr dosyasını metin olarak aç ve satır satır oku
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # -------------------------------------------------------------------------
    # Basit "anahtar = değer" satırlarını ayrıştır
    # -------------------------------------------------------------------------
    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith(";"):
            continue

        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip().lower()
            value = value.strip()

            if key == "lines":
                metadata["lines"] = int(value)

            elif key == "samples":
                metadata["samples"] = int(value)

            elif key == "bands":
                metadata["bands"] = int(value)

            elif key == "data type":
                metadata["data_type"] = int(value)

            elif key == "interleave":
                metadata["interleave"] = value.lower()

    # -------------------------------------------------------------------------
    # wavelength listesini özel olarak ayrıştır
    # -------------------------------------------------------------------------
    wavelength_match = re.search(
        r"wavelength\s*=\s*\{([^}]+)\}", content, re.IGNORECASE | re.DOTALL
    )

    if wavelength_match:
        raw_wl = wavelength_match.group(1)
        wavelengths = [
            float(w.strip()) for w in re.split(r"[,\s]+", raw_wl) if w.strip()
        ]
        metadata["wavelengths"] = wavelengths
    else:
        metadata["wavelengths"] = []
        print("Uyarı: .hdr dosyasında wavelength bilgisi bulunamadı.")

    return metadata


# =============================================================================
# YARDIMCI FONKSİYON: envi_dtype_to_numpy
# Görev: ENVI'nin kendi veri tipi kodunu numpy dtype'a çevir
# =============================================================================


def envi_dtype_to_numpy(envi_code):
    """
    ENVI veri tipi kodu numpy dtype nesnesine dönüştürülmüştür.

    ENVI kod tablosu:
        1  → uint8, 2  → int16, 3  → int32, 4  → float32,
        5  → float64, 12 → uint16, 13 → uint32
    """

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
    }

    if envi_code not in dtype_map:
        print(
            f"Uyarı: Bilinmeyen ENVI veri tipi kodu ({envi_code}). float32 varsayılıyor."
        )
        return np.float32

    return dtype_map[envi_code]


# =============================================================================
# FONKSİYON 2: load_dat
# Görev: .dat binary dosyasını okuyup (lines, samples, bands) array'e dönüştür
#
# GÜNCELLEME: .hdr'daki boyutlar .dat dosyasıyla uyuşmuyorsa
# gerçek boyutlar .dat dosya boyutundan hesaplanmaktadır.
# Ryckewaert veri setindeki 2020-09-10_* serisi yaprakların
# .hdr dosyalarında lines/samples yanlış yazılmıştır.
# =============================================================================


def load_dat(dat_path, metadata):
    """
    ENVI .dat (binary) dosyası okunmuş ve 3D numpy array döndürülmüştür.

    Eğer .hdr'daki lines×samples×bands değeri .dat dosya boyutuyla
    uyuşmuyorsa, bant sayısı sabit tutularak gerçek lines ve samples
    değerleri .dat dosyasından hesaplanmıştır.

    Parametreler:
        dat_path (str)  : .dat dosyasının yolu
        metadata (dict) : parse_hdr() fonksiyonundan dönen metadata sözlüğü

    Döndürür:
        numpy.ndarray: (lines, samples, bands) şeklinde float32 array
    """

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f".dat dosyası bulunamadı: {dat_path}")

    # Metadata'dan boyutları çıkar
    lines = metadata["lines"]
    samples = metadata["samples"]
    bands = metadata["bands"]
    interleave = metadata.get("interleave", "bil")

    # ENVI veri tipini numpy dtype'a çevir
    envi_code = metadata.get("data_type", 4)
    dtype = envi_dtype_to_numpy(envi_code)

    # -------------------------------------------------------------------------
    # .dat dosyasını binary olarak oku
    # -------------------------------------------------------------------------
    print(f".dat dosyası okunuyor: {dat_path}")
    raw_data = np.fromfile(dat_path, dtype=dtype)

    # Toplam piksel sayısını hesapla ve kontrol et
    expected_total = lines * samples * bands
    actual_total = raw_data.size

    # -------------------------------------------------------------------------
    # BOYUT UYUŞMAZLIĞI KONTROLÜ
    # .hdr yanlış olabilir — .dat dosyasından gerçek boyutu hesapla
    # -------------------------------------------------------------------------
    if actual_total != expected_total:
        print(
            f"UYARI: .hdr boyutları ({lines}×{samples}×{bands} = {expected_total:,}) "
            f".dat dosyasıyla ({actual_total:,}) uyuşmuyor!"
        )

        # Bant sayısı doğru varsayılır — gerçek piksel sayısını hesapla
        if actual_total % bands != 0:
            raise ValueError(
                f"Boyut düzeltilemedi! .dat boyutu ({actual_total}) "
                f"bant sayısına ({bands}) tam bölünemiyor."
            )

        total_pixels = actual_total // bands  # toplam mekânsal piksel

        # BIL formatında: data = (lines, bands, samples)
        # Yani actual_total = lines * bands * samples
        # total_pixels = lines * samples
        # lines ve samples'ı bulmak için kare kök dene
        side = int(np.sqrt(total_pixels))

        # Kare değilse farklı lines×samples kombinasyonları dene
        if side * side == total_pixels:
            new_lines = side
            new_samples = side
        else:
            # Kare değil — total_pixels'in çarpanlarını bul
            # En yakın kare köke yakın çarpanı seç
            new_lines = None
            for candidate in range(side, 0, -1):
                if total_pixels % candidate == 0:
                    new_lines = candidate
                    new_samples = total_pixels // candidate
                    break

            if new_lines is None:
                raise ValueError(
                    f"Boyut düzeltilemedi! {total_pixels} piksel için "
                    f"uygun lines×samples kombinasyonu bulunamadı."
                )

        print(f"  → Düzeltilmiş boyutlar: {new_lines}×{new_samples}×{bands}")

        # Metadata'yı güncelle
        lines = new_lines
        samples = new_samples
        metadata["lines"] = lines
        metadata["samples"] = samples

    # -------------------------------------------------------------------------
    # Interleave formatına göre array'i yeniden şekillendir
    # -------------------------------------------------------------------------
    if interleave == "bil":
        print(
            "BIL formatı tespit edildi. (lines, bands, samples) → (lines, samples, bands)"
        )
        data = raw_data.reshape((lines, bands, samples))
        data = data.transpose(0, 2, 1)

    elif interleave == "bsq":
        print(
            "BSQ formatı tespit edildi. (bands, lines, samples) → (lines, samples, bands)"
        )
        data = raw_data.reshape((bands, lines, samples))
        data = data.transpose(1, 2, 0)

    elif interleave == "bip":
        print("BIP formatı tespit edildi. Doğrudan reshape yapılıyor...")
        data = raw_data.reshape((lines, samples, bands))

    else:
        raise ValueError(f"Bilinmeyen interleave formatı: '{interleave}'")

    print(f"Yükleme tamamlandı. Array şekli: {data.shape}  |  dtype: {data.dtype}")
    return data.astype(np.float32)


# =============================================================================
# YARDIMCI FONKSİYON: load_envi
# Görev: .hdr ve .dat yollarını birlikte al, her ikisini de yükle
# =============================================================================


def load_envi(hdr_path, dat_path=None):
    """
    .hdr ve .dat dosyaları birlikte yüklenmiştir.

    Eğer dat_path verilmezse, .hdr yolundaki uzantıyı .dat ile
    değiştirerek tahmin edilmiştir.

    Parametreler:
        hdr_path (str)       : .hdr dosyasının yolu
        dat_path (str, opt.) : .dat dosyasının yolu

    Döndürür:
        tuple: (data, metadata)
            - data     : (lines, samples, bands) numpy array
            - metadata : parse_hdr()'dan dönen sözlük
    """

    if dat_path is None:
        dat_path = os.path.splitext(hdr_path)[0] + ".dat"
        print(f".dat yolu otomatik belirlendi: {dat_path}")

    print(f"Header okunuyor: {hdr_path}")
    metadata = parse_hdr(hdr_path)

    print(f"  → Satır (lines)  : {metadata.get('lines')}")
    print(f"  → Sütun (samples): {metadata.get('samples')}")
    print(f"  → Bant (bands)   : {metadata.get('bands')}")
    print(f"  → Format         : {metadata.get('interleave', '?').upper()}")
    print(f"  → Dalga boyu sayısı: {len(metadata.get('wavelengths', []))}")

    data = load_dat(dat_path, metadata)

    return data, metadata


# =============================================================================
# TEST BLOĞU
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: parse_hdr")
    print("=" * 60)

    HDR_PATH = "yaprak.hdr"
    DAT_PATH = "yaprak.dat"

    if not os.path.exists(HDR_PATH):
        print(f"'{HDR_PATH}' bulunamadı. Sahte veriyle test yapılıyor...\n")

        sahte_hdr = """ENVI
description = { Test görüntüsü }
lines = 4
samples = 4
bands = 3
data type = 4
interleave = bil
wavelength = {
  450.0, 550.0, 650.0
}
"""
        with open("test_sahte.hdr", "w") as f:
            f.write(sahte_hdr)

        meta = parse_hdr("test_sahte.hdr")
        print("Metadata:", meta)
        print()

        print("=" * 60)
        print("TEST 2: load_dat (sahte veri)")
        print("=" * 60)

        sahte_array = np.arange(48, dtype=np.float32).reshape(4, 3, 4)
        sahte_array.tofile("test_sahte.dat")

        data = load_dat("test_sahte.dat", meta)
        print(f"Yüklenen array şekli  : {data.shape}")
        print(f"Beklenen şekil        : (4, 4, 3)")
        print(f"Şekil doğru mu?       : {data.shape == (4, 4, 3)}")
        print()

        os.remove("test_sahte.hdr")
        os.remove("test_sahte.dat")

    else:
        print("=" * 60)
        print("GERÇEK VERİ TESTİ")
        print("=" * 60)

        data, meta = load_envi(HDR_PATH, DAT_PATH)

        print()
        print("Sonuçlar:")
        print(f"  Array şekli  : {data.shape}")
        print(f"  Array dtype  : {data.dtype}")
        print(f"  Min değer    : {data.min():.4f}")
        print(f"  Max değer    : {data.max():.4f}")
        print(f"  İlk 5 dalga boyu: {meta['wavelengths'][:5]}")
        print(f"  Son 5 dalga boyu: {meta['wavelengths'][-5:]}")
