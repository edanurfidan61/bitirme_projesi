================================================================================
  MODÜL 0 — ALTYAPI: load_envi.py
================================================================================

  Aşama    : 0 (Altyapı)
  Dosya    : load_envi.py
  Durum    :  Tamamlandı

================================================================================
  AMAÇ
================================================================================

  ENVI formatındaki hiperspektral görüntü dosyalarını Python'a yüklemek.

  ENVI formatı iki dosyadan oluşur:
    .hdr  → Metin tabanlı başlık dosyası (metadata: boyutlar, bant sayısı,
             dalga boyları, veri tipi, interleave formatı)
    .dat  → Binary veri dosyası (piksel değerleri ham sayılar olarak)

  Çıktı: (512, 512, 204) boyutunda float32 numpy array
         → (satır, sütun, bant) formatında

================================================================================
  FONKSİYONLAR
================================================================================

  parse_hdr(hdr_path)
  ───────────────────
    Girdi  : .hdr dosyasının yolu (str)
    Çıktı  : metadata sözlüğü (dict)
             → lines, samples, bands, interleave, data_type, wavelengths

    Ne yapar:
      - .hdr dosyasını satır satır okur
      - "anahtar = değer" çiftlerini ayrıştırır
      - wavelength listesini regex ile çok satırlı bloktan çıkarır
      - Tüm bilgileri tek bir dict'te toplar

  envi_dtype_to_numpy(envi_code)
  ──────────────────────────────
    Girdi  : ENVI veri tipi kodu (int)
    Çıktı  : numpy dtype nesnesi

    Kod tablosu:
      1 → uint8,  2 → int16,  4 → float32,  12 → uint16  vb.

  load_dat(dat_path, metadata)
  ────────────────────────────
    Girdi  : .dat dosyasının yolu + parse_hdr()'dan dönen metadata
    Çıktı  : (lines, samples, bands) numpy array

    Ne yapar:
      - Binary dosyayı np.fromfile ile okur
      - Interleave formatına göre reshape + transpose yapar:
        BIL: (lines, bands, samples) → transpose → (lines, samples, bands)
        BSQ: (bands, lines, samples) → transpose → (lines, samples, bands)
        BIP: doğrudan (lines, samples, bands)
      - float32'ye dönüştürüp döndürür

  load_envi(hdr_path, dat_path=None)
  ──────────────────────────────────
    Girdi  : .hdr yolu (zorunlu), .dat yolu (isteğe bağlı — verilmezse
             .hdr uzantısını .dat ile değiştirerek tahmin eder)
    Çıktı  : (data, metadata) tuple

    Kolaylık fonksiyonu — parse_hdr + load_dat'ı tek seferde çağırır.

================================================================================
  KULLANIM ÖRNEĞİ
================================================================================

  from load_envi import load_envi

  hdr = r'C:\...\results\yaprak.hdr'
  data, meta = load_envi(hdr)

  print(data.shape)              # (512, 512, 204)
  print(meta['wavelengths'][:5]) # [397.32, 400.20, 403.09, ...]
  print(meta['interleave'])      # 'bil'

================================================================================
  TEST
================================================================================

  Dosyayı doğrudan çalıştırılarak test edilebilir:

    python load_envi.py

  Eğer gerçek .hdr/.dat dosyası yoksa, sahte veriyle otomatik test yapar:
    - 4×4 piksel, 3 bant, BIL formatında sahte dosya oluşturur
    - Yükleyip shape kontrolü yapar
    - Geçici dosyaları temizler

  Gerçek dosya varsa:
    - Yükleme sonrası shape, dtype, min/max, dalga boyu bilgisi yazdırır

================================================================================
  BAĞIMLILIKLAR
================================================================================

  numpy  — array işlemleri ve binary okuma
  re     — regex ile wavelength bloğu ayrıştırma
  os     — dosya varlık kontrolü ve yol işlemleri

================================================================================
  NOTLAR
================================================================================

  • Bu modül diğer tüm modüllerin temelini oluşturur
  • visualize.py, indices.py, features.py hepsi load_envi'yi import eder
  • Ryckewaert veri setindeki dosyalar BIL interleave ve uint16 (data_type=12)
    formatındadır; load_dat bunları otomatik olarak float32'ye çevirir
  • results/ klasöründeki ön-hesaplanmış yansıma dosyaları doğrudan
    bu modülle okunabilir (calibrate.py adımını atlamak mümkün)

================================================================================