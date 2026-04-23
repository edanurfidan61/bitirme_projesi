================================================================================
  MODÜL 2 — SPEKTRAL İNDEKSLER: indices.py
================================================================================

  Aşama    : 2 (Spektral İndeksler)
  Dosya    : indices.py
  Durum    : Tamamlandı

================================================================================
  AMAÇ
================================================================================

  Hiperspektral yansıma verisinden bitki fizyolojisini yansıtan
  spektral indeksler hesaplanmıştır. Her indeks, belirli dalga
  boylarındaki yansıma değerlerinin matematiksel kombinasyonundan
  oluşmaktadır.

  Hesaplanan 5 indeks:

    İndeks    Amaç                          Kaynak
    ──────    ────────────────────────────   ──────────────────────────
    NDVI      Genel bitki sağlığı           Rouse et al. (1974)
    GNDVI     Klorofil hassasiyeti          Gitelson et al. (1996)
    ARI       Flavonol / antosiyanin        Gitelson et al. (2001)
    RVSI      Kırmızı kenar stresi          Merton & Huntington (1999)
    ZTM       Klorofil içeriği              Zarco-Tejada et al. (2001)

================================================================================
  FONKSİYONLAR
================================================================================

  get_band(wavelengths, target_nm)
  ────────────────────────────────
    Girdi  : dalga boyu listesi (list) + hedef nm değeri (float)
    Çıktı  : en yakın bant indeksi (int)

    İstenen dalga boyuna en yakın bant indeksi hesaplanmıştır.
    Tüm indeks fonksiyonları bu yardımcı fonksiyonu kullanmaktadır.

    Not: visualize.py modülünde de aynı mantıkta bir find_band()
    fonksiyonu bulunmaktadır. İki modül birbirinden bağımsız
    çalışabilmesi için her birinde ayrı tanımlanmıştır.

  calc_ndvi(data, wavelengths)
  ────────────────────────────
    Formül : (R800 − R670) / (R800 + R670)
    Çıktı  : (lines, samples) float array, değer aralığı [-1, +1]

    Normalized Difference Vegetation Index hesaplanmıştır.
    Sağlıklı bitkilerde NDVI yüksek (~0.7–0.9), stresli veya
    ölü dokularda düşük (< 0.3) değer almaktadır.

  calc_gndvi(data, wavelengths)
  ─────────────────────────────
    Formül : (R800 − R550) / (R800 + R550)
    Çıktı  : (lines, samples) float array, değer aralığı [-1, +1]

    Green NDVI hesaplanmıştır. Klasik NDVI'ya göre klorofil
    değişimlerine daha hassas olduğu bildirilmiştir. R670 yerine
    R550 (yeşil bant) kullanılmıştır.

  calc_ari(data, wavelengths)
  ───────────────────────────
    Formül : 1/R550 − 1/R700
    Çıktı  : (lines, samples) float array

    Anthocyanin Reflectance Index hesaplanmıştır.
    Yüksek ARI değeri flavonol/antosiyanin birikmesine işaret
    etmektedir. Sağlıklı yapraklarda savunma pigmentleri daha
    yoğun bulunmaktadır.

    Dikkat: Sıfıra bölme koruması eklenmiştir (R550 veya R700
    sıfır olduğunda sonuç 0.0 olarak atanmıştır).

    Literatür: Gitelson et al. (2001), AL-Saddik et al. (2017)

  calc_rvsi(data, wavelengths)
  ────────────────────────────
    Formül : (R714 + R752) / 2 − R733
    Çıktı  : (lines, samples) float array

    Red-edge Vegetation Stress Index hesaplanmıştır.
    Kırmızı kenar bölgesindeki (700–750 nm) eğrilik değişimini
    ölçmektedir. Stresli bitkilerde RVSI artan değerler
    göstermektedir.

  calc_ztm(data, wavelengths)
  ───────────────────────────
    Formül : R750 / R710
    Çıktı  : (lines, samples) float array

    Zarco-Tejada & Miller indeksi hesaplanmıştır.
    Klorofil içeriğiyle güçlü korelasyon gösterdiği
    bildirilmiştir. Sağlıklı yapraklarda ZTM daha yüksek
    değerler almaktadır.

    Dikkat: Sıfıra bölme koruması eklenmiştir.

  calc_all_indices(data, wavelengths)
  ───────────────────────────────────
    Girdi  : hyperspektral küp + dalga boyu listesi
    Çıktı  : dict — 5 indeksin adı → (lines, samples) array eşlemesi

    Tüm indeksler tek seferde hesaplanmıştır. Döndürülen sözlük:
      {"NDVI": ..., "GNDVI": ..., "ARI": ..., "RVSI": ..., "ZTM": ...}

    features.py modülü bu fonksiyonu kullanarak indeks
    ortalamalarını hesaplayacaktır.

================================================================================
  KULLANIM ÖRNEĞİ
================================================================================

  import sys
  sys.path.append(r'..\modul_0')
  from load_envi import load_envi
  from indices import calc_all_indices, calc_ari

  # Veriyi yükle
  data, meta = load_envi(r'C:\...\results\yaprak.hdr')
  wl = meta['wavelengths']

  # Tüm indeksleri hesapla
  indices = calc_all_indices(data, wl)
  print(indices['ARI'].shape)       # (512, 512)
  print(indices['NDVI'].mean())     # ~0.75

  # Tek indeks hesapla
  ari_map = calc_ari(data, wl)

================================================================================
  BAĞIMLILIKLAR
================================================================================

  numpy  — array işlemleri ve matematiksel hesaplamalar

  Modül içi:
    modul_0/load_envi.py → veri yükleme (test bloğunda kullanılmıştır)
    modul_1/visualize.py → indeks haritası görselleştirme (test bloğunda
                           plot_index_map ile kullanılmıştır)

================================================================================
  LİTERATÜR BAĞLANTISI
================================================================================

  • NDVI: Rouse et al. (1974) tarafından önerilmiştir. En yaygın
    kullanılan vejetasyon indeksidir. Bitki canlılığının genel
    göstergesi olarak kabul edilmiştir.

  • GNDVI: Gitelson et al. (1996) klorofil duyarlılığını artırmak
    amacıyla NDVI'nın yeşil bant varyantını geliştirmiştir.

  • ARI: Gitelson et al. (2001) antosiyanin/flavonol pigmentlerini
    tespit etmek için önermiştir. Bu projede flavonol tahmininin
    temel spektral göstergesi olarak kullanılmıştır.

  • RVSI: Merton & Huntington (1999) kırmızı kenar bölgesindeki
    stres belirtilerini yakalamak için geliştirmiştir.

  • ZTM: Zarco-Tejada et al. (2001) klorofil içeriğiyle yüksek
    korelasyon gösteren basit bir oran indeksi olarak önermiştir.

  • Tüm indeksler: Okyere et al. (2023) özellik seçim pipeline'ında
    benzer indeksleri girdi olarak kullanmıştır.

================================================================================
  NOTLAR
================================================================================

  • Sıfıra bölme durumları np.where ile korunmuştur — payda sıfır
    olduğunda sonuç 0.0 olarak atanmıştır

  • İndeks değerleri piksel bazında hesaplanmıştır; yaprak ortalaması
    features.py modülünde alınacaktır

  • Ryckewaert veri setinin spektral aralığı 397–1004 nm olduğundan
    tüm indeks bantları (450–800 nm arası) kapsam dahilindedir

  • get_band() fonksiyonu, formüldeki sabit nm değerlerini (örn: R800)
    veri setindeki en yakın gerçek dalga boyuyla eşleştirmektedir;
    bu yaklaşım farklı kamera çözünürlüklerinde de çalışmaktadır

================================================================================