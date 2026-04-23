================================================================================
  MODÜL 4 — ÖZELLİK ÇIKARIMI: features.py
================================================================================

  Aşama    : 4 (Özellik Çıkarımı)
  Dosya    : features.py
  Durum    : ✅ Tamamlandı

================================================================================
  AMAÇ
================================================================================

  Her yaprak için tek bir özellik vektörü çıkarılmıştır.
  Bu vektör, makine öğrenmesi modellerine girdi olarak
  kullanılacaktır.

  Özellik vektörü iki bileşenden oluşturulmuştur:

    1. Spektral ortalama vektörü (204 boyut)
       → Yaprak maskesi içindeki piksellerin her bant için
         ortalaması alınmıştır

    2. İndeks ortalamaları (5 boyut)
       → NDVI, GNDVI, ARI, RVSI, ZTM indekslerinin yaprak
         maskesi üzerindeki ortalamaları hesaplanmıştır

    Toplam: 204 + 5 = 209 boyutlu özellik vektörü

  Bu yaklaşım, (512, 512, 204) boyutundaki 3D görüntü küpünü
  tek bir (209,) vektöre indirgemektedir. dataset.py modülünde
  204 yaprak için bu işlem tekrarlanarak X (204, 209) özellik
  matrisi oluşturulacaktır.

================================================================================
  FONKSİYONLAR
================================================================================

  extract_spectral_means(data, mask)
  ──────────────────────────────────
    Girdi  : (lines, samples, bands) 3D array + boolean yaprak maskesi
    Çıktı  : (bands,) float64 vektör — her bantın yaprak ortalaması

    Yaprak maskesi içindeki pikseller seçilmiştir (data[mask] → 2D array).
    Her bant (sütun) boyunca ortalama alınmıştır (axis=0).

    Ryckewaert veri seti için çıktı: (204,) vektör
    Her eleman, ilgili dalga boyundaki ortalama yansıma değerini
    temsil etmektedir.

    Literatür: Burnett et al. (2021) PLSR modelleri için bu
    yaklaşımı uygulamıştır.

  extract_index_means(data, wavelengths, mask)
  ─────────────────────────────────────────────
    Girdi  : 3D array + dalga boyu listesi + boolean yaprak maskesi
    Çıktı  : (5,) float64 vektör — 5 indeksin yaprak ortalaması

    indices.py modülündeki calc_all_indices() ile 5 indeks haritası
    hesaplanmıştır. Her indeks haritasının yaprak maskesi içindeki
    ortalaması alınmıştır.

    Çıktı sırası: [NDVI, GNDVI, ARI, RVSI, ZTM]

  extract_features(data, wavelengths, mask)
  ──────────────────────────────────────────
    Girdi  : 3D array + dalga boyu listesi + boolean yaprak maskesi
    Çıktı  : (209,) float64 vektör — birleştirilmiş özellik vektörü

    Üst düzey fonksiyon — extract_spectral_means() ve
    extract_index_means() çağrılarak sonuçlar np.concatenate
    ile birleştirilmiştir.

    Bu fonksiyon dataset.py tarafından her yaprak için
    çağrılacaktır.

  get_feature_names(wavelengths)
  ──────────────────────────────
    Girdi  : dalga boyu listesi
    Çıktı  : 209 elemanlı isim listesi (list of str)

    Özellik matrisinin sütun isimlerini döndürür.
    İlk 204 eleman: "band_397.32", "band_400.20", ...
    Son 5 eleman: "NDVI", "GNDVI", "ARI", "RVSI", "ZTM"

    Bu isimler, model.py'de feature importance grafiklerinde
    ve CSV çıktılarında kullanılacaktır.

================================================================================
  KULLANIM ÖRNEĞİ
================================================================================

  import sys
  sys.path.append(r'..\modul_0')
  sys.path.append(r'..\modul_1')
  sys.path.append(r'..\modul_2')
  from load_envi import load_envi
  from visualize import make_leaf_mask
  from features import extract_features, get_feature_names

  # Veriyi yükle
  data, meta = load_envi(r'C:\...\results\yaprak.hdr')
  wl = meta['wavelengths']

  # Yaprak maskesi oluştur
  mask = make_leaf_mask(data)

  # Özellik vektörü çıkar
  feat = extract_features(data, wl, mask)
  print(feat.shape)         # (209,)
  print(feat[:5])           # İlk 5 spektral ortalama

  # Özellik isimleri
  names = get_feature_names(wl)
  print(names[:3])          # ['band_397.32', 'band_400.20', 'band_403.09']
  print(names[-3:])         # ['ARI', 'RVSI', 'ZTM']

================================================================================
  BAĞIMLILIKLAR
================================================================================

  numpy  — array işlemleri ve birleştirme

  Modül içi:
    modul_0/load_envi.py   → veri yükleme (test bloğunda)
    modul_1/visualize.py   → make_leaf_mask (yaprak maskesi)
    modul_2/indices.py     → calc_all_indices (indeks hesaplama)

================================================================================
  LİTERATÜR BAĞLANTISI
================================================================================

  • Burnett et al. (2021): PLSR regresyonu için yaprak düzeyinde
    ortalama spektral vektör kullanılmıştır. Bu projede aynı
    yaklaşım uygulanmıştır.

  • Okyere et al. (2023): Özellik seçim pipeline'ında spektral
    bantlar ve vejetasyon indeksleri birlikte kullanılmıştır.
    Bu projede de aynı birleştirme stratejisi izlenmiştir.

  • TÜBİTAK kontağı: Özellik matrisinin (N_piksel, bands) formatında
    CSV/transposed yapıda tutulması önerilmiştir. Bu modülde yaprak
    düzeyinde ortalama alınarak (1, 209) vektöre indirgenmiştir.

================================================================================
  NOTLAR
================================================================================

  • Özellik vektörü float64 hassasiyetinde tutulmuştur (PLSR için önemli)

  • Yaprak maskesi visualize.py'den import edilmektedir; eşik değeri
    (threshold=0.10) orada tanımlanmıştır

  • İndeks sırası sabittir: [NDVI, GNDVI, ARI, RVSI, ZTM]
    Bu sıra dataset.py ve model.py'de de korunmalıdır

  • Boş maske durumu kontrol edilmiştir — yaprak pikseli bulunamazsa
    uyarı verilmekte ve sıfır vektörü döndürülmektedir

================================================================================