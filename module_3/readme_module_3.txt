================================================================================
  MODÜL 3 — KALİBRASYON: calibrate.py
================================================================================

  Aşama    : 3 (Kalibrasyon)
  Dosya    : calibrate.py
  Durum    : ⏭️ Atlandı (ön-hesaplanmış yansıma verileri kullanılmıştır)

================================================================================
  AÇIKLAMA
================================================================================

  Kalibrasyon adımında ham görüntü verisi (raw DN) yansıma değerine
  dönüştürülmektedir:

    R = (scene − dark) / (white − dark)

  Ancak Ryckewaert et al. (2023) veri setinde bu adım üretici
  tarafından zaten uygulanmıştır. Her yaprak klasöründeki results/
  alt dizininde ön-hesaplanmış yansıma dosyaları (.dat + .hdr)
  bulunmaktadır.

  Bu nedenle calibrate.py modülü yazılmamıştır. Pipeline'da
  load_envi.py ile doğrudan results/ klasöründeki dosyalar
  okunmuştur.

================================================================================
  GELECEKTE GEREKİRSE
================================================================================

  Farklı bir veri seti veya Specim FX10 kamerasından alınan ham
  görüntüler kullanıldığında bu modülün yazılması gerekecektir.

  Beklenen fonksiyonlar:
    - load_dark_reference()   → karanlık referans okuması
    - load_white_reference()  → beyaz referans okuması
    - calibrate_to_reflectance(scene, dark, white) → yansıma hesabı

================================================================================