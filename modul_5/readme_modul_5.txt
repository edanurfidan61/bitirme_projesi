================================================================================
  MODÜL 5 — VERİ SETİ OLUŞTURMA: dataset.py
================================================================================

  Aşama    : 5 (Veri Seti Oluşturma)
  Dosya    : dataset.py
  Durum    :  Tamamlandı

================================================================================
  AMAÇ
================================================================================

  204 yaprak görüntüsünün tamamı toplu olarak işlenmiş ve makine
  öğrenmesi modellerine hazır bir veri seti oluşturulmuştur.

  İşlem adımları:
    1. description-2.tab dosyasından ground truth değerleri okunmuştur
       (Dualex ölçümleri: Chl, Flav, Anth, NBI)
    2. Her yaprak klasöründeki results/ dosyası load_envi ile yüklenmiştir
    3. Her yaprak için features.py ile 209 boyutlu özellik vektörü
       çıkarılmıştır
    4. Dosya adı üzerinden ground truth ile eşleştirme yapılmıştır
    5. Stres etiketleri (y_stress) flavonol değerine göre atanmıştır

  Çıktılar:
    X         → (204, 209) özellik matrisi
    y_chl     → (204,)     klorofil hedef değerleri
    y_flav    → (204,)     flavonol hedef değerleri
    y_stress  → (204,)     stres sınıf etiketleri

  Bu dosyalar .npy ve .csv formatında kaydedilmiştir.
  model.py modülü bu kaydedilmiş dosyaları doğrudan yükleyerek
  her seferinde 204 yaprağı yeniden işlemekten kaçınacaktır.

================================================================================
  FONKSİYONLAR
================================================================================

  load_ground_truth(tab_path)
  ───────────────────────────
    Girdi  : description-2.tab dosyasının yolu (str)
    Çıktı  : pandas DataFrame

    Tab-separated ground truth dosyası okunmuştur. Sütunlar:
      - filename    : yaprak görüntüsünün adı
      - Chl         : klorofil değeri (Dualex µg/cm²)
      - Flav        : flavonol değeri (Dualex indeksi)
      - Anth        : antosiyanin değeri
      - NBI         : azot denge indeksi
      - variety     : çeşit adı (7 çeşitten biri)

    Cerovic et al. (2012) tarafından geliştirilen Dualex sensörü
    ile ölçülmüş değerler kullanılmıştır.

  find_leaf_folders(data_dir)
  ───────────────────────────
    Girdi  : veri seti kök dizini (str)
    Çıktı  : list of (klasör_adı, hdr_yolu) tuple'ları

    Data/ dizini altındaki yaprak klasörleri taranmıştır.
    Her klasördeki results/ alt dizininde .hdr dosyası aranmıştır.
    Bulunan dosyalar (klasör_adı, tam_hdr_yolu) olarak listelenmiştir.

  assign_stress_labels(flav_values, threshold_low, threshold_high)
  ────────────────────────────────────────────────────────────────
    Girdi  : flavonol değerleri array'i + iki eşik değeri
    Çıktı  : stres etiketleri array'i (int)

    Flavonol değerine göre stres kategorileri atanmıştır:
      0 → "sağlıklı"      (Flav ≥ threshold_high)
      1 → "hafif stres"    (threshold_low ≤ Flav < threshold_high)
      2 → "orta stres"     (Ph. Eur. eşiği civarı)
      3 → "şiddetli stres" (Flav < threshold_low)

    Ph. Eur. bağlantısı: ~3.5% toplam flavonoid eşiğinin altı
    ilaç kalitesi açısından FAIL anlamına gelmektedir.

  build_dataset(data_dir, tab_path, output_dir)
  ──────────────────────────────────────────────
    Girdi  : veri seti kök dizini + ground truth dosya yolu
             + çıktı dizini
    Çıktı  : (X, y_chl, y_flav, y_stress, feature_names) tuple

    Ana pipeline fonksiyonudur. Tüm adımları sırasıyla çalıştırır:
      1. Ground truth okunur
      2. Yaprak klasörleri taranır
      3. Her yaprak için: yükleme → maske → özellik çıkarımı
      4. Ground truth ile eşleştirme yapılır
      5. Stres etiketleri atanır
      6. Sonuçlar .npy ve .csv olarak kaydedilir

    İlerleme çubuğu (progress) ile hangi yaprağın işlendiği
    ekrana yazdırılmıştır.

  load_saved_dataset(output_dir)
  ──────────────────────────────
    Girdi  : kaydedilmiş dosyaların dizini (str)
    Çıktı  : (X, y_chl, y_flav, y_stress, feature_names) tuple

    Daha önce build_dataset() ile kaydedilmiş .npy dosyalarını
    yükler. model.py modülü bu fonksiyonu kullanacaktır.

================================================================================
  KULLANIM ÖRNEĞİ
================================================================================

  from dataset import build_dataset, load_saved_dataset

  # İlk kez: veri setini oluştur ve kaydet
  X, y_chl, y_flav, y_stress, names = build_dataset(
      data_dir  = r'C:\...\ryckewaert_dataset\Dataset\Data',
      tab_path  = r'C:\...\description-2.tab',
      output_dir = r'C:\...\BITIRME_PROJESI\dataset_output'
  )

  print(X.shape)        # (204, 209)
  print(y_flav.shape)   # (204,)

  # Sonraki sefer: kaydedilmiş dosyaları yükle (hızlı)
  X, y_chl, y_flav, y_stress, names = load_saved_dataset(
      r'C:\...\BITIRME_PROJESI\dataset_output'
  )

================================================================================
  ÇIKTI DOSYALARI
================================================================================

  build_dataset() aşağıdaki dosyaları output_dir'e kaydetmiştir:

  .npy dosyaları (numpy binary — hızlı yükleme):
    X.npy             → (204, 209) özellik matrisi
    y_chl.npy         → (204,)     klorofil değerleri
    y_flav.npy        → (204,)     flavonol değerleri
    y_stress.npy      → (204,)     stres etiketleri
    feature_names.npy → (209,)     özellik isimleri

  .csv dosyası (okunabilir format — kontrol amaçlı):
    dataset_full.csv  → tüm veriler tek tabloda
                        Sütunlar: filename, variety, Chl, Flav,
                        stress_label, band_397.32, ..., ZTM

================================================================================
  BAĞIMLILIKLAR
================================================================================

  numpy   — array işlemleri ve .npy dosya kayıt/yükleme
  pandas  — ground truth okuma ve CSV çıktısı
  os/glob — dosya sistemi tarama

  Modül içi:
    modul_0/load_envi.py   → hyperspektral veri yükleme
    modul_1/visualize.py   → make_leaf_mask (yaprak maskesi)
    modul_4/features.py    → extract_features, get_feature_names

================================================================================
  LİTERATÜR BAĞLANTISI
================================================================================

  • Ryckewaert et al. (2023): Veri seti yapısı ve ground truth
    formatı bu kaynaktan alınmıştır. description-2.tab dosyası
    Dualex ölçüm sonuçlarını içermektedir.

  • Cerovic et al. (2012): Dualex sensörünün çalışma prensibi
    ve ölçüm metodolojisi bu kaynakta açıklanmıştır.

  • EMA/HMPC/464682/2016: Ph. Eur. monografındaki ~3.5% toplam
    flavonoid eşiği, stres etiketlemesinde referans olarak
    kullanılmıştır.

================================================================================
  NOTLAR
================================================================================

  • build_dataset() uzun sürebilir (204 yaprak × yükleme + özellik
    çıkarımı). Bu nedenle sonuçlar kaydedilmiş ve load_saved_dataset()
    ile hızlıca yüklenebilir hâle getirilmiştir.

  • Dosya adı eşleştirmesi büyük/küçük harf duyarsız yapılmıştır

  • Eşleşmeyen yapraklar (ground truth'ta olmayan) atlanmış ve
    uyarı mesajı yazdırılmıştır

  • Stres eşik değerleri veri setinin Flav dağılımına göre
    ayarlanabilir — varsayılan değerler Ryckewaert verisi için
    uygun bulunmuştur

================================================================================