================================================================================
  MODÜL 1 — GÖRSELLEŞTİRME: visualize.py
================================================================================

  Aşama    : 1 (Görselleştirme)
  Dosya    : visualize.py
  Durum    : Tamamlandı

================================================================================
  AMAÇ
================================================================================

  Hiperspektral görüntü küpünden görsel çıktılar üretilmiştir.
  Üç temel görselleştirme fonksiyonu yazılmıştır:

    1. True-color RGB sentezi
    2. Yaprak maskesi oluşturma (arka plan ayrımı)
    3. False-color indeks haritası (ARI, NDVI vb.)

  Bu modül, hem veri kalitesinin gözle kontrol edilmesi hem de
  bitirme tezi için görsel materyallerin üretilmesi amacıyla
  geliştirilmiştir.

================================================================================
  FONKSİYONLAR
================================================================================

  find_band(wavelengths, target_nm)
  ─────────────────────────────────
    Girdi  : dalga boyu listesi (list) + hedef nm değeri (float)
    Çıktı  : en yakın bant indeksi (int)

    İstenen dalga boyuna en yakın bant indeksi hesaplanmıştır.
    Tüm diğer fonksiyonlar tarafından kullanılmaktadır.

  make_rgb(data, wavelengths)
  ───────────────────────────
    Girdi  : (512, 512, 204) hyperspektral küp + dalga boyu listesi
    Çıktı  : (512, 512, 3) uint8 RGB görüntüsü

    Üç bant seçilerek true-color RGB görüntüsü sentezlenmiştir:
      Kırmızı kanal → ~670 nm
      Yeşil kanal   → ~550 nm
      Mavi kanal    → ~450 nm

    Her kanal 0–255 aralığına normalize edilmiştir.
    Ryckewaert et al. (2023) veri setindeki RGBSCENE.png dosyaları
    ile karşılaştırma yapılabilir.

  make_leaf_mask(data, threshold=0.10)
  ────────────────────────────────────
    Girdi  : (512, 512, 204) hyperspektral küp
    Çıktı  : (512, 512) boolean maske

    Tüm bantlar üzerinden ortalama yansıma hesaplanmıştır.
    Eşik değerinin (varsayılan 0.10) üzerindeki pikseller yaprak
    olarak işaretlenmiştir. Morfolojik açma (opening) işlemi ile
    gürültü pikselleri temizlenmiştir.

    Bu maske, features.py ve indices.py modüllerinde arka plan
    piksellerinin hesaplamalardan dışlanması için kullanılmaktadır.

  plot_rgb(rgb, title, save_path)
  ───────────────────────────────
    Girdi  : (512, 512, 3) RGB array + başlık + kayıt yolu
    Çıktı  : matplotlib figürü (ekranda gösterim veya .png kaydı)

    Oluşturulan RGB görüntüsü matplotlib ile görselleştirilmiştir.
    save_path verilmişse .png olarak kaydedilmiştir.

  plot_index_map(index_array, mask, title, cmap, save_path)
  ──────────────────────────────────────────────────────────
    Girdi  : (512, 512) indeks haritası + yaprak maskesi + başlık
             + renk haritası adı + kayıt yolu
    Çıktı  : matplotlib figürü

    Spektral indeks değerleri false-color olarak görselleştirilmiştir.
    Yaprak maskesi dışındaki pikseller şeffaf bırakılmıştır.
    Colorbar ile değer skalası eklenmiştir.

    Önerilen renk haritaları:
      ARI   → 'RdYlGn'  (yüksek=yeşil/sağlıklı, düşük=kırmızı/stresli)
      NDVI  → 'RdYlGn'
      RVSI  → 'coolwarm'

  plot_spectral_profile(data, mask, wavelengths, title, save_path)
  ────────────────────────────────────────────────────────────────
    Girdi  : hyperspektral küp + maske + dalga boyları + başlık
    Çıktı  : matplotlib figürü

    Yaprak maskesi içindeki piksellerin ortalama spektral profili
    çizilmiştir. X ekseni dalga boyu (nm), Y ekseni yansıma değeri.
    Standart sapma bandı (±1σ) gri alan olarak eklenmiştir.

================================================================================
  KULLANIM ÖRNEĞİ
================================================================================

  import sys
  sys.path.append(r'..\modul_0')
  from load_envi import load_envi
  from visualize import make_rgb, make_leaf_mask, plot_rgb, plot_index_map

  # Veriyi yükle
  data, meta = load_envi(r'C:\...\results\yaprak.hdr')
  wl = meta['wavelengths']

  # RGB sentezle ve göster
  rgb = make_rgb(data, wl)
  plot_rgb(rgb, title="Yaprak — True Color")

  # Yaprak maskesini oluştur
  mask = make_leaf_mask(data)

  # Spektral profili çiz
  plot_spectral_profile(data, mask, wl, title="Ortalama Yansıma")

================================================================================
  BAĞIMLILIKLAR
================================================================================

  numpy       — array işlemleri
  matplotlib  — görselleştirme
  scipy.ndimage — morfolojik işlemler (binary_opening)

  Modül içi:
    modul_0/load_envi.py → veri yükleme (test bloğunda kullanılmıştır)

================================================================================
  LİTERATÜR BAĞLANTISI
================================================================================

  • True-color RGB sentezi: Ryckewaert et al. (2023) kendi RGB
    preview'larını benzer şekilde üretmiştir (results/ içinde
    RGBSCENE.png dosyaları mevcuttur)

  • Yaprak maskesi: Tüm HSI çalışmalarında standart ön işlem
    adımı olarak uygulanmaktadır

  • ARI false-color haritası: Gitelson et al. (2001) tarafından
    önerilmiş olup flavonol/antosiyanin dağılımının mekânsal
    olarak görselleştirilmesinde kullanılmıştır

  • Spektral profil çizimi: AL-Saddik et al. (2017) sağlıklı vs
    stresli yaprakların spektral farklılıklarını bu yöntemle
    göstermiştir

================================================================================
  NOTLAR
================================================================================

  • Bu modül doğrudan model eğitimi için kullanılmamaktadır;
    amacı verinin görsel olarak keşfedilmesi ve tez için
    figürlerin üretilmesidir

  • Yaprak maskesi fonksiyonu (make_leaf_mask) ileride
    features.py ve dataset.py modüllerinde yeniden kullanılacaktır

  • Eşik değeri (threshold=0.10) Ryckewaert veri seti için
    uygun bulunmuştur; farklı veri setlerinde ayarlanması
    gerekebilir

================================================================================