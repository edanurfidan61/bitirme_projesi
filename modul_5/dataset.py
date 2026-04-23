"""
dataset.py
==========
204 yaprak görüntüsünün tamamını toplu olarak işleyerek makine öğrenmesi
modellerine hazır bir veri seti oluşturmak için yazılmıştır.

İşlem akışı:
  1. description-2.tab → ground truth (Chl, Flav, Anth, NBI) okunur
  2. Her yaprak klasöründeki results/ dosyası yüklenir
  3. Her yaprak için segmentasyon (arka plan + beyaz disk eleme) yapılır
  4. Her yaprak için 209 boyutlu özellik vektörü çıkarılır
  5. Dosya adıyla ground truth eşleştirilir
  6. Stres etiketleri atanır
  7. Sonuçlar .npy ve .csv olarak kaydedilir

GÜNCELLEME:
  - prep_steps parametresi build_dataset'e eklenmiştir
  - Segmentasyon yöntemi parametrik hale getirilmiştir
  - İlerleme bilgisi iyileştirilmiştir

Çıktılar:
  X         → (N, 209) özellik matrisi
  y_chl     → (N,)     klorofil hedef değerleri
  y_flav    → (N,)     flavonol hedef değerleri
  y_stress  → (N,)     stres sınıf etiketleri (0–3)

Literatür:
    - Ryckewaert et al. (2023): Veri seti yapısı ve ground truth
    - Cerovic et al. (2012): Dualex sensör metodolojisi
    - EMA/HMPC/464682/2016: Ph. Eur. flavonoid eşiği
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import time

# =============================================================================
# Diğer modülleri import edebilmek için üst dizin yolları eklendi
# =============================================================================

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

for _mod in ["modul_0", "modul_1", "modul_2", "modul_3", "modul_4"]:
    _mod_dir = os.path.join(_PROJECT_DIR, _mod)
    if os.path.isdir(_mod_dir) and _mod_dir not in sys.path:
        sys.path.insert(0, _mod_dir)

# Geçerli dizin de eklenir
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from load_envi import load_envi
from features import extract_features, get_feature_names

# Segmentasyon modülünü yükle — hybrid yöntemiyle beyaz disk elenir
try:
    from segmentation import best_mask
    HAS_SEGMENTATION = True
    print("[dataset.py] segmentation.py başarıyla yüklendi (hybrid yöntem kullanılacak).")
except ImportError:
    HAS_SEGMENTATION = False
    print("[dataset.py] UYARI: segmentation.py bulunamadı, visualize.py'deki yedek maskeleme kullanılacak.")
    from visualize import make_leaf_mask as best_mask


# =============================================================================
# FONKSİYON 1: load_ground_truth
# =============================================================================

def load_ground_truth(tab_path):
    """
    Ground truth dosyası okunmuştur (description-2.tab).
    """
    if not os.path.exists(tab_path):
        raise FileNotFoundError(
            f"Ground truth dosyası bulunamadı: {tab_path}\n"
            f"Ryckewaert veri setindeki 'description-2.tab' dosyasının "
            f"yolunu kontrol edin."
        )

    df = pd.read_csv(tab_path, sep='\t')

    print(f"Ground truth yüklendi: {tab_path}")
    print(f"  Satır sayısı  : {len(df)}")
    print(f"  Sütunlar      : {list(df.columns)}")

    # Sütun adlarını standartlaştır
    df.columns = df.columns.str.strip()

    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["file", "filename", "file_name", "image"]:
            col_map[col] = "filename"
        elif col_lower in ["variety", "cultivar", "grape_variety"]:
            col_map[col] = "variety"

    if col_map:
        df = df.rename(columns=col_map)
        print(f"  Yeniden adlandırılan sütunlar: {col_map}")

    return df


# =============================================================================
# FONKSİYON 2: find_leaf_folders
# =============================================================================

def find_leaf_folders(data_dir):
    """
    Data/ dizini altındaki yaprak klasörleri taranmıştır.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Veri seti dizini bulunamadı: {data_dir}")

    results = []

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        results_dir = os.path.join(folder_path, "results")
        if not os.path.isdir(results_dir):
            continue

        hdr_files = glob.glob(os.path.join(results_dir, "*.hdr"))

        if len(hdr_files) == 0:
            print(f"  UYARI: {folder_name}/results/ içinde .hdr bulunamadı, atlanıyor.")
            continue

        hdr_path = hdr_files[0]
        results.append((folder_name, hdr_path))

    print(f"Yaprak klasörleri tarandı: {len(results)} yaprak bulundu")
    return results


# =============================================================================
# FONKSİYON 3: assign_stress_labels
# =============================================================================

def assign_stress_labels(flav_values, threshold_low=0.8, threshold_high=1.5):
    """
    Flavonol değerlerine göre stres etiketleri atanmıştır.

    Sınıflandırma:
      Flav ≥ threshold_high       → 0 ("sağlıklı")
      threshold_low ≤ Flav < high → 1 ("hafif stres")
      threshold_low*0.6 ≤ Flav    → 2 ("orta stres")
      Flav < threshold_low*0.6    → 3 ("şiddetli stres")
    """
    labels = np.zeros(len(flav_values), dtype=int)

    for i, flav in enumerate(flav_values):
        if flav >= threshold_high:
            labels[i] = 0
        elif flav >= threshold_low:
            labels[i] = 1
        elif flav >= threshold_low * 0.6:
            labels[i] = 2
        else:
            labels[i] = 3

    label_names = ["sağlıklı", "hafif stres", "orta stres", "şiddetli stres"]
    print("Stres etiketleri atandı:")
    for cls in range(4):
        count = np.sum(labels == cls)
        print(f"  Sınıf {cls} ({label_names[cls]:15s}) → {count} yaprak")

    return labels


# =============================================================================
# FONKSİYON 4: build_dataset (GÜNCELLENMİŞ)
# =============================================================================

def build_dataset(data_dir, tab_path, output_dir=None, 
                  prep_steps=None, seg_method="hybrid"):
    """
    204 yaprak görüntüsü toplu olarak işlenmiş ve ML-ready veri seti
    oluşturulmuştur.

    İşlem sırası:
      1. Ground truth okunur
      2. Yaprak klasörleri taranır
      3. Her yaprak için:
         a. load_envi ile hyperspektral küp yüklenir
         b. best_mask ile yaprak maskesi oluşturulur (arka plan + beyaz disk elenir)
         c. extract_features ile 209 boyutlu özellik vektörü çıkarılır
            (prep_steps uygulanarak)
      4. Ground truth ile eşleştirilir
      5. Stres etiketleri atanır
      6. Sonuçlar kaydedilir

    Parametreler:
        data_dir (str)      : Data/ dizininin yolu
        tab_path (str)      : description-2.tab dosyasının yolu
        output_dir (str)    : Çıktı dizini (None ise kaydetme yapılmaz)
        prep_steps (list)   : Ön işleme adımları (örn: ["sg", "snv"])
                              None ise ön işleme uygulanmaz
        seg_method (str)    : Segmentasyon yöntemi ("hybrid", "ndvi", "pca", "kmeans")
                              Varsayılan: "hybrid" (arka plan + beyaz disk elenir)

    Döndürür:
        tuple : (X, y_chl, y_flav, y_stress, feature_names)
    """

    print("=" * 60)
    print("VERİ SETİ OLUŞTURMA BAŞLATILDI")
    print("=" * 60)
    
    if prep_steps:
        print(f"  Ön işleme adımları: {prep_steps}")
    else:
        print(f"  Ön işleme: YOK (ham veri)")
    print(f"  Segmentasyon yöntemi: {seg_method}")
    
    start_time = time.time()

    # -------------------------------------------------------------------------
    # ADIM 1: Ground truth oku
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Ground truth okunuyor...")
    gt_df = load_ground_truth(tab_path)

    if "filename" in gt_df.columns:
        gt_df["filename_lower"] = gt_df["filename"].astype(str).str.strip().str.lower()
    else:
        print("UYARI: 'filename' sütunu bulunamadı.")
        print(f"  Mevcut sütunlar: {list(gt_df.columns)}")
        print("  İlk sütun filename olarak varsayılıyor.")
        first_col = gt_df.columns[0]
        gt_df["filename_lower"] = gt_df[first_col].astype(str).str.strip().str.lower()

    # -------------------------------------------------------------------------
    # ADIM 2: Yaprak klasörlerini tara
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Yaprak klasörleri taranıyor...")
    leaf_folders = find_leaf_folders(data_dir)

    if len(leaf_folders) == 0:
        raise ValueError(
            f"Hiç yaprak klasörü bulunamadı: {data_dir}\n"
            f"Dizin yapısını kontrol edin: Data/<yaprak_adı>/results/*.hdr"
        )

    # -------------------------------------------------------------------------
    # ADIM 3: Her yaprak için özellik çıkarımı
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Özellik çıkarımı başlatılıyor ({len(leaf_folders)} yaprak)...")

    # İlk yaprağı yükle → dalga boyu listesini al
    _, first_meta = load_envi(leaf_folders[0][1])
    wavelengths = first_meta["wavelengths"]
    feature_names = get_feature_names(wavelengths)
    n_features = len(feature_names)   # 209

    all_features = []
    all_folder_names = []
    skipped = []

    for i, (folder_name, hdr_path) in enumerate(leaf_folders):

        progress = f"[{i+1}/{len(leaf_folders)}]"
        print(f"\n{'='*50}")
        print(f"{progress} İşleniyor: {folder_name}")

        try:
            # a. Hyperspektral küpü yükle
            data, meta = load_envi(hdr_path)

            # b. Yaprak maskesi oluştur (arka plan + beyaz disk eleme)
            mask = best_mask(data, meta["wavelengths"], method=seg_method)
            
            # Maske kontrolü
            leaf_count = np.sum(mask)
            if leaf_count == 0:
                print(f"  UYARI: {folder_name} için maske boş! Atlanıyor.")
                skipped.append(folder_name)
                continue
            
            leaf_ratio = leaf_count / mask.size * 100
            print(f"  Maske: {leaf_count} yaprak pikseli ({leaf_ratio:.1f}%)")

            # c. Özellik vektörü çıkar (prep_steps aktarılıyor!)
            feat = extract_features(data, meta["wavelengths"], mask, 
                                    prep_steps=prep_steps)

            # Boyut kontrolü
            if feat.shape[0] != n_features:
                print(f"  UYARI: Beklenen {n_features} özellik, "
                      f"alınan {feat.shape[0]}. Atlanıyor.")
                skipped.append(folder_name)
                continue

            all_features.append(feat)
            all_folder_names.append(folder_name)

        except Exception as e:
            print(f"  HATA: {folder_name} işlenemedi → {e}")
            skipped.append(folder_name)
            continue

    # Listeyi numpy array'e çevir
    X = np.array(all_features, dtype=np.float64)

    print(f"\n{'='*50}")
    print(f"Özellik çıkarımı tamamlandı:")
    print(f"  Başarılı : {len(all_folder_names)} yaprak")
    print(f"  Atlanan  : {len(skipped)} yaprak")
    print(f"  X şekli  : {X.shape}")

    if skipped:
        print(f"  Atlanan yapraklar: {skipped}")

    # -------------------------------------------------------------------------
    # ADIM 4: Ground truth ile eşleştirme
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Ground truth eşleştirmesi yapılıyor...")

    y_chl    = np.full(len(all_folder_names), np.nan, dtype=np.float64)
    y_flav   = np.full(len(all_folder_names), np.nan, dtype=np.float64)
    varieties = []
    matched_count = 0
    unmatched = []

    for i, folder_name in enumerate(all_folder_names):
        folder_lower = folder_name.strip().lower()

        # Tam eşleştirme
        match = gt_df[gt_df["filename_lower"] == folder_lower]

        if len(match) == 0:
            # Kısmi eşleştirme
            match = gt_df[gt_df["filename_lower"].str.contains(folder_lower, na=False)]

        if len(match) > 0:
            row = match.iloc[0]

            for col in ["Chl", "chl", "Chlorophyll", "chlorophyll"]:
                if col in row.index:
                    y_chl[i] = float(row[col])
                    break

            for col in ["Flav", "flav", "Flavonol", "flavonol"]:
                if col in row.index:
                    y_flav[i] = float(row[col])
                    break

            for col in ["variety", "Variety", "cultivar"]:
                if col in row.index:
                    varieties.append(str(row[col]))
                    break
            else:
                varieties.append("unknown")

            matched_count += 1
        else:
            varieties.append("unknown")
            unmatched.append(folder_name)

    print(f"  Eşleşen  : {matched_count} / {len(all_folder_names)}")
    if unmatched:
        print(f"  Eşleşmeyen yapraklar ({len(unmatched)}): {unmatched[:10]}...")

    nan_chl  = np.sum(np.isnan(y_chl))
    nan_flav = np.sum(np.isnan(y_flav))
    if nan_chl > 0:
        print(f"  UYARI: {nan_chl} yaprakta Chl değeri eksik (NaN)")
    if nan_flav > 0:
        print(f"  UYARI: {nan_flav} yaprakta Flav değeri eksik (NaN)")

    # -------------------------------------------------------------------------
    # ADIM 5: Stres etiketleri ata
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Stres etiketleri atanıyor...")

    flav_for_labels = np.nan_to_num(y_flav, nan=0.0)
    y_stress = assign_stress_labels(flav_for_labels)

    # -------------------------------------------------------------------------
    # KAYDETME
    # -------------------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y_chl.npy"), y_chl)
        np.save(os.path.join(output_dir, "y_flav.npy"), y_flav)
        np.save(os.path.join(output_dir, "y_stress.npy"), y_stress)
        np.save(os.path.join(output_dir, "feature_names.npy"),
                np.array(feature_names))

        print(f"\n.npy dosyaları kaydedildi: {output_dir}")

        # CSV
        df_features = pd.DataFrame(X, columns=feature_names)
        df_features.insert(0, "filename", all_folder_names)
        df_features.insert(1, "variety", varieties)
        df_features.insert(2, "Chl", y_chl)
        df_features.insert(3, "Flav", y_flav)
        df_features.insert(4, "stress_label", y_stress)

        csv_path = os.path.join(output_dir, "dataset_full.csv")
        df_features.to_csv(csv_path, index=False)
        print(f".csv dosyası kaydedildi: {csv_path}")

    # -------------------------------------------------------------------------
    # Süre bilgisi
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nToplam süre: {minutes} dk {seconds:.1f} sn")

    print("=" * 60)
    print("VERİ SETİ OLUŞTURMA TAMAMLANDI")
    print("=" * 60)

    return X, y_chl, y_flav, y_stress, feature_names


# =============================================================================
# FONKSİYON 5: load_saved_dataset
# =============================================================================

def load_saved_dataset(output_dir):
    """
    Daha önce build_dataset() ile kaydedilmiş .npy dosyaları yüklenmiştir.
    """
    print(f"Kaydedilmiş veri seti yükleniyor: {output_dir}")

    X             = np.load(os.path.join(output_dir, "X.npy"))
    y_chl         = np.load(os.path.join(output_dir, "y_chl.npy"))
    y_flav        = np.load(os.path.join(output_dir, "y_flav.npy"))
    y_stress      = np.load(os.path.join(output_dir, "y_stress.npy"))
    feature_names = np.load(os.path.join(output_dir, "feature_names.npy"),
                            allow_pickle=True).tolist()

    print(f"  X            : {X.shape}")
    print(f"  y_chl        : {y_chl.shape}")
    print(f"  y_flav       : {y_flav.shape}")
    print(f"  y_stress     : {y_stress.shape}  (sınıflar: {np.unique(y_stress)})")
    print(f"  feature_names: {len(feature_names)} isim")
    print("Yükleme tamamlandı.")

    return X, y_chl, y_flav, y_stress, feature_names


# =============================================================================
# TEST BLOĞU
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("dataset.py TESTİ")
    print("=" * 60)

    DATA_DIR   = r"Data"
    TAB_PATH   = r"description-2.tab"
    OUTPUT_DIR = r"dataset_output"

    # Test 1: Ground truth
    if os.path.exists(TAB_PATH):
        print("\nTEST 1: Ground truth okuma")
        gt = load_ground_truth(TAB_PATH)
        print(f"  İlk 3 satır:")
        print(gt.head(3).to_string(index=False))
    else:
        print(f"\n'{TAB_PATH}' bulunamadı. Sahte veriyle test yapılıyor...\n")
        fake_gt = pd.DataFrame({
            "filename": [f"leaf_{i:03d}" for i in range(10)],
            "variety": ["Cabernet"] * 5 + ["Merlot"] * 5,
            "Chl": np.random.uniform(20, 45, 10),
            "Flav": np.random.uniform(0.3, 2.0, 10),
        })
        print("Sahte ground truth:")
        print(fake_gt.head().to_string(index=False))

    # Test 2: Stres etiketleme
    print("\nTEST 2: assign_stress_labels")
    test_flav = np.array([2.0, 1.8, 1.2, 0.9, 0.6, 0.3, 0.1])
    test_labels = assign_stress_labels(test_flav)
    print(f"  Flav    : {test_flav}")
    print(f"  Etiketler: {test_labels}")

    # Test 3: Tam pipeline
    if os.path.exists(DATA_DIR) and os.path.exists(TAB_PATH):
        print("\nTEST 3: Tam pipeline (build_dataset)")
        X, y_chl, y_flav, y_stress, names = build_dataset(
            data_dir=DATA_DIR,
            tab_path=TAB_PATH,
            output_dir=OUTPUT_DIR,
            prep_steps=["sg", "snv"],      # ← ÖN İŞLEME AKTİF
            seg_method="hybrid"             # ← HİBRİT SEGMENTASYON
        )
        print(f"\nSonuçlar:")
        print(f"  X.shape       : {X.shape}")
        print(f"  y_chl aralığı : {np.nanmin(y_chl):.2f} – {np.nanmax(y_chl):.2f}")
        print(f"  y_flav aralığı: {np.nanmin(y_flav):.2f} – {np.nanmax(y_flav):.2f}")
        print(f"  Sınıf dağılımı: {dict(zip(*np.unique(y_stress, return_counts=True)))}")
    else:
        print(f"\nTEST 3: Atlandı (veri seti dizini bulunamadı)")

    print("\nTüm testler tamamlandı.")