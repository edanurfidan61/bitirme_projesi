"""
fix_fe_outputs.py — Geçici yama
Konum: modul_6/ kökünde

Problem:
  Model scriptleri results.npz dosyalarını model_outputs/<alt_klasör>/
  altına kaydediyor. --fe modunda bunlar model_outputs_fe/ altına gitmesi lazım.

Çözüm:
  Bu script model_outputs/ altındaki results.npz + görselleri
  model_outputs_fe/ altındaki doğru konuma kopyalar.

Kullanım:
  python fix_fe_outputs.py
  (pipeline_models.py --fe çalıştırdıktan HEMEN SONRA çalıştır)
"""

import os
import shutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# kaynak → hedef klasör eşleşmeleri
FOLDER_MAP = {
    "plsr": "plsr",
    "rf_reg": "rf_reg",
    "gbr": "gbr",
    "svr": "svr",
    "rf_cls": "rf_cls",
    "svm_cls": "svm_cls",
    "gbc": "gbc",
}

SRC_BASE = os.path.join(_THIS_DIR, "model_outputs")
DST_BASE = os.path.join(_THIS_DIR, "model_outputs_fe")


def copy_results():
    print("model_outputs/ → model_outputs_fe/ kopyalanıyor...")
    for folder in FOLDER_MAP:
        src = os.path.join(SRC_BASE, folder)
        dst = os.path.join(DST_BASE, folder)
        if not os.path.isdir(src):
            print(f"  ATLA (bulunamadı): {src}")
            continue
        os.makedirs(dst, exist_ok=True)
        copied = 0
        for fname in os.listdir(src):
            s = os.path.join(src, fname)
            d = os.path.join(dst, fname)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                copied += 1
        print(f"  {folder:<12s}: {copied} dosya kopyalandı → {dst}")
    print(
        "\nTamamlandı. Şimdi pipeline_models.py --fe --compare-only çalıştırabilirsin."
    )


if __name__ == "__main__":
    copy_results()
