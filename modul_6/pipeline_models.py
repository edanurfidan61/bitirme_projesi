"""
pipeline_models.py — Tüm modelleri çalıştır + karşılaştır
Konum: modul_6/ kökünde

Çalıştırma:
  python pipeline_models.py                        # ham veri, 4-sınıf
  python pipeline_models.py --fe                   # FE verisi, 4-sınıf
  python pipeline_models.py --fe --binary          # FE verisi, binary Ph.Eur. sınıflandırma
  python pipeline_models.py --fe --compare-only    # sadece karşılaştırma tablosu
"""

import numpy as np
import os, sys, time, importlib, argparse

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODUL6_DIR = _THIS_DIR
sys.path.insert(0, _MODUL6_DIR)

for subdir in ["Regression", "classification"]:
    sub_path = os.path.join(_MODUL6_DIR, subdir)
    if os.path.isdir(sub_path) and sub_path not in sys.path:
        sys.path.insert(0, sub_path)


def _resolve_dirs(use_fe: bool):
    if use_fe:
        dataset_dir = os.path.join(_MODUL6_DIR, "dataset_output_fe")
        outputs_dir = os.path.join(_MODUL6_DIR, "model_outputs_fe")
        tag = "[FE]"
    else:
        dataset_dir = os.path.join(_MODUL6_DIR, "dataset_output")
        outputs_dir = os.path.join(_MODUL6_DIR, "model_outputs")
        tag = "[HAM]"

    if not os.path.isdir(dataset_dir):
        print(f"HATA: Veri klasörü bulunamadı: {dataset_dir}")
        if use_fe:
            print("  Önce `python feature_engineering.py` çalıştırın.")
        sys.exit(1)

    return dataset_dir, outputs_dir, tag


def run_all(use_fe: bool = False, binary: bool = False):
    dataset_dir, outputs_dir, tag = _resolve_dirs(use_fe)

    print("=" * 70)
    print(f"TÜM MODELLER ÇALIŞTIRILIYOR  {tag}")
    print(f"  Veri          : {dataset_dir}")
    print(f"  Çıktılar      : {outputs_dir}")
    print(f"  Sınıflandırma : {'binary Ph.Eur.' if binary else '4-sınıf y_stress'}")
    print("=" * 70)
    start = time.time()

    # Env değişkenlerini set et — model scriptleri buradan okur
    os.environ["_PIPELINE_DATASET_DIR"] = dataset_dir
    os.environ["_PIPELINE_OUTPUTS_DIR"] = outputs_dir
    os.environ["_PIPELINE_BINARY_CLS"]  = "1" if binary else "0"

    modules = [
        ("PLSR Regresyon",         "model_plsr"),
        ("RF Regresyon",           "model_rf_regressor"),
        ("Gradient Boosting Reg.", "model_gbr"),
        ("SVR Regresyon",          "model_svr"),
        ("RF Sınıflandırma",      "model_rf_classify"),
        ("SVM Sınıflandırma",     "model_svm_classify"),
        ("GBC Sınıflandırma",     "model_gbc"),
    ]

    for name, mod_name in modules:
        print(f"\n\n{'#'*70}")
        print(f"# {name}  {tag}")
        print(f"{'#'*70}")
        try:
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)
            mod.main()
        except Exception as e:
            print(f"  HATA: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n\nTüm modeller tamamlandı {tag}. Süre: {elapsed/60:.1f} dk")
    return outputs_dir


def compare(outputs_dir: str = None, use_fe: bool = False, binary: bool = False):
    import matplotlib.pyplot as plt

    if outputs_dir is None:
        _, outputs_dir, _ = _resolve_dirs(use_fe)
    compare_dir = os.path.join(outputs_dir, "compare")

    print(f"\n\n{'='*70}")
    print("KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*70}")
    os.makedirs(compare_dir, exist_ok=True)

    tag = "FE" if use_fe else "Ham"

    # --- REGRESYON ---
    reg_models = [("PLSR", "plsr"), ("RF Reg.", "rf_reg"), ("GBR", "gbr"), ("SVR", "svr")]

    print(f"\n  {'Model':<15s} {'Chl R²':>8s} {'Chl RMSE':>9s} {'Chl RPD':>8s} │ "
          f"{'Flav R²':>8s} {'Flav RMSE':>10s} {'Flav RPD':>9s}")
    print(f"  {'─'*15} {'─'*8} {'─'*9} {'─'*8} │ {'─'*8} {'─'*10} {'─'*9}")

    reg_data = []
    for name, folder in reg_models:
        path = os.path.join(outputs_dir, folder, "results.npz")
        if os.path.exists(path):
            r    = np.load(path)
            cr2  = float(r['chl_r2']);   crmse = float(r['chl_rmse']);  crpd = float(r['chl_rpd'])
            fr2  = float(r['flav_r2']);  frmse = float(r['flav_rmse']); frpd = float(r['flav_rpd'])
            print(f"  {name:<15s} {cr2:8.4f} {crmse:9.4f} {crpd:8.2f} │ "
                  f"{fr2:8.4f} {frmse:10.4f} {frpd:9.2f}")
            reg_data.append((name, cr2, crmse, crpd, fr2, frmse, frpd))
        else:
            print(f"  {name:<15s}  — sonuç bulunamadı")

    # --- SINIFLANDIRMA ---
    cls_label = "binary Ph.Eur." if binary else "4-sınıf"
    cls_models = [("RF Cls.", "rf_cls"), ("SVM Cls.", "svm_cls"), ("GBC", "gbc")]

    print(f"\n  Sınıflandırma modu: {cls_label}")
    print(f"  {'Model':<15s} {'Accuracy':>10s}")
    print(f"  {'─'*15} {'─'*10}")

    cls_data = []
    for name, folder in cls_models:
        path = os.path.join(outputs_dir, folder, "results.npz")
        if os.path.exists(path):
            r   = np.load(path)
            acc = float(r['accuracy'])
            print(f"  {name:<15s} {acc*100:9.1f}%")
            cls_data.append((name, acc))
        else:
            print(f"  {name:<15s}  — sonuç bulunamadı")

    # --- GÖRSELLEŞTİRME ---
    if reg_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        names   = [d[0] for d in reg_data]
        chl_r2  = [d[1] for d in reg_data]
        flav_r2 = [d[4] for d in reg_data]

        for ax, r2_vals, title in [
            (axes[0], chl_r2,  f"Klorofil R²  ({tag})"),
            (axes[1], flav_r2, f"Flavonol R²  ({tag})"),
        ]:
            colors = ['forestgreen' if r > 0.7 else 'orange' if r > 0.4 else 'red' for r in r2_vals]
            ax.barh(names, r2_vals, color=colors, edgecolor='k')
            ax.set_xlabel("R²"); ax.set_title(title, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.axvline(0.7, color='gray', linestyle=':', alpha=0.5)
            for i, v in enumerate(r2_vals):
                ax.text(max(v + 0.02, 0.05), i, f"{v:.3f}", va='center', fontsize=10)
            ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(compare_dir, "regression_r2_compare.png"), dpi=150, bbox_inches='tight')
        plt.show(); plt.close()

    if cls_data:
        random_baseline = 50 if binary else 25
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [d[0] for d in cls_data]
        accs  = [d[1] * 100 for d in cls_data]
        colors = ['forestgreen' if a > 70 else 'orange' if a > 50 else 'red' for a in accs]
        ax.barh(names, accs, color=colors, edgecolor='k')
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"Sınıflandırma ({cls_label}) — {tag}", fontweight='bold')
        ax.axvline(random_baseline, color='gray', linestyle=':', alpha=0.5,
                   label=f'Random ({random_baseline}%)')
        for i, v in enumerate(accs):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)
        ax.set_xlim(0, 100)
        ax.legend(); ax.grid(True, axis='x', alpha=0.3); plt.tight_layout()
        fig.savefig(os.path.join(compare_dir, "classification_compare.png"), dpi=150, bbox_inches='tight')
        plt.show(); plt.close()

    print(f"\n  Görseller: {compare_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tüm modelleri çalıştır ve karşılaştır")
    parser.add_argument("--fe", action="store_true",
                        help="dataset_output_fe/ kullan (özellik mühendisliği sonrası)")
    parser.add_argument("--binary", action="store_true",
                        help="Sınıflandırmada Ph.Eur. 3.5 binary modu kullan")
    parser.add_argument("--compare-only", action="store_true",
                        help="Sadece karşılaştırma tablosu/grafik üret")
    args = parser.parse_args()

    if args.compare_only:
        compare(use_fe=args.fe, binary=args.binary)
    else:
        outputs_dir = run_all(use_fe=args.fe, binary=args.binary)
        compare(outputs_dir=outputs_dir, use_fe=args.fe, binary=args.binary)


if __name__ == "__main__":
    main()