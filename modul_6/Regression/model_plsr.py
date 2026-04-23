"""
model_plsr.py — PLSR Regresyon
Konum: modul_6/Regression/model_plsr.py

Çalıştırma (tekil):
  python model_plsr.py                   # dataset_output/ kullanır
Çalıştırma (pipeline üzerinden):
  python pipeline_models.py --fe         # dataset_output_fe/ ve model_outputs_fe/ kullanır
"""
import numpy as np
import os, sys

# --- PATH AYARI ---
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))   # Regression/
_MODUL6_DIR = os.path.dirname(_THIS_DIR)                    # modul_6/
sys.path.insert(0, _MODUL6_DIR)

from utils_model import load_dataset, regression_metrics, plot_regression, pheur_pass_fail

# pipeline_models.py --fe çalıştırıldığında bu env değişkenleri set edilmiş olur.
# Tekil çalıştırmada env yoktur → varsayılan (ham veri) path'ler kullanılır.
DATASET_DIR = os.environ.get(
    "_PIPELINE_DATASET_DIR",
    os.path.join(_MODUL6_DIR, "dataset_output")
)
OUTPUT_DIR = os.environ.get(
    "_PIPELINE_OUTPUTS_DIR",
    os.path.join(_MODUL6_DIR, "model_outputs")
)
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "plsr")


def run_plsr_single(X, y, target_name, output_dir):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    import matplotlib.pyplot as plt

    valid = ~np.isnan(y)
    X_c, y_c = X[valid], y[valid]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    max_n = min(20, X_s.shape[1], X_s.shape[0] - 5)
    best_rmse, best_n = 999, 5
    rmse_list = []
    for n in range(1, max_n + 1):
        pls = PLSRegression(n_components=n, scale=False)
        yp  = cross_val_predict(pls, X_s, y_c, cv=5).ravel()
        rmse = np.sqrt(np.mean((y_c - yp) ** 2))
        rmse_list.append(rmse)
        if rmse < best_rmse:
            best_rmse, best_n = rmse, n
    print(f"  Optimal bileşen: {best_n} (RMSE={best_rmse:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(rmse_list) + 1), rmse_list, 'b-o', markersize=3)
    ax.axvline(best_n, color='r', linestyle='--', label=f'Optimal={best_n}')
    ax.set_xlabel("Bileşen Sayısı"); ax.set_ylabel("CV-RMSE")
    ax.set_title(f"PLSR Bileşen Seçimi — {target_name}", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"component_{target_name}.png"), dpi=150)
    plt.show(); plt.close()

    model = PLSRegression(n_components=best_n, scale=False)
    y_cv  = cross_val_predict(model, X_s, y_c, cv=5).ravel()
    metrics = regression_metrics(y_c, y_cv, label=f"PLSR {target_name} (CV)")
    plot_regression(
        y_c, y_cv,
        title=f"PLSR — {target_name} (5-Fold CV)",
        xlabel=f"Gerçek {target_name}", ylabel=f"Tahmin {target_name}",
        save_path=os.path.join(output_dir, f"scatter_{target_name}.png"),
    )
    return {"metrics": metrics, "optimal_n": best_n, "y_true": y_c, "y_pred": y_cv}


def main():
    print("=" * 60); print("PLSR REGRESYON"); print("=" * 60)
    print(f"  Veri   : {DATASET_DIR}")
    print(f"  Çıktı  : {OUTPUT_DIR}")

    X, y_chl, y_flav, y_stress, feat = load_dataset(DATASET_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    res_chl  = run_plsr_single(X, y_chl,  "Chl",  OUTPUT_DIR)
    res_flav = run_plsr_single(X, y_flav, "Flav", OUTPUT_DIR)
    pheur_pass_fail(res_flav["y_true"], res_flav["y_pred"])

    print(f"\nPLSR ÖZET")
    print(f"  Chl:  R²={res_chl['metrics']['R2']:.4f}, RMSE={res_chl['metrics']['RMSE']:.4f}, RPD={res_chl['metrics']['RPD']:.2f}")
    print(f"  Flav: R²={res_flav['metrics']['R2']:.4f}, RMSE={res_flav['metrics']['RMSE']:.4f}, RPD={res_flav['metrics']['RPD']:.2f}")

    np.savez(
        os.path.join(OUTPUT_DIR, "results.npz"),
        chl_r2=res_chl['metrics']['R2'],   chl_rmse=res_chl['metrics']['RMSE'],
        chl_rpd=res_chl['metrics']['RPD'], chl_n=res_chl['optimal_n'],
        flav_r2=res_flav['metrics']['R2'], flav_rmse=res_flav['metrics']['RMSE'],
        flav_rpd=res_flav['metrics']['RPD'], flav_n=res_flav['optimal_n'],
    )


if __name__ == "__main__":
    main()