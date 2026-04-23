"""
model_svr.py — Support Vector Regresyon
Konum: modul_6/Regression/

Çalıştırma (tekil):
  python model_svr.py
Çalıştırma (pipeline üzerinden):
  python pipeline_models.py --fe
"""

import numpy as np
import os, sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODUL6_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _MODUL6_DIR)

from utils_model import (
    load_dataset,
    regression_metrics,
    plot_regression,
    pheur_pass_fail,
)

DATASET_DIR = os.environ.get(
    "_PIPELINE_DATASET_DIR", os.path.join(_MODUL6_DIR, "dataset_output")
)
OUTPUT_DIR = os.environ.get(
    "_PIPELINE_OUTPUTS_DIR", os.path.join(_MODUL6_DIR, "model_outputs")
)
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "svr")


def run_svr(X, y, target_name, output_dir):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold

    valid = ~np.isnan(y)
    X_c, y_c = X[valid], y[valid]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    print(f"\n  [{target_name}] Hyperparameter arama...")
    param_grid = {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.001],
        "epsilon": [0.01, 0.05, 0.1],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(SVR(), param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X_s, y_c)
    print(f"  En iyi: {grid.best_params_}, CV R²={grid.best_score_:.4f}")

    y_cv = cross_val_predict(grid.best_estimator_, X_s, y_c, cv=cv)
    metrics = regression_metrics(y_c, y_cv, label=f"SVR {target_name} (CV)")
    plot_regression(
        y_c,
        y_cv,
        title=f"SVR — {target_name} (5-Fold CV)",
        xlabel=f"Gerçek {target_name}",
        ylabel=f"Tahmin {target_name}",
        save_path=os.path.join(output_dir, f"scatter_{target_name}.png"),
    )
    return {"metrics": metrics, "y_true": y_c, "y_pred": y_cv}


def main():
    print("=" * 60)
    print("SVR REGRESYON")
    print("=" * 60)
    print(f"  Veri   : {DATASET_DIR}")
    print(f"  Çıktı  : {OUTPUT_DIR}")

    X, y_chl, y_flav, y_stress, feat = load_dataset(DATASET_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    res_chl = run_svr(X, y_chl, "Chl", OUTPUT_DIR)
    res_flav = run_svr(X, y_flav, "Flav", OUTPUT_DIR)
    pheur_pass_fail(res_flav["y_true"], res_flav["y_pred"])

    print(f"\nSVR ÖZET")
    print(
        f"  Chl:  R²={res_chl['metrics']['R2']:.4f}, RMSE={res_chl['metrics']['RMSE']:.4f}, RPD={res_chl['metrics']['RPD']:.2f}"
    )
    print(
        f"  Flav: R²={res_flav['metrics']['R2']:.4f}, RMSE={res_flav['metrics']['RMSE']:.4f}, RPD={res_flav['metrics']['RPD']:.2f}"
    )

    np.savez(
        os.path.join(OUTPUT_DIR, "results.npz"),
        chl_r2=res_chl["metrics"]["R2"],
        chl_rmse=res_chl["metrics"]["RMSE"],
        chl_rpd=res_chl["metrics"]["RPD"],
        flav_r2=res_flav["metrics"]["R2"],
        flav_rmse=res_flav["metrics"]["RMSE"],
        flav_rpd=res_flav["metrics"]["RPD"],
    )


if __name__ == "__main__":
    main()
