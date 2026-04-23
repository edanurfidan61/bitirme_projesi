"""
model_rf_classify.py — RF Sınıflandırma
Konum: modul_6/classification/

Çalıştırma (tekil):
  python model_rf_classify.py              # 4-sınıf, y_stress etiketi
  python model_rf_classify.py --binary     # 2-sınıf, Ph.Eur. 3.5 eşiği (y_flav)
Çalıştırma (pipeline üzerinden):
  python pipeline_models.py --fe
"""
import numpy as np
import os, sys, argparse

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODUL6_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _MODUL6_DIR)

from utils_model import load_dataset, classify_metrics, plot_confusion, plot_feature_importance

# ── Env-aware paths ──────────────────────────────────────────────────────────
DATASET_DIR = os.environ.get(
    "_PIPELINE_DATASET_DIR",
    os.path.join(_MODUL6_DIR, "dataset_output")
)
OUTPUT_DIR = os.environ.get(
    "_PIPELINE_OUTPUTS_DIR",
    os.path.join(_MODUL6_DIR, "model_outputs")
)
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "rf_cls")

# Pipeline'dan binary mod env değişkeni ile de tetiklenebilir
_BINARY_ENV = os.environ.get("_PIPELINE_BINARY_CLS", "0") == "1"

PHEUR_THRESHOLD = 3.5
CLASS_NAMES_4   = ["yüksek flav", "orta-yüksek", "orta-düşük", "düşük flav"]
CLASS_NAMES_2   = ["FAIL (Flav<3.5)", "PASS (Flav≥3.5)"]


def make_binary_labels(y_flav, threshold=PHEUR_THRESHOLD):
    """Ph.Eur. eşiğine göre 0/1 etiket üret."""
    return (y_flav >= threshold).astype(int)


def main(binary: bool = False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold

    binary = binary or _BINARY_ENV

    print("=" * 60)
    print(f"RF SINIFLANDIRMA  ({'binary Ph.Eur.' if binary else '4-sınıf y_stress'})")
    print("=" * 60)
    print(f"  Veri   : {DATASET_DIR}")
    print(f"  Çıktı  : {OUTPUT_DIR}")

    X, y_chl, y_flav, y_stress, feat = load_dataset(DATASET_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Etiket seçimi ──────────────────────────────────────────────────────
    if binary:
        valid = ~np.isnan(y_flav)
        y_c   = make_binary_labels(y_flav[valid])
        class_names = CLASS_NAMES_2
        print(f"  PASS (≥3.5): {y_c.sum()}  FAIL (<3.5): {(1-y_c).sum()}")
    else:
        valid = ~np.isnan(y_flav)
        y_c   = y_stress[valid]
        class_names = CLASS_NAMES_4

    X_c = X[valid]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    # ── GridSearch ─────────────────────────────────────────────────────────
    param_grid = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 4],
    }
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1,
    )
    grid.fit(X_s, y_c)
    print(f"  En iyi: {grid.best_params_}, acc={grid.best_score_:.4f}")

    y_cv = cross_val_predict(grid.best_estimator_, X_s, y_c, cv=cv)
    res  = classify_metrics(y_c, y_cv, class_names, label="RF (CV)")

    suffix = "binary" if binary else "4cls"
    plot_confusion(
        res["confusion_matrix"], class_names,
        title=f"RF — Confusion Matrix ({suffix})",
        save_path=os.path.join(OUTPUT_DIR, f"confusion_rf_{suffix}.png"),
    )
    grid.best_estimator_.fit(X_s, y_c)
    plot_feature_importance(
        grid.best_estimator_.feature_importances_, feat, top_n=20,
        title=f"RF — Feature Importance ({suffix})",
        save_path=os.path.join(OUTPUT_DIR, f"importance_rf_{suffix}.png"),
    )
    np.savez(
        os.path.join(OUTPUT_DIR, "results.npz"),
        accuracy=res['accuracy'],
        cm=res['confusion_matrix'],
        binary=binary,
    )
    print(f"\n  Accuracy: {res['accuracy']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", action="store_true",
                        help="Ph.Eur. 3.5 eşiğine göre 2-sınıf sınıflandırma")
    args = parser.parse_args()
    main(binary=args.binary)