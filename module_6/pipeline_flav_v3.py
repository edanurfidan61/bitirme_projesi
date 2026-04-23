"""
pipeline_flav_v3.py — Flavonol Odaklı Genişletilmiş Model Pipeline
Konum: modul_6/ kökünde

Yenilikler:
  - dataset_output_fe_v3 kullanır (252 özellik, ARI kompozit dahil)
  - Karar Ağacı (Decision Tree) görselleştirmesiyle
  - ExtraTreesRegressor / ExtraTreesClassifier
  - Stacking (RF + GBR + SVR → Ridge meta-model)
  - y_stress'i ek özellik olarak dener
  - Tüm sonuçları karşılaştırma tablosuna ekler

Çalıştırma:
  python pipeline_flav_v3.py
  python pipeline_flav_v3.py --stress    # y_stress özellik olarak ekle
  python pipeline_flav_v3.py --tree_plot # Karar ağacı görselini kaydet
"""

import numpy as np
import os, sys, argparse, time

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# ── Regresyon modelleri
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

# ── Sınıflandırma modelleri
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

DATASET_DIR = os.path.join(_THIS_DIR, "dataset_output_fe_v3")
OUTPUT_DIR = os.path.join(_THIS_DIR, "model_outputs_fe_v3")
PHEUR_THR = 3.5
CV_FOLDS = 5
eps = 1e-6


# ── YARDIMCILAR ─────────────────────────────────────────────────────────────
def load_data(use_stress=False):
    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    y_chl = np.load(os.path.join(DATASET_DIR, "y_chl.npy"))
    y_flav = np.load(os.path.join(DATASET_DIR, "y_flav.npy"))
    y_stress = np.load(os.path.join(DATASET_DIR, "y_stress.npy"))
    feat = np.load(
        os.path.join(DATASET_DIR, "feature_names.npy"), allow_pickle=True
    ).tolist()

    # NaN satırlarını at
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    X = X[valid]
    y_chl = y_chl[valid]
    y_flav = y_flav[valid]
    y_stress = y_stress[valid]

    if use_stress:
        # y_stress'i ek sütun olarak X'e ekle
        stress_col = y_stress.reshape(-1, 1)
        X = np.hstack([X, stress_col])
        feat = feat + ["y_stress"]
        print(f"  [+] y_stress özellik olarak eklendi → X={X.shape}")

    # NaN/Inf temizle
    X = np.where(np.isfinite(X), X, 0.0)

    print(
        f"  X: {X.shape}, Chl: {np.nanmin(y_chl):.1f}–{np.nanmax(y_chl):.1f}, "
        f"Flav: {np.nanmin(y_flav):.2f}–{np.nanmax(y_flav):.2f}"
    )
    print(
        f"  PASS (≥{PHEUR_THR}): {(y_flav>=PHEUR_THR).sum()}  "
        f"FAIL (<{PHEUR_THR}): {(y_flav<PHEUR_THR).sum()}"
    )
    return X, y_chl, y_flav, y_stress, feat


def reg_metrics(y_true, y_pred, label=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / (rmse + eps)
    nonz = y_true != 0
    mape = np.mean(np.abs((y_true[nonz] - y_pred[nonz]) / y_true[nonz])) * 100
    if label:
        print(
            f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  RPD={rpd:.2f}  MAPE={mape:.1f}%"
        )
    return dict(R2=r2, RMSE=rmse, RPD=rpd, MAPE=mape)


def pheur_acc(y_true, y_pred):
    return accuracy_score(
        (y_true >= PHEUR_THR).astype(int), (y_pred >= PHEUR_THR).astype(int)
    )


def save_dir(name):
    p = os.path.join(OUTPUT_DIR, name)
    os.makedirs(p, exist_ok=True)
    return p


# ── 1. KARAR AĞACI (görselleştirme + regresyon + sınıflandırma) ─────────────
def run_decision_tree(X, y_chl, y_flav, feat, tree_plot=False):
    print("\n" + "#" * 65)
    print("# KARAR AĞACI (Decision Tree)")
    print("#" * 65)
    out = save_dir("decision_tree")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    for target, y, label in [("Chl", y_chl, "Chl"), ("Flav", y_flav, "Flav")]:
        print(f"\n  [{target}] Grid search...")
        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
        gs = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X, y)
        best = gs.best_estimator_
        print(f"  En iyi: {gs.best_params_}, CV R²={gs.best_score_:.4f}")

        y_pred = cross_val_predict(best, X, y, cv=cv)
        m = reg_metrics(y, y_pred, f"DT {target} (CV)")
        results[target] = m

        if target == "Flav":
            acc = pheur_acc(y, y_pred)
            print(f"  Ph.Eur. PASS/FAIL: Accuracy={acc*100:.1f}%")
            results["pheur_acc"] = acc

        # Görselleştirme
        if tree_plot and target == "Flav":
            # Kısaltılmış ağaç (max_depth=4 görsel için)
            dt_vis = DecisionTreeRegressor(
                max_depth=4, min_samples_leaf=4, random_state=42
            )
            dt_vis.fit(X, y)

            # Top 20 özellik ismi (uzun isimleri kısalt)
            short_feat = [f[:20] for f in feat]

            fig, ax = plt.subplots(figsize=(28, 10))
            plot_tree(
                dt_vis,
                feature_names=short_feat,
                filled=True,
                rounded=True,
                fontsize=7,
                max_depth=4,
                ax=ax,
            )
            ax.set_title(
                "Karar Ağacı — Flavonol (max_depth=4)", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            path = os.path.join(out, "dt_flav_tree.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            print(f"  Ağaç görseli kaydedildi: {path}")

            # Metin formatı (ilk 3 seviye)
            rules = export_text(dt_vis, feature_names=short_feat, max_depth=3)
            rules_path = os.path.join(out, "dt_flav_rules.txt")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(rules)
            print(f"  Karar kuralları: {rules_path}")

    # Sınıflandırma (DT)
    print(f"\n  [DT Sınıflandırma — Ph.Eur. binary]")
    y_bin = (y_flav >= PHEUR_THR).astype(int)
    cv_s = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    param_cls = {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [2, 4, 8],
        "class_weight": [None, "balanced"],
    }
    gs_cls = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_cls,
        cv=cv_s,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs_cls.fit(X, y_bin)
    best_cls = gs_cls.best_estimator_
    print(f"  En iyi: {gs_cls.best_params_}, CV acc={gs_cls.best_score_:.4f}")
    y_pred_cls = cross_val_predict(best_cls, X, y_bin, cv=cv_s)
    acc_cls = accuracy_score(y_bin, y_pred_cls)
    print(f"  [DT Cls (CV)]  Accuracy={acc_cls*100:.1f}%")
    print(
        classification_report(
            y_bin, y_pred_cls, target_names=["FAIL", "PASS"], zero_division=0
        )
    )
    results["cls_acc"] = acc_cls

    if tree_plot:
        dt_cls_vis = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=42
        )
        dt_cls_vis.fit(X, y_bin)
        short_feat = [f[:20] for f in feat]
        fig, ax = plt.subplots(figsize=(26, 10))
        plot_tree(
            dt_cls_vis,
            feature_names=short_feat,
            class_names=["FAIL", "PASS"],
            filled=True,
            rounded=True,
            fontsize=7,
            max_depth=4,
            ax=ax,
        )
        ax.set_title(
            "Karar Ağacı Sınıflandırma — Ph.Eur. PASS/FAIL (max_depth=4)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(out, "dt_cls_tree.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"  Sınıflandırma ağacı: {path}")

    return results


# ── 2. EXTRA TREES ──────────────────────────────────────────────────────────
def run_extra_trees(X, y_chl, y_flav, feat):
    print("\n" + "#" * 65)
    print("# EXTRA TREES Regresyon + Sınıflandırma")
    print("#" * 65)
    out = save_dir("extra_trees")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    for target, y, label in [("Chl", y_chl, "Chl"), ("Flav", y_flav, "Flav")]:
        print(f"\n  [{target}] Grid search...")
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [5, 8, None],
            "min_samples_leaf": [1, 2, 4],
        }
        gs = GridSearchCV(
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X, y)
        best = gs.best_estimator_
        print(f"  En iyi: {gs.best_params_}, CV R²={gs.best_score_:.4f}")

        y_pred = cross_val_predict(best, X, y, cv=cv)
        m = reg_metrics(y, y_pred, f"ET {target} (CV)")
        results[target] = m

        if target == "Flav":
            acc = pheur_acc(y, y_pred)
            print(f"  Ph.Eur. PASS/FAIL: Accuracy={acc*100:.1f}%")
            results["pheur_acc"] = acc

            # Feature importance
            importances = best.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                range(20), importances[top_idx][::-1], color="darkorange", edgecolor="k"
            )
            ax.set_yticks(range(20))
            ax.set_yticklabels([feat[i] for i in top_idx][::-1], fontsize=9)
            ax.set_xlabel("Önem")
            ax.set_title("ExtraTrees — Flav Feature Importance", fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(out, "et_flav_importance.png"), dpi=150)
            plt.show()
            plt.close(fig)

    # Sınıflandırma
    print(f"\n  [ET Sınıflandırma]")
    y_bin = (y_flav >= PHEUR_THR).astype(int)
    cv_s = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    gs_cls = GridSearchCV(
        ExtraTreesClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [200, 300],
            "max_depth": [5, 8, None],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=cv_s,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs_cls.fit(X, y_bin)
    best_cls = gs_cls.best_estimator_
    y_pred_cls = cross_val_predict(best_cls, X, y_bin, cv=cv_s)
    acc_cls = accuracy_score(y_bin, y_pred_cls)
    print(f"  En iyi: {gs_cls.best_params_}, CV acc={gs_cls.best_score_:.4f}")
    print(f"  [ET Cls (CV)]  Accuracy={acc_cls*100:.1f}%")
    print(
        classification_report(
            y_bin, y_pred_cls, target_names=["FAIL", "PASS"], zero_division=0
        )
    )
    results["cls_acc"] = acc_cls
    return results


# ── 3. RF (v3 veriyle yeniden) ───────────────────────────────────────────────
def run_rf_v3(X, y_chl, y_flav, feat):
    print("\n" + "#" * 65)
    print("# RF Regresyon (v3 veri)")
    print("#" * 65)
    out = save_dir("rf_v3")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    for target, y, label in [("Chl", y_chl, "Chl"), ("Flav", y_flav, "Flav")]:
        print(f"\n  [{target}] Grid search...")
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 8, None],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X, y)
        best = gs.best_estimator_
        print(f"  En iyi: {gs.best_params_}, CV R²={gs.best_score_:.4f}")
        y_pred = cross_val_predict(best, X, y, cv=cv)
        m = reg_metrics(y, y_pred, f"RF {target} v3 (CV)")
        results[target] = m
        if target == "Flav":
            acc = pheur_acc(y, y_pred)
            print(f"  Ph.Eur. PASS/FAIL: Accuracy={acc*100:.1f}%")
            results["pheur_acc"] = acc

    # Sınıflandırma
    y_bin = (y_flav >= PHEUR_THR).astype(int)
    cv_s = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    gs_cls = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [200, 300],
            "max_depth": [5, None],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=cv_s,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs_cls.fit(X, y_bin)
    y_pred_cls = cross_val_predict(gs_cls.best_estimator_, X, y_bin, cv=cv_s)
    acc_cls = accuracy_score(y_bin, y_pred_cls)
    print(f"\n  [RF Cls v3]  Accuracy={acc_cls*100:.1f}%")
    print(
        classification_report(
            y_bin, y_pred_cls, target_names=["FAIL", "PASS"], zero_division=0
        )
    )
    results["cls_acc"] = acc_cls
    return results


# ── 4. SVM v3 ───────────────────────────────────────────────────────────────
def run_svm_v3(X, y_chl, y_flav):
    print("\n" + "#" * 65)
    print("# SVM (v3 veri)")
    print("#" * 65)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    for target, y, label in [("Chl", y_chl, "Chl"), ("Flav", y_flav, "Flav")]:
        gs = GridSearchCV(
            pipe,
            {
                "svr__C": [1, 10, 100],
                "svr__gamma": [0.001, 0.01],
                "svr__epsilon": [0.05, 0.1],
                "svr__kernel": ["rbf"],
            },
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X, y)
        print(f"  [{target}] En iyi: {gs.best_params_}, R²={gs.best_score_:.4f}")
        y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv)
        m = reg_metrics(y, y_pred, f"SVR {target} v3 (CV)")
        results[target] = m
        if target == "Flav":
            acc = pheur_acc(y, y_pred)
            print(f"  Ph.Eur. PASS/FAIL: Accuracy={acc*100:.1f}%")
            results["pheur_acc"] = acc

    # SVM Sınıflandırma
    y_bin = (y_flav >= PHEUR_THR).astype(int)
    cv_s = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pipe_cls = Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(class_weight="balanced"))]
    )
    gs_cls = GridSearchCV(
        pipe_cls,
        {"svc__C": [1, 10, 100], "svc__gamma": [0.001, 0.01], "svc__kernel": ["rbf"]},
        cv=cv_s,
        scoring="accuracy",
        n_jobs=-1,
    )
    gs_cls.fit(X, y_bin)
    y_pred_cls = cross_val_predict(gs_cls.best_estimator_, X, y_bin, cv=cv_s)
    acc_cls = accuracy_score(y_bin, y_pred_cls)
    print(f"\n  [SVM Cls v3]  Accuracy={acc_cls*100:.1f}%")
    print(
        classification_report(
            y_bin, y_pred_cls, target_names=["FAIL", "PASS"], zero_division=0
        )
    )
    results["cls_acc"] = acc_cls
    return results


# ── 5. STACKING ──────────────────────────────────────────────────────────────
def run_stacking(X, y_chl, y_flav):
    print("\n" + "#" * 65)
    print("# STACKING (RF + GBR + SVR → Ridge meta)")
    print("#" * 65)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    estimators_reg = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            ),
        ),
        ("svr", SVR(C=10, gamma=0.001, kernel="rbf")),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            ),
        ),
    ]

    for target, y, label in [("Chl", y_chl, "Chl"), ("Flav", y_flav, "Flav")]:
        stack = StackingRegressor(
            estimators=estimators_reg,
            final_estimator=Ridge(alpha=1.0),
            cv=CV_FOLDS,
            passthrough=False,
            n_jobs=-1,
        )
        y_pred = cross_val_predict(stack, Xs, y, cv=cv)
        m = reg_metrics(y, y_pred, f"Stack {target} (CV)")
        results[target] = m
        if target == "Flav":
            acc = pheur_acc(y, y_pred)
            print(f"  Ph.Eur. PASS/FAIL: Accuracy={acc*100:.1f}%")
            results["pheur_acc"] = acc

    # Stacking Sınıflandırma
    print("\n  [Stacking Sınıflandırma]")
    y_bin = (y_flav >= PHEUR_THR).astype(int)
    cv_s = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    estimators_cls = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            ),
        ),
        (
            "gbr",
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
            ),
        ),
        ("svc", SVC(C=10, gamma=0.001, kernel="rbf", probability=True)),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            ),
        ),
    ]
    stack_cls = StackingClassifier(
        estimators=estimators_cls,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=CV_FOLDS,
        passthrough=False,
        n_jobs=-1,
    )
    y_pred_cls = cross_val_predict(stack_cls, Xs, y_bin, cv=cv_s)
    acc_cls = accuracy_score(y_bin, y_pred_cls)
    print(f"  [Stack Cls (CV)]  Accuracy={acc_cls*100:.1f}%")
    print(
        classification_report(
            y_bin, y_pred_cls, target_names=["FAIL", "PASS"], zero_division=0
        )
    )
    results["cls_acc"] = acc_cls
    return results


# ── KARŞILAŞTIRMA TABLOSU ────────────────────────────────────────────────────
def print_comparison(all_results, baseline):
    print("\n" + "=" * 75)
    print("KARŞILAŞTIRMA TABLOSU (v3 — FE + Yeni Modeller)")
    print("=" * 75)
    print(
        f"\n  {'Model':<22}  {'Chl R²':>7}  {'Chl RMSE':>9}  {'Chl RPD':>8} │"
        f"  {'Flav R²':>8}  {'Flav RMSE':>10}  {'Flav RPD':>9}"
    )
    print(f"  {'─'*22}  {'─'*7}  {'─'*9}  {'─'*8} │  {'─'*8}  {'─'*10}  {'─'*9}")

    # Baseline (eski FE)
    print(f"\n  [Eski FE — Baseline]")
    for name, r in baseline.items():
        if "Chl" in r and "Flav" in r:
            print(
                f"  {name:<22}  {r['Chl']['R2']:>7.4f}  {r['Chl']['RMSE']:>9.4f}  "
                f"{r['Chl']['RPD']:>8.2f} │  {r['Flav']['R2']:>8.4f}  "
                f"{r['Flav']['RMSE']:>10.4f}  {r['Flav']['RPD']:>9.2f}"
            )

    print(f"\n  [v3 FE — Yeni Modeller]")
    for name, r in all_results.items():
        if "Chl" in r and "Flav" in r:
            flav_r2 = r["Flav"]["R2"]
            delta = flav_r2 - baseline.get(name, {}).get("Flav", {}).get("R2", flav_r2)
            delta_s = f"(+{delta:.3f})" if delta > 0 else f"({delta:.3f})"
            print(
                f"  {name:<22}  {r['Chl']['R2']:>7.4f}  {r['Chl']['RMSE']:>9.4f}  "
                f"{r['Chl']['RPD']:>8.2f} │  {flav_r2:>8.4f} {delta_s}  "
                f"{r['Flav']['RMSE']:>10.4f}  {r['Flav']['RPD']:>9.2f}"
            )

    print(f"\n  {'Model':<22}  {'Ph.Eur. Acc (regr)':>20}  {'Cls Acc':>10}")
    print(f"  {'─'*22}  {'─'*20}  {'─'*10}")
    for name, r in all_results.items():
        pa = r.get("pheur_acc", float("nan"))
        ca = r.get("cls_acc", float("nan"))
        print(f"  {name:<22}  {pa*100:>20.1f}%  {ca*100:>10.1f}%")
    print("=" * 75)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(use_stress=False, tree_plot=False):
    t0 = time.time()
    print("=" * 65)
    print("FLAVONOL ODAKLI PIPELINE v3")
    print(f"  Veri   : {DATASET_DIR}")
    print(f"  Çıktı  : {OUTPUT_DIR}")
    print(f"  Stres  : {'AÇIK' if use_stress else 'KAPALI'}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y_chl, y_flav, y_stress, feat = load_data(use_stress=use_stress)

    # Baseline (önceki sonuçlar — manuel girildi)
    baseline = {
        "RF Reg.": {
            "Chl": {"R2": 0.8270, "RMSE": 3.8692, "RPD": 2.40},
            "Flav": {"R2": 0.2740, "RMSE": 0.4893, "RPD": 1.17},
        },
        "GBR": {
            "Chl": {"R2": 0.8242, "RMSE": 3.9003, "RPD": 2.38},
            "Flav": {"R2": 0.2835, "RMSE": 0.4862, "RPD": 1.18},
        },
        "SVR": {
            "Chl": {"R2": 0.8096, "RMSE": 4.0584, "RPD": 2.29},
            "Flav": {"R2": 0.3407, "RMSE": 0.4663, "RPD": 1.23},
        },
        "SVM Cls.": {},
    }

    all_results = {}

    # Modelleri çalıştır
    all_results["DT"] = run_decision_tree(X, y_chl, y_flav, feat, tree_plot)
    all_results["ET"] = run_extra_trees(X, y_chl, y_flav, feat)
    all_results["RF v3"] = run_rf_v3(X, y_chl, y_flav, feat)
    all_results["SVR v3"] = run_svm_v3(X, y_chl, y_flav)
    all_results["Stacking"] = run_stacking(X, y_chl, y_flav)

    print_comparison(all_results, baseline)
    print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dk")
    print(f"Görseller  : {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stress", action="store_true", help="y_stress'i özellik olarak ekle"
    )
    parser.add_argument(
        "--tree_plot",
        action="store_true",
        help="Karar ağacı görselini kaydet (yavaşlatır)",
    )
    args = parser.parse_args()
    main(use_stress=args.stress, tree_plot=args.tree_plot)
