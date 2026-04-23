"""
utils_model.py — Ortak yardımcı fonksiyonlar
Bu dosya modul_6/ kökünde bulunur. Tüm model scriptleri bunu import eder.
"""

import numpy as np
import os


def load_dataset(dataset_dir):
    """Kaydedilmiş .npy dosyalarını yükler."""
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    y_chl = np.load(os.path.join(dataset_dir, "y_chl.npy"))
    y_flav = np.load(os.path.join(dataset_dir, "y_flav.npy"))
    y_stress = np.load(os.path.join(dataset_dir, "y_stress.npy"))
    feature_names = np.load(
        os.path.join(dataset_dir, "feature_names.npy"), allow_pickle=True
    ).tolist()
    print(
        f"  X: {X.shape}, Chl: {np.nanmin(y_chl):.1f}–{np.nanmax(y_chl):.1f}, "
        f"Flav: {np.nanmin(y_flav):.2f}–{np.nanmax(y_flav):.2f}"
    )
    return X, y_chl, y_flav, y_stress, feature_names


def regression_metrics(y_true, y_pred, label=""):
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse if rmse > 0 else 0.0
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if np.sum(nonzero) > 0
        else 0.0
    )
    if label:
        print(
            f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  RPD={rpd:.2f}  MAPE={mape:.1f}%"
        )
    return {"R2": r2, "RMSE": rmse, "RPD": rpd, "MAPE": mape}


def classify_metrics(y_true, y_pred, class_names=None, label=""):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    if label:
        print(f"  [{label}]  Accuracy={acc:.4f} ({acc*100:.1f}%)")
        print(report)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def plot_regression(
    y_true, y_pred, title="", xlabel="Gerçek", ylabel="Tahmin", save_path=None
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5, s=40)
    lims = [
        min(y_true.min(), y_pred.min()) * 0.9,
        max(y_true.max(), y_pred.max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse if rmse > 0 else 0
    ax.text(
        0.05,
        0.92,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nRPD = {rpd:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion(cm, class_names=None, title="Confusion Matrix", save_path=None):
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = [f"Sınıf {i}" for i in range(cm.shape[0])]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12
            )
    ax.set_xlabel("Tahmin", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9, rotation=20, ha="right")
    ax.set_yticklabels(class_names, fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_feature_importance(
    importances, feature_names, top_n=20, title="Feature Importance", save_path=None
):
    import matplotlib.pyplot as plt

    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.barh(range(top_n), top_values[::-1], color="steelblue", edgecolor="k")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Önem", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def pheur_pass_fail(y_true, y_pred, threshold=3.5):
    from sklearn.metrics import accuracy_score, confusion_matrix

    true_pass = (y_true >= threshold).astype(int)
    pred_pass = (y_pred >= threshold).astype(int)
    acc = accuracy_score(true_pass, pred_pass)
    cm = confusion_matrix(true_pass, pred_pass, labels=[0, 1])
    print(f"  Ph.Eur. PASS/FAIL (eşik={threshold}): Accuracy={acc*100:.1f}%")
    return {"accuracy": acc, "confusion_matrix": cm}
