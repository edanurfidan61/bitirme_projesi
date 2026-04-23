"""
eda.py — Keşifsel Veri Analizi
Bu dosya modul_6/ kökünde bulunur.
Çalıştırma: python eda.py
"""

import numpy as np
import pandas as pd
import os, sys

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# --- PATH AYARI ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODUL6_DIR = _THIS_DIR  # eda.py modul_6 kökünde
sys.path.insert(0, _MODUL6_DIR)

DATASET_DIR = os.path.join(_MODUL6_DIR, "dataset_output")
OUTPUT_DIR = os.path.join(_MODUL6_DIR, "model_outputs", "eda")
INDEX_NAMES = ["NDVI", "GNDVI", "ARI", "RVSI", "ZTM"]


def load_data():
    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    y_chl = np.load(os.path.join(DATASET_DIR, "y_chl.npy"))
    y_flav = np.load(os.path.join(DATASET_DIR, "y_flav.npy"))
    y_stress = np.load(os.path.join(DATASET_DIR, "y_stress.npy"))
    feat_names = np.load(
        os.path.join(DATASET_DIR, "feature_names.npy"), allow_pickle=True
    ).tolist()
    csv_path = os.path.join(DATASET_DIR, "dataset_full.csv")
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        varieties = (
            df_csv["variety"].tolist()
            if "variety" in df_csv.columns
            else ["?"] * len(y_chl)
        )
        symptoms = (
            df_csv["symptom"].tolist()
            if "symptom" in df_csv.columns
            else ["?"] * len(y_chl)
        )
    else:
        varieties = ["?"] * len(y_chl)
        symptoms = ["?"] * len(y_chl)
    print(f"Veri yüklendi: X={X.shape}")
    return X, y_chl, y_flav, y_stress, feat_names, varieties, symptoms


def plot_target_distributions(y_chl, y_flav, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    valid = y_chl[~np.isnan(y_chl)]
    axes[0].hist(valid, bins=25, color="forestgreen", edgecolor="k", alpha=0.7)
    axes[0].axvline(
        valid.mean(), color="red", linestyle="--", label=f"Ort={valid.mean():.1f}"
    )
    axes[0].set_xlabel("Klorofil (µg/cm²)")
    axes[0].set_ylabel("Yaprak Sayısı")
    axes[0].set_title("Klorofil Dağılımı", fontweight="bold")
    axes[0].legend()
    axes[0].text(
        0.95,
        0.95,
        f"n={len(valid)}\nort={valid.mean():.1f}\nstd={valid.std():.1f}",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow"),
    )

    valid = y_flav[~np.isnan(y_flav)]
    axes[1].hist(valid, bins=25, color="darkorange", edgecolor="k", alpha=0.7)
    axes[1].axvline(
        valid.mean(), color="red", linestyle="--", label=f"Ort={valid.mean():.2f}"
    )
    axes[1].axvline(3.5, color="blue", linestyle=":", linewidth=2, label="Ph.Eur.=3.5")
    axes[1].set_xlabel("Flavonol")
    axes[1].set_ylabel("Yaprak Sayısı")
    axes[1].set_title("Flavonol Dağılımı", fontweight="bold")
    axes[1].legend()
    axes[1].text(
        0.05,
        0.95,
        f"n={len(valid)}\nort={valid.mean():.2f}\nstd={valid.std():.2f}",
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"),
    )
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "target_distributions.png"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()
    plt.close(fig)


def plot_boxplots(y_chl, y_flav, varieties, symptoms, save_dir=None):
    df = pd.DataFrame(
        {"Chl": y_chl, "Flav": y_flav, "variety": varieties, "symptom": symptoms}
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, col, by, title in [
        (axes[0, 0], "Chl", "variety", "Klorofil × Çeşit"),
        (axes[0, 1], "Flav", "variety", "Flavonol × Çeşit"),
        (axes[1, 0], "Chl", "symptom", "Klorofil × Semptom"),
        (axes[1, 1], "Flav", "symptom", "Flavonol × Semptom"),
    ]:
        df.boxplot(column=col, by=by, ax=ax, grid=False, patch_artist=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        plt.sca(ax)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        if "Flav" in col:
            ax.axhline(3.5, color="blue", linestyle=":")
    plt.suptitle("Çeşit ve Semptom Bazlı Dağılımlar", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "boxplots.png"), dpi=150, bbox_inches="tight"
        )
    plt.show()
    plt.close(fig)


def plot_band_correlation(X, y_chl, y_flav, feat_names, save_dir=None):
    n_bands = len(feat_names) - len(INDEX_NAMES)
    X_bands = X[:, :n_bands]
    wavelengths = np.array(
        [float(n.replace("band_", "")) for n in feat_names[:n_bands]]
    )
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    corr_chl = np.array(
        [np.corrcoef(X_bands[valid, i], y_chl[valid])[0, 1] for i in range(n_bands)]
    )
    corr_flav = np.array(
        [np.corrcoef(X_bands[valid, i], y_flav[valid])[0, 1] for i in range(n_bands)]
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, corr, color, fill, title in [
        (axes[0], corr_chl, "forestgreen", "green", "Bant-Klorofil Korelasyon"),
        (axes[1], corr_flav, "darkorange", "orange", "Bant-Flavonol Korelasyon"),
    ]:
        ax.plot(wavelengths, corr, color=color, linewidth=1.5)
        ax.fill_between(wavelengths, corr, 0, alpha=0.2, color=fill)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.5)
        ax.axhline(-0.5, color="red", linestyle=":", alpha=0.5)
        ax.set_ylabel("Pearson r")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        top5 = np.argsort(np.abs(corr))[-5:]
        for idx in top5:
            ax.annotate(
                f"{wavelengths[idx]:.0f}nm\nr={corr[idx]:.2f}",
                xy=(wavelengths[idx], corr[idx]),
                fontsize=7,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
            )
    axes[1].set_xlabel("Dalga Boyu (nm)")
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "band_correlation.png"), dpi=150, bbox_inches="tight"
        )
    plt.show()
    plt.close(fig)
    return corr_chl, corr_flav, wavelengths


def plot_index_correlation(X, y_chl, y_flav, feat_names, save_dir=None):
    n_bands = len(feat_names) - len(INDEX_NAMES)
    X_idx = X[:, n_bands:]
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    all_names = INDEX_NAMES + ["Chl", "Flav"]
    all_data = np.column_stack(
        [X_idx[valid], y_chl[valid].reshape(-1, 1), y_flav[valid].reshape(-1, 1)]
    )
    corr = np.corrcoef(all_data.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    for i in range(len(all_names)):
        for j in range(len(all_names)):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{corr[i,j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax.set_xticks(range(len(all_names)))
    ax.set_yticks(range(len(all_names)))
    ax.set_xticklabels(all_names, fontsize=10, rotation=30, ha="right")
    ax.set_yticklabels(all_names, fontsize=10)
    ax.set_title("İndeks + Hedef Korelasyon Matrisi", fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "index_correlation.png"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()
    plt.close(fig)


def plot_chl_vs_flav(y_chl, y_flav, symptoms, save_dir=None):
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_symp = sorted(set(s for s, v in zip(symptoms, valid) if v))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_symp)))
    for sym, c in zip(unique_symp, colors):
        mask = np.array([s == sym and v for s, v in zip(symptoms, valid)])
        ax.scatter(
            y_chl[mask],
            y_flav[mask],
            c=[c],
            label=sym,
            alpha=0.7,
            edgecolors="k",
            linewidth=0.3,
            s=40,
        )
    r = np.corrcoef(y_chl[valid], y_flav[valid])[0, 1]
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axhline(3.5, color="blue", linestyle=":", alpha=0.5, label="Ph.Eur.")
    ax.set_xlabel("Klorofil (µg/cm²)")
    ax.set_ylabel("Flavonol")
    ax.set_title("Klorofil vs Flavonol", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, "chl_vs_flav.png"), dpi=150, bbox_inches="tight"
        )
    plt.show()
    plt.close(fig)


def main():
    print("=" * 60)
    print("KEŞİFSEL VERİ ANALİZİ (EDA)")
    print("=" * 60)
    X, y_chl, y_flav, y_stress, feat, var, sym = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_target_distributions(y_chl, y_flav, OUTPUT_DIR)
    plot_boxplots(y_chl, y_flav, var, sym, OUTPUT_DIR)
    plot_band_correlation(X, y_chl, y_flav, feat, OUTPUT_DIR)
    plot_index_correlation(X, y_chl, y_flav, feat, OUTPUT_DIR)
    plot_chl_vs_flav(y_chl, y_flav, sym, OUTPUT_DIR)
    print(f"\nEDA tamamlandı. Görseller: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
