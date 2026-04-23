"""
pipeline_flav_v5_pro.py
Katmanlı Mimari (Layered Architecture):
Layer 1: Stress Probability Estimation (Olasılık Bazlı Stres Tahmini)
Layer 2: Augmented Feature Space (Orijinal Veri + Stres Olasılığı)
Layer 3: Optimized Ensemble Regression (Flavonol Tahmini)
Görselleştirme: Antileakage klasörüne kayıt.
"""

import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix

# --- DİZİN AYARLARI ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_THIS_DIR, "dataset_output_fe_v3")
PLOT_DIR = os.path.join(_THIS_DIR, "antileakage")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    y_chl = np.load(os.path.join(DATASET_DIR, "y_chl.npy"))
    y_flav = np.load(os.path.join(DATASET_DIR, "y_flav.npy"))
    y_stress = np.load(os.path.join(DATASET_DIR, "y_stress.npy"))
    feat = np.load(os.path.join(DATASET_DIR, "feature_names.npy"), allow_pickle=True).tolist()
    
    valid = ~np.isnan(y_chl) & ~np.isnan(y_flav)
    # y_stress zaten X içindeyse sızıntıyı önlemek için çıkar
    if "y_stress" in feat:
        idx = feat.index("y_stress")
        X = np.delete(X, idx, axis=1)
        feat.pop(idx)
        
    return X[valid], y_chl[valid], y_flav[valid], y_stress[valid], feat

def layer1_stress_probability(X, y_stress):
    """Katman 1: Stres Olasılığını Hesapla"""
    print("\n[Layer 1] Stres Olasılık Modeli (Ensemble) Çalışıyor...")
    
    # Daha güçlü bir sınıflandırıcı yapısı
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 'predict_proba' kullanarak olasılıkları al (sızıntısız cross-validation ile)
    probs = cross_val_predict(clf, X, y_stress, cv=cv, method='predict_proba')
    
    # Binary tahminleri de al (raporlama için)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_stress, preds)
    print(f"  -> Katman 1 Doğruluğu: {acc*100:.1f}%")
    
    # Grafik 1: Karmaşıklık Matrisi
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_stress, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Stres Sınıflandırma (Acc: {acc*100:.1f}%)")
    plt.savefig(os.path.join(PLOT_DIR, "v5_stress_confusion.png"))
    plt.close()
    
    # Sadece 'stres var' olasılığını (sütun 1) döndür
    return probs[:, 1].reshape(-1, 1)

def layer2_flavonol_regression(X_orig, stress_probs, y_flav):
    """Katman 2: Olasılık Destekli Regresyon"""
    print("\n[Layer 2] Flavonol Regresyonu Başlıyor...")
    
    # Orijinal özelliklere stres olasılığını bir 'ipucu' olarak ekle
    X_combined = np.hstack([X_orig, stress_probs])
    
    # Güçlü bir Regresyon Modeli (Extra Trees + GBR hibriti gibi düşünebiliriz)
    reg = ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=42)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(reg, X_combined, y_flav, cv=cv)
    
    # Metrikler
    r2 = r2_score(y_flav, y_pred)
    rmse = np.sqrt(mean_squared_error(y_flav, y_pred))
    rpd = np.std(y_flav) / rmse
    
    # Grafik 2: Prediction vs Actual
    plt.figure(figsize=(7,7))
    plt.scatter(y_flav, y_pred, alpha=0.6, color='darkgreen')
    plt.plot([y_flav.min(), y_flav.max()], [y_flav.min(), y_flav.max()], 'r--', lw=2)
    plt.xlabel("Gerçek Flavonol (mg/g)")
    plt.ylabel("Tahmin Edilen Flavonol (mg/g)")
    plt.title(f"Flav Tahmin Başarısı (R²: {r2:.3f})")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "v5_flav_regression_perf.png"))
    plt.close()
    
    return r2, rmse, rpd

def main():
    t0 = time.time()
    X, y_chl, y_flav, y_stress, feat = load_data()
    
    # ADIM 1: Olasılık türet
    stress_probs = layer1_stress_probability(X, y_stress)
    
    # ADIM 2: Regresyonu çalıştır
    r2, rmse, rpd = layer2_flavonol_regression(X, stress_probs, y_flav)
    
    print("\n" + "="*50)
    print("V5 PRO - LAYERED ARCHITECTURE RESULTS")
    print(f"Flav R²:   {r2:.4f}")
    print(f"Flav RMSE: {rmse:.4f}")
    print(f"Flav RPD:  {rpd:.2f}")
    print("="*50)
    print(f"Grafikler kaydedildi: {PLOT_DIR}")
    print(f"Toplam süre: {(time.time()-t0)/60:.2f} dk")

if __name__ == "__main__":
    main()