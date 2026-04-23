"""
preprocessing.py
================
Yaprak piksellerinin yansıma spektrumlarına uygulanan ön işleme yöntemleri.

Bu yöntemler segmentasyon (maskeleme) SONRASI uygulanmıştır.
Amaç: gürültü azaltma, baz çizgisi düzeltme, saçılma düzeltme.

Pipeline sırası:
  Ham veri → [segmentasyon] → yaprak pikselleri
           → [spektral ön işleme] → temiz spektrum
           → [özellik çıkarımı] → model

Yöntemler:
  1. Savitzky-Golay (SG)     — gürültü yumuşatma (smoothing)
  2. SNV                     — Standard Normal Variate (saçılma düzeltme)
  3. First Derivative (1D)   — baz çizgisi kaldırma + tepe vurgulama
  4. MSC                     — Multiplicative Scatter Correction

Her fonksiyon (N, bands) şeklinde 2D array alır ve aynı şekilde döndürür.
Tek piksel (1D) veya tüm yaprak pikselleri (2D) verilebilir.

Kullanım örneği:
    from preprocessing import savitzky_golay, snv, first_derivative, msc
    from preprocessing import apply_pipeline

    # Yaprak piksellerini al (maskeleme sonrası)
    leaf_pixels = data[mask]   # (N, 204)

    # Tek yöntem
    smoothed = savitzky_golay(leaf_pixels)

    # Zincirleme: SG → SNV
    processed = apply_pipeline(leaf_pixels, ["sg", "snv"])

Literatür:
    - Savitzky & Golay (1964): Polinom tabanlı yumuşatma
    - Barnes et al. (1989): SNV ve MSC
    - Burnett et al. (2021): PLSR için ön işleme önerileri
    - Rinnan et al. (2009): Ön işleme yöntemlerinin karşılaştırması
"""

import numpy as np
from scipy.signal import savgol_filter

# =============================================================================
# YÖNTEM 1: SAVITZKY-GOLAY FİLTRESİ
# =============================================================================


def savitzky_golay(spectra, window_length=11, polyorder=2, deriv=0):
    """
    Savitzky-Golay filtresi uygulanmıştır.

    Polinom tabanlı hareketli pencere ile gürültü yumuşatma yapılmıştır.
    Spektral şekli korurken yüksek frekanslı gürültüyü bastırmaktadır.

    Parametreler:
        spectra (ndarray)       : (N, bands) veya (bands,) spektrum array
        window_length (int)     : pencere genişliği (tek sayı, varsayılan: 11)
        polyorder (int)         : polinom derecesi (varsayılan: 2)
        deriv (int)             : türev derecesi (0=yumuşatma, 1=1.türev)

    Döndürür:
        ndarray : aynı şekilde işlenmiş spektrum

    Not:
        window_length > polyorder olmalıdır.
        Daha büyük pencere → daha fazla yumuşatma ama detay kaybı.
        Daha küçük pencere → gürültü kalır ama detay korunur.

    Literatür:
        Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation
        of data by simplified least squares procedures.
    """

    # 1D girdi kontrolü — tek piksel ise 2D'ye çevir
    single = spectra.ndim == 1
    if single:
        spectra = spectra.reshape(1, -1)

    # scipy.signal.savgol_filter: axis=1 → bant ekseni boyunca filtrele
    result = savgol_filter(
        spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1
    )

    if single:
        result = result.flatten()

    return result


# =============================================================================
# YÖNTEM 2: SNV — Standard Normal Variate
# =============================================================================


def snv(spectra):
    """
    SNV (Standard Normal Variate) uygulanmıştır.

    Her piksel spektrumunu kendi ortalamasını çıkarıp standart
    sapmasına bölerek normalize etmektedir.

    Formül (her satır/piksel için):
        SNV(x) = (x − mean(x)) / std(x)

    Amaç:
      - Parçacık boyutu kaynaklı saçılma etkisini düzeltir
      - Her pikseli kendi içinde normalize eder
      - Özellikle NIR bölgesindeki baz çizgisi kaymasını giderir

    Parametreler:
        spectra (ndarray) : (N, bands) veya (bands,) spektrum array

    Döndürür:
        ndarray : aynı şekilde SNV uygulanmış spektrum

    Literatür:
        Barnes, R.J. et al. (1989). Standard normal variate transformation
        and de-trending of near-infrared diffuse reflectance spectra.
    """

    single = spectra.ndim == 1
    if single:
        spectra = spectra.reshape(1, -1)

    # Her satır (piksel) için ortalama ve standart sapma
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)

    # Standart sapma sıfırsa bölme hatası önlenir
    stds[stds == 0] = 1.0

    result = (spectra - means) / stds

    if single:
        result = result.flatten()

    return result


# =============================================================================
# YÖNTEM 3: FIRST DERIVATIVE (1. TÜREV)
# =============================================================================


def first_derivative(spectra, spacing=1.0):
    """
    Birinci türev hesaplanmıştır.

    Ardışık bantlar arasındaki fark alınarak baz çizgisi sabit kaymaları
    kaldırılmıştır. Spektral tepe ve çukurlar vurgulanmıştır.

    Savitzky-Golay tabanlı türev (deriv=1) daha yumuşak sonuç verir.
    Bu fonksiyon basit np.diff kullanmaktadır.

    Parametreler:
        spectra (ndarray) : (N, bands) veya (bands,) spektrum array
        spacing (float)   : bant aralığı (nm, varsayılan: 1.0)

    Döndürür:
        ndarray : (N, bands-1) veya (bands-1,) türev spektrum
                  Not: çıktı 1 bant daha kısadır!

    Alternatif: SG tabanlı türev için savitzky_golay(spectra, deriv=1)
    kullanılmalıdır — aynı bant sayısını korur ve daha yumuşaktır.

    Literatür:
        Rinnan, Å. et al. (2009). Review of the most common
        pre-processing techniques for near-infrared spectra.
    """

    single = spectra.ndim == 1
    if single:
        spectra = spectra.reshape(1, -1)

    # np.diff: ardışık bantlar arasındaki fark
    result = np.diff(spectra, axis=1) / spacing

    if single:
        result = result.flatten()

    return result


def sg_first_derivative(spectra, window_length=11, polyorder=2):
    """
    Savitzky-Golay tabanlı birinci türev hesaplanmıştır.

    np.diff'ten farklı olarak:
      - Gürültü yumuşatılarak türev alınmıştır
      - Bant sayısı korunmaktadır (çıktı aynı boyutta)
      - Daha kararlı ve güvenilir sonuç vermektedir

    Parametreler:
        spectra (ndarray)   : (N, bands) veya (bands,) spektrum array
        window_length (int) : pencere genişliği (varsayılan: 11)
        polyorder (int)     : polinom derecesi (varsayılan: 2)

    Döndürür:
        ndarray : aynı şekilde türev spektrum (bant sayısı korunur)
    """
    return savitzky_golay(
        spectra, window_length=window_length, polyorder=polyorder, deriv=1
    )


# =============================================================================
# YÖNTEM 4: MSC — Multiplicative Scatter Correction
# =============================================================================


def msc(spectra, reference=None):
    """
    MSC (Multiplicative Scatter Correction) uygulanmıştır.

    Her pikselin spektrumunu referans spektruma göre düzeltir.
    Saçılma kaynaklı çarpımsal ve toplamsal etkileri gidermektedir.

    Yöntem (her piksel için):
      1. Referans spektruma lineer regresyon uygula: x_i = a + b * x_ref
      2. Düzeltme: x_corrected = (x_i − a) / b

    Referans verilmezse tüm piksellerin ortalaması kullanılmıştır.

    Parametreler:
        spectra (ndarray)   : (N, bands) spektrum array
        reference (ndarray) : (bands,) referans spektrum
                              None → ortalama spektrum kullanılmıştır

    Döndürür:
        ndarray : (N, bands) MSC uygulanmış spektrum

    Literatür:
        Geladi, P. et al. (1985). Linearization and scatter-correction
        for near-infrared reflectance spectra of meat.
    """

    single = spectra.ndim == 1
    if single:
        spectra = spectra.reshape(1, -1)

    n_samples, n_bands = spectra.shape

    # Referans spektrum: verilmezse ortalama kullan
    if reference is None:
        reference = np.mean(spectra, axis=0)

    result = np.zeros_like(spectra, dtype=np.float64)

    for i in range(n_samples):
        # Lineer regresyon: x_i = a + b * reference
        # np.polyfit(reference, x_i, 1) → [b, a]
        coeffs = np.polyfit(reference, spectra[i, :], 1)
        b = coeffs[0]  # eğim
        a = coeffs[1]  # kesişim

        # b sıfırsa düzeltme yapılmaz
        if abs(b) < 1e-10:
            result[i, :] = spectra[i, :]
        else:
            result[i, :] = (spectra[i, :] - a) / b

    if single:
        result = result.flatten()

    return result


# =============================================================================
# ZİNCİRLEME ÖN İŞLEME PIPELINE
# =============================================================================


def apply_pipeline(spectra, steps):
    """
    Birden fazla ön işleme yöntemi sırasıyla uygulanmıştır.

    Parametreler:
        spectra (ndarray) : (N, bands) spektrum array
        steps (list)      : yöntem isimleri listesi
                            Geçerli isimler:
                              "sg"    → Savitzky-Golay yumuşatma
                              "snv"   → Standard Normal Variate
                              "d1"    → SG tabanlı 1. türev
                              "msc"   → Multiplicative Scatter Correction

    Döndürür:
        ndarray : işlenmiş spektrum (bant sayısı korunmaktadır)

    Kullanım:
        # SG yumuşatma → SNV normalizasyon
        processed = apply_pipeline(leaf_pixels, ["sg", "snv"])

        # SNV → 1. türev
        processed = apply_pipeline(leaf_pixels, ["snv", "d1"])

    Önerilen kombinasyonlar (Burnett et al. 2021):
        PLSR için: ["sg", "snv"] veya ["msc", "sg"]
        Sınıflandırma için: ["sg"] veya ["snv", "d1"]
    """

    result = spectra.copy().astype(np.float64)

    for step in steps:
        step = step.lower().strip()

        if step == "sg":
            result = savitzky_golay(result)
            print(f"  ön işleme: Savitzky-Golay yumuşatma uygulandı")

        elif step == "snv":
            result = snv(result)
            print(f"  ön işleme: SNV uygulandı")

        elif step == "d1":
            result = sg_first_derivative(result)
            print(f"  ön işleme: SG 1. türev uygulandı")

        elif step == "msc":
            result = msc(result)
            print(f"  ön işleme: MSC uygulandı")

        else:
            raise ValueError(
                f"Bilinmeyen ön işleme adımı: '{step}'. "
                f"Geçerli seçenekler: sg, snv, d1, msc"
            )

    return result


# =============================================================================
# TEST BLOĞU
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ÖN İŞLEME TESTİ")
    print("=" * 60)

    # Sahte spektrum verisi: 10 piksel, 204 bant
    np.random.seed(42)
    fake_spectra = np.random.rand(10, 204).astype(np.float64) * 0.5 + 0.1

    # Saçılma efekti simüle et (farklı piksellere çarpımsal kayma)
    for i in range(10):
        scale = 0.8 + 0.4 * np.random.rand()
        offset = 0.05 * np.random.rand()
        fake_spectra[i, :] = fake_spectra[i, :] * scale + offset

    print(f"\nGirdi şekli: {fake_spectra.shape}")
    print(f"Girdi aralığı: {fake_spectra.min():.4f} – {fake_spectra.max():.4f}")

    # --- TEST 1: Savitzky-Golay ---
    print("\nTEST 1: Savitzky-Golay")
    sg_result = savitzky_golay(fake_spectra)
    print(f"  Çıktı şekli: {sg_result.shape}")
    print(f"  Aralık: {sg_result.min():.4f} – {sg_result.max():.4f}")

    # --- TEST 2: SNV ---
    print("\nTEST 2: SNV")
    snv_result = snv(fake_spectra)
    print(f"  Çıktı şekli: {snv_result.shape}")
    print(f"  Satır ortalaması (≈0 olmalı): {snv_result[0].mean():.6f}")
    print(f"  Satır std (≈1 olmalı): {snv_result[0].std():.6f}")

    # --- TEST 3: First Derivative ---
    print("\nTEST 3: SG 1. türev")
    d1_result = sg_first_derivative(fake_spectra)
    print(f"  Çıktı şekli: {d1_result.shape}  (bant sayısı korunmalı)")

    # --- TEST 4: MSC ---
    print("\nTEST 4: MSC")
    msc_result = msc(fake_spectra)
    print(f"  Çıktı şekli: {msc_result.shape}")

    # --- TEST 5: Pipeline ---
    print("\nTEST 5: Pipeline (SG → SNV)")
    pipe_result = apply_pipeline(fake_spectra, ["sg", "snv"])
    print(f"  Çıktı şekli: {pipe_result.shape}")

    # --- TEST 6: Tek piksel ---
    print("\nTEST 6: Tek piksel")
    single = fake_spectra[0, :]
    sg_single = savitzky_golay(single)
    snv_single = snv(single)
    print(f"  SG  şekli: {sg_single.shape}")
    print(f"  SNV şekli: {snv_single.shape}")

    print("\nTüm testler tamamlandı.")
