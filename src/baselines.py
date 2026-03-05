# src/baselines.py
# ============================================================
# Các mô hình Baseline để so sánh với Freq-MiniRocket
# ============================================================

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.transformations.collection.feature_based import Catch22


class MiniRocketBaseline:
    """MiniRocket gốc — Baseline chính."""

    def __init__(self, num_kernels=10_000, random_state=42, ridge_alphas=None):
        self.ridge_alphas = ridge_alphas or [1e-3, 1e-1, 1.0, 10.0, 100.0]
        self._rocket   = MiniRocket(num_kernels=num_kernels, random_state=random_state)
        self._scaler   = StandardScaler()
        self._clf      = RidgeClassifierCV(
            alphas=self.ridge_alphas, class_weight="balanced"
        )

    def fit(self, X_train, y_train):
        self._rocket.fit(X_train)
        F = self._rocket.transform(X_train)
        F = self._scaler.fit_transform(F)
        self._clf.fit(F, y_train)
        return self

    def predict(self, X_test):
        F = self._rocket.transform(X_test)
        F = self._scaler.transform(F)
        return self._clf.predict(F)


class Catch22Baseline:
    """Catch22 — Baseline giải thích được."""

    def __init__(self, ridge_alphas=None):
        self.ridge_alphas = ridge_alphas or [1e-3, 1e-1, 1.0, 10.0, 100.0]
        self._catch22 = Catch22(replace_nans=True)
        self._scaler  = StandardScaler()
        self._clf     = RidgeClassifierCV(
            alphas=self.ridge_alphas, class_weight="balanced"
        )

    def fit(self, X_train, y_train):
        F = self._catch22.fit_transform(X_train)
        F = self._scaler.fit_transform(F)
        self._clf.fit(F, y_train)
        return self

    def predict(self, X_test):
        F = self._catch22.transform(X_test)
        F = self._scaler.transform(F)
        return self._clf.predict(F)


class FreqOnlyBaseline:
    """
    Chỉ dùng nhánh freq-domain — để phân tích ablation:
    Liệu freq features một mình có ích không?
    """

    def __init__(self, num_kernels=10_000, freq_method="fft",
                 random_state=42, ridge_alphas=None):
        from .freq_transform import FrequencyTransformer
        self.ridge_alphas = ridge_alphas or [1e-3, 1e-1, 1.0, 10.0, 100.0]
        self._freq_tf  = FrequencyTransformer(method=freq_method, normalize=True)
        self._rocket   = MiniRocket(num_kernels=num_kernels, random_state=random_state)
        self._scaler   = StandardScaler()
        self._clf      = RidgeClassifierCV(
            alphas=self.ridge_alphas, class_weight="balanced"
        )

    def fit(self, X_train, y_train):
        X_freq = self._freq_tf.transform(X_train)
        self._rocket.fit(X_freq)
        F = self._rocket.transform(X_freq)
        F = self._scaler.fit_transform(F)
        self._clf.fit(F, y_train)
        return self

    def predict(self, X_test):
        X_freq = self._freq_tf.transform(X_test)
        F = self._rocket.transform(X_freq)
        F = self._scaler.transform(F)
        return self._clf.predict(F)
