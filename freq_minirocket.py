# src/freq_minirocket.py
# ============================================================
# Freq-MiniRocket: Dual-Domain MiniRocket
#
# Kiến trúc tổng thể:
#
#   X (n, C, T)
#       │
#       ├─────────────────────────────────┐
#       │  Nhánh Time-Domain              │  Nhánh Freq-Domain
#       │                                 │
#       ▼                                 ▼
#   MiniRocket(X)              FrequencyTransformer(X) → X_freq
#       │                                 │
#       ▼                                 ▼
#   F_time (n, 10_000)          MiniRocket(X_freq)
#                                         │
#                                         ▼
#                                 F_freq (n, 10_000)
#       │                                 │
#       └──────────── concat ─────────────┘
#                          │
#                          ▼
#                   F_combined (n, 20_000)
#                          │
#                          ▼
#                  Ridge Classifier / RF
#                          │
#                          ▼
#                     Nhãn phân loại
# ============================================================

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from aeon.transformations.collection.convolution_based import MiniRocket

from .freq_transform import FrequencyTransformer


class FreqMiniRocket:
    """
    Freq-MiniRocket: Phân loại chuỗi thời gian dual-domain.

    Parameters
    ----------
    num_kernels : int
        Số kernel MiniRocket cho mỗi nhánh (mặc định 10,000)
    freq_method : str
        Phương pháp biến đổi tần số: 'fft', 'stft', hoặc 'both'
    use_log_magnitude : bool
        Dùng log(1+|FFT|) thay vì |FFT|
    use_phase : bool
        Thêm thông tin pha vào nhánh tần số
    ridge_alphas : list
        Danh sách alpha để RidgeCV tự chọn
    random_state : int
    """

    def __init__(
        self,
        num_kernels: int = 10_000,
        freq_method: str = "fft",
        use_log_magnitude: bool = True,
        use_phase: bool = False,
        nperseg: int = 32,
        noverlap: int = 16,
        ridge_alphas: list = None,
        random_state: int = 42,
    ):
        self.num_kernels      = num_kernels
        self.freq_method      = freq_method
        self.use_log_magnitude = use_log_magnitude
        self.use_phase        = use_phase
        self.nperseg          = nperseg
        self.noverlap         = noverlap
        self.ridge_alphas     = ridge_alphas or [1e-3, 1e-1, 1.0, 10.0, 100.0]
        self.random_state     = random_state

        # Khởi tạo các thành phần
        self._rocket_time = None
        self._rocket_freq = None
        self._freq_transformer = FrequencyTransformer(
            method=freq_method,
            use_log=use_log_magnitude,
            use_phase=use_phase,
            nperseg=nperseg,
            noverlap=noverlap,
            normalize=True,
        )
        self._classifier = RidgeClassifierCV(
            alphas=self.ridge_alphas,
            class_weight="balanced",
        )
        self._scaler = StandardScaler()
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Huấn luyện toàn bộ pipeline dual-domain.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, C, T)
        y_train : np.ndarray, shape (n,)
        """
        print("[FreqMiniRocket] Bắt đầu huấn luyện...")

        # ── Nhánh 1: Time-domain ──────────────────────────────────────
        print("  [1/4] MiniRocket time-domain...")
        self._rocket_time = MiniRocket(
            num_kernels=self.num_kernels,
            random_state=self.random_state,
        )
        self._rocket_time.fit(X_train)
        F_time = self._rocket_time.transform(X_train)  # (n, 10_000)

        # ── Nhánh 2: Freq-domain ──────────────────────────────────────
        print(f"  [2/4] Biến đổi tần số [{self.freq_method}]...")
        X_freq = self._freq_transformer.transform(X_train)  # (n, C', T')

        print("  [3/4] MiniRocket freq-domain...")
        self._rocket_freq = MiniRocket(
            num_kernels=self.num_kernels,
            random_state=self.random_state,
        )
        self._rocket_freq.fit(X_freq)
        F_freq = self._rocket_freq.transform(X_freq)  # (n, 10_000)

        # ── Fusion & Classification ───────────────────────────────────
        print("  [4/4] Ghép đặc trưng và huấn luyện Ridge...")
        F_combined = np.concatenate([F_time, F_freq], axis=1)  # (n, 20_000)
        F_scaled   = self._scaler.fit_transform(F_combined)
        self._classifier.fit(F_scaled, y_train)

        # Lưu metadata
        self.feature_dim_time = F_time.shape[1]
        self.feature_dim_freq = F_freq.shape[1]
        self.best_alpha_      = self._classifier.alpha_
        self.is_fitted        = True

        print(f"  ✓ Xong! Alpha Ridge tốt nhất: {self.best_alpha_:.4f}")
        print(f"  ✓ Tổng số đặc trưng: {F_combined.shape[1]:,}")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        F_combined = self._extract_features(X_test)
        F_scaled   = self._scaler.transform(F_combined)
        return self._classifier.predict(F_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Trả về điểm quyết định (decision scores).
        Ridge không có predict_proba chuẩn, dùng decision_function.
        """
        self._check_fitted()
        F_combined = self._extract_features(X_test)
        F_scaled   = self._scaler.transform(F_combined)
        return self._classifier.decision_function(F_scaled)

    # ------------------------------------------------------------------
    # Feature Analysis
    # ------------------------------------------------------------------
    def get_feature_weights(self) -> dict:
        """
        Trả về trọng số Ridge để phân tích đặc trưng nào quan trọng.
        Cho phép ta so sánh: đặc trưng time-domain hay freq-domain có
        trọng số lớn hơn?

        Returns
        -------
        dict với keys: 'time_weights', 'freq_weights', 'time_importance', 'freq_importance'
        """
        self._check_fitted()
        W = self._classifier.coef_  # shape (n_classes, n_features) hoặc (n_features,)
        if W.ndim > 1:
            W = np.abs(W).mean(axis=0)  # Lấy trung bình qua các lớp
        else:
            W = np.abs(W)

        time_w = W[:self.feature_dim_time]
        freq_w = W[self.feature_dim_time:]

        return {
            "time_weights":     time_w,
            "freq_weights":     freq_w,
            "time_importance":  time_w.sum() / W.sum(),  # Tỉ lệ đóng góp
            "freq_importance":  freq_w.sum() / W.sum(),
            "time_top_idx":     np.argsort(time_w)[::-1][:20],
            "freq_top_idx":     np.argsort(freq_w)[::-1][:20],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Trích xuất đặc trưng kép từ input X."""
        F_time  = self._rocket_time.transform(X)
        X_freq  = self._freq_transformer.transform(X)
        F_freq  = self._rocket_freq.transform(X_freq)
        return np.concatenate([F_time, F_freq], axis=1)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi .fit() trước.")

    def __repr__(self):
        return (
            f"FreqMiniRocket("
            f"num_kernels={self.num_kernels}, "
            f"freq_method='{self.freq_method}', "
            f"use_log={self.use_log_magnitude}, "
            f"use_phase={self.use_phase})"
        )
