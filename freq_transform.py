# src/freq_transform.py
# ============================================================
# Biến đổi Miền Tần Số — Thành phần Cốt lõi của Freq-MiniRocket
#
# Lý thuyết:
#   Cho chuỗi x(t) độ dài T, biến đổi DFT cho:
#       X[k] = Σ_{t=0}^{T-1} x(t) · e^{-j2πkt/T}
#
#   Magnitude spectrum: |X[k]| phản ánh "năng lượng" tại tần số f_k = k·fs/T
#   Phase spectrum:     ∠X[k] phản ánh vị trí pha tại tần số đó
#
#   Giả thuyết của Freq-MiniRocket:
#       Các kernel nhị phân {-1,+1} của MiniRocket trong time domain
#       tương đương với bộ lọc băng thông trong freq domain,
#       nhưng KHÔNG tối ưu hóa để bắt tần số cụ thể.
#       → Cung cấp biểu diễn freq domain trực tiếp bổ sung thông tin
#         mà time-domain kernels bỏ sót.
# ============================================================

import numpy as np
from scipy.signal import stft as scipy_stft
from typing import Literal


class FrequencyTransformer:
    """
    Biến đổi chuỗi thời gian sang biểu diễn miền tần số.

    Parameters
    ----------
    method : {'fft', 'stft', 'both'}
        - 'fft'  : FFT toàn cục (1D, nhanh)
        - 'stft' : Short-Time Fourier Transform (2D, giữ thông tin thời gian)
        - 'both' : Kết hợp cả hai
    use_log : bool
        Áp dụng log(1 + |X|) — giảm ảnh hưởng của các đỉnh cực lớn
    use_phase : bool
        Thêm thông tin pha vào bộ đặc trưng
    nperseg : int
        Độ dài cửa sổ STFT (chỉ dùng khi method='stft' hoặc 'both')
    noverlap : int
        Số điểm chồng lấp giữa các cửa sổ STFT
    normalize : bool
        Chuẩn hóa output về [0, 1] per channel
    """

    def __init__(
        self,
        method: Literal["fft", "stft", "both"] = "fft",
        use_log: bool = True,
        use_phase: bool = False,
        nperseg: int = 32,
        noverlap: int = 16,
        normalize: bool = True,
    ):
        self.method    = method
        self.use_log   = use_log
        self.use_phase = use_phase
        self.nperseg   = nperseg
        self.noverlap  = noverlap
        self.normalize = normalize

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_channels, n_timepoints)

        Returns
        -------
        X_freq : np.ndarray, shape (n_samples, n_channels_out, n_freq_bins)
            Biểu diễn tần số dưới dạng "chuỗi thời gian" mới,
            sẵn sàng đưa vào MiniRocket.
        """
        if self.method == "fft":
            return self._apply_fft(X)
        elif self.method == "stft":
            return self._apply_stft(X)
        elif self.method == "both":
            X_fft  = self._apply_fft(X)   # shape (n, C, T//2+1)
            X_stft = self._apply_stft(X)  # shape (n, C*F, W)
            # Pad/truncate STFT ve cung do dai time voi FFT
            T_fft  = X_fft.shape[2]
            T_stft = X_stft.shape[2]
            if T_stft < T_fft:
                pad = np.zeros((X_stft.shape[0], X_stft.shape[1], T_fft - T_stft), dtype=X_stft.dtype)
                X_stft = np.concatenate([X_stft, pad], axis=2)
            elif T_stft > T_fft:
                X_stft = X_stft[:, :, :T_fft]
            return np.concatenate([X_fft, X_stft], axis=1)
        else:
            raise ValueError(f"method phải là 'fft', 'stft', hoặc 'both'. Nhận: {self.method}")

    # ------------------------------------------------------------------
    # FFT Transform
    # ------------------------------------------------------------------
    def _apply_fft(self, X: np.ndarray) -> np.ndarray:
        """
        Tính magnitude (và optionally phase) spectrum qua FFT.

        Output shape: (n_samples, n_channels * n_components, T//2+1)
        với n_components = 2 nếu use_phase=True, else 1
        """
        n_samples, n_channels, T = X.shape
        n_freq = T // 2 + 1  # Chỉ lấy phần một chiều (real signal)

        # Số lượng kênh output
        n_out = n_channels * (2 if self.use_phase else 1)
        X_freq = np.zeros((n_samples, n_out, n_freq), dtype=np.float32)

        for i in range(n_samples):
            for c in range(n_channels):
                # Tính FFT — scipy rfft chỉ trả về nửa phổ cho tín hiệu thực
                fft_vals = np.fft.rfft(X[i, c, :])

                # Magnitude spectrum
                magnitude = np.abs(fft_vals)
                if self.use_log:
                    magnitude = np.log1p(magnitude)

                X_freq[i, c, :] = magnitude

                # Phase spectrum (tùy chọn)
                if self.use_phase:
                    phase = np.angle(fft_vals)  # radians trong [-π, π]
                    # Chuẩn hóa phase về [0, 1]
                    phase = (phase + np.pi) / (2 * np.pi)
                    X_freq[i, n_channels + c, :] = phase

        if self.normalize:
            X_freq = self._normalize_per_channel(X_freq)

        return X_freq

    # ------------------------------------------------------------------
    # STFT Transform
    # ------------------------------------------------------------------
    def _apply_stft(self, X: np.ndarray) -> np.ndarray:
        """
        Tính Short-Time Fourier Transform — giữ lại cả thông tin
        thời gian VÀ tần số (spectrogram).

        Mỗi kênh gốc → nhiều kênh tần số trong output
        Output shape: (n_samples, n_channels * n_freq_bins, n_time_frames)
        """
        n_samples, n_channels, T = X.shape

        # Tính trước kích thước output — lấy trực tiếp từ Zxx.shape
        # scipy_stft trả về (f, t, Zxx); Zxx.shape = (n_freq_bins, n_time_frames)
        _, _, Zxx_dummy = scipy_stft(
            X[0, 0, :],
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window="hann",
        )
        n_freq_bins, n_time_frames = Zxx_dummy.shape

        # Mỗi kênh gốc → n_freq_bins kênh mới, mỗi kênh dài n_time_frames
        X_stft = np.zeros(
            (n_samples, n_channels * n_freq_bins, n_time_frames),
            dtype=np.float32
        )

        for i in range(n_samples):
            for c in range(n_channels):
                _, _, Zxx = scipy_stft(
                    X[i, c, :],
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    window="hann",
                )
                # Zxx.shape = (n_freq_bins, n_time_frames)
                magnitude = np.abs(Zxx)
                if self.use_log:
                    magnitude = np.log1p(magnitude)

                start = c * n_freq_bins
                end   = start + n_freq_bins
                # Gán đúng chiều: (n_freq_bins, n_time_frames)
                X_stft[i, start:end, :] = magnitude  # shape phải khớp

        if self.normalize:
            X_stft = self._normalize_per_channel(X_stft)

        return X_stft

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_per_channel(X: np.ndarray) -> np.ndarray:
        """Min-max normalize mỗi kênh về [0, 1], tránh div-by-zero."""
        X_norm = X.copy()
        for c in range(X.shape[1]):
            min_val = X[:, c, :].min()
            max_val = X[:, c, :].max()
            if max_val - min_val > 1e-8:
                X_norm[:, c, :] = (X[:, c, :] - min_val) / (max_val - min_val)
        return X_norm

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Tính toán shape output mà không cần chạy thực sự."""
        n_samples, n_channels, T = input_shape
        n_freq = T // 2 + 1
        if self.method == "fft":
            n_ch_out = n_channels * (2 if self.use_phase else 1)
            return (n_samples, n_ch_out, n_freq)
        elif self.method == "stft":
            _, freqs, _ = scipy_stft(
                np.zeros(T), nperseg=self.nperseg, noverlap=self.noverlap
            )
            return (n_samples, n_channels * len(freqs), None)
        else:
            return None  # 'both' — phức tạp, cần chạy thực
