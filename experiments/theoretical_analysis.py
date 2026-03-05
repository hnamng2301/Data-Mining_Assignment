# experiments/theoretical_analysis.py
# ============================================================
# Phân tích Lý thuyết: Đáp ứng Tần số của Kernel MiniRocket
#
# Luận điểm khoa học cốt lõi:
#   Kernel nhị phân {-1, +1} của MiniRocket có thể được phân tích
#   như một bộ lọc số (digital filter) thông qua biến đổi Fourier
#   của hàm xung (impulse response) tương ứng.
#   Bằng cách vẽ frequency response của 84 kernel cố định, ta chứng
#   minh được "vùng mù tần số" — từ đó biện luận tại sao cung cấp
#   biểu diễn freq-domain trực tiếp lại bổ sung thêm thông tin.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Các kernel nhị phân cố định của MiniRocket (84 kernel)
# Trích từ paper: Dempster et al., "MiniRocket", 2021
# Biểu diễn dưới dạng index vào bộ weights {-1, +1}
# ──────────────────────────────────────────────────────────────
MINIROCKET_WEIGHTS = np.array([-1.0, 2.0])  # Hai giá trị duy nhất được dùng

# 84 patterns nhị phân (dạng rút gọn — thực tế kernel có length 9)
# Tái tạo từ mô tả trong paper: mỗi kernel có đúng 2 giá trị +2 và phần còn lại -1
def generate_minirocket_kernels(n_kernels: int = 84, kernel_length: int = 9):
    """
    Tái tạo xấp xỉ các kernel nhị phân của MiniRocket.
    Thực tế MiniRocket dùng tổ hợp C(9,3) = 84 cách chọn 3 vị trí có weight +2.
    """
    from itertools import combinations

    positions = list(range(kernel_length))
    kernels   = []

    # Tất cả tổ hợp chọn 3 vị trí từ 9 để đặt weight +2
    for combo in combinations(positions, 3):
        kernel = np.full(kernel_length, -1.0)
        for pos in combo:
            kernel[pos] = 2.0
        kernels.append(kernel)

    return np.array(kernels[:n_kernels])  # Chỉ lấy 84 kernel đầu


def compute_frequency_response(kernel: np.ndarray, n_fft: int = 512):
    """
    Tính frequency response H(f) của một kernel (impulse response h[n]).

    H(f) = Σ h[n] · e^{-j2πfn}   (DTFT)

    Trả về:
        freqs : np.ndarray — tần số chuẩn hóa [0, 0.5]
        magnitude : np.ndarray — |H(f)|
        phase : np.ndarray — ∠H(f) (radians)
    """
    # Zero-pad kernel lên n_fft điểm để tăng độ phân giải tần số
    h_padded  = np.zeros(n_fft)
    h_padded[:len(kernel)] = kernel

    H = np.fft.rfft(h_padded)
    freqs     = np.fft.rfftfreq(n_fft)
    magnitude = np.abs(H)
    phase     = np.angle(H)

    return freqs, magnitude, phase


def plot_kernel_frequency_responses(n_kernels: int = 84, save: bool = True):
    """
    Vẽ frequency response của tất cả 84 kernel MiniRocket.
    Tô màu mỗi kernel theo năng lượng tần số thấp (phân loại bandpass type).
    """
    kernels = generate_minirocket_kernels(n_kernels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Tất cả frequency responses (overlay) ---
    ax1 = axes[0]
    low_freq_energy = []

    for kernel in kernels:
        freqs, mag, _ = compute_frequency_response(kernel)
        # Phân loại kernel theo năng lượng: low-pass, high-pass, band-pass
        cutoff_idx = len(freqs) // 4  # f < 0.125 = "low freq"
        lfe = mag[:cutoff_idx].sum() / (mag.sum() + 1e-8)
        low_freq_energy.append(lfe)

    # Chuẩn hóa để lấy màu
    lfe_norm = np.array(low_freq_energy)
    lfe_norm = (lfe_norm - lfe_norm.min()) / (lfe_norm.max() - lfe_norm.min() + 1e-8)
    cmap = cm.get_cmap("coolwarm")

    for i, kernel in enumerate(kernels):
        freqs, mag, _ = compute_frequency_response(kernel)
        mag_norm = mag / (mag.max() + 1e-8)  # Chuẩn hóa magnitude
        ax1.plot(freqs, mag_norm, alpha=0.3, linewidth=0.8,
                 color=cmap(lfe_norm[i]))

    # Colorbar giả
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Low-freq energy ratio\n(Xanh = High-pass, Đỏ = Low-pass)", fontsize=9)

    ax1.set_xlabel("Tần số Chuẩn hóa f (0=DC, 0.5=Nyquist)", fontsize=11)
    ax1.set_ylabel("|H(f)| chuẩn hóa", fontsize=11)
    ax1.set_title(f"Đáp ứng Tần số của {n_kernels} Kernel MiniRocket\n"
                  f"(Mỗi đường = 1 kernel)", fontsize=12, fontweight="bold")
    ax1.set_xlim([0, 0.5])

    # --- Plot 2: Tổng hợp coverage (average + std band) ---
    ax2 = axes[1]
    all_responses = []
    for kernel in kernels:
        freqs, mag, _ = compute_frequency_response(kernel)
        mag_norm = mag / (mag.max() + 1e-8)
        all_responses.append(mag_norm)

    all_responses = np.array(all_responses)
    mean_resp = all_responses.mean(axis=0)
    std_resp  = all_responses.std(axis=0)

    ax2.plot(freqs, mean_resp, color="#4C72B0", linewidth=2, label="Trung bình")
    ax2.fill_between(freqs, mean_resp - std_resp, mean_resp + std_resp,
                     alpha=0.25, color="#4C72B0", label="±1 Std")

    # Highlight vùng có coverage thấp (vùng mù tiềm năng)
    low_coverage_mask = mean_resp < (mean_resp.mean() * 0.5)
    if low_coverage_mask.any():
        ax2.fill_between(freqs, 0, 1,
                         where=low_coverage_mask,
                         alpha=0.15, color="red",
                         label="Vùng Coverage Thấp")

    ax2.axhline(y=mean_resp.mean(), color="gray", linestyle="--",
                linewidth=1, alpha=0.5, label="Ngưỡng trung bình")

    ax2.set_xlabel("Tần số Chuẩn hóa f", fontsize=11)
    ax2.set_ylabel("|H(f)| trung bình", fontsize=11)
    ax2.set_title("Tổng hợp Frequency Coverage của Tất cả Kernel\n"
                  "(Vùng đỏ = Tiềm năng bổ sung bởi Freq Branch)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim([0, None])

    plt.suptitle("Phân tích Lý thuyết: Tại sao Freq-MiniRocket Bổ sung Thông tin?\n"
                 "Kernel MiniRocket không phủ đều toàn bộ phổ tần số",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "kernel_freq_response.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()


def compute_kernel_stats(n_kernels: int = 84):
    """
    Tính thống kê về frequency coverage của các kernel MiniRocket.
    In ra bảng phân loại kernel: Low-pass / High-pass / Band-pass.
    """
    kernels = generate_minirocket_kernels(n_kernels)

    stats = {"low_pass": 0, "high_pass": 0, "band_pass": 0}
    for kernel in kernels:
        freqs, mag, _ = compute_frequency_response(kernel)
        n      = len(freqs)
        low_e  = mag[:n//4].sum()
        mid_e  = mag[n//4:3*n//4].sum()
        high_e = mag[3*n//4:].sum()
        total  = mag.sum() + 1e-8

        if low_e / total > 0.5:
            stats["low_pass"] += 1
        elif high_e / total > 0.5:
            stats["high_pass"] += 1
        else:
            stats["band_pass"] += 1

    print("\n=== Phân loại Kernel MiniRocket theo Tần số ===")
    print(f"  Low-pass  (năng lượng tập trung tần số thấp): {stats['low_pass']:>3} kernels "
          f"({stats['low_pass']/n_kernels*100:.1f}%)")
    print(f"  Band-pass (năng lượng tập trung tần số giữa): {stats['band_pass']:>3} kernels "
          f"({stats['band_pass']/n_kernels*100:.1f}%)")
    print(f"  High-pass (năng lượng tập trung tần số cao):  {stats['high_pass']:>3} kernels "
          f"({stats['high_pass']/n_kernels*100:.1f}%)")
    print(f"\n  → Biểu diễn FFT trực tiếp bổ sung thông tin KHÔNG bị ràng buộc")
    print(f"    bởi cấu trúc kernel nhị phân, đặc biệt ở dải tần số trung-cao.")
    return stats


if __name__ == "__main__":
    compute_kernel_stats()
    plot_kernel_frequency_responses()
