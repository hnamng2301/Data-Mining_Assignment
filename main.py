# main.py
# ============================================================
# Điểm vào chính — chạy toàn bộ pipeline Freq-MiniRocket
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASETS
from src.data_loader import load_uea_dataset
from src.freq_minirocket import FreqMiniRocket
from experiments.run_experiments import run_all, run_ablation
from experiments.visualization import (
    plot_accuracy_comparison,
    plot_improvement_heatmap,
    plot_feature_importance,
    plot_time_vs_freq_sample,
    plot_speed_comparison,
)
from experiments.theoretical_analysis import (
    plot_kernel_frequency_responses,
    compute_kernel_stats,
)


def main():
    print("=" * 65)
    print("  Freq-MiniRocket: Dual-Domain Time Series Classification")
    print("=" * 65)

    # ── Bước 0: Phân tích lý thuyết kernel ──────────────────────
    print("\n[Phase 0] Phân tích Lý thuyết Kernel MiniRocket...")
    compute_kernel_stats()
    plot_kernel_frequency_responses(save=True)

    # ── Bước 1: Chạy thực nghiệm toàn bộ ────────────────────────
    print("\n[Phase 1] Chạy Thực nghiệm...")
    df_results = run_all(datasets=DATASETS, save=True)

    # ── Bước 2: Vẽ biểu đồ so sánh ──────────────────────────────
    print("\n[Phase 2] Tạo Biểu đồ So sánh...")
    plot_accuracy_comparison(df_results, save=True)
    plot_improvement_heatmap(df_results, save=True)
    plot_speed_comparison(df_results, save=True)

    # ── Bước 3: Phân tích đặc trưng trên một dataset tiêu biểu ──
    print("\n[Phase 3] Phân tích Đặc trưng Chi tiết...")
    target_dataset = "EthanolConcentration"
    X_train, y_train, X_test, y_test = load_uea_dataset(target_dataset)

    # Trực quan hóa time vs freq domain
    plot_time_vs_freq_sample(X_train, y_train, target_dataset, n_classes=3, save=True)

    # Feature importance analysis
    model = FreqMiniRocket(freq_method="fft", num_kernels=10_000)
    plot_feature_importance(model, X_train, y_train, target_dataset, save=True)

    # ── Bước 4: Ablation study ───────────────────────────────────
    print("\n[Phase 4] Ablation Study...")
    df_ablation = run_ablation(target_dataset)

    print("\n✅ Hoàn tất! Kiểm tra thư mục 'figures/' và 'results/' để xem output.")


if __name__ == "__main__":
    main()
