# experiments/visualization.py
# ============================================================
# Vẽ biểu đồ phân tích kết quả Freq-MiniRocket
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. So sánh Accuracy giữa các mô hình
# ──────────────────────────────────────────────────────────────
def plot_accuracy_comparison(df: pd.DataFrame, save=True):
    """
    Grouped bar chart: Accuracy của mỗi mô hình trên từng dataset.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    models   = df["model"].unique()
    datasets = df["dataset"].unique()
    x        = np.arange(len(datasets))
    width    = 0.8 / len(models)

    colors = sns.color_palette("Set2", len(models))
    for i, (model, color) in enumerate(zip(models, colors)):
        accs = []
        for ds in datasets:
            row = df[(df["model"] == model) & (df["dataset"] == ds)]
            accs.append(row["accuracy"].values[0] if len(row) > 0 else np.nan)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model, color=color, alpha=0.85)

        # Ghi giá trị lên đỉnh cột
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{acc:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("So sánh Accuracy: Freq-MiniRocket vs Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 2. Cải thiện tương đối của Freq-MiniRocket so với MiniRocket
# ──────────────────────────────────────────────────────────────
def plot_improvement_heatmap(df: pd.DataFrame, save=True):
    """
    Heatmap: Δ Accuracy = FreqMiniRocket - MiniRocket trên mỗi dataset.
    Màu xanh = cải thiện, Màu đỏ = giảm.
    """
    freq_models = [m for m in df["model"].unique() if "FreqMiniRocket" in m]
    datasets    = df["dataset"].unique()

    # Guard: Nếu không có FreqMiniRocket nào chạy thành công → bỏ qua
    if len(freq_models) == 0:
        print("  ⚠️  plot_improvement_heatmap: Không có dữ liệu FreqMiniRocket để vẽ heatmap.")
        return

    # Guard: Nếu MiniRocket baseline cũng không có → không thể tính delta
    if "MiniRocket" not in df["model"].values:
        print("  ⚠️  plot_improvement_heatmap: Không có baseline MiniRocket để so sánh.")
        return

    delta_data = {}
    for model in freq_models:
        deltas = []
        for ds in datasets:
            base = df[(df["model"] == "MiniRocket") & (df["dataset"] == ds)]["accuracy"]
            new  = df[(df["model"] == model) & (df["dataset"] == ds)]["accuracy"]
            if len(base) > 0 and len(new) > 0:
                deltas.append(round(new.values[0] - base.values[0], 4))
            else:
                deltas.append(np.nan)
        delta_data[model] = deltas

    delta_df = pd.DataFrame(delta_data, index=datasets)

    # Guard: Nếu toàn bộ delta là NaN → không vẽ được heatmap
    if delta_df.isnull().all().all():
        print("  ⚠️  plot_improvement_heatmap: Tất cả giá trị delta đều là NaN, bỏ qua.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(freq_models) * 2.5), max(5, len(datasets) * 0.8)))
    sns.heatmap(
        delta_df, annot=True, fmt=".3f", center=0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        vmin=delta_df.min().min(), vmax=delta_df.max().max(),  # tránh all-NaN crash
        cbar_kws={"label": "Δ Accuracy (vs MiniRocket)"}
    )
    ax.set_title("Cải thiện Tương đối của Freq-MiniRocket so với MiniRocket Gốc",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Biến thể Freq-MiniRocket", fontsize=11)
    ax.set_ylabel("Dataset", fontsize=11)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "improvement_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 3. Phân tích Tầm quan trọng Đặc trưng (Time vs Freq Weight)
# ──────────────────────────────────────────────────────────────
def plot_feature_importance(model, X_train, y_train, dataset_name: str, save=True):
    """
    Vẽ biểu đồ phân tích trọng số Ridge:
    - Tỉ lệ đóng góp Time vs Freq
    - Top 20 kernel quan trọng nhất mỗi nhánh
    """
    model.fit(X_train, y_train)
    weights = model.get_feature_weights()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Pie chart: Tỉ lệ đóng góp ---
    ax0 = axes[0]
    sizes  = [weights["time_importance"], weights["freq_importance"]]
    labels = [f"Time-domain\n({sizes[0]*100:.1f}%)", f"Freq-domain\n({sizes[1]*100:.1f}%)"]
    colors = ["#4C72B0", "#DD8452"]
    ax0.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11})
    ax0.set_title(f"Đóng góp Time vs Freq\n({dataset_name})", fontweight="bold")

    # --- Bar chart: Top time-domain kernels ---
    ax1 = axes[1]
    top_t  = weights["time_top_idx"][:20]
    top_tw = weights["time_weights"][top_t]
    ax1.barh(range(20), top_tw[::-1], color="#4C72B0", alpha=0.8)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([f"Kernel #{i}" for i in top_t[::-1]], fontsize=8)
    ax1.set_xlabel("Trọng số |W| trung bình")
    ax1.set_title("Top 20 Kernel Time-Domain", fontweight="bold")

    # --- Bar chart: Top freq-domain kernels ---
    ax2 = axes[2]
    top_f  = weights["freq_top_idx"][:20]
    top_fw = weights["freq_weights"][top_f]
    ax2.barh(range(20), top_fw[::-1], color="#DD8452", alpha=0.8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([f"Kernel #{i}" for i in top_f[::-1]], fontsize=8)
    ax2.set_xlabel("Trọng số |W| trung bình")
    ax2.set_title("Top 20 Kernel Freq-Domain", fontweight="bold")

    plt.suptitle(f"Phân tích Tầm quan trọng Đặc trưng — Freq-MiniRocket\nDataset: {dataset_name}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fname = f"feature_importance_{dataset_name}.png"
        path  = os.path.join(FIGURES_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 4. Trực quan hóa FFT của mẫu dữ liệu
# ──────────────────────────────────────────────────────────────
def plot_time_vs_freq_sample(X: np.ndarray, y: np.ndarray,
                              dataset_name: str, n_classes=3, save=True):
    """
    Vẽ chuỗi thời gian gốc và phổ tần số FFT cho mỗi lớp.
    Giúp trực quan hóa: liệu các lớp có phân biệt nhau trong freq domain không?
    """
    classes = np.unique(y)[:n_classes]
    fig, axes = plt.subplots(len(classes), 2,
                              figsize=(12, 3 * len(classes)),
                              sharex="col")

    colors = sns.color_palette("tab10", len(classes))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        idx = np.where(y == cls)[0][0]  # Lấy mẫu đầu tiên của lớp này
        x_sample = X[idx, 0, :]        # Kênh đầu tiên

        T    = len(x_sample)
        freqs = np.fft.rfftfreq(T)
        fft_mag = np.log1p(np.abs(np.fft.rfft(x_sample)))

        # Time domain
        axes[i, 0].plot(x_sample, color=color, linewidth=1.2)
        axes[i, 0].set_ylabel(f"Lớp {cls}", fontsize=10, color=color, fontweight="bold")
        axes[i, 0].set_xlabel("Timestep" if i == len(classes) - 1 else "")

        # Freq domain
        axes[i, 1].plot(freqs, fft_mag, color=color, linewidth=1.2)
        axes[i, 1].fill_between(freqs, fft_mag, alpha=0.2, color=color)
        axes[i, 1].set_xlabel("Tần số (chuẩn hóa)" if i == len(classes) - 1 else "")

    axes[0, 0].set_title("Miền Thời gian (Time Domain)", fontweight="bold")
    axes[0, 1].set_title("Miền Tần số — log|FFT| (Freq Domain)", fontweight="bold")

    plt.suptitle(f"So sánh Biểu diễn Time vs Freq — {dataset_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"time_vs_freq_{dataset_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 5. Train time comparison
# ──────────────────────────────────────────────────────────────
def plot_speed_comparison(df: pd.DataFrame, save=True):
    """So sánh thời gian huấn luyện giữa các mô hình."""
    avg_time = df.groupby("model")["train_time"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#4C72B0" if "Freq" not in m else "#DD8452" for m in avg_time.index]
    bars    = ax.barh(avg_time.index, avg_time.values, color=colors, alpha=0.85)

    for bar, val in zip(bars, avg_time.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=10)

    baseline_patch = mpatches.Patch(color="#4C72B0", label="Baseline")
    ours_patch     = mpatches.Patch(color="#DD8452", label="Freq-MiniRocket (ours)")
    ax.legend(handles=[baseline_patch, ours_patch], fontsize=10)

    ax.set_xlabel("Thời gian Huấn luyện Trung bình (giây)", fontsize=11)
    ax.set_title("So sánh Tốc độ Huấn luyện", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "speed_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {path}")
    plt.show()
