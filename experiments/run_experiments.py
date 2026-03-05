# experiments/run_experiments.py
# ============================================================
# Chạy toàn bộ thực nghiệm và lưu kết quả
# ============================================================

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATASETS, MINIROCKET_NUM_KERNELS, RANDOM_STATE, RIDGE_ALPHAS, RESULTS_DIR
from src.data_loader import load_uea_dataset
from src.baselines import MiniRocketBaseline, Catch22Baseline, FreqOnlyBaseline
from src.freq_minirocket import FreqMiniRocket


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=""):
    """Huấn luyện và đánh giá một mô hình, trả về dict kết quả."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"    [{model_name}] Acc={acc:.4f} | F1={f1:.4f} | "
          f"Train={train_time:.1f}s | Infer={infer_time:.3f}s")

    return {
        "model":      model_name,
        "accuracy":   round(acc, 4),
        "f1_weighted": round(f1, 4),
        "train_time": round(train_time, 2),
        "infer_time": round(infer_time, 4),
    }


def run_all(datasets=None, save=True):
    """
    Chạy tất cả mô hình trên tất cả dataset.
    Trả về DataFrame kết quả.
    """
    datasets = datasets or DATASETS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    # Định nghĩa các mô hình cần so sánh
    model_configs = {
        "MiniRocket":        lambda: MiniRocketBaseline(
                                 num_kernels=MINIROCKET_NUM_KERNELS,
                                 random_state=RANDOM_STATE,
                                 ridge_alphas=RIDGE_ALPHAS),
        "Catch22":           lambda: Catch22Baseline(ridge_alphas=RIDGE_ALPHAS),
        "FreqOnly-FFT":      lambda: FreqOnlyBaseline(
                                 num_kernels=MINIROCKET_NUM_KERNELS,
                                 freq_method="fft",
                                 random_state=RANDOM_STATE,
                                 ridge_alphas=RIDGE_ALPHAS),
        "FreqMiniRocket-FFT":  lambda: FreqMiniRocket(
                                 num_kernels=MINIROCKET_NUM_KERNELS,
                                 freq_method="fft",
                                 use_log_magnitude=True,
                                 use_phase=False,
                                 random_state=RANDOM_STATE,
                                 ridge_alphas=RIDGE_ALPHAS),
        "FreqMiniRocket-STFT": lambda: FreqMiniRocket(
                                 num_kernels=MINIROCKET_NUM_KERNELS,
                                 freq_method="stft",
                                 use_log_magnitude=True,
                                 random_state=RANDOM_STATE,
                                 ridge_alphas=RIDGE_ALPHAS),
        "FreqMiniRocket+Phase": lambda: FreqMiniRocket(
                                 num_kernels=MINIROCKET_NUM_KERNELS,
                                 freq_method="fft",
                                 use_log_magnitude=True,
                                 use_phase=True,
                                 random_state=RANDOM_STATE,
                                 ridge_alphas=RIDGE_ALPHAS),
    }

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            X_train, y_train, X_test, y_test = load_uea_dataset(dataset_name)
        except Exception as e:
            print(f"  ⚠️  Không tải được dataset {dataset_name}: {e}")
            continue

        for model_name, model_factory in model_configs.items():
            try:
                result = evaluate_model(
                    model_factory(),
                    X_train, y_train, X_test, y_test,
                    model_name=model_name
                )
                result["dataset"] = dataset_name
                result["n_train"]    = X_train.shape[0]
                result["n_channels"] = X_train.shape[1]
                result["n_time"]     = X_train.shape[2]
                all_results.append(result)
            except Exception as e:
                print(f"  ⚠️  Lỗi khi chạy {model_name}: {e}")

    df = pd.DataFrame(all_results)

    if save and len(df) > 0:
        path = os.path.join(RESULTS_DIR, "results_full.csv")
        df.to_csv(path, index=False)
        print(f"\n✓ Kết quả đã lưu vào: {path}")

    return df


def run_ablation(dataset_name: str):
    """
    Ablation study trên một dataset:
    Phân tích đóng góp của từng thành phần.
    """
    print(f"\n[Ablation Study] Dataset: {dataset_name}")
    X_train, y_train, X_test, y_test = load_uea_dataset(dataset_name)

    configs = {
        "Time only (MiniRocket)":    MiniRocketBaseline(),
        "Freq only (FFT-Rocket)":    FreqOnlyBaseline(freq_method="fft"),
        "Time + Freq (FFT) [OURS]":  FreqMiniRocket(freq_method="fft"),
        "Time + Freq (STFT) [OURS]": FreqMiniRocket(freq_method="stft"),
        "Time + Freq + Phase [OURS]": FreqMiniRocket(freq_method="fft", use_phase=True),
    }

    results = []
    for name, model in configs.items():
        r = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        r["dataset"] = dataset_name
        results.append(r)

    df = pd.DataFrame(results)
    print("\n" + df[["model", "accuracy", "f1_weighted", "train_time"]].to_string(index=False))
    return df


if __name__ == "__main__":
    # Chạy nhanh trên 2 dataset để kiểm tra pipeline
    df = run_all(datasets=["BasicMotions", "EthanolConcentration"])
    print("\n=== Kết quả Tổng hợp ===")
    print(df.groupby("model")[["accuracy", "f1_weighted"]].mean().round(4))
