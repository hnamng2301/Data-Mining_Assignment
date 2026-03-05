# src/data_loader.py
# ============================================================
# Tải và tiền xử lý dữ liệu UEA Multivariate Time Series
# ============================================================

import numpy as np
import os
from aeon.datasets import load_classification


def load_uea_dataset(name: str, cache_dir: str = ".data_cache"):
    """
    Tải dataset UEA, trả về (X_train, y_train, X_test, y_test).

    Returns
    -------
    X_train, X_test : np.ndarray, shape (n_samples, n_channels, n_timepoints)
    y_train, y_test : np.ndarray, shape (n_samples,)
    """
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[DataLoader] Đang tải dataset: {name} ...")
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test   = load_classification(name, split="test")

    # aeon trả về numpy array shape (n, c, t) — chuẩn cho pipeline của ta
    # Xử lý NaN nếu có (một số dataset UEA có độ dài không đều)
    X_train = _handle_nan(X_train)
    X_test  = _handle_nan(X_test)

    n_train, n_channels, n_time = X_train.shape
    n_test  = X_test.shape[0]
    n_classes = len(np.unique(y_train))

    print(f"  Train: {n_train} mẫu | Test: {n_test} mẫu")
    print(f"  Kênh: {n_channels} | Độ dài chuỗi: {n_time} | Lớp: {n_classes}")

    return X_train, y_train, X_test, y_test


def _handle_nan(X: np.ndarray) -> np.ndarray:
    """
    Thay thế NaN bằng giá trị nội suy tuyến tính theo từng kênh.
    Nếu toàn bộ kênh là NaN thì thay bằng 0.
    """
    if not np.isnan(X).any():
        return X

    X = X.copy()
    for i in range(X.shape[0]):           # mỗi mẫu
        for c in range(X.shape[1]):       # mỗi kênh
            series = X[i, c, :]
            nan_mask = np.isnan(series)
            if nan_mask.all():
                X[i, c, :] = 0.0
            elif nan_mask.any():
                idx = np.arange(len(series))
                series[nan_mask] = np.interp(
                    idx[nan_mask], idx[~nan_mask], series[~nan_mask]
                )
                X[i, c, :] = series
    return X


def get_dataset_info(X_train, y_train):
    """Trả về dict thông tin tóm tắt về dataset."""
    return {
        "n_train":    X_train.shape[0],
        "n_channels": X_train.shape[1],
        "n_time":     X_train.shape[2],
        "n_classes":  len(np.unique(y_train)),
    }
