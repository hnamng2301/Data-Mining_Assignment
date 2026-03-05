# config.py
# ============================================================
# Cấu hình trung tâm cho toàn bộ dự án Freq-MiniRocket
# ============================================================

# --- Tập dữ liệu UEA Multivariate ---
# Chọn các dataset có đặc điểm tần số rõ ràng để minh chứng lợi thế
DATASETS = [
    # Dataset tần số cao / dao động rõ rệt → kỳ vọng Freq-MiniRocket tốt hơn
    "EthanolConcentration",     # Cảm biến quang học, tần số quan trọng
    "NATOPS",                   # Cử chỉ tay, tín hiệu tuần hoàn
    "RacketSports",             # Cảm biến gia tốc, rung động
    "Heartbeat",                # ECG/PCG — frequency-domain cực kỳ quan trọng
    "SelfRegulationSCP1",       # EEG — điển hình frequency domain

    # Dataset hình thái rõ rệt → kỳ vọng Time-domain vẫn đủ mạnh
    "BasicMotions",             # Chuyển động cơ bản, hình thái đơn giản
    "ArticularyWordRecognition",# Cử động khớp
    "AtrialFibrillation",       # ECG nhịp tim
]

# --- Cấu hình MiniRocket ---
MINIROCKET_NUM_KERNELS = 10_000   # Số kernel chuẩn của MiniRocket

# --- Cấu hình Frequency Transform ---
FFT_CONFIG = {
    "use_log_magnitude": True,     # Log(|FFT|+1) — ổn định hơn về mặt số học
    "use_phase": False,            # Thêm phase info (tùy chọn, tăng đặc trưng)
    "normalize": True,             # Chuẩn hóa sau FFT
}

STFT_CONFIG = {
    "window": "hann",
    "nperseg": 32,                 # Độ dài cửa sổ STFT
    "noverlap": 16,                # Độ chồng lấp
    "use_log": True,
}

# --- Cấu hình Thực nghiệm ---
N_SPLITS = 5                       # Số fold cross-validation
RANDOM_STATE = 42
RIDGE_ALPHAS = [1e-3, 1e-1, 1.0, 10.0, 100.0]  # Grid search alpha

# --- Đường dẫn ---
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
DATA_CACHE_DIR = ".data_cache"
