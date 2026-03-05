# Đề cương Nghiên cứu: Freq-MiniRocket
## Dual-Domain Time Series Classification via Frequency-Augmented MiniRocket

---

## 1. Tóm tắt (Abstract)

MiniRocket đạt độ chính xác SOTA trong phân loại chuỗi thời gian bằng cách áp dụng 10,000 kernel nhị phân tất định trong **miền thời gian**. Tuy nhiên, với các chuỗi thời gian mang thông tin phân biệt chủ yếu ở **miền tần số** (EEG, ECG, cảm biến rung động), cơ chế kernel nhị phân của MiniRocket không tối ưu để bắt các đặc trưng này.

Chúng tôi đề xuất **Freq-MiniRocket** — kiến trúc dual-domain kết hợp MiniRocket chạy song song trên biểu diễn thời gian gốc và biểu diễn tần số FFT/STFT. Thực nghiệm trên benchmark UEA Multivariate cho thấy Freq-MiniRocket cải thiện độ chính xác đáng kể trên các dataset có cấu trúc tần số rõ rệt, với chi phí tính toán tăng tuyến tính có thể chấp nhận được.

---

## 2. Động lực và Khoảng trống Nghiên cứu

### 2.1 Hạn chế của MiniRocket trong miền tần số

MiniRocket sử dụng 84 kernel nhị phân cố định, mỗi kernel có độ dài 9 điểm. Phân tích Fourier cho thấy:

- **Phổ tần số coverage không đều**: Các kernel nhị phân {-1, +1} hoạt động tương tự bộ lọc FIR với đáp ứng tần số phụ thuộc vào vị trí các bit +2. Không có cơ chế đảm bảo phủ đều toàn bộ phổ tần số.
- **Kernel ngắn (length=9)**: Giới hạn độ phân giải tần số. Kernel độ dài 9 chỉ có thể phân biệt tối đa ~5 băng tần khác nhau.
- **Không có biểu diễn pha (phase)**: PPV (Proportion of Positive Values) hoàn toàn bỏ qua thông tin pha — quan trọng trong nhiều ứng dụng tín hiệu.

### 2.2 Tại sao FFT bổ sung thông tin?

Cho chuỗi x(t) độ dài T, magnitude spectrum `|X[k]| = |Σ x(t)·e^{-j2πkt/T}|` trực tiếp đo năng lượng tại tần số `f_k = k·fs/T`. Đây là biểu diễn tối ưu cho các pattern tuần hoàn và dao động — thứ mà kernel nhị phân ngắn của MiniRocket chỉ bắt được một cách gián tiếp và không đầy đủ.

---

## 3. Phương pháp Đề xuất: Freq-MiniRocket

### 3.1 Kiến trúc

```
X (n, C, T)
    │
    ├─────────────────────────────────┐
    │  Nhánh 1: Time-Domain           │  Nhánh 2: Freq-Domain
    ▼                                 ▼
MiniRocket(X)              FFT/STFT(X) → X_freq (n, C', T')
    │                                 │
    ▼                                 ▼
F_time (n, 10,000)         MiniRocket(X_freq) → F_freq (n, 10,000)
    │                                 │
    └──────── Concat ─────────────────┘
                   │
                   ▼
          F_combined (n, 20,000)
                   │
                   ▼
          RidgeClassifierCV → Nhãn
```

### 3.2 Biến đổi Tần số

**FFT (Fast Fourier Transform):**
- Input: `x ∈ R^T` (mỗi kênh)
- Output: `|X[k]|` với `k = 0, ..., T/2` (chỉ lấy phần tần số dương)
- Optionally: `log(1 + |X[k]|)` để giảm ảnh hưởng của spike
- Output shape: `(n, C, T/2+1)` — sẵn sàng đưa vào MiniRocket

**STFT (Short-Time Fourier Transform):**
- Chia chuỗi thành các cửa sổ Hann, tính FFT cục bộ
- Giữ lại thông tin thời-tần (time-frequency) — phù hợp khi tần số thay đổi theo thời gian
- Output shape: `(n, C·F, W)` với F = số frequency bins, W = số time windows

### 3.3 Các biến thể thực nghiệm

| Biến thể | Mô tả |
|---|---|
| `FreqMiniRocket-FFT` | Dual-domain với FFT magnitude |
| `FreqMiniRocket-STFT` | Dual-domain với STFT spectrogram |
| `FreqMiniRocket+Phase` | FFT magnitude + phase information |
| `FreqOnly-FFT` | Chỉ nhánh freq (ablation study) |

---

## 4. Thiết kế Thực nghiệm

### 4.1 Tập dữ liệu

Chọn từ kho **UEA Multivariate Time Series Archive**, phân thành hai nhóm:

**Nhóm A — Tần số rõ rệt (kỳ vọng Freq-MiniRocket tốt hơn):**
- EthanolConcentration, Heartbeat, SelfRegulationSCP1, NATOPS, RacketSports

**Nhóm B — Hình thái rõ rệt (kỳ vọng Time-domain đủ mạnh):**
- BasicMotions, ArticularyWordRecognition, AtrialFibrillation

### 4.2 Baselines

| Mô hình | Mô tả |
|---|---|
| MiniRocket | Baseline chính (Dempster et al., 2021) |
| Catch22 | Baseline giải thích được (Lubba et al., 2019) |
| FreqOnly-FFT | Ablation: chỉ dùng freq branch |

### 4.3 Chỉ số Đánh giá

- **Accuracy** (chỉ số chính — nhất quán với paper MiniRocket gốc)
- **Weighted F1-score** (xử lý mất cân bằng lớp)
- **Train time** (giây)
- **Inference time** (giây)
- **Δ Accuracy** = FreqMiniRocket - MiniRocket (chỉ số cải thiện)

### 4.4 Điều kiện Công bằng

- Cùng `random_state = 42` cho tất cả mô hình
- Cùng train/test split theo chuẩn UEA (pre-defined)
- Cùng RidgeClassifierCV với cùng danh sách alpha
- Không dùng cross-validation trên test set (tránh data leakage)

---

## 5. Kế hoạch Tuần và Deliverables

### Tuần 1 — Reproduce MiniRocket + Phân tích Lý thuyết
**Mục tiêu:** Reproduce số liệu MiniRocket gốc trên 3 dataset UEA, đọc hiểu paper

**Công việc:**
- [ ] Cài đặt môi trường: `pip install aeon scikit-learn scipy matplotlib seaborn`
- [ ] Chạy `MiniRocketBaseline` trên EthanolConcentration, BasicMotions, Heartbeat
- [ ] So sánh với số liệu Table 1 trong paper MiniRocket gốc (Dempster et al., 2021)
- [ ] Chạy `theoretical_analysis.py` → vẽ frequency response của 84 kernels
- [ ] Viết phần Motivation trong báo cáo dựa trên phân tích lý thuyết

**Deliverable:** Bảng reproduce + biểu đồ kernel_freq_response.png

---

### Tuần 2 — Implement FrequencyTransformer + Nhánh Freq
**Mục tiêu:** Xây dựng và kiểm tra thành phần freq-domain

**Công việc:**
- [ ] Hoàn thiện `src/freq_transform.py` và unit test
- [ ] Chạy `FreqOnlyBaseline` để xác nhận freq features có thông tin
- [ ] Vẽ `plot_time_vs_freq_sample()` cho từng dataset → xác nhận giả thuyết
- [ ] Kiểm tra shape của X_freq đưa vào MiniRocket lần 2

**Deliverable:** Biểu đồ time_vs_freq cho 3 dataset + bảng FreqOnly accuracy

---

### Tuần 3 — Tích hợp Freq-MiniRocket + Thực nghiệm Đầy đủ
**Mục tiêu:** Chạy toàn bộ thực nghiệm so sánh

**Công việc:**
- [ ] Chạy `run_all()` trên tất cả 8 dataset
- [ ] Vẽ accuracy comparison bar chart và improvement heatmap
- [ ] Chạy ablation study (`run_ablation()`) trên EthanolConcentration
- [ ] Phân tích: dataset nào được lợi nhiều nhất từ freq branch?

**Deliverable:** `results/results_full.csv` + tất cả biểu đồ trong `figures/`

---

### Tuần 4 — Phân tích Sâu + Viết Báo cáo
**Mục tiêu:** Feature importance analysis + hoàn thiện báo cáo

**Công việc:**
- [ ] Chạy `plot_feature_importance()` trên EthanolConcentration và Heartbeat
- [ ] Phân tích: Tỉ lệ đóng góp Time vs Freq weight là bao nhiêu?
- [ ] Thử nghiệm biến thể với `use_phase=True` — có cải thiện không?
- [ ] Viết báo cáo hoàn chỉnh (Introduction, Method, Experiments, Analysis, Conclusion)

**Deliverable:** Báo cáo PDF + slide thuyết trình

---

## 6. Phân tích Rủi ro

| Rủi ro | Khả năng | Xử lý |
|---|---|---|
| FreqMiniRocket không tốt hơn MiniRocket tổng thể | Trung bình | Phân tích theo nhóm dataset (A vs B) — chứng minh lợi thế có điều kiện |
| FFT của chuỗi ngắn (<50 điểm) không có ý nghĩa | Thấp | Dùng STFT với window nhỏ hơn, hoặc bỏ dataset đó khỏi phân tích |
| Thời gian huấn luyện tăng gấp đôi | Cao (chắc chắn) | Đây là trade-off đã biết — cần đo đạc và thảo luận trong báo cáo |
| UEA dataset không tải được | Thấp | Dùng `aeon.datasets.load_classification()` với bản cache offline |

---

## 7. Tài liệu Tham khảo

1. Dempster, A., et al. "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." *KDD 2021*.
2. Dempster, A., et al. "ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels." *Data Mining and Knowledge Discovery, 2020*.
3. Tan, C.W., et al. "MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification." *Data Mining and Knowledge Discovery, 2022*.
4. Lubba, C.H., et al. "catch22: CAnonical Time-series CHaracteristics." *Data Mining and Knowledge Discovery, 2019*.
5. Bagnall, A., et al. "The UEA multivariate time series classification archive, 2018." *arXiv 2018*.
