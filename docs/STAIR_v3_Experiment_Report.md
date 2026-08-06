# STAIR-v3: Báo Cáo Phân Tích Kết Quả Thực Nghiệm & Bài Học Kỹ Thuật

**Tên mô hình:** STAIR-v3 (STAIR-ClipFuse / ClipFuse-Consensus)  
**Ngày báo cáo:** 2026-08-06  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Đơn vị thực nghiệm:** Kaggle Environment (Tesla T4 / PyTorch 2.10 / freerec 0.9.7)

---

## 1. TỔNG QUAN THỰC NGHIỆM

Thực nghiệm STAIR-v3 được tiến hành trên 2 tập dữ liệu Amazon2014 (*Baby* và *Sports*) với 500 epochs training cho mỗi mô hình. Bản thực nghiệm này giải quyết 3 điểm nghẽn kỹ thuật từ STAIR-v2:
1. Thay đổi cơ chế tính điểm tin cậy từ L2-norm thô (bị lệch do preprocessing) sang **kNN-Discriminability** ($1 - \text{mean\_knn\_sim}$).
2. Áp dụng ngưỡng cắt **Connectivity Floor $\delta = 0.3$** nhằm chống Modality Collapse.
3. Sửa lỗi trôi dữ liệu giữa CPU và GPU trong PyTorch sparse graph.

---

## 2. PHÁT HIỆN KỸ THUẬT QUAN TRỌNG: LỖI ĐỌC LOG (LOG PARSER ARTIFACT)

Khi phân tích kết quả đầu ra từ notebook, hệ thống gặp một **lỗi hiển thị biểu đồ nghiêm trọng** do hàm `parse_log()`:

```text
Parsing Baby V3...  best={'Recall@10': 3.1, 'Recall@20': 0.0361, 'NDCG@10': 0.0129, 'NDCG@20': 0.016}
Parsing Sports V3... best={'Recall@10': 3.1, 'Recall@20': 0.0511, 'NDCG@10': 0.018, 'NDCG@20': 0.0223}
```

### Nguyên nhân lỗi hiển thị `Recall@10 = 3.1`:
1. Biểu thức chính quy (Regex) `(?i)best.*?Recall@10.*?([0-9]+\.[0-9]+)` đã bắt nhầm chuỗi **`3.12` (từ thông tin hệ thống `Python 3.12.13`)** hoặc **`3.1` (từ thời gian chạy `Done in 3.1 min`)** nằm ở đầu file log trước khi huấn luyện xong.
2. Giá trị `3.1000` ($310\%$) sai lệch này đã làm cho trục $Y$ của biểu đồ so sánh bị đẩy lên mốc $3.5$.
3. Hệ quả: Các cột chỉ số thực tế ($0.03 \to 0.11$) bị nén nhỏ lại chỉ còn vài mm ở đáy biểu đồ, tạo ra cảm giác biểu đồ bị "mất hình/trống chỉ số".

---

## 3. KẾT QUẢ THỰC NGHIỆM THỰC TẾ (ACTUAL METRICS)

Sau khi đọc trực tiếp các dòng log huấn luyện chuẩn từ `freerec` ở các epoch tối ưu:

- **Tập dữ liệu Baby (Epoch 185):**
  - `Recall@10: 0.0605`, `Recall@20: 0.0926`, `NDCG@10: 0.0330`, `NDCG@20: 0.0412`
- **Tập dữ liệu Sports (Epoch 480):**
  - `Recall@10: 0.0718`, `Recall@20: 0.1087`, `NDCG@10: 0.0390`, `NDCG@20: 0.0484`

### Bảng So Sánh Chỉ Số Chuẩn Xác

| Dataset | Metric | STAIR Baseline | STAIR-v1 (GCL) | STAIR-v2 (DyFuse) | **STAIR-v3 (ClipFuse - Thực tế)** | **Δ v3 vs Base** | **Δ v3 vs v2** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baby** | **Recall@10** | 0.068560 | 0.069459 | 0.057800 | **0.060500** | -11.75% | **+4.67%** |
| | **Recall@20** | 0.103420 | 0.104700 | 0.089800 | **0.092600** | -10.46% | **+3.12%** |
| | **NDCG@10** | 0.036335 | 0.036535 | 0.030900 | **0.033000** | -9.18% | **+6.80%** |
| | **NDCG@20** | 0.045143 | 0.045531 | 0.038800 | **0.041200** | -8.73% | **+6.19%** |
| **Sports** | **Recall@10** | 0.074310 | 0.074541 | 0.066900 | **0.071800** | -3.38% | **+7.32%** |
| | **Recall@20** | 0.111900 | 0.112394 | 0.100200 | **0.108700** | -2.86% | **+8.48%** |
| | **NDCG@10** | 0.040200 | 0.040429 | 0.036600 | **0.039000** | -2.99% | **+6.56%** |
| | **NDCG@20** | 0.050050 | 0.050400 | 0.044900 | **0.048400** | -3.30% | **+7.80%** |

---

## 4. PHÂN TÍCH NGUYÊN NHÂN VÀ BÀI HỌC KHOA HỌC

### 4.1 Điểm sáng: Hồi phục so với STAIR-v2 (+3.1% đến +8.5%)
* Chiếm ưu thế rõ rệt so với v2 trên cả 2 dataset: Baby Recall@20 tăng từ `0.0898` lên `0.0926` (+3.1%), Sports Recall@20 tăng từ `0.1002` lên `0.1087` (+8.5%).
* **Nguyên nhân:** kNN-Discriminability ($1 - \text{mean\_knn\_sim}$) đã khắc phục triệt để hiện tượng Modality Collapse của v2, đưa phương thức Text thoát khỏi mức sụp đổ 4.4%.

### 4.2 Hạn chế: Tại sao v3 vẫn kém hơn Baseline gốc (-2.9% đến -10.4%)?

1. **Sự suy giảm do San bằng Tỷ lệ 50-50 (Prior Dilution):**
   * Trong STAIR gốc, tham số `--num-neighbors 5-1` ($k_t=5, k_v=1$) tạo ra tỷ lệ **83.3% Text : 16.7% Visual**. Đây là tri thức miền (domain prior) cốt lõi vì đặc trưng văn bản trên Amazon chuẩn hơn đặc trưng hình ảnh rất nhiều.
   * Ở v3.0, confidence đưa tỷ lệ về **51.4% Visual : 48.6% Text**. Việc đưa 50% ảnh nhiễu vào đồ thị đã làm loãng 83% liên kết văn bản chất lượng cao.

2. **Triệt tiêu Tín hiệu Đồng thuận Đa phương thức (Consensus Loss):**
   * Trong STAIR gốc, các cạnh xuất hiện ở **CẢ HAI** đồ thị kNN được cộng dồn trọng số lên `2.0`.
   * Ở v3.0, việc binarize toàn bộ đồ thị về `1.0` đã vô tình xóa bỏ tín hiệu ưu tiên cho các liên kết ngữ nghĩa đồng thuận đa phương thức.

---

## 5. GIẢI PHÁP ĐÃ TRIỂN KHAI TRONG STAIR-v3.1

Từ các phát hiện trên, phiên bản **STAIR-v3.1 (ClipFuse-Consensus)** đã được thiết kế và triển khai:
1. **Prior-Preserving Weighting:** Bảo toàn tỷ lệ cấu trúc $5:1$ ($W_t = 5 \cdot c_t, W_v = 1 \cdot c_v$), đưa tỷ lệ Text trở lại mốc **82.5%** thay vì 48.6%.
2. **Cross-Modal Consensus Boosting ($\alpha=0.5$):** Nhân hệ số $(1 + \alpha)$ đối với các cạnh đồng thuận xuất hiện ở cả Text và Visual kNN.
3. **Fix Log Parser:** Sửa hoàn toàn regex trong notebook để bắt đúng số liệu thực tế, triệt tiêu lỗi hiển thị `Recall@10 = 3.1`.
