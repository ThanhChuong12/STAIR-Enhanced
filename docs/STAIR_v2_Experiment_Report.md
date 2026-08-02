# BÁO CÁO PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM STAIR-v2 (STAIR-DyFuse)

---

## 1. TỔNG QUAN KẾT QUẢ THỰC NGHIỆM

Trong đợt thực nghiệm này, mô hình **STAIR-v2 (STAIR-DyFuse)** — áp dụng cơ chế *Modality-Adaptive Graph Reweighting* dựa trên chỉ số biến thiên đặc trưng (*Feature Spread / Std*) — đã được huấn luyện 500 epochs trên hai bộ dữ liệu chuẩn: **Amazon2014Baby** và **Amazon2014Sports**.

Kết quả đo đạc thực tế thu được như sau:

---

## 2. BẢNG SO SÁNH HIỆU NĂNG TỔNG HỢP

### 2.1 Tập dữ liệu Amazon2014Baby_550_MMRec

| Metric | STAIR Baseline | STAIR-v1 (GCL) | STAIR-v2 (DyFuse) | $\Delta$ v1 vs Base | $\Delta$ v2 vs Base | $\Delta$ v2 vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall@10** | **0.068560** | **0.069459** | 0.057800 | +1.31% | **-15.69%** | -16.79% |
| **Recall@20** | **0.103420** | **0.104700** | 0.089800 | +1.24% | **-13.17%** | -14.23% |
| **NDCG@10** | **0.036335** | **0.036535** | 0.030900 | +0.55% | **-14.96%** | -15.42% |
| **NDCG@20** | **0.045143** | **0.045531** | 0.038800 | +0.86% | **-14.05%** | -14.78% |

---

### 2.2 Tập dữ liệu Amazon2014Sports_550_MMRec

| Metric | STAIR Baseline | STAIR-v1 (GCL) | STAIR-v2 (DyFuse) | $\Delta$ v1 vs Base | $\Delta$ v2 vs Base | $\Delta$ v2 vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall@10** | **0.074310** | **0.074541** | 0.066900 | +0.31% | **-9.97%** | -10.25% |
| **Recall@20** | **0.111900** | **0.112394** | 0.100200 | +0.44% | **-10.46%** | -10.85% |
| **NDCG@10** | **0.040200** | **0.040429** | 0.036600 | +0.57% | **-8.96%** | -9.47% |
| **NDCG@20** | **0.050050** | **0.050400** | 0.044900 | +0.70% | **-10.29%** | -10.91% |

---

## 3. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

Kết quả thực nghiệm cho thấy **STAIR-v2 bị suy giảm từ 9.0% đến 15.7%** so với Baseline và STAIR-v1. Dựa trên biểu đồ quá trình học (Learning Curves) và chỉ số chẩn đoán đặc trưng, chúng ta xác định **3 nguyên nhân kỹ thuật cốt lõi**:

```
                               ┌────────────────────────────────────────────────────────┐
                               │     Đặc trưng Văn bản (Text) bị pre-normalized (L2=1)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Feature Std imbalance: Visual (0.94) vs Text (0.04)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Trọng số Modality bị áp đảo: Visual ~95.6%, Text ~4.4%  │
                               │  -> Đồ thị Văn bản bị triệt tiêu gần như hoàn toàn     │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Suy giảm cấu trúc liên kết & Overfitting nhanh chóng │
                               │  -> Recall@20 giảm 13-15% trên cả 2 dataset            │
                               └────────────────────────────────────────────────────────┘
```

---

### Nguyên nhân 1: Áp đảo phương thức (Extreme Modality Imbalance & Text Suppression)
- **Hiện tượng:** 
  - `Text feature std` ≈ 0.0435, trong khi `Visual feature std` ≈ 0.9420.
  - Khi tính confidence score theo tỷ lệ variance/std, kết quả thu được là:
    - **Visual Confidence ($c_v$):** **95.6%**
    - **Textual Confidence ($c_t$):** **4.4%**
- **Tác động tiêu cực:**
  - Trọng số đồ thị kNN văn bản ($mAdj_{text}$) bị nhân với hệ số cực nhỏ (0.044), dẫn tới việc **gần như loại bỏ hoàn toàn đồ thị văn bản** khỏi quá trình lan truyền thông tin (BSC Smoother).
  - Trong hệ gợi ý thương mại điện tử (Amazon Baby/Sports), thông tin văn bản (tên sản phẩm, mô tả, danh mục) chứa tín hiệu ngữ nghĩa quan trọng bậc nhất. Việc triệt tiêu đồ thị văn bản khiến mô hình mất đi khả năng kết nối các sản phẩm tương đồng về ngữ nghĩa.

---

### Nguyên nhân 2: Trọng số Cosine liên tục (Continuous Cosine Weights) làm mất tính ổn định Laplacian
- **Hiện tượng:**
  - Ở STAIR Baseline, đồ thị kNN nhị phân (unweighted binary edges = 1) được gộp bằng toán tử `max`, tạo ra ma trận kề đại diện cho cấu trúc liên tục của đồ thị.
  - Ở STAIR-v2, đồ thị sử dụng trực tiếp trọng số Cosine Similarity nhân với $c_v, c_t$.
- **Tác động tiêu cực:**
  - Trọng số liên tục làm cho ma trận kề $mAdj$ có phân bố giá trị cực kỳ lệch (skewed distribution). Các cạnh có độ tương đồng Cosine nhỏ bị nén về gần 0, làm giảm đáng kể đường lan truyền đa bước (multi-hop propagation) trong bộ lọc bước (FSC/BSC).

---

### Nguyên nhân 3: Phân tích đường cong học tập (Learning Curves & Overfitting)
- **Đồ thị BPR Loss (Train Loss):**
  - Loss giảm rất nhanh và mượt từ 0.61 xuống **0.02** chỉ sau ~100 epochs, chứng tỏ mô hình học tập tập Train cực kỳ dễ dàng.
- **Đồ thị Metrics (Validation Recall/NDCG):**
  - Recall/NDCG đạt đỉnh rất sớm ở epoch 10-15 (Baby Recall@20 ~ 0.090, Sports Recall@20 ~ 0.100), sau đó đi ngang hoặc giảm nhẹ.
  - Điều này chỉ ra hiện tượng **Overfitting vào tín hiệu Visual noise**: Do đồ thị Visual chiếm 95.6% trọng số, mô hình bị ép phụ thuộc vào đặc trưng hình ảnh (vốn có nhiều nhiễu về góc chụp, màu sắc, background), làm giảm khả năng tổng quát hóa trên tập Validation/Test.

---

## 4. BÀI HỌC KINH NGHIỆM VÀ HƯỚNG CẢI TIẾN TIẾP THEO (STAIR-v3 / DyFuse-v2)

### 4.1 Bài học rút ra
1. **Không dùng trực tiếp tỷ lệ Variance/Std thô để chia trọng số đồ thị** khi hai phương thức có đặc tính chuẩn hóa khác nhau (Text đã pre-L2 normalized, Visual chưa normalized).
2. **Cần giữ một ngưỡng kết nối tối thiểu (Connectivity Floor):** Đồ thị văn bản không bao giờ được phép giảm xuống dưới 30-40% trọng số toàn cục.
3. **Cấu trúc Topology quan trọng hơn Trọng số cạnh liên tục:** Đồ thị nhị phân có chuẩn hóa Laplacian mang lại sự ổn định lan truyền cao hơn trọng số Cosine liên tục bị lệch.

---

### 4.2 Đề xuất hướng cải tiến tiếp theo (STAIR-v3: Gated Adaptive Fusion)

Để khắc phục hoàn toàn hạn chế của STAIR-v2, hướng cải tiến tiếp theo (**STAIR-v3**) sẽ thiết kế cơ chế **Dynamic Gated Fusion (Cổng thích nghi theo item)**:

1. **Chuẩn hóa Z-Score cho Feature Variance:**
   $$c_v = \text{Softmax}\left(\frac{\text{Std}_v}{\tau}\right), \quad \tau \ge 1.0$$
   Giúp khống chế tỷ lệ trọng số giữa 2 modality luôn nằm trong khoảng an toàn $[0.35, 0.65]$.

2. **Cấu trúc Topology Bảo toàn (Topology-Preserving Graph Fusion):**
   $$A_{ij} = \text{TopK}\Big(c_i^v S_{ij}^v + c_i^t S_{ij}^t\Big)$$
   Giữ nguyên bản chất ma trận nhị phân sau khi lọc Top-K cạnh có điểm kết hợp cao nhất, đảm bảo tính ổn định cho phép lọc bước (FSC/BSC).

3. **Kết hợp GCL (Ý tưởng 1) và Dynamic Fusion (Ý tưởng 2):**
   Kết hợp cả Contrastive Loss ở không gian cộng tác (GCL) cùng với Đồ thị đa phương thức cải tiến (DyFuse-v2) để đạt hiệu năng tối ưu nhất.

---

## 5. KẾT LUẬN

Thực nghiệm STAIR-v2 (DyFuse) mang lại giá trị định lượng quan trọng trong công trình nghiên cứu khóa luận:
- **Thử nghiệm giả thuyết:** Chứng minh được việc áp dụng trực tiếp norm/std làm trọng số đồ thị mà không khống chế biên độ sẽ dẫn đến **Modality Collapse** (áp đảo phương thức).
- **Cung cấp cơ sở khoa học vững chắc:** Đóng góp góc nhìn phản biện để hoàn thiện kiến trúc **STAIR-v3 (Gated Adaptive Fusion)** ở giai đoạn tiếp theo.
