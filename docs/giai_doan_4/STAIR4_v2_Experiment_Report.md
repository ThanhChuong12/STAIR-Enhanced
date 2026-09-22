# BÁO CÁO PHÂN TÍCH TOÀN DIỆN KẾT QUẢ THỰC NGHIỆM GIAI ĐOẠN 4: STAIR4-v2 (BSF-POCL)
# BOUNDED SPECTRAL FILTERING & PHASE-OVERLAP COHERENT CONTRASTIVE LEARNING
### Đối Soát Thực Nghiệm Với STAIR Baseline Trên Amazon Baby & Amazon Sports; Giải Phẫu Nguyên Nhân Suy Thoái Hiệu Năng (-32% đến -50%); Động Lực Học Kẹt Loss & Định Hướng Chiến Lược (v2.1 Ablation vs. Quyết Định Chạy Electronics)

---

**Đề tài:** Recommender Systems using Graph Representation: Multi-modal  
**Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Sinh viên thực hiện:**  
- Lê Hà Thanh Chương (MSSV: 23120195)  
- Bùi Trung Hiếu (MSSV: 23120257)  
**Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo  
**Mã nguồn triển khai:** [`ThanhChuong12/STAIR-Enhanced`](https://github.com/ThanhChuong12/STAIR-Enhanced) (Branch: `main`)  
**Tài liệu thiết kế phương pháp:** [`docs/giai_doan_4/STAIR4_v2_Report.md`](STAIR4_v2_Report.md)  
**Tệp nhật ký thực nghiệm đối soát:**  
- `logs/GD4/STAIR4-v2/Amazon2014Baby_550_MMRec/0922074606/log.txt` (Amazon Baby — 500 Epochs, Ablation A6)  
- `logs/GD4/STAIR4-v2/Amazon2014Sports_550_MMRec/0922083722/log.txt` (Amazon Sports — 500 Epochs, Ablation A6)  
- `notebook/P4/sta-v2.ipynb` (Notebook thực thi và phân tích đồ họa telemetry)  
**Ngày báo cáo:** 22/09/2026  
**Trạng thái kiểm định:** ⚠️ **EMPIRICAL AUDIT COMPLETE — NEGATIVE RESULT DETECTED & CRITICAL POST-MORTEM ANALYZED**

---

## MỤC LỤC BÁO CÁO

1. [TỔNG QUAN QUẢN TRỊ & MA TRẬN ĐỐI SOÁT BASELINE (EXECUTIVE SUMMARY & MASTER AUDIT MATRIX)](#1-tổng-quan-quản-trị--ma-trận-đối-soát-baseline-executive-summary--master-audit-matrix)
   - 1.1. Mục tiêu thiết kế của STAIR4-v2 (BSF-POCL)
   - 1.2. Ma trận số liệu đối chuẩn thực tế STAIR Baseline vs. STAIR4-v2 (A6)
   - 1.3. Đánh giá sơ bộ: Hiện tượng suy thoái hiệu năng nghiêm trọng (-32% đến -50%)
2. [PHÂN TÍCH CHI TIẾT THỰC NGHIỆM TRÊN AMAZON BABY & AMAZON SPORTS](#2-phân-tích-chi-tiết-thực-nghiệm-trên-amazon-baby--amazon-sports)
   - 2.1. Amazon Baby: Suy giảm -41.8% Recall@20 và hiện tượng kẹt BPR Loss ở mức 0.405
   - 2.2. Amazon Sports: Suy giảm -32.6% Recall@20 và trần hiệu năng thấp
   - 2.3. Động lực học suy giảm hàm mất mát & Lịch trình điều tiết (Ramp Schedule)
3. [GIẢI PHÃU NGUYÊN NHÂN HỌC THUẬT & TOÁN HỌC (POST-MORTEM ROOT CAUSE ANALYSIS)](#3-giải-phẫu-nguyên-nhân-học-thuật--toán-học-post-mortem-root-cause-analysis)
   - 3.1. Xung đột Gradient & Can nhiễu Triệt tiêu (Destructive Interference) giữa POCL và BPR
   - 3.2. Sự suy biến của Ánh xạ Pha $\arctan$ và Hàm Coherent Fidelity không âm
   - 3.3. Tác động bất lợi của BSF (Bounded Spectral Filtering) lên tính chọn lọc phổ của BSC
   - 3.4. Động lực học Góc xoay Givens: Góc xoay nhỏ ($< 0.15$ rad) nhưng tạo ma sát tối ưu
4. [HỒ SƠ TIÊU THỤ VRAM & TELEMETRY PHẦN CỨNG (HARDWARE TELEMETRY)](#4-hồ-sơ-tiêu-thụ-vram--telemetry-phần-cứng-hardware-telemetry)
   - 4.1. Đỉnh tiêu thụ VRAM đo thật: 875.2 MiB (Baby) & 1077.2 MiB (Sports)
   - 4.2. Độ phẳng tuyệt đối: Tối ưu bộ nhớ thành công nhưng không đồng nghĩa với hiệu năng gợi ý cao
5. [ĐỊNH HƯỚNG CHIẾN LƯỢC & CÂU HỎI QUYẾT ĐỊNH: V2.1 HAY CHẠY ELECTRONICS?](#5-định-hướng-chiến-lược--câu-hỏi-quyết-định-v21-hay-chạy-electronics)
   - 5.1. Quyết định 1: Tuyệt đối KHÔNG chạy phiên bản v2 hiện tại trên Amazon Electronics
   - 5.2. Quyết định 2: Kế hoạch tinh chỉnh STAIR4-v2.1 (Ablation Diagnostics trên tập Baby)
   - 5.3. Định vị học thuật của Kết quả Âm tính (Negative Result) trong Luận văn tốt nghiệp

---

## 1. TỔNG QUAN QUẢN TRỊ & MA TRẬN ĐỐI SOÁT BASELINE (EXECUTIVE SUMMARY & MASTER AUDIT MATRIX)

### 1.1. Mục tiêu thiết kế của STAIR4-v2 (BSF-POCL)

Theo tài liệu đặc tả [`docs/giai_doan_4/STAIR4_v2_Report.md`](STAIR4_v2_Report.md), phiên bản **STAIR4-v2 (Bounded Spectral Filtering with Phase-Overlap Contrastive Learning — BSF-POCL)** được đề xuất như một hướng tiếp cận vật lý toán học nhằm:
1. **Trụ cột A — Bounded Spectral Filtering (BSF):** Bổ sung một bộ lọc phổ có biên co rút bậc hai $Q_t(H) = I - \frac{t^2}{2}H^2$ (với $H = \frac{I - S}{2}$) vào bộ làm mịn gradient `AdamWSEvo`, hòa trộn với toán tử BSC gốc theo tỷ lệ $\zeta = 0.10$ và thời gian phổ $t = 0.5$, nhằm kiểm soát suy hao tần số cao mà không làm bùng nổ norm gradient.
2. **Trụ cột B — Phase-Overlap Coherent Contrastive Learning (POCL):** Chuyển đổi embedding thực sang vector trạng thái lượng tử giả lập trên hình cầu phức $\mathbb{C}^d$ qua ánh xạ pha $\phi(x) = \arctan(\bar{x})$, áp dụng phép quay Givens học được $G(\theta)$ trên tầng tích chập lân cận, và tính độ tương đồng bằng hàm coherent fidelity $K(x, y) = |\psi(x)^\dagger G \psi(y)|^2$ kết hợp giám sát đa dương tính (multi-positive InfoNCE) với trọng số ramp $\lambda_{\text{cl}} \to 0.001$.

Thực nghiệm chính thức (cấu hình Ablation A6) đã được thực thi đầy đủ 500 epochs trên 2 tập benchmark: **Amazon Baby** (ID: `0922074606`) và **Amazon Sports** (ID: `0922083722`).

---

### 1.2. Ma trận số liệu đối chuẩn thực tế STAIR Baseline vs. STAIR4-v2 (A6)

Bảng 1.1 đối chiếu trực diện kết quả đo thật từ log chạy của STAIR Baseline, phiên bản GĐ4 STAIR-CNLGCL v1-R, STAIR-MHD v3 và **STAIR4-v2 (BSF-POCL)**:

#### Bảng 1.1: Ma trận đối soát đa thế hệ trên Amazon Sports & Amazon Baby

| Tập Dữ Liệu | Thước Đo Metric | STAIR Baseline (03_stair.tex) | STAIR GĐ4-v1-R (CNLGCL) | STAIR-MHD v3 (Hypergraph) | **STAIR4-v2 (A6 - Đo Thật)** | $\Delta$ vs Baseline | Đánh Giá Học Thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Amazon Sports** | **Recall@1** | 0.0143 | **0.0149** | 0.0146 | **0.0094** *(Best Ep 490)* | **-34.27%** 🔻🔻 | Suy thoái nghiêm trọng |
| *(35,598 Users* | **Recall@10** | 0.0743 | 0.0747 | **0.0748** | **0.0482** *(Best Ep 490)* | **-35.13%** 🔻🔻 | Suy thoái nghiêm trọng |
| *18,357 Items* | **Recall@20** | 0.1111 | 0.1124 | **0.1129** | **0.0749** *(Best Ep 490)* | **-32.58%** 🔻🔻 | Mất 32.6% năng lực gợi ý |
| *Độ thưa 99.95%)* | **NDCG@10** | 0.0405 | **0.0411** | 0.0409 | **0.0262** *(Best Ep 490)* | **-35.31%** 🔻🔻 | Thứ hạng bị xáo trộn |
| *(Best Ep: 490)* | **NDCG@20** | 0.0500 | 0.0509 | **0.0508** | **0.0331** *(Best Ep 490)* | **-33.80%** 🔻🔻 | Sụt giảm nặng nề |
| | *Test R@20 (Ep 500)*| 0.1111 | 0.1124 | 0.1126 | **0.0752** *(Ep 500)* | -32.31% | Không cải thiện ở cuối |
| | *BPR Loss (Final)*| ~0.024 | 0.0632 | **0.0249** | **0.1769** *(Ep 500)* | $+637\%$ | Kẹt ở mức cao |
| | *Pipeline Peak VRAM*| — | 867.8 MB | 430.0 MiB | **1077.2 MiB (1.05 GiB)** | — | VRAM ổn định, phẳng |
| | *Thời gian train* | ~54 min | 48.8 min | ~2.23 h | **6583.6 s (~1.83 h)** | — | Tốc độ 13.0 s/epoch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Amazon Baby** | **Recall@1** | 0.0113 | **0.0125** | 0.0124 | **0.0067** *(Best Ep 340)* | **-40.71%** 🔻🔻 | Sụp đổ Top-1 |
| *(19,445 Users* | **Recall@10** | 0.0674 | **0.0674** | 0.0672 | **0.0339** *(Best Ep 340)* | **-49.70%** 🔻🔻 | Mất một nửa độ chuẩn xác |
| *7,050 Items* | **Recall@20** | **0.1042** | 0.1027 | 0.1038 | **0.0606** *(Best Ep 340)* | **-41.84%** 🔻🔻 | Sụp đổ tương tự v4 |
| *Mật độ 0.117%)* | **NDCG@10** | 0.0359 | 0.0360 | **0.0361** | **0.0186** *(Best Ep 340)* | **-48.19%** 🔻🔻 | Mất 48.2% điểm xếp hạng |
| *(Best Ep: 340)* | **NDCG@20** | 0.0454 | 0.0451 | **0.0455** | **0.0255** *(Best Ep 340)* | **-43.83%** 🔻🔻 | Rơi sâu dưới Baseline |
| | *Test R@20 (Ep 500)*| 0.1042 | 0.1033 | 0.1038 | **0.0587** *(Ep 500)* | -43.67% | Tiếp tục thoái lui |
| | *BPR Loss (Final)*| ~0.022 | 0.1808 | **0.1327** | **0.4059** *(Ep 500)* | $+1745\%$ | Kẹt nghiêm trọng |
| | *Pipeline Peak VRAM*| — | 652.0 MB | 236.0 MiB | **875.2 MiB (0.85 GiB)** | — | VRAM ổn định, phẳng |
| | *Thời gian train* | ~25 min | 21.8 min | ~56.2 min | **3028.4 s (~50.5 min)** | — | Tốc độ 6.0 s/epoch |

---

### 1.3. Đánh giá sơ bộ: Hiện tượng suy thoái hiệu năng nghiêm trọng (-32% đến -50%)

Kết quả thực nghiệm chính thức cho thấy một bức tranh **hoàn toàn tiêu cực về mặt độ đo gợi ý**:
1. Trên **Amazon Sports**: Tất cả các chỉ số giảm từ **$32.5\%$ đến $35.3\%$**. Đỉnh Recall@20 chỉ đạt $0.0749$ (so với Baseline $0.1111$).
2. Trên **Amazon Baby**: Mức sụt giảm thậm chí còn tồi tệ hơn, mất từ **$40.7\%$ đến $49.7\%$**. NDCG@20 tụt dốc xuống **$0.0255$** (so với Baseline $0.0454$).
3. **Hiện tượng Kẹt Loss (Optimization Stagnation):** Hàm mất mát BPR không thể hội tụ. Trên Baby, BPR loss bị kẹt cứng ở mức **$0.405$** suốt từ Epoch 100 đến 500; trên Sports, BPR loss kẹt ở mức **$0.177$**.

Đây là một **Negative Result (Kết quả Phủ định)** rõ ràng và dứt khoát. Trong phương pháp luận nghiên cứu khoa học, việc nhận diện và mổ xẻ nguyên nhân của một kết quả phủ định có giá trị tương đương với một kết quả tích cực, giúp trả lời các câu hỏi bản chất về mặt hình học biểu diễn.

---

## 2. PHÂN TÍCH CHI TIẾT THỰC NGHIỆM TRÊN AMAZON BABY & AMAZON SPORTS

### 2.1. Amazon Baby: Suy giảm -41.8% Recall@20 và hiện tượng kẹt BPR Loss ở mức 0.405

Nhật ký huấn luyện `logs/GD4/STAIR4-v2/Amazon2014Baby_550_MMRec/0922074606/log.txt` ghi nhận tiến trình kiểm định chi tiết:

#### Bảng 2.1: Các mốc kiểm định then chốt của STAIR4-v2 trên Amazon Baby

| Epoch | BPR Loss | Valid Recall@10 | Valid Recall@20 | Valid NDCG@10 | Valid NDCG@20 | Test Recall@20 | Test NDCG@20 | Trạng Thái Động Lực Học |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 0.0210 | 0.0344 | 0.0108 | 0.0152 | — | — | Khởi tạo SVD Whitening cơ sở |
| **10** | 0.5124 | 0.0305 | 0.0521 | 0.0165 | 0.0219 | — | — | Pha khởi động, $\lambda_{\text{cl}}=0, \zeta=0$ |
| **30** | 0.4312 | 0.0335 | 0.0572 | 0.0182 | 0.0242 | — | — | Hoàn tất Ramp, $\lambda_{\text{cl}}=0.001, \zeta=0.1$ |
| **100**| 0.4102 | 0.0341 | 0.0580 | 0.0187 | 0.0247 | — | — | **Bắt đầu kẹt Loss hoàn toàn** |
| **200**| 0.4068 | 0.0345 | 0.0588 | 0.0189 | 0.0250 | — | — | Dao động quanh mức nền |
| **340**| **0.4055** | **0.0340** | **0.0595** | **0.0189** | **0.0253** | **0.0606** | **0.0255** | **BEST VALID CHECKPOINT (-43.8%)** |
| **400**| 0.4037 | 0.0346 | 0.0581 | 0.0190 | 0.0250 | — | — | Trạng thái trơ lỳ gradient |
| **500**| **0.4058** | **0.0341** | **0.0579** | **0.0189** | **0.0249** | **0.0587** | **0.0251** | **Kết thúc chu trình 500 epochs** |

**Nhận định chuyên sâu:**
- Tại Epoch 0 (chưa bật POCL và BSF), mô hình có Valid NDCG@20 là $0.0152$.
- Sau 30 epochs đầu (khi bật ramp $\lambda_{\text{cl}} \to 0.001$ và $\zeta \to 0.10$), mô hình tăng lên $0.0242$.
- **Kể từ Epoch 30 trở đi, mô hình gần như ngừng tiến bộ!** Từ Epoch 30 đến Epoch 500 (suốt 470 epochs), NDCG@20 chỉ nhích nhẹ từ $0.0242$ lên đỉnh $0.0253$ rồi dao động đi ngang.
- BPR loss hoàn toàn không thể giảm xuống dưới ngưỡng $0.403$. Đây là minh chứng rõ ràng cho thấy gradient của nhánh phụ trợ đã triệt tiêu độ dốc của hàm mục tiêu chính.

---

### 2.2. Amazon Sports: Suy giảm -32.6% Recall@20 và trần hiệu năng thấp

Nhật ký huấn luyện `logs/GD4/STAIR4-v2/Amazon2014Sports_550_MMRec/0922083722/log.txt` ghi nhận:

#### Bảng 2.2: Các mốc kiểm định then chốt của STAIR4-v2 trên Amazon Sports

| Epoch | BPR Loss | Valid Recall@10 | Valid Recall@20 | Valid NDCG@10 | Valid NDCG@20 | Test Recall@20 | Test NDCG@20 | Trạng Thái Động Lực Học |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 0.0253 | 0.0511 | 0.0142 | 0.0223 | — | — | Khởi tạo SVD Whitening cơ sở |
| **10** | 0.3850 | 0.0385 | 0.0620 | 0.0198 | 0.0258 | — | — | Pha khởi động BPR thuần |
| **30** | 0.2840 | 0.0421 | 0.0665 | 0.0220 | 0.0282 | — | — | Hoàn tất Ramp, $\lambda_{\text{cl}}=0.001, \zeta=0.1$ |
| **100**| 0.2085 | 0.0450 | 0.0701 | 0.0236 | 0.0299 | — | — | Tốc độ giảm Loss bắt đầu chậm lại |
| **200**| 0.1870 | 0.0461 | 0.0715 | 0.0241 | 0.0305 | — | — | Chạm trần bão hòa sớm |
| **350**| 0.1805 | 0.0465 | 0.0722 | 0.0245 | 0.0310 | — | — | Dao động đi ngang |
| **490**| **0.1779** | **0.0466** | **0.0729** | **0.0247** | **0.0313** | **0.0749** | **0.0331** | **BEST VALID CHECKPOINT (-33.8%)** |
| **500**| **0.1769** | **0.0457** | **0.0719** | **0.0245** | **0.0311** | **0.0752** | **0.0333** | **Kết thúc chu trình 500 epochs** |

**Nhận định chuyên sâu:**
- Mặc dù trên Sports, BPR loss có giảm được từ $0.46 \to 0.177$, nhưng mức $0.177$ này vẫn cao gấp hơn $7\times$ so với mức hội tụ $0.0249$ của STAIR Baseline và STAIR-MHD v3.
- Kết quả Test Recall@20 đạt $0.0749$, thấp hơn rất nhiều so với con số kỷ lục $0.1129$ của STAIR-MHD v3 hay $0.1111$ của STAIR Baseline.

---

### 2.3. Động lực học suy giảm hàm mất mát & Lịch trình điều tiết (Ramp Schedule)

Quan sát biểu đồ động lực học huấn luyện trong tệp ảnh `notebook/P4/sta-v2.ipynb` (Figure 1):
1. **Pha 1 — Khởi động BPR (Epochs 0–10):**
   Trong 10 epochs đầu khi $\lambda_{\text{cl}} = 0$ và $\zeta = 0$, mô hình học nhanh và giảm loss dốc đứng bình thường.
2. **Pha 2 — Kích hoạt Ramp (Epochs 11–30):**
   Khi $\lambda_{\text{cl}}$ tăng tuyến tính từ $0 \to 0.001$ và $\zeta$ tăng từ $0 \to 0.10$, đường cong validation bắt đầu bị uốn cong và chậm lại rõ rệt.
3. **Pha 3 — Bão hòa và Kẹt Loss (Epochs 31–500):**
   Đường cong Training Loss của Amazon Baby đi ngang phẳng lì ở mức $0.405$, đường cong NDCG@20 đạt đỉnh giả (pseudo-plateau) ở mức $0.0253$. Trên Amazon Sports, loss chậm lại từ epoch 100 và kẹt ở $0.177$.

---

## 3. GIẢI PHÃU NGUYÊN NHÂN HỌC THUẬT & TOÁN HỌC (POST-MORTEM ROOT CAUSE ANALYSIS)

Tại sao một thiết kế toán học trông rất chặt chẽ trên lý thuyết (BSF co rút phổ, POCL bảo toàn norm Hilbert) lại dẫn tới sự sụp đổ thực nghiệm nghiêm trọng như vậy? Dưới đây là phân tích pháp y giải tích:

### 3.1. Xung đột Gradient & Can nhiễu Triệt tiêu (Destructive Interference) giữa POCL và BPR

1. **Bản chất của hàm mục tiêu BPR:**
   BPR tối ưu hóa tích vô hướng trong không gian Euclide:
   $$s(u, i) = Z_u^T Z_i = \sum_{j=0}^{d-1} Z_{u, j} Z_{i, j}$$
   Gradient của BPR trên cặp $(u, i^+, i^-)$ đẩy vector $Z_{i^+}$ cùng hướng với $Z_u$ và đẩy $Z_{i^-}$ ngược hướng với $Z_u$.

2. **Bản chất của nhánh POCL:**
   POCL chuyển đổi $E^{(0)}$ và $E^{(1)}$ qua ánh xạ pha $\phi = \arctan(\bar{x})$, chuẩn hóa trên mặt cầu phức $S_{\mathbb{C}}^{d-1}$, và tính:
   $$K(u, i) = |\psi_u^\dagger G \psi_i|^2$$
   Gradient của POCL tác động ngược về $E^{(0)}$ theo hướng làm cho phân phối pha của nút người dùng và sản phẩm đồng pha (coherent phase alignment).

3. **Hiện tượng Can nhiễu Triệt tiêu (Destructive Interference):**
   - Trong không gian thực $\mathbb{R}^d$, hai vector có thể có tích vô hướng Euclide lớn nhờ độ dài (magnitude) và góc nhọn.
   - Nhưng trong không gian pha $\mathbb{C}^d$, hàm $\arctan(\bar{x})$ nén toàn bộ giá trị thực $(-\infty, +\infty)$ vào khoảng mở $(-\pi/2, +\pi/2)$.
   - Hai vector có độ dài Euclide rất khác nhau có thể bị ánh xạ thành cùng một vector pha. Ngược lại, hai vector có cùng hướng Euclide nhưng khác biệt nhỏ về giá trị trung bình từng tọa độ có thể tạo ra độ lệch pha lớn.
   - **Hậu quả:** Gradient từ $\nabla \mathcal{L}_{\text{POCL}}$ liên tục kéo các vector embedding quay theo các chiều tọa độ trực giao với hướng dốc của $\nabla \mathcal{L}_{\text{BPR}}$. Tổng vector cập nhật:
     $$\Delta_{\text{total}} = \Delta_{\text{BPR}} + \lambda_{\text{cl}} \Delta_{\text{POCL}}$$
     bị triệt tiêu lẫn nhau ở các thành phần tần số thấp quan trọng, khiến mô hình rơi vào trạng thái bế tắc (limit cycle / optimization jam), dẫn tới việc BPR loss không thể giảm xuống dưới $0.405$.

---

### 3.2. Sự suy biến của Ánh xạ Pha $\arctan$ và Hàm Coherent Fidelity không âm

1. **Hiện tượng Bão hòa Gradient của $\arctan$:**
   Đạo hàm của hàm ánh xạ pha là:
   $$\frac{\partial \phi}{\partial \bar{x}} = \frac{1}{1 + \bar{x}^2}$$
   Khi các giá trị embedding lớn dần trong quá trình huấn luyện ($|\bar{x}| \ge 2$), đạo hàm này giảm nhanh về $0$ ($\le 0.20$). Mạng nơ-ron bị mất tín hiệu gradient truyền ngược từ nhánh đối sánh pha về embedding gốc.

2. **Hàm Coherent Fidelity Luôn Không Âm ($K \in [0, 1]$):**
   Khác với hàm Cosine Similarity truyền thống trong NLGCL/InfoNCE có miền giá trị $[-1, 1]$ (cho phép phạt mạnh các mẫu âm bằng giá trị âm), hàm Coherent Fidelity định nghĩa bởi bình phương mô-đun:
   $$K(u, i) = |\psi_u^\dagger G \psi_i|^2 \ge 0, \quad \forall u, i$$
   Khi tất cả các cặp âm đều có độ tương đồng $K \ge 0$, lực đẩy (repulsive force) trong mẫu số của InfoNCE bị suy yếu, không đủ tạo ra khoảng cách phân tách (margin) cần thiết trên mặt cầu, dẫn đến hiện tượng co cụm biểu diễn.

---

### 3.3. Tác động bất lợi của BSF (Bounded Spectral Filtering) lên tính chọn lọc phổ của BSC

1. **Cơ chế gốc của STAIR BSC:**
   Toán tử làm mịn gradient gốc của STAIR là một đa thức Neumann có hệ số phụ thuộc chặt chẽ vào chỉ số chiều $j \in \{0, \dots, d-1\}$:
   $$P_{b_j, L}(S) = \frac{\sum_{l=0}^L b_j^l S^l}{\sum_{l=0}^L b_j^l}, \quad \text{với } b_j = 0.1 + 0.9(j/d)^\gamma$$
   Cơ chế này phân tách rõ rệt: các chiều đầu tiên ($j \approx 0$) có $b_j$ nhỏ, hầu như không bị làm mịn (bảo toàn tín hiệu tần số cao); các chiều cuối ($j \approx d-1$) có $b_j$ lớn, bị làm mịn mạnh (lọc nhiễu đồ thị).

2. **BSF làm mất tính phân tách chiều:**
   Toán tử $Q_t(H) = I - \frac{t^2}{2} H^2$ là một toán tử **đẳng hướng (isotropic)**, tác động đồng đều lên tất cả các chiều $j$ với cùng một hệ số suy hao $t = 0.5$.
   Khi trộn $T = (1 - \zeta) P_{\text{BSC}} + \zeta Q_t$ với $\zeta = 0.10$:
   - Tại các chiều đầu tiên (nơi cần giữ sắc nét tần số cao), toán tử $Q_t$ đã áp đặt thêm một lượng suy hao $(1 - t^2/2) = 0.875$.
   - Điều này làm mờ đi các đặc trưng sắc nét mà bộ mã hóa FSC đã dày công tách biệt, gây ra hiện tượng làm mịn quá mức (Over-smoothing) trên không gian gradient.

---

### 3.4. Động lực học Góc xoay Givens: Góc xoay nhỏ ($< 0.15$ rad) nhưng tạo ma sát tối ưu

Quan sát Figure 5 (b) (Panel b):
- Quỹ đạo của các góc xoay Givens $\theta_k$ chỉ biến thiên trong khoảng từ $-0.05$ đến $+0.15$ radian (tương đương góc quay cực nhỏ từ $-2.8^\circ$ đến $+8.6^\circ$).
- Điều này chứng minh rằng:
  1. Phép quay Givens 2D **hầu như không tìm thấy hướng xoay có ý nghĩa** để tối ưu hóa hàm đối sánh.
  2. Tuy nhiên, sự hiện diện của các tham số $\theta_k$ trong đồ thị tính toán autograd tạo ra một bề mặt mất mát phi lồi gồ ghề (rugged loss landscape), làm chậm tốc độ lan truyền gradient của các tầng tích chập chính.

---

## 4. HỒ SƠ TIÊU THỤ VRAM & TELEMETRY PHẦN CỨNG (HARDWARE TELEMETRY)

### 4.1. Đỉnh tiêu thụ VRAM đo thật: 875.2 MiB (Baby) & 1077.2 MiB (Sports)

Nhật ký đo đạc từ thư viện `pynvml` và biểu đồ Figure 6 ghi nhận:

#### Bảng 4.1: Bảng tổng kết chi phí phần cứng của STAIR4-v2 trên GPU Tesla T4 (16 GB)

| Tập Dữ Liệu | Số Items | Pure Tensor Peak (`max_allocated`) | Pipeline Total Peak (NVML) | Tỷ Lệ Chiếm Dụng VRAM T4 | Thời Gian 500 Epochs | Trạng Thái Bộ Nhớ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Amazon Baby** | 7,050 | **~191.2 MiB** | **875.2 MiB (0.85 GiB)** | **5.47%** | **3028.4 s (~50.5 min)** | ✅ Phẳng mượt tuyệt đối |
| **Amazon Sports** | 18,357 | **~365.5 MiB** | **1077.2 MiB (1.05 GiB)** | **6.73%** | **6583.6 s (~1.83 h)** | ✅ Phẳng mượt tuyệt đối |
| **Amazon Electronics**| 63,001 | — | — *(Chưa chạy)* | — | — *(Dự kiến ~8h)* | ⏸️ Tạm hoãn theo khuyến nghị |

---

### 4.2. Độ phẳng tuyệt đối: Tối ưu bộ nhớ thành công nhưng không đồng nghĩa với hiệu năng gợi ý cao

- **Thành công về mặt kỹ thuật phần mềm:** Đồ thị VRAM tại Figure 6 duy trì độ phẳng hoàn hảo suốt 500 epochs trên cả hai tập dữ liệu. Không hề có hiện tượng rò rỉ bộ nhớ (memory leak) hay xung nhọn tức thời (transient spike).
- **Bài học học thuật đắt giá:** *Một mô hình có thể tối ưu hóa bộ nhớ cực tốt và chạy rất ổn định về mặt phần cứng, nhưng vẫn có thể thất bại hoàn toàn về mặt chất lượng biểu diễn toán học.* Độ phẳng của VRAM chỉ chứng minh kiến trúc pipeline tính toán ma trận thưa hợp lý, không chứng minh được tính đúng đắn của hàm mục tiêu.

---

## 5. ĐỊNH HƯỚNG CHIẾN LƯỢC & CÂU HỎI QUYẾT ĐỊNH: V2.1 HAY CHẠY ELECTRONICS?

Căn cứ vào toàn bộ bằng chứng thực nghiệm trên, nhóm nghiên cứu đưa ra câu trả lời dứt khoát cho câu hỏi chiến lược:

### 5.1. Quyết định 1: Tuyệt đối KHÔNG chạy phiên bản v2 hiện tại trên Amazon Electronics

- **Lý do kỹ thuật & chi phí:**
  - Tập Amazon Electronics có quy mô khổng lồ: $63,001$ sản phẩm, gần $1.7$ triệu tương tác.
  - Quá trình huấn luyện 500 epochs trên Electronics tiêu tốn **khoảng 8.0 đến 8.5 giờ GPU liên tục** trên Kaggle (chạm sát ngưỡng giới hạn 9h/session).
  - Khi bản v2 đã thất bại đồng loạt trên cả tập thưa (Sports, giảm $-32.6\%$) và tập dày (Baby, giảm $-41.8\%$), việc chạy tiếp trên Electronics chắc chắn $100\%$ sẽ cho ra kết quả sụp đổ tương tự.
  - **Hành động:** Tiết kiệm toàn bộ hạn ngạch tính toán GPU của Kaggle; không nạp job huấn luyện v2 trên Electronics.

---

### 5.2. Quyết định 2: Kế hoạch tinh chỉnh STAIR4-v2.1 (Ablation Diagnostics trên tập Baby)

Nếu nhóm nghiên cứu muốn cứu vãn hoặc kiểm chứng độc lập từng trụ cột của thiết kế v2, chúng ta **phải thực hiện phân rã thành phần (Decoupled Ablation)** trong phiên bản **STAIR4-v2.1**, và **chỉ thử nghiệm trên tập dữ liệu nhỏ Amazon Baby (chỉ mất ~15-20 phút)**:

#### 3 Thí nghiệm Bóc Tách Cốt Lõi cho v2.1:

1. **Ablation 1 — BSF-Only ($\lambda_{\text{cl}} = 0$, $\zeta = 0.10$, $t = 0.5$):**
   - Tắt hoàn toàn nhánh POCL (không tính pha, không quay Givens, không có InfoNCE).
   - Chỉ giữ lại bộ lọc BSF trong `AdamWSEvo`.
   - **Mục đích:** Trả lời câu hỏi: *Liệu bản thân BSF có làm hỏng mô hình không, hay BSF vẫn tốt nhưng bị POCL kéo sập?*
2. **Ablation 2 — POCL-Only ($\lambda_{\text{cl}} = 0.001$, $\zeta = 0$):**
   - Tắt hoàn toàn BSF, dùng lại toán tử BSC gốc của STAIR.
   - Chỉ bật nhánh đối sánh pha POCL.
   - **Mục đích:** Đo lường chính xác mức độ tổn hại do xung đột gradient của POCL gây ra.
3. **Ablation 3 — POCL-Calibrated ($\lambda_{\text{cl}} = 10^{-4}$ hoặc $10^{-5}$):**
   - Hạ trọng số POCL xuống mức siêu nhỏ để kiểm tra xem ở mức can thiệp rất nhẹ, pha có mang lại tín hiệu bổ trợ nào không.

> **Quy tắc nghiệm thu v2.1:** Nếu sau 100 epochs trên Amazon Baby, phiên bản v2.1 không thể đạt được ít nhất **$95\%$ mức NDCG@20 của Baseline ($> 0.0430$)**, chúng ta sẽ chính thức dừng toàn bộ việc phát triển nhánh này để tập trung nguồn lực vào hai kiến trúc thành công vượt bậc là **STAIR-CNLGCL v1-R** và **STAIR-MHD v3**.

---

### 5.3. Định vị học thuật của Kết quả Âm tính (Negative Result) trong Luận văn tốt nghiệp

Trong văn bản Khóa luận tốt nghiệp, kết quả của STAIR4-v2 không hề bị bỏ đi mà sẽ được đưa vào **Mục 4.5: Phân tích Nghiên cứu Bóc Tách & Các Hướng Tiếp Cận Thất Bại (Ablation Studies & Negative Results Analysis)** với các đóng góp học thuật sâu sắc:
1. **Minh chứng thực nghiệm về Giới hạn của Lượng tử hóa Biểu diễn (Limits of Phase Encoding):**
   Chứng minh rằng việc cố gắng ép buộc tính kết hợp pha (Phase Coherence) trong không gian Hilbert phức lên các mô hình gợi ý đồ thị dựa trên tích vô hướng Euclide tạo ra hiện tượng **Can nhiễu Triệt tiêu Gradient (Destructive Gradient Interference)**, khiến hàm mục tiêu BPR bị tê liệt.
2. **Làm nổi bật tính ưu việt của STAIR-MHD v3:**
   Sự thất bại của cách tiếp cận phase-matching càng làm sáng tỏ lý do vì sao cơ chế **Multi-Head Hypergraph Disentanglement (STAIR-MHD v3)** lại thành công rực rỡ (đạt kỷ lục SOTA Sports $0.1129$): vì STAIR-MHD v3 xử lý bài toán lọc nhiễu ở cấp độ cấu trúc nhóm (hyperedge group-level) kết hợp gating thích ứng điều kiện hành vi, thay vì can thiệp thô bạo vào góc pha tọa độ của từng vector embedding.

---

## TỔNG KẾT VÀ KIẾN NGHỊ HÀNH ĐỘNG

1. **Về mặt số liệu:** Báo cáo xác nhận trung thực kết quả đo thật của STAIR4-v2 (A6): Test NDCG@20 đạt $0.0255$ trên Baby ($-43.8\%$) và $0.0331$ trên Sports ($-33.8\%$).
2. **Về mặt huấn luyện trên Electronics:** **DỪNG NGAY LẬP TỨC**, tuyệt đối không chạy v2 trên Electronics để tránh lãng phí 8.5 giờ GPU.
3. **Về mặt tinh chỉnh v2.1:** Triển khai một script ablation siêu gọn trên Baby (chạy 100 epochs, ~15 phút) để bóc tách riêng BSF-only và POCL-only, hoàn tất bằng chứng cho mục Negative Study của Luận văn tốt nghiệp.
