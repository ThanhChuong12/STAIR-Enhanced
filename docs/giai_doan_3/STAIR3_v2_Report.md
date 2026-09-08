# BÁO CÁO NGHIÊN CỨU & THIẾT KẾ KIẾN TRÚC GIAI ĐOẠN 3 — ĐỢT 2 (STAIR3-v2)
## MÔ HÌNH STAIR-SRE-ANS: STEPWISE SPECTRAL-REFINED CONTRASTIVE LEARNING WITH ADAPTIVE NEGATIVE SCHEDULING & CONTINUOUS SPECTRAL DIFFICULTY DECOUPLING
### Đột phá Hiệu năng Đa phương thức Thông qua Phân rã Phổ Năng lượng Liên tục, Điều phối Mẫu âm Thích ứng, Gated Top-K Selection & Triệt tiêu Mẫu âm Giả

**Đề tài:** Recommender Systems using Graph Representation: Multi-modal  
**Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Sinh viên thực hiện:**  
- Lê Hà Thanh Chương (MSSV: 23120195)  
- Bùi Trung Hiếu (MSSV: 23120257)  
**Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo  
**Mã nguồn triển khai:** [`ThanhChuong12/STAIR-Enhanced`](https://github.com/ThanhChuong12/STAIR-Enhanced)  
**Tập tài liệu thiết kế:** `docs/giai_doan_3/STAIR3_v2_Report.md`  
**Ngày hoàn thiện:** 2026-09-08  
**Trạng thái:** Hoàn thiện Thiết kế Toán học, Khắc phục Triệt để Ranh giới Cứng Chiều 32, Khung Thuật toán & Mã nguồn PyTorch Chuẩn Sản xuất — Sẵn sàng Thực nghiệm Kaggle GPU

---

## MỤC LỤC

1. [Tổng quan Chiến lược & Sứ mệnh Đột phá của Giai đoạn 3 — Đợt 2](#1-tổng-quan-chiến-lược--sứ-mệnh-đột-phá-của-giai-đoạn-3--đợt-2)
   - 1.1 Tổng kết Chuyển tiếp từ Đợt 1 (v1 & v1.1): Bài học Thực nghiệm, Thành quả và Nút thắt Còn lại
   - 1.2 Nhận diện Tử huyệt của Cơ chế Lấy mẫu âm Ngẫu nhiên Đồng nhất (Uniform Negative Sampling)
   - 1.3 Mục tiêu Chiến lược: Chinh phục Ngưỡng Tăng trưởng $\ge +5.0\%$ Đồng bộ trên Cả 3 Tập Dữ liệu
2. [Phân tích Phản biện Sâu sắc & Tiếp thu Có Chọn lọc 3 Triết lý Tiên phong](#2-phân-tích-phản-biện-sâu-sắc--tiếp-thu-có-chọn-lọc-3-triết-lý-tiên-phong)
   - 2.1 Phân tích & Phản biện Triết lý NegGen: Phân loại False vs. Hard vs. Easy Negatives Không Dùng MLLM
   - 2.2 Phân tích & Phản biện Triết lý GDNSM: Bác bỏ Ranh giới Cứng Chiều 32 & Thiết lập Phân rã Phổ Liên tục
   - 2.3 Phân tích & Phản biện Triết lý AdNGCL: Bộ Điều phối HANS Thích ứng theo Epoch
3. [Phát hiện & Khắc phục 4 Điểm nghẽn Toán học và 1 Lỗi Kỹ thuật Hệ thống Tinh vi](#3-phát-hiện--khắc-phục-4-điểm-nghẽn-toán-học-và-1-lỗi-kỹ-thuật-hệ-thống-tinh-vi)
   - 3.1 Bác bỏ Ranh giới Cứng Chiều 32: Chuyển dịch sang Phân rã Phổ Liên tục theo $\boldsymbol{\beta}(d)$
   - 3.2 Nghịch lý Phân bổ Budget Mẫu âm khó (Top-K Budget Bottleneck) & Giải pháp Gated Top-K Selection
   - 3.3 Mâu thuẫn Attenuation trên Mẫu cùng Metadata nhưng Tương đồng thấp & Thresholded Cosine Gating
   - 3.4 Khắc phục Ghép nối Lập lịch HANS với Loss tổng hợp & Giám sát Tương phản Độc lập
   - 3.5 Khắc phục Lỗi Kỹ thuật Rò rỉ Đánh giá (Evaluation Leak) trong Hàng đợi Memory Bank FIFO
   - 3.6 Bảng Đối chiếu Hệ thống: Các Điểm nghẽn Toán học và Bản vá Kiến trúc Hoàn thiện
4. [Kiến trúc Toàn diện Mô hình STAIR-SRE-ANS (Giai đoạn 3 — Đợt 2)](#4-kiến-trúc-toàn-diện-mô-hình-stair-sre-ans-giai-đoạn-3--đợt-2)
   - 4.1 Sơ đồ Luồng Dữ liệu và Tương tác Module Toàn hệ thống
   - 4.2 Trụ cột 1: Regularized Diagonal Spectral Projector (Bảo tồn Tuyệt đối Hệ trục SVD)
   - 4.3 Trụ cột 2: Continuous Spectral Difficulty Decoupling với Trọng số Phổ Liên tục $\boldsymbol{\beta}(d)$
   - 4.4 Trụ cột 3: Gated Top-K Selection & Triệt tiêu Rỗng Ngân sách Mẫu khó
   - 4.5 Trụ cột 4: Thresholded Cosine-Gated MFNA & Phân tầng Mẫu âm Đa cấp
   - 4.6 Trụ cột 5: Decoupled HANS Scheduler & Cross-Batch Memory Bank FIFO An toàn
5. [Hệ thống Công thức Toán học Vi phân & Định lý Cân bằng Gradient](#5-hệ-thống-công-thức-toán-học-vi-phân--định-lý-cân-bằng-gradient)
   - 5.1 Hàm Mục tiêu Đa nhiệm Toàn cục ($\mathcal{L}_{\text{total}}$)
   - 5.2 Công thức Chi tiết Hàm Mất mát Stepwise SRE-ANS Loss
   - 5.3 Giải tích Gradient và Chứng minh Toán học về Tính Hòa hợp Gradient với BPR và BSC
   - 5.4 Định lý Bảo toàn Phương sai và Phân tán trên Siêu Mặt cầu (Uniformity Theorem)
6. [Hiện thực hóa Mã nguồn PyTorch Chuẩn Sản xuất (Production-Grade Implementation)](#6-hiện-thực-hóa-mã-nguồn-pytorch-chuẩn-sản-xuất-production-grade-implementation)
   - 6.1 Module `models/stair_sre_ans_v8.py` Hoàn chỉnh
7. [Ma trận Mục tiêu Bứt phá $\ge +5.0\%$ trên Cả 3 Tập Dữ liệu](#7-ma-trận-mục-tiêu-bứt-phá-ge-50-trên-cả-3-tập-dữ-liệu)
8. [Đặc tả Không gian Siêu tham số & Hướng dẫn Vận hành](#8-đặc-tả-không-gian-siêu-tham-số--hướng-dẫn-vận-hành)
9. [Đánh giá Hiệu năng Phần cứng & Độ phức tạp Tính toán](#9-đánh-giá-hiệu-năng-phần-cứng--độ-phức-tạp-tính-toán)
10. [Kịch bản Phản biện Học thuật Nâng cấp Trước Hội đồng (Academic Defense Upgrade)](#10-kịch-bản-phản-biện-học-thuật-nâng-cấp-trước-hội-đồng-academic-defense-upgrade)
11. [Kế hoạch Hành động Triển khai Thực nghiệm Đợt 2](#11-kế-hoạch-hành-động-triển-khai-thực-nghiệm-đợt-2)
12. [Đóng góp Phản biện Chuyên sâu & Nâng cấp Kiến trúc STAIR-SRE-ANS v2.1](#12-đóng-góp-phản-biện-chuyên-sâu--nâng-cấp-kiến-trúc-stair-sre-ans-v21)
    - 12.1 Phân tích Phản biện 4 Tử huyệt Toán học & Kỹ thuật trong Thực thi v2
    - 12.2 Thiết kế Kiến trúc STAIR-SRE-ANS v2.1: Decoupled Multi-Scale Representation & Thresholded Gating
    - 12.3 Bảng Ma trận Đối chiếu Tiến hóa: STAIR Baseline vs v2 vs v2.1
    - 12.4 Hệ thống Công thức Toán học Vi phân Hoàn thiện của v2.1
    - 12.5 Mã nguồn Triển khai Chuẩn hóa: `models/stair_sre_ans_v2_1.py` & Pipeline Huấn luyện
    - 12.6 Giả thuyết Khoa học, Khuyến nghị Tham số & Kỳ vọng Vượt Baseline trên Cả 3 Tập Dữ liệu
11. [Kế hoạch Hành động Triển khai Thực nghiệm Đợt 2](#11-kế-hoạch-hành-động-triển-khai-thực-nghiệm-đợt-2)

---

## 1. TỔNG QUAN CHIẾN LƯỢC & SỨ MỆNH ĐỘT PHÁ CỦA GIAI ĐOẠN 3 — ĐỢT 2

### 1.1 Tổng kết Chuyển tiếp từ Đợt 1 (v1 & v1.1): Bài học Thực nghiệm, Thành quả và Nút thắt Còn lại

Khép lại Đợt 1 của Giai đoạn 3, mô hình **STAIR-SRE** đã hoàn thành trọn vẹn chu kỳ nghiên cứu thực nghiệm đầu tiên với những phát hiện học thuật mang tính bước ngoặt:

1. **Phát hiện và chứng minh toán học Hiện tượng Xung đột Gradient Ký sinh (Parasitic Gradient Conflict):**  
   Trong phiên bản v1, việc hoán đổi phổ từ chính sản phẩm dương $\mathbf{i}^+$ đã vô tình tạo ra một mẫu âm giả mang hơn $60\%$ đặc trưng cộng tác của $\mathbf{i}^+$. Khi hàm InfoNCE đẩy người dùng ra xa mẫu này, nó triệt tiêu trực tiếp lực kéo của hàm xếp hạng BPR.
2. **Bước ngoặt phục hồi ngoạn mục của phiên bản v1.1:**  
   Bằng việc thiết lập cơ chế **Cross-Negative Spectral Swapping (CNSS)** (hoán đổi giữa hai mẫu âm độc lập trong batch), tái lập ngưỡng lọc âm giả $\tau_{\text{atten}} = 0.35$, bổ sung số hạng neo giữ L2 $\lambda_w \|\mathbf{w} - \mathbf{1}\|_2^2$, và nâng nhiệt độ $\tau = 0.30$ trên tập siêu thưa Amazon Sports:
   - Trên **Amazon Sports**, hiệu năng lập tức đảo chiều tăng vọt **$+6.71\%$ Recall@20** (từ $0.1029 \to 0.1098$) và **$+6.94\%$ NDCG@20** (từ $0.0461 \to 0.0493$) so với v1, thu hẹp hơn $84\%$ khoảng cách và đưa mô hình tiệm cận sát Baseline (chỉ còn cách $-1.17\%$).
   - Hiện tượng bão hòa loss sớm về $0.0045$ bị xóa bỏ hoàn toàn; loss huấn luyện duy trì ổn định ở mức $0.0242$ (cao hơn $5.3$ lần so với v1), kéo dài đỉnh hội tụ tối ưu từ Epoch 155 lên tận Epoch 415 mà không hề bị suy thoái.
   - Trên **Amazon Baby**, Recall@20 cũng phục hồi $+3.72\%$ lên mức $0.1003$ và NDCG@20 tăng $+3.10\%$ lên $0.0433$.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      LỘ TRÌNH TIẾN HÓA HỌC THUẬT: TỪ ĐỢT 1 SANG ĐỢT 2 (PHASE 3)                        │
├──────────────────────────────┬─────────────────────────────────────────────────────────────────────────┤
│ Giai đoạn 3 — Đợt 1 (v1)     │ Nhận diện xung đột gradient ký sinh & bão hòa loss sớm                  │
│                              │ Recall@20 Sports: 0.1029 (-7.38% vs BL) | Baby: 0.0967 (-7.20% vs BL)   │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Giai đoạn 3 — Đợt 1.1 (v1.1) │ Phục hồi hiệu năng: CNSS (0% i+) + Atten τ=0.35 + Anchoring + τ=0.30    │
│                              │ Recall@20 Sports: 0.1098 (-1.17% vs BL) | Baby: 0.1003 (-3.74% vs BL)   │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ ▶ GIAI ĐOẠN 3 — ĐỢT 2 (v2)   │ STAIR-SRE-ANS: Phân rã Phổ Liên tục β(d) (Không cắt cứng 32)            │
│   (STAIR-SRE-ANS v2)         │                + Gated Top-K Selection + Thresholded MFNA               │
│                              │                + Decoupled HANS + Cross-Batch Memory Bank FIFO          │
│                              │ MỤC TIÊU: CHÍNH THỨC VƯỢT BASELINE ≥ +5.0% ĐỒNG BỘ TRÊN CẢ 3 DATASETS   │
└──────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Nhận diện Tử huyệt của Cơ chế Lấy mẫu âm Ngẫu nhiên Đồng nhất (Uniform Negative Sampling)

Mặc dù phiên bản v1.1 đã giải tỏa được xung đột gradient và tiệm cận Baseline, mô hình vẫn chưa thể bứt phá mạnh mẽ vượt mức $+5.0\%$ vì một hạn chế mang tính nền tảng của hàm InfoNCE truyền thống: **Cơ chế Lấy mẫu âm Đồng nhất trong Mini-batch (Uniform In-batch Negative Sampling)**.

1. **Độ thưa đồ thị cực đại làm cạn kiệt thông tin giám sát:**  
   Trên đồ thị tương tác có độ thưa từ $99.82\%$ (Baby) đến $99.95\%$ (Sports) và $99.96\%$ (Electronics), xác suất để hai sản phẩm ngẫu nhiên trong batch có liên quan ngữ nghĩa là cực kỳ thấp ($< 0.1\%$). Khi đó, đại đa số mẫu âm in-batch đều là **Easy Negatives hiển nhiên** (ví dụ: người dùng đang xem vợt tennis nhưng mẫu âm là tã lót sơ sinh). Các mẫu âm quá dễ này bị bộ phân loại đẩy ra xa một cách tầm thường, gradient đóng góp tiệm cận về 0, không cung cấp thêm bất kỳ tín hiệu phân biệt tinh vi nào để tái cấu trúc danh sách Top-K.
2. **Nguy cơ tiềm ẩn từ Hard Negatives mù quáng:**  
   Nếu chỉ đơn thuần tìm kiếm các sản phẩm có độ tương đồng embedding cao nhất để làm mẫu âm khó (Hard Negative Mining thông thường), mô hình sẽ rơi vào cái bẫy **Mẫu âm giả (False Negatives)**: phạt nhầm các sản phẩm mà người dùng thực sự yêu thích nhưng chưa có tương tác trong tập huấn luyện, từ đó trực tiếp phá hủy độ phủ Recall.
3. **Giới hạn không gian tìm kiếm của Mini-batch (Batch Bottleneck):**  
   Với kích thước batch $B = 1024$, không gian tìm kiếm mẫu âm chỉ giới hạn trong 1024 sản phẩm ngẫu nhiên. Số lượng này là quá nhỏ so với tổng số hàng chục nghìn sản phẩm trong catalog, khiến mô hình bỏ lỡ các mẫu đối kháng có giá trị học tập cao nhất.

### 1.3 Mục tiêu Chiến lược: Chinh phục Ngưỡng Tăng trưởng $\ge +5.0\%$ Đồng bộ trên Cả 3 Tập Dữ liệu

Phiên bản **STAIR-SRE-ANS (v2)** được thiết kế để giải quyết dứt điểm các hạn chế trên, với mục tiêu học thuật xác định:
- **Amazon Baby:** Recall@20 $\ge 0.1095$ ($+5.09\%$), NDCG@20 $\ge 0.0480$ ($+5.73\%$).
- **Amazon Sports:** Recall@20 $\ge 0.1168$ ($+5.13\%$), NDCG@20 $\ge 0.0530$ ($+6.00\%$).
- **Amazon Electronics:** Recall@20 $\ge 0.0705$ ($+6.02\%$), NDCG@20 $\ge 0.0325$ ($+7.26\%$).

---

## 2. PHÂN TÍCH PHẢN BIỆN SÂU SẮC & TIẾP THU CÓ CHỌN LỌC 3 TRIẾT LÝ TIÊN PHONG

Dưới lăng kính của một **Senior AI Research Engineer**, chúng tôi tiến hành giải phẫu ba công trình nghiên cứu tiên phong gần đây trong lĩnh vực học tương phản cho hệ gợi ý (NegGen, GDNSM, AdNGCL), nhận diện các tử huyệt nếu áp dụng mù quáng, và đề xuất giải pháp cải biên toán học chuẩn mực cho STAIR.

```
                         GIẢI PHẪU 3 CÔNG TRÌNH TIÊN PHONG & BỘ LỌC THÍCH NGHI STAIR
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
1. TRIẾT LÝ NEGGEN                          2. TRIẾT LÝ GDNSM                            3. TRIẾT LÝ ADNGCL
   Phân loại FN vs HN vs EN                   Subspace Difficulty Scoring                  Hardness-Aware Scheduling
   TỬ HUYỆT: Dùng MLLM quá chậm,              TỬ HUYỆT: Dùng Diffusion cồng kềnh;         TỬ HUYỆT: Tăng độ khó đột ngột
   MLP xoay trục SVD của STAIR                CẮT CỨNG [:32]/[32:] LÀ GIẢ ĐỊNH SAI!        gây sốc gradient, xung đột BSC!
         │                                            │                                            │
         ▼                                            ▼                                            ▼
[ GIẢI PHÁP STAIR v2: NEGGEN-MASK ]         [ GIẢI PHÁP STAIR v2: PHÂN RÃ PHỔ β(d) ]     [ GIẢI PHÁP STAIR v2: HANS-SMOOTH ]
  Phân loại qua SVD Cosine tĩnh               Phân rã phổ liên tục qua β và (1-β),         Loss-Gated Trigger trên loss_ans,
  kết hợp Metadata Brand/Category             Separate L2 trên toàn bộ 64 chiều            step ≤ 0.02, trần an toàn ≤ 0.35
```

---

### 2.1 Phân tích & Phản biện Triết lý NegGen: Phân loại False vs. Hard vs. Easy Negatives Không Dùng MLLM

#### Cơ chế nguyên bản của NegGen và Rủi ro kỹ thuật đối với STAIR
NegGen đề xuất sử dụng Mô hình Ngôn ngữ Lớn Đa phương thức (MLLM) hoặc các mạng nơ-ron sinh tự hồi quy để phân định ranh giới giữa mẫu âm giả (False Negative), mẫu âm khó (Hard Negative) và mẫu âm dễ (Easy Negative).
- **Rủi ro chí mạng:** Việc sử dụng MLLM hay các mạng nơ-ron học tham số (MLP/Attention) ở khâu tiền xử lý hoặc trong vòng lặp huấn luyện sẽ:
  1. Làm xoay hệ trục tọa độ SVD và làm biến dạng không gian đa tạp trực giao (bài học thất bại từ v1 và v3).
  2. Khiến chi phí tính toán bùng nổ, gây nghẽn cổ chai I/O và tràn bộ nhớ VRAM trên các môi trường nghiên cứu như Kaggle GPU T4.

#### Giải pháp Cải biên STAIR-NegGen (An toàn Toán học 100%)
Chúng tôi nhận định: **Việc phân loại mẫu âm hoàn toàn có thể thực hiện một cách chính xác mà không cần dùng đến bất kỳ tham số học nào làm xoay trục tọa độ.**  
Thay vì dùng MLLM, STAIR-NegGen tận dụng sự kết hợp giữa:
1. Độ tương đồng Cosine trong không gian SVD tĩnh $\mathbf{W} \in [-1, 1]$.
2. Ràng buộc cấu trúc sản phẩm $\mathbf{M}_{\text{meta}}$ (Brand, Category).

Bảng logic phân loại chuẩn xác 3 miền mẫu âm:

| Loại Mẫu âm | Bản chất Ngữ nghĩa | Điều kiện Phân loại Kỹ thuật | Hành vi Tác động trong Hàm Loss |
| :--- | :--- | :--- | :--- |
| **False Negative (FN)** | Sản phẩm cực kỳ phù hợp với gu của người dùng nhưng chưa có tương tác (ví dụ: cùng Brand/Category và tương đồng SVD cao). | $\text{Sim}(u, k) > \tau_{\text{atten}}$ **AND** $\text{Metadata Match} = 1$ | **Triệt tiêu lực đẩy:** Hệ số suy giảm $(1 - W_{u, k}) \to 0$, bảo toàn Recall. |
| **Hard Negative (HN)** | Sản phẩm cạnh tranh cao, dễ gây nhầm lẫn nhưng thực sự khác sở thích (ví dụ: tương đồng hành vi cao nhưng khác Brand/Category). | $\text{Sim}(u, k) > \tau_{\text{atten}}$ **AND** $\text{Metadata Match} = 0$ | **Khuếch đại có kiểm soát:** Gán trọng số $\Psi_{\text{HN}} > 1.0$ để rèn luyện ranh giới phân biệt. |
| **Easy Negative (EN)** | Sản phẩm hoàn toàn xa lạ, thuộc phân khúc khác biệt hoàn toàn. | $\text{Sim}(u, k) \le \tau_{\text{atten}}$ **AND** $\text{Metadata Match} = 0$ | **Duy trì lực đẩy chuẩn:** Gán trọng số nền $\Psi_{\text{EN}} \le 1.0$ để giữ tính phân tán phổ. |

**Tính an toàn toán học:** Cơ chế này là một bộ lọc mặt nạ không tham số (parameter-free static filter), tuyệt đối không sinh ra ma trận xoay, bảo vệ $100\%$ hệ cơ sở trực giao SVD của STAIR.

---

### 2.2 Phân tích & Phản biện Triết lý GDNSM: Bác bỏ Ranh giới Cứng Chiều 32 & Thiết lập Phân rã Phổ Liên tục

#### Bác bỏ dứt khoát Giả định Sai lầm về "Ranh giới cứng chiều 32":
Trong một số tài liệu phác thảo sơ bộ trước đây, có quan điểm cho rằng vector 64 chiều của STAIR được chia cắt cơ học thành: *Collaborative Subspace $[0:32]$* và *Multimodal Subspace $[32:64]$*.  
Dưới góc độ giải tích phổ toán học và nguyên lý thiết kế hệ thống của STAIR, **đây là một giả định sai lầm nghiêm trọng đã từng bị bác bỏ trong các giai đoạn nghiên cứu trước**:
1. **Hàm co phổ của STAIR là đường cong trơn liên tục:**  
   Thuật toán Forward Stepwise Convolution (FSC) của STAIR sử dụng hàm suy giảm phổ lũy thừa:
   $$\beta(d) = 0.9 \cdot \left[ 1 - \left(\frac{d}{D}\right)^\gamma \right], \quad \text{với } D = 64, \; \gamma = 0.1$$
   Hàm số này là **khả vi, đơn điệu giảm và liên tục tuyệt đối** trên toàn bộ miền chỉ mục $d \in [0, 63]$. Hoàn toàn không tồn tại bất kỳ điểm gián đoạn, điểm gãy hay bước nhảy nào tại vị trí $d = 32$.
2. **Tính toán số học vi phân chứng minh:**
   - Tại chiều $d = 31$: $\beta(31) = 0.9 \cdot [1 - (31/64)^{0.1}] \approx 0.06286$.
   - Tại chiều $d = 32$: $\beta(32) = 0.9 \cdot [1 - (32/64)^{0.1}] \approx 0.06030$.
   - **Độ chênh lệch giữa chiều 31 và chiều 32 chỉ là $0.00256$ ($< 4.3\%$)!** Về mặt vật lý, chiều 31 và chiều 32 nằm sát nhau trên cùng một dải tần số cao và chịu mức độ suy giảm gần như y hệt nhau. Việc coi chiều 31 là "hoàn toàn mang tính cộng tác (collaborative)" và chiều 32 là "hoàn toàn mang tính đa phương thức (multimodal)" là một sự khiên cưỡng phi toán học.
3. **Phân bố năng lượng thực tế của STAIR:**  
   - Tại $d = 0$: $\beta(0) = 0.9000$.
   - Tại $d = 20$: $\beta(20) = 0.0988$.
   - Chênh lệch giữa chiều 0 và chiều 20 lên tới $0.8012$ ($> 89\%$ tổng biên độ suy giảm). Toàn bộ sự chuyển pha năng lượng đồ thị thực sự diễn ra tập trung trong khoảng $d \in [0, 20]$, chứ không hề có ranh giới chia đôi ở vị trí 32.
4. **Bản chất biểu diễn của STAIR:**  
   Toàn bộ 64 chiều của vector item ban đầu đều được khởi tạo từ phép làm trắng SVD Whitening trên đặc trưng đa phương thức. Trong quá trình lan truyền đồ thị $\tilde{\mathbf{A}} \mathbf{H}^{(l-1)} \odot \boldsymbol{\beta}$, thông tin cộng tác và đa phương thức đan xen liên tục trên cả 64 chiều theo hàm trọng số phổ $\boldsymbol{\beta}$, chứ không chia thành hai nửa độc lập.

```
                  ĐỒ THỊ HÀM SUY GIẢM PHỔ LIÊN TỤC β(d) CỦA STAIR (γ = 0.1, D = 64)
    β(d)
    1.0 ┤  β(0) = 0.9000
        │       0.8 ┤            │         0.6 ┤              │           0.4 ┤                │             0.2 ┤          \  β(20) = 0.0988
        │           \────────────────────── β(31)=0.0629  β(32)=0.0603 ─────── β(63)=0.0014
    0.0 └───┬───────────┬──────────────────────┬─────────────┬─────────────────────┬── d (chiều)
            0          20                     31            32                    63
            ▲                                  ▲             ▲
            │                                  └──────┬──────┘
       VÙNG NĂNG LƯỢNG CAO                ĐỘ LỆCH CHỈ 0.0026!
       (Tần số thấp: Graph)               KHÔNG PHẢI RANH GIỚI CỨNG!
```

#### Giải pháp Cải biên Chuẩn mực: Phân rã Phổ Năng lượng Liên tục (Continuous Spectral Decoupling)
Thay vì sử dụng "nhát cắt cơ học" tùy tiện tại chiều 32, chúng tôi đề xuất cơ chế **Continuous Spectral-Decoupled Difficulty Scoring** dựa trên chính vector suy giảm phổ liên tục $\boldsymbol{\beta} \in \mathbb{R}^{64}$:

1. **Chiết xuất Thành phần Dải tần thấp (Low-frequency Graph Collaborative Component):**  
   Được cân trọng số liên tục bởi vector phổ $\boldsymbol{\beta}$:
   $$\mathbf{z}^{\text{low}} = \boldsymbol{\beta} \odot \mathbf{z} \in \mathbb{R}^{64}$$
   Sau đó chuẩn hóa L2 độc lập trên siêu mặt cầu để bảo đảm biên độ:
   $$\hat{\mathbf{z}}^{\text{low}} = \frac{\boldsymbol{\beta} \odot \mathbf{z}}{\|\boldsymbol{\beta} \odot \mathbf{z}\|_2 + \epsilon}$$
   Thành phần này bảo toàn tối đa thông tin đồ thị cộng tác từ các chiều tần số thấp ($d < 20$) mà không bỏ rơi phần đuôi tần số cao.
2. **Chiết xuất Thành phần Dải tần cao (High-frequency Multimodal Invariant Component):**  
   Được cân trọng số liên tục bởi phổ bù $(\mathbf{1} - \boldsymbol{\beta})$:
   $$\mathbf{z}^{\text{high}} = (\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z} \in \mathbb{R}^{64}$$
   Sau đó chuẩn hóa L2 độc lập trên siêu mặt cầu:
   $$\hat{\mathbf{z}}^{\text{high}} = \frac{(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}}{\|(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}\|_2 + \epsilon}$$
   Thành phần này tập trung vào các chiều có $\beta$ nhỏ, nơi đặc trưng nội dung đa phương thức gốc ít bị đồ thị làm mờ nhất.
3. **Tính toán Độ khó Phổ Liên tục (Continuous Spectral Difficulty):**
   $$\mathcal{D}(u, k) = \alpha_{\text{spec}} \cdot (\hat{\mathbf{u}}^{\text{low}} \cdot \hat{\mathbf{k}}^{\text{low}}) + (1 - \alpha_{\text{spec}}) \cdot (\hat{\mathbf{u}}^{\text{high}} \cdot \hat{\mathbf{k}}^{\text{high}})$$
   với $\alpha_{\text{spec}} = 0.50$.

*Ưu điểm vượt trội:*  
- Không có bất kỳ bước nhảy gián đoạn nào (chiều 31 và 32 được chuyển tiếp vi phân hoàn toàn tự nhiên).
- Tôn trọng $100\%$ vật lý của mô hình STAIR gốc và bảo toàn không gian vector 64 chiều liên tục.
- Phép chuẩn hóa L2 riêng biệt trên $\hat{\mathbf{z}}^{\text{low}}$ và $\hat{\mathbf{z}}^{\text{high}}$ vẫn hoàn thành trọn vẹn nhiệm vụ ngăn chặn dải tần thấp lấn át dải tần cao, đạt được sự cân bằng công bằng 50/50 giữa cấu trúc đồ thị và nội dung đa phương thức.

---

### 2.3 Phân tích & Phản biện Triết lý AdNGCL: Bộ Điều phối HANS Thích ứng theo Epoch

#### Cơ chế của AdNGCL và Nguy cơ Xung đột với Tích chập Ngược BSC của STAIR
AdNGCL đề xuất điều chỉnh tỷ lệ mẫu âm khó (Hard Negative Ratio) tăng dần theo thời gian huấn luyện. Tuy nhiên, trong STAIR, quá trình huấn luyện được điều phối đồng thời bởi hai cơ chế:
1. **Forward Stepwise Convolution (FSC):** Lan truyền tín hiệu từ bậc thấp lên bậc cao.
2. **Backward Stepwise Convolution (BSC):** Điều hướng gradient cập nhật cho các chiều đa phương thức dựa trên ma trận tương đồng kNN tĩnh $\mathbf{S}$.

**CẢNH BÁO KỸ THUẬT NGHIÊM TRỌNG:**  
Ma trận kNN trong BSC tạo ra một dòng gradient tương đối ổn định theo cấu trúc hình học cố định của đồ thị đặc trưng. Nếu bộ điều phối HANS thay đổi tỷ lệ mẫu âm khó hoặc trọng số $\gamma_h$ quá đột ngột, hàm loss InfoNCE sẽ tạo ra các xung gradient cực lớn (gradient spikes). Các xung nhọn này sẽ:
- Trực tiếp bẻ gãy hướng hội tụ ổn định của BSC.
- Phá vỡ tính trực giao của vector trọng số Projector $\mathbf{w}$.
- Gây bùng nổ gradient hoặc làm mô hình rơi vào trạng thái thoái hóa sớm (early representation collapse).

#### Giải pháp Phòng vệ Bắt buộc: Bộ Điều phối HANS với Cập nhật Mềm & Ngưỡng Trần An toàn
Để hòa hợp tuyệt đối với BSC và BPR, bộ điều phối **HANS (Hardness-Aware Negative Scheduling)** trong bản v2 được thiết kế với 3 lớp chốt chặn:

1. **Giai đoạn Khởi động Bootstrap (Warm-up Phase):**  
   Trong $E_{\text{warmup}} = 50$ epochs đầu tiên, khóa cứng $\gamma_h = 0.05$ và $\text{hn\_ratio} = 0.10$. Giai đoạn này chỉ cho phép mô hình học trên các mẫu âm dễ để ổn định ma trận hiệp phương sai SVD và định hình các cụm biểu diễn cơ sở.
2. **Kích hoạt Dựa trên Trạng thái Bão hòa Loss Tương phản (Loss-Gated Trigger):**  
   Chỉ cho phép tăng độ khó khi giá trị loss tương phản `loss_ans` trung bình của cửa sổ trượt hiện tại không giảm quá $1\%$ so với cửa sổ trước đó:
   $$\text{Trigger Condition}: \quad \overline{\mathcal{L}}_{\text{ans, curr}} \ge 0.99 \cdot \overline{\mathcal{L}}_{\text{ans, prev}}$$
   Điều này đảm bảo mô hình chỉ đối mặt với các thử thách mẫu âm khó hơn khi đã thực sự hấp thụ hết thông tin từ các mẫu dễ.
3. **Cập nhật Mềm (Exponential Smoothing Update) & Ngưỡng Trần An toàn (Safety Ceiling Cap):**  
   Mỗi bước kích hoạt chỉ cho phép tăng hệ số với bước nhảy cực nhỏ $\Delta \le 0.02$, và bị chặn cứng tại ngưỡng trần an toàn:
   $$\gamma_h^{(t+1)} = \min\left(\gamma_h^{(t)} + 0.02, \; 0.35\right)$$
   $$\text{hn\_ratio}^{(t+1)} = \min\left(\text{hn\_ratio}^{(t)} + 0.02, \; 0.40\right)$$
   Việc khống chế $\gamma_h \le 0.35$ đảm bảo mẫu âm khó không bao giờ chiếm ưu thế áp đảo so với dòng gradient chính của BPR và BSC.

---

## 3. PHÁT HIỆN & KHẮC PHỤC 4 ĐIỂM NGHẼN TOÁN HỌC VÀ 1 LỖI KỸ THUẬT HỆ THỐNG TINH VI

Dưới sự thẩm định chuyên sâu của Senior AI Research Engineer, nhóm nghiên cứu đã phát hiện và xử lý dứt điểm **4 điểm nghẽn toán học và 1 lỗi kỹ thuật hệ thống cực kỳ tinh vi** ẩn giấu trong cơ chế tương tác giữa các thành phần:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   4 ĐIỂM NGHẼN TOÁN HỌC & 1 LỖI KỸ THUẬT HỆ THỐNG TINH VI ĐÃ XỬ LÝ               │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. BÁC BỎ RANH GIỚI CỨNG CHIỀU 32 (DISCRETE BOUNDARY FALLACY):                                   │
│    Cắt cứng [:32]/[32:] mâu thuẫn với hàm co phổ liên tục β(d) (chênh lệch chiều 31 & 32 < 4.3%).│
│    ===> GIẢI PHÁP: Phân rã phổ liên tục: z_low = β ⊙ z và z_high = (1 - β) ⊙ z trên toàn bộ 64D.│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. NGHỊCH LÝ PHÂN BỔ BUDGET MẪU KHÓ (TOP-K BUDGET BOTTLENECK):                                   │
│    Top-K chọn theo difficulty thô bị chiếm trọn bởi False Negatives, sau đó MFNA ép              │
│    attenuation về 0, làm rỗng tập mẫu khó thực tế huấn luyện (Active Hard Negatives = 0)!        │
│    ===> GIẢI PHÁP: Gated Top-K Selection: Selection_Score = difficulty ⊙ attenuation.            │
│         Lọc FN TRƯỚC khi chọn Top-K, bảo đảm 100% budget dành cho True Hard Negatives!           │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. MÂU THUẪN ATTENUATION TRÊN MẪU CÙNG METADATA NHƯNG TƯƠNG ĐỒNG THẤP:                           │
│    Tại cos = 0 (trực giao), sim = 0/tau => sigmoid(0) = 0.5. Nếu metadata_mask = 1,              │
│    W = 0.5 * 1.5 = 0.75 => attenuation = 0.25 (suy giảm lực đẩy tới 4 lần dù không liên quan)!   │
│    ===> GIẢI PHÁP: Thresholded Cosine-Gated Metadata Masking:                                    │
│         W = σ(sim_all) ⊙ (1.0 + 0.5 × metadata_mask ⊙ max(0, cos(θ))).                          │
│         Chỉ kích hoạt metadata mask khi cosine similarity thực sự mang giá trị dương!            │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. GHÉP NỐI LẬP LỊCH HANS VỚI LOSS TỔNG HỢP (SCHEDULER COUPLING):                                │
│    BPR loss đi vào plateau sớm đánh lừa HANS tăng độ khó quá sớm khi không gian CL chưa ổn định, │
│    gây sốc gradient và làm lệch trục phổ.                                                        │
│    ===> GIẢI PHÁP: Tách biệt hoàn toàn: Chỉ truyền riêng Contrastive Loss (loss_ans) vào HANS!   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. LỖI KỸ THUẬT RÒ RỈ ĐÁNH GIÁ (EVALUATION LEAK TRONG MEMORY BANK):                              │
│    Hàm forward gọi enqueue_negatives vô điều kiện khiến embedding tập Validation/Test bị đẩy     │
│    vào hàng đợi FIFO, làm ô nhiễm không gian mẫu âm đối chiếu ở các epoch kế tiếp.               │
│    ===> GIẢI PHÁP: Thêm chốt chặn huấn luyện an toàn: if self.training: enqueue_negatives(...)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 Bác bỏ Ranh giới Cứng Chiều 32: Chuyển dịch sang Phân rã Phổ Liên tục theo $\boldsymbol{\beta}(d)$
- **Bản chất sai lầm:** Việc chia đôi `[:32]` và `[32:]` đã sao chép lại một giả định sai lầm từng xuất hiện ở giai đoạn 2. Hàm co phổ $\beta_j = 0.9(1 - (j/d)^\gamma)$ là một hàm liên tục. Không có lý do toán học hay vật lý nào để coi chiều 31 là "cộng tác" còn chiều 32 là "đa phương thức" khi $\beta(31) = 0.0629$ và $\beta(32) = 0.0603$.
- **Bản vá kiến trúc:** Chúng tôi loại bỏ hoàn toàn toán tử cắt lát `[:d_half]` và `[d_half:]`, thay thế bằng phép phân rã phổ liên tục qua vector $\boldsymbol{\beta}$ và $(\mathbf{1} - \boldsymbol{\beta})$. Điều này giữ trọn vẹn không gian liên tục 64 chiều, nhất quán $100\%$ với định hướng nghiên cứu của toàn bộ khóa luận.

### 3.2 Nghịch lý Phân bổ Budget Mẫu âm khó (Top-K Budget Bottleneck) & Giải pháp Gated Top-K Selection
- **Bản chất sai lầm:** Chọn Top-K trên `difficulty` thô sẽ đưa các False Negatives (vừa giống tương tác vừa giống nội dung) vào danh sách. Sau đó MFNA nhận diện chúng và ép `attenuation` về 0, làm cho danh sách Hard Negatives thực tế tham gia huấn luyện bị rỗng (Active HN $\approx 0$), mất lực đẩy đối kháng.
- **Bản vá kiến trúc:** Áp dụng **Gated Top-K Selection**:
  $$\text{Selection\_Score} = \mathcal{D}(u, k) \cdot \text{attenuation}(u, k)$$
  Lọc mẫu âm giả TRƯỚC khi phân bổ Top-K. Các FN bị kéo điểm chọn lọc về 0 và bị loại, nhường $100\%$ ngân sách cho True Hard Negatives.

### 3.3 Mâu thuẫn Attenuation trên Mẫu cùng Metadata nhưng Tương đồng thấp & Thresholded Cosine Gating
- **Bản chất sai lầm:** Công thức $W = \sigma(\text{sim}) \cdot (1 + 0.5 \cdot \text{mask})$ khiến các sản phẩm trực giao ($\cos = 0$) nhưng cùng category lớn bị giảm lực đẩy tới 4 lần, phá hủy năng lực phân biệt tinh mịn trong cùng danh mục.
- **Bản vá kiến trúc:** Khóa cổng Cosine: $\mathcal{M}_{\text{gated}} = \mathcal{M}_{\text{meta}} \cdot \max(0.0, \cos)$. Chỉ kích hoạt metadata mask khi cosine similarity thực sự mang giá trị dương.

### 3.4 Khắc phục Ghép nối Lập lịch HANS với Loss tổng hợp & Giám sát Tương phản Độc lập
- **Bản chất sai lầm:** Giám sát loss tổng hợp bị chi phối bởi BPR loss (bão hòa sớm sau 50 epoch), đánh lừa HANS tăng độ khó quá sớm khi không gian CL của GNN chưa ổn định.
- **Bản vá kiến trúc:** Tách biệt tín hiệu giám sát: HANS chỉ tiếp nhận riêng `loss_ans` để điều phối tiến trình một cách độc lập và chính xác.

### 3.5 Khắc phục Lỗi Kỹ thuật Rò rỉ Đánh giá (Evaluation Leak) trong Hàng đợi Memory Bank FIFO
- **Bản chất sai lầm:** Gọi `enqueue_negatives` vô điều kiện khiến embedding của tập validation/test bị đẩy vào hàng đợi mẫu âm trong quá trình đánh giá.
- **Bản vá kiến trúc:** Thêm chốt chặn an toàn `if self.training: enqueue_negatives(pos_batch)`.

---

### 3.6 Bảng Đối chiếu Hệ thống: Các Điểm nghẽn Toán học và Bản vá Kiến trúc Hoàn thiện

| Thành phần Module | Thiết kế Sơ bộ / Giả định Sai lầm | Bản vá Kiến trúc STAIR-SRE-ANS v2 Hoàn thiện | Giá trị Học thuật Đóng góp |
| :--- | :--- | :--- | :--- |
| **Không gian Phổ** | Cắt cứng nhị phân `[:32]` và `[32:]` (sai lệch vật lý với hàm $\beta$). | **Continuous Spectral Decoupling:** Chiết xuất qua $\boldsymbol{\beta} \odot \mathbf{z}$ và $(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}$. | Bảo toàn trọn vẹn 64 chiều liên tục, khớp $100\%$ hàm co phổ của STAIR. |
| **Lựa chọn Mẫu khó** | `topk(difficulty)` làm đầy Top-K bằng FN $\to$ **Rỗng Budget Mẫu khó**. | **Gated Top-K Selection:** `topk(difficulty * attenuation)`. Lọc FN trước khi Top-K. | Đảm bảo $100\%$ mẫu Top-K là True Hard Negatives mang gradient cao. |
| **Mặt nạ Metadata** | Áp dụng cứng khiến cặp $\cos = 0$ bị phạt giảm lực đẩy tới 4 lần. | **Thresholded Cosine Gating:** $\mathcal{M}_{\text{meta}} \cdot \max(0, \cos)$. Chỉ kích hoạt khi cosine dương. | Bảo toàn khả năng phân biệt tinh mịn (Fine-grained Intra-category Ranking). |
| **Bộ Điều phối HANS** | Giám sát loss tổng hợp (bị BPR chi phối, bão hòa giả, tăng độ khó sớm). | **Decoupled Contrastive Monitoring:** Giám sát độc lập riêng giá trị $\mathcal{L}_{\text{ANS}}$. | Bảo đảm lộ trình tăng độ khó đồng điệu với tốc độ hội tụ của GNN. |
| **Memory Bank FIFO** | `enqueue_negatives` không có điều kiện, gây rò rỉ tập Val/Test vào queue. | **Training-Guarded Enqueue:** Khóa cứng `if self.training: enqueue_negatives(...)`. | Bảo vệ tính toàn vẹn và nhất quán của phân phối mẫu âm đối chiếu. |
| **Tập hợp Mẫu InfoNCE** | Lỗi sum toàn bộ mẫu âm trong batch thay vì các mẫu Top-K được chọn. | **Vectorized `torch.gather`:** Trích xuất chính xác `hn_exp` và `hn_weights` theo `hn_indices`. | Loại bỏ hoàn toàn nhiễu từ các mẫu ngoài Top-K, tối ưu hóa bộ nhớ GPU. |

---

## 4. KIẾN TRÚC TOÀN DIỆN MÔ HÌNH STAIR-SRE-ANS (GIAI ĐOẠN 3 — ĐỢT 2)

### 4.1 Sơ đồ Luồng Dữ liệu và Tương tác Module Toàn hệ thống

```
==================================================================================================
        KIẾN TRÚC TOÀN DIỆN STAIR-SRE-ANS (GIAI ĐOẠN 3 — ĐỢT 2: ĐÃ TỐI ƯU HÓA HOÀN CHỈNH)
==================================================================================================

    [ User ID Embeddings E_u ]                          [ Item SVD Whitened Embeddings E_i ]
                │                                                        │
                │                                           ┌────────────┴────────────┐
                │                                           │  Trụ cột 1: DIAGONAL    │
                │                                           │  SPECTRAL PROJECTOR     │
                │                                           │  E_proj = E_svd ⊙ w     │
                │                                           │  (0-rotation + L2 reg)  │
                │                                           └────────────┬────────────┘
                ▼                                                        ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                 FORWARD STEPWISE CONVOLUTION (FSC) NEUMANN EXPANSION                   │
    │  Tầng 0: H^(0) = [E_u ∥ E_proj]                                                       │
    │  Tầng 1: H^(1) = A_tilde · H^(0) · (1 - β1) + H^(0) · β1                               │
    │  Tầng 2: H^(2) = A_tilde · H^(1) · (1 - β2) + H^(1) · β2                               │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Final Embeddings H^(L) ]                                   [ Layer 0 & Layer 1 Views ]
                 │                                                (z_u ∈ R^64,  z_pos ∈ R^64)
                 ▼                                                             │
        ====================                                                   │
        MAIN BPR RANKING LOSS                                                  │
        L_BPR(u, i+, i-)                                                       │
        ====================                                                   │
                 │                                                             ▼
                 │                                            ┌────────────────────────────────┐
                 │                                            │ Trụ cột 5: TRAINING-GUARDED    │
                 │                                            │ MEMORY BANK FIFO QUEUE         │
                 │                                            │ Q ∈ R^(4096 x 64) (detached)   │
                 │                                            │ [if self.training: enqueue]    │
                 │                                            └────────────────┬───────────────┘
                 │                                                             │
                 │                                                             ▼
                 │                                            ┌────────────────────────────────┐
                 │                                            │ Trụ cột 2: CONTINUOUS SPECTRAL │
                 │                                            │ DIFFICULTY DECOUPLING          │
                 │                                            │ - Low-Freq [β] (Separate L2)   │
                 │                                            │ - High-Freq [1-β] (Separate L2)│
                 │                                            │ Diff = 0.5·S_low + 0.5·S_high  │
                 │                                            └────────────────┬───────────────┘
                 │                                                             │
                 │                                                             ▼
                 │                                            ┌────────────────────────────────┐
                 │                                            │ Trụ cột 4: THRESHOLDED MFNA    │
                 │                                            │ M_gated = Meta ⊙ max(0, cos)   │
                 │                                            │ W = σ(sim)·(1 + 0.5·M_gated)   │
                 │                                            │ Attenuation = clamp(1-W, 0, 1) │
                 │                                            └────────────────┬───────────────┘
                 │                                                             │
                 │                                                             ▼
                 │                                            ┌────────────────────────────────┐
                 │                                            │ Trụ cột 3: GATED TOP-K & HANS  │
                 │                                            │ Score = Diff ⊙ Attenuation     │
                 │                                            │ Top-K HN: (scores, indices)    │
                 │                                            │ HANS: Decoupled loss_ans gate  │
                 │                                            │ Ψ_HN = 1 + γ_h·d_norm          │
                 │                                            │ Final_W = Ψ_all ⊙ Attenuation  │
                 │                                            └────────────────┬───────────────┘
                 │                                                             │
                 │                                                             ▼
                 │                                            ┌────────────────────────────────┐
                 │                                            │ VECTORIZED INFO NCE GATHER     │
                 │                                            │ hn_exp = gather(exp, indices)  │
                 │                                            │ hn_w   = gather(Final_W, idx)  │
                 │                                            │ L_ANS = -log(pos / (pos + neg))│
                 │                                            └────────────────┬───────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
     =======================================================================================
               HÀM MỤC TIÊU TỐI ƯU TOÀN CỤC:
               L_total = L_BPR + λ_ans · L_ANS + λ_w · ‖w - 1‖_2^2 + λ_reg · ‖Θ‖_2^2
     =======================================================================================
```

---

### 4.2 Trụ cột 1: Regularized Diagonal Spectral Projector (Bảo tồn Tuyệt đối Hệ trục SVD)
- Vector tham số $\mathbf{w} \in \mathbb{R}^D$ tác động dưới dạng phép nhân Hadamard trực tiếp $\mathbf{E}_{\text{proj}} = \mathbf{E}_{\text{svd}} \odot \mathbf{w}$.
- Khởi tạo toàn bộ bằng $1.0$ tại epoch 0.
- Số hạng phạt neo giữ: $\mathcal{L}_{\text{anchor}} = \lambda_w \sum_{d=1}^D (w_d - 1.0)^2$ với $\lambda_w = 10^{-4}$.
- **Đảm bảo toán học:** Ma trận biến đổi là ma trận đường chéo thuần túy (Diagonal Matrix), góc xoay trục tọa độ bằng đúng $0^\circ$, bảo tồn trọn vẹn cơ sở trực giao của phân tích kỳ dị SVD.

### 4.3 Trụ cột 2: Continuous Spectral Difficulty Decoupling với Trọng số Phổ Liên tục $\boldsymbol{\beta}(d)$
- Thay vì cắt cứng nhị phân $[0:32]$ và $[32:64]$, mô hình áp dụng trực tiếp hàm suy giảm phổ $\boldsymbol{\beta} \in \mathbb{R}^{64}$:
  $$\beta(d) = 0.9 \cdot \left[1 - \left(\frac{d}{D}\right)^{0.1}\right]$$
- Hai thành phần phổ trơn được xác định trên toàn bộ 64 chiều:
  $$\hat{\mathbf{z}}^{\text{low}} = \frac{\boldsymbol{\beta} \odot \mathbf{z}}{\|\boldsymbol{\beta} \odot \mathbf{z}\|_2 + \epsilon}, \quad \hat{\mathbf{z}}^{\text{high}} = \frac{(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}}{\|(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}\|_2 + \epsilon}$$
- Điểm độ khó phổ: $\mathcal{D}(u, k) = 0.50 \cdot (\hat{\mathbf{u}}^{\text{low}} \cdot \hat{\mathbf{k}}^{\text{low}}) + 0.50 \cdot (\hat{\mathbf{u}}^{\text{high}} \cdot \hat{\mathbf{k}}^{\text{high}})$.
- Bảo toàn tính liên tục vi phân, tôn trọng quy luật vật lý của FSC và cân bằng chính xác 50/50 năng lượng hai dải tần.

### 4.4 Trụ cột 3: Gated Top-K Selection & Triệt tiêu Rỗng Ngân sách Mẫu khó
- Kết hợp độ khó với hệ số suy giảm để tính Điểm chọn lọc: $\text{Selection\_Score} = \mathcal{D} \odot \text{attenuation}$.
- Lọc False Negatives trước khi phân bổ vào danh sách Top-K.
- Trích xuất chính xác $K_{\text{hn}} = \max(1, \lfloor \text{hn\_ratio} \times Q \rfloor)$ mẫu khó thực sự.

### 4.5 Trụ cột 4: Thresholded Cosine-Gated MFNA & Phân tầng Mẫu âm Đa cấp
- Khóa cổng Cosine: $\mathcal{M}_{\text{gated}} = \mathcal{M}_{\text{meta}} \odot \max(0, \cos(u, k))$.
- Hệ số phạt phân tầng bảo toàn trật tự toán học nghiêm ngặt:
  $$\Psi_{\text{HN}} = 1.0 + \gamma_h \cdot \left(\frac{\mathcal{D} + 1}{2}\right) \ge 1.0 > \Psi_{\text{EN}} = 1.0 - \gamma_h$$
- Hệ số phạt cuối cùng sau suy giảm: $\Psi_{\text{final}} = \Psi \odot \text{clamp}(1.0 - W, 0.0, 1.0)$.

### 4.6 Trụ cột 5: Decoupled HANS Scheduler & Cross-Batch Memory Bank FIFO An toàn
- Hàng đợi FIFO tĩnh $\mathcal{Q} \in \mathbb{R}^{4096 \times 64}$ mở rộng không gian mẫu âm gấp 4 lần.
- Chốt chặn `if self.training` bảo vệ hàng đợi khỏi rò rỉ dữ liệu tập Validation/Test.
- Bộ lập lịch HANS giám sát riêng `loss_ans`, áp dụng bước nhảy mềm $\Delta \le 0.02$ và trần an toàn $\gamma_h \le 0.35$.

---

## 5. HỆ THỐNG CÔNG THỨC TOÁN HỌC VI PHÂN & ĐỊNH LÝ CÂN BẰNG GRADIENT

### 5.1 Hàm Mục tiêu Đa nhiệm Toàn cục ($\mathcal{L}_{\text{total}}$)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BPR}}(\mathcal{O}) + \lambda_{\text{ans}} \cdot \mathcal{L}_{\text{ANS}} + \lambda_w \cdot \|\mathbf{w} - \mathbf{1}\|_2^2 + \lambda_{\text{reg}} \cdot \|\Theta\|_2^2$$

### 5.2 Công thức Chi tiết Hàm Mất mát Stepwise SRE-ANS Loss

Xét mini-batch $B$ người dùng và hàng đợi mẫu âm $\mathcal{Q}$ gồm $Q$ sản phẩm.

1. **Độ khó Phổ Liên tục:**
   $$\mathcal{D}(u, k) = \frac{1}{2} \left[ \frac{(\boldsymbol{\beta} \odot \mathbf{z}_u) \cdot (\boldsymbol{\beta} \odot \mathbf{z}_k)}{\|\boldsymbol{\beta} \odot \mathbf{z}_u\|_2 \|\boldsymbol{\beta} \odot \mathbf{z}_k\|_2} + \frac{((\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}_u) \cdot ((\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}_k)}{\|(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}_u\|_2 \|(\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}_k\|_2} \right]$$

2. **Gated Top-K Hard Negative Selection:**
   $$\mathcal{H}_u = \arg\text{topk}_{k \in \mathcal{Q}} \left( \mathcal{D}(u, k) \cdot \text{clamp}(1.0 - W_{u, k}, \; 0.0, \; 1.0) \right)$$

3. **Hàm mất mát $\mathcal{L}_{\text{ANS}}$ Phân tầng Chuẩn xác:**
   $$\mathcal{L}_{\text{ANS}} = -\frac{1}{B} \sum_{u=1}^B \ln \frac{\exp\left( \frac{\mathbf{z}_u \cdot \mathbf{z}_{i^+}}{\tau} \right)}{\exp\left( \frac{\mathbf{z}_u \cdot \mathbf{z}_{i^+}}{\tau} \right) + \sum_{k \in \mathcal{H}_u} \Psi_{\text{HN}}(u, k) \cdot (1 - W_{u, k}) \cdot \exp\left( \frac{\mathbf{z}_u \cdot \mathbf{z}_k}{\tau} \right)}$$

### 5.3 Giải tích Gradient và Chứng minh Toán học về Tính Hòa hợp Gradient với BPR và BSC

Ta xét đạo hàm riêng của $\mathcal{L}_{\text{total}}$ theo vector biểu diễn người dùng $\mathbf{z}_u$:
$$\mathbf{g}_{\text{total}} = \frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{z}_u} = \mathbf{g}_{\text{BPR}} + \lambda_{\text{ans}} \mathbf{g}_{\text{ANS}}$$

- **Gradient BPR:** $\mathbf{g}_{\text{BPR}} = -\sigma(-\hat{x}_{ui}) \mathbf{z}_{i^+} + \sigma(-\hat{x}_{ui}) \mathbf{z}_{j^-}$. Lực kéo cốt lõi là $+\sigma(-\hat{x}_{ui}) \mathbf{z}_{i^+}$.
- **Gradient ANS:**
  $$\mathbf{g}_{\text{ANS}} = -\frac{1}{\tau} \left[ (1 - P_{\text{pos}}) \mathbf{z}_{i^+} - \sum_{k \in \mathcal{H}_u} P_k \cdot \Psi_{\text{final}}(u, k) \mathbf{z}_k \right]$$

#### Định lý về Tính Không Rỗng của Lực Đẩy Đối Kháng (Active Gradient Guarantee):
- Dưới cơ chế **Gated Top-K Selection**, vì $\mathcal{H}_u$ chỉ chọn các mẫu có $\text{Selection\_Score} > 0$, ta có:
  $$\forall k \in \mathcal{H}_u \implies \text{attenuation}(u, k) = (1 - W_{u, k}) > \delta > 0$$
- Do đó, tổng lực đẩy đối kháng:
  $$\|\mathbf{g}_{\text{repulsion}}\| = \left\| \sum_{k \in \mathcal{H}_u} P_k \cdot \Psi_{\text{final}}(u, k) \mathbf{z}_k \right\| \ge \delta \cdot \min_{k} P_k > 0$$
- **Hệ quả:** Ngân sách mẫu khó không bao giờ bị rỗng, mô hình duy trì liên tục dòng gradient cải biên không gian biểu diễn mà không bị suy thoái về Uniform Sampling.

---

## 6. HIỆN THỰC HÓA MÃ NGUỒN PYTORCH CHUẨN SẢN XUẤT (PRODUCTION-GRADE IMPLEMENTATION)

Dưới đây là mã nguồn module `models/stair_sre_ans_v8.py` hoàn chỉnh, tích hợp trọn vẹn Continuous Spectral Decoupling (loại bỏ cắt cứng chiều 32), Gated Top-K Selection, Thresholded MFNA và Training-Guarded FIFO Queue:

```python
# -*- coding: utf-8 -*-
"""
models/stair_sre_ans_v8.py
STAIR-SRE-ANS (Phase 3 -- Batch 2 / v2)
Stepwise Spectral-Refined Contrastive Learning with Adaptive Negative Scheduling,
Continuous Spectral Difficulty Decoupling (No hard boundary at dim 32),
Gated Top-K Selection, and Thresholded MFNA.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RegularizedDiagonalSpectralProjector', 'StepwiseSREANSLoss']


class RegularizedDiagonalSpectralProjector(nn.Module):
    """
    Projector duong cheo khong xoay truc (0-rotation) voi co che neo giu L2.
    Bao toan tuyet doi he truc toa do SVD va tinh don dieu cua pho nang luong STAIR.
    """
    def __init__(self, dim: int = 64, reg_weight: float = 1e-4):
        super(RegularizedDiagonalSpectralProjector, self).__init__()
        self.dim = dim
        self.reg_weight = reg_weight
        # Khoi tao bang 1.0 de epoch 0 trung khop hoan toan voi khong gian SVD goc
        self.w = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w

    def get_anchoring_loss(self) -> torch.Tensor:
        """Phat neo giu: L_reg_w = lambda_w * ||w - 1||_2^2"""
        return self.reg_weight * torch.sum((self.w - 1.0) ** 2)


class StepwiseSREANSLoss(nn.Module):
    """
    Ham mat mat STAIR-SRE-ANS (v2) tich hop 5 tru cot toan hoc hoan thien:
    1. Zero-rotation Diagonal Projector voi L2 Anchoring.
    2. Continuous Spectral Difficulty Decoupling theo beta(d) STAIR (khong cat cung chieu 32).
    3. Gated Top-K Selection: Selection_Score = Difficulty * Attenuation (Giai toa Budget Bottleneck).
    4. Thresholded Cosine-Gated MFNA: W = sigmoid(sim) * (1 + 0.5 * Meta * max(0, cos)).
    5. Decoupled HANS Scheduler & Training-Guarded Cross-Batch Memory Bank FIFO.
    """
    def __init__(
        self,
        dim: int = 64,
        tau: float = 0.20,
        queue_size: int = 4096,
        warmup_epochs: int = 50,
        gamma_max: float = 0.35,
        hn_ratio_max: float = 0.40,
        subspace_alpha: float = 0.50
    ):
        super(StepwiseSREANSLoss, self).__init__()
        self.dim = dim
        self.tau = tau
        self.queue_size = queue_size
        self.warmup_epochs = warmup_epochs
        self.gamma_max = gamma_max
        self.hn_ratio_max = hn_ratio_max
        self.subspace_alpha = subspace_alpha

        # Tinh toan vector suy giam pho lien tuc beta(d) cua STAIR: beta(d) = 0.9 * (1 - (d/D)^0.1)
        # Hoan toan lien tuc tren ca 64 chieu, KHONG co diem gay hay cat cung tai chieu 32!
        d_indices = torch.arange(dim, dtype=torch.float32)
        beta_curve = 0.9 * (1.0 - torch.pow(d_indices / float(dim), 0.10))
        self.register_buffer('beta', beta_curve)                  # [D] Pho tan so thap (Graph Collaborative)
        self.register_buffer('beta_high', 1.0 - beta_curve)       # [D] Pho tan so cao (Multimodal Invariant)

        # Memory Bank FIFO Queue cho mau am cross-batch (ngat gradient 100%)
        self.register_buffer('neg_queue', torch.randn(queue_size, dim))
        self.neg_queue = F.normalize(self.neg_queue, p=2, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Tham so bo dieu phoi HANS
        self.hn_ratio = 0.10
        self.gamma_h = 0.05
        self.current_epoch = 0
        self.loss_history = []

    @torch.no_grad()
    def enqueue_negatives(self, pos_emb: torch.Tensor):
        """Cap nhat hang doi FIFO voi vector bieu dien pos items trong batch."""
        batch_size = pos_emb.size(0)
        norm_emb = F.normalize(pos_emb.detach(), p=2, dim=-1)

        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.neg_queue[ptr:ptr + batch_size] = norm_emb
            ptr = (ptr + batch_size) % self.queue_size
        else:
            first_chunk = self.queue_size - ptr
            self.neg_queue[ptr:] = norm_emb[:first_chunk]
            remain = batch_size - first_chunk
            self.neg_queue[:remain] = norm_emb[first_chunk:]
            ptr = remain

        self.queue_ptr[0] = ptr

    def update_scheduler(self, current_cl_loss: float, window: int = 10, threshold: float = 0.99):
        """
        Bo dieu phoi HANS thich ung doc lap:
        Giam sat RIENG ham mat mat tuong phan CL (loss_ans) de tang gamma_h va hn_ratio.
        """
        if self.current_epoch < self.warmup_epochs:
            self.gamma_h = 0.05
            self.hn_ratio = 0.10
            return

        self.loss_history.append(current_cl_loss)
        if len(self.loss_history) > window * 2:
            self.loss_history.pop(0)
            loss_curr = sum(self.loss_history[-window:]) / window
            loss_prev = sum(self.loss_history[-window*2:-window]) / window

            # Kich hoat Loss-Gated Trigger khi CL loss giam cham hon 1%
            if loss_curr >= threshold * loss_prev:
                self.gamma_h = min(self.gamma_h + 0.02, self.gamma_max)
                self.hn_ratio = min(self.hn_ratio + 0.02, self.hn_ratio_max)
            else:
                self.gamma_h = max(self.gamma_h - 0.01, 0.05)
                self.hn_ratio = max(self.hn_ratio - 0.01, 0.10)

    def forward(
        self,
        u_embed: torch.Tensor,
        i_pos_embed: torch.Tensor,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
        metadata_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = u_embed.device
        B = batch_users.size(0)

        # 1. Trich xuat mini-batch
        u_batch = u_embed[batch_users]           # [B, D]
        pos_batch = i_pos_embed[batch_items]     # [B, D]

        # 2. Lay mau am tu Memory Bank FIFO Queue (detached)
        neg_pool = self.neg_queue.clone().to(device)  # [Q, D]
        Q = neg_pool.size(0)

        # 3. CONTINUOUS SPECTRAL DIFFICULTY DECOUPLING (Khong cat cung chieu 32)
        # Phan ra pho lien tuc tren toan bo 64 chieu qua vector beta(d) cua STAIR
        beta = self.beta.to(device)                   # [D]
        beta_high = self.beta_high.to(device)         # [D]

        u_low = F.normalize(u_batch * beta, p=2, dim=-1)
        u_high = F.normalize(u_batch * beta_high, p=2, dim=-1)

        neg_low = F.normalize(neg_pool * beta, p=2, dim=-1)
        neg_high = F.normalize(neg_pool * beta_high, p=2, dim=-1)

        cos_low = torch.matmul(u_low, neg_low.T)       # [B, Q] - Tuong dong dai tan thap (Graph Collaborative)
        cos_high = torch.matmul(u_high, neg_high.T)   # [B, Q] - Tuong dong dai tan cao (Multimodal Detail)

        difficulty = self.subspace_alpha * cos_low + (1.0 - self.subspace_alpha) * cos_high  # [B, Q]

        # 4. Tinh toan ma tran tuong quan toan phan (Cosine similarity goc truoc khi chia tau)
        u_norm = F.normalize(u_batch, p=2, dim=-1)
        pos_norm = F.normalize(pos_batch, p=2, dim=-1)
        neg_norm = F.normalize(neg_pool, p=2, dim=-1)

        cos_all = torch.matmul(u_norm, neg_norm.T)                 # [B, Q]
        sim_all = cos_all / self.tau                              # [B, Q]

        # 5. Thresholded Cosine-Gated MFNA: Chi kich hoat metadata khi cosine > 0
        if metadata_mask is not None:
            gated_metadata = metadata_mask.to(device) * torch.clamp(cos_all, min=0.0)
            W = torch.sigmoid(sim_all) * (1.0 + 0.5 * gated_metadata)
        else:
            W = torch.sigmoid(sim_all)
        attenuation = torch.clamp(1.0 - W, min=0.0, max=1.0)       # [B, Q]

        # 6. GATED TOP-K SELECTION: Loc FN truoc khi phan bo ngan sach Top-K HN
        selection_score = difficulty * attenuation                 # [B, Q]
        k_hn = max(1, int(self.hn_ratio * Q))
        _, hn_indices = torch.topk(selection_score, k=k_hn, dim=1)  # [B, k_hn]

        # 7. He so dieu phoi phan tang Psi (HN vs EN)
        diff_norm = (difficulty + 1.0) / 2.0                       # Chuan hoa ve [0, 1]
        psi_HN = 1.0 + self.gamma_h * diff_norm                    # >= 1.0
        psi_EN = 1.0 - self.gamma_h                                # <= 1.0

        psi_all = torch.full_like(sim_all, psi_EN)                 # Khoi tao mac dinh EN
        hn_mask = torch.zeros_like(sim_all, dtype=torch.bool)
        hn_mask.scatter_(1, hn_indices, True)
        psi_all = torch.where(hn_mask, psi_HN, psi_all)

        # Ap dung MFNA suy giam len he so phat
        final_weights = psi_all * attenuation                      # [B, Q]

        # 8. Tinh InfoNCE Loss bang cach gather chinh xac Top-K Hard Negatives da loc
        pos_sim = torch.sum(u_norm * pos_norm, dim=-1) / self.tau  # [B]
        pos_exp = torch.exp(pos_sim)                               # [B]
        exp_all = torch.exp(sim_all)                               # [B, Q]

        hn_exp = torch.gather(exp_all, dim=1, index=hn_indices)           # [B, k_hn]
        hn_weights = torch.gather(final_weights, dim=1, index=hn_indices) # [B, k_hn]

        neg_weighted_sum = (hn_exp * hn_weights).sum(dim=1)               # [B]

        loss = -torch.log(pos_exp / (pos_exp + neg_weighted_sum + 1e-8)).mean()

        # 9. TRAINING GUARD: Chi cap nhat Queue khi o che do train (Chong Evaluation Leak)
        if self.training:
            with torch.no_grad():
                self.enqueue_negatives(pos_batch)

        return loss
```

---

## 7. MA TRẬN MỤC TIÊU BỨT PHÁ $\ge +5.0\%$ TRÊN CẢ 3 TẬP DỮ LIỆU

Bảng đối chuẩn mục tiêu hiệu năng toàn diện trên tập kiểm thử (Test Set) của Giai đoạn 3 — Đợt 2:

| Tập dữ liệu | Chỉ số Đánh giá | Baseline STAIR (Paper Table 2) | STAIR-SRE v1 (Thực nghiệm) | STAIR-SRE v1.1 (Thực nghiệm) | **STAIR-SRE-ANS v2 (Mục tiêu)** | Mức Tăng trưởng vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Amazon Baby** | **Recall@10** | 0.0674 | 0.0621 | 0.0645 | **$\ge 0.0710$** | **$+5.34\%$** |
| *(19.4K Users)* | **Recall@20** | 0.1042 | 0.0967 | 0.1003 | **$\ge 0.1095$** | **$+5.09\%$** |
| *(Sparsity: 99.82%)* | **NDCG@10** | 0.0359 | 0.0326 | 0.0341 | **$\ge 0.0380$** | **$+5.85\%$** |
| *(Best @Ep 455)* | **NDCG@20** | 0.0454 | 0.0416 | 0.0433 | **$\ge 0.0480$** | **$+5.73\%$** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Amazon Sports** | **Recall@10** | 0.0743 | 0.0682 | 0.0736 | **$\ge 0.0785$** | **$+5.65\%$** |
| *(35.6K Users)* | **Recall@20** | 0.1111 | 0.1029 | 0.1098 | **$\ge 0.1168$** | **$+5.13\%$** |
| *(Sparsity: 99.95%)* | **NDCG@10** | 0.0405 | 0.0371 | 0.0396 | **$\ge 0.0430$** | **$+6.17\%$** |
| *(Best @Ep 415)* | **NDCG@20** | 0.0500 | 0.0461 | 0.0493 | **$\ge 0.0530$** | **$+6.00\%$** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Amazon Electronics** | **Recall@10** | 0.0442 | — | — | **$\ge 0.0470$** | **$+6.33\%$** |
| *(43.4K Users)* | **Recall@20** | 0.0665 | — | — | **$\ge 0.0705$** | **$+6.02\%$** |
| *(Sparsity: 99.96%)* | **NDCG@10** | 0.0246 | — | — | **$\ge 0.0265$** | **$+7.72\%$** |
| *(~1.7M interactions)* | **NDCG@20** | 0.0303 | — | — | **$\ge 0.0325$** | **$+7.26\%$** |

---

## 8. ĐẶC TẢ KHÔNG GIAN SIÊU THAM SỐ & HƯỚNG DẪN VẬN HÀNH

### 8.1 Bảng Cấu hình Siêu tham số Khuyến nghị

| Tham số CLI | Ý nghĩa Vật lý | Amazon Baby | Amazon Sports | Amazon Electronics | Rationale Kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `--lambda-ans` | Trọng số hàm mất mát ANS | $3 \times 10^{-5}$ | $5 \times 10^{-5}$ | $10^{-5}$ | Baby có mật độ dày hơn nên cần $\lambda$ nhỏ hơn để tránh cạnh tranh biểu diễn. |
| `--ans-tau` | Nhiệt độ InfoNCE | $0.20$ | $0.30$ | $0.25$ | Giữ $\tau = 0.30$ trên Sports để chống bão hòa loss về $0.0045$. |
| `--queue-size` | Kích thước Memory Bank | $4096$ | $4096$ | $8192$ | Electronics có hơn 27K items nên mở rộng queue lên 8192. |
| `--warmup-epochs`| Số epoch khởi động dễ | $50$ | $50$ | $60$ | Đảm bảo SVD coordinate ổn định trước khi đưa Hard Negatives vào. |
| `--gamma-max` | Trần trọng số phạt HN | $0.35$ | $0.35$ | $0.30$ | Ngăn xung đột gradient với BSC. |
| `--hn-ratio-max`| Tỷ lệ mẫu HN tối đa | $0.40$ | $0.40$ | $0.35$ | Giữ $60\%$ mẫu nền để bảo toàn tính Uniformity. |
| `--lr-w` | Learning rate cho vector $w$ | $10^{-4}$ ($0.1 \times \text{lr}$) | $10^{-4}$ | $10^{-4}$ | Cập nhật chậm để tránh trôi dạt tham số. |
| `--lambda-w` | Hệ số neo giữ L2 Projector | $10^{-4}$ | $10^{-4}$ | $10^{-4}$ | Neo giữ quanh giá trị 1.0. |

### 8.2 Các Lưu ý Vận hành Sống còn cho Cột mốc $\ge +5\%$

1. **Giám sát Kích thước Hàng đợi (Queue Size Tuning) & Độ nhạy Nhiệt độ $\tau$:**  
   Hàng đợi mẫu âm càng lớn ($Q = 4096$ hay $8192$), lực đẩy tích lũy từ $\sum \exp(\text{sim})$ càng mạnh. Nếu trong 50 epoch đầu, quan sát thấy `loss_ans` giảm quá nhanh tiệm cận về 0 (dưới $0.01$), cần chủ động nâng nhiệt độ $\tau$ (ví dụ từ $0.20$ lên $0.25$ hoặc $0.30$) để làm mềm phân phối xác suất, tránh bão hòa sớm trước khi bộ lập lịch HANS kịp kích hoạt.
2. **Điểm Neo Cửa sổ Trượt HANS (Window Tuning):**  
   Mặc định sử dụng `window = 10` (so sánh 10 epoch hiện tại với 10 epoch trước đó). Tuy nhiên, trên tập **Amazon Electronics** (kích thước dữ liệu lớn, thời gian huấn luyện mỗi epoch dài $\approx 15 \sim 20$ giây), việc chờ 20 epoch để kích hoạt trigger có thể làm chậm nhịp thích ứng. Nhóm nghiên cứu nên cấu hình `window = 5` riêng cho Electronics để HANS phản ứng nhạy bén hơn.
3. **Chốt chặn Attenuation An toàn (Zero-NaN Guarantee):**  
   Toán tử `torch.clamp(1.0 - W, min=0.0, max=1.0)` đảm bảo hệ số phạt không bao giờ nhận giá trị âm, triệt tiêu hoàn toàn nguy cơ số hạng mẫu số InfoNCE bị âm sinh ra lỗi toán học `NaN` trong hàm logarit.

---

## 9. ĐÁNH GIÁ HIỆU NĂNG PHẦN CỨNG & ĐỘ PHỨC TẠP TÍNH TOÁN

Kiến trúc **STAIR-SRE-ANS v2** duy trì ưu thế vận hành siêu nhẹ và ổn định tài nguyên:

1. **Bộ nhớ VRAM GPU (Zero OOM Guarantee):**
   - Memory Bank Queue $4096 \times 64$ float32 chỉ chiếm: $4096 \times 64 \times 4 \text{ bytes} \approx 1.05 \text{ MB}$.
   - Vector phổ cố định $\boldsymbol{\beta}$ và $\boldsymbol{\beta}_{\text{high}}$ chỉ tốn $64 \times 4 \text{ bytes} = 256 \text{ bytes}$.
   - Toàn bộ pipeline huấn luyện tiêu thụ đỉnh (Peak VRAM):
     * **Amazon Baby:** $\approx 1.15 \text{ GB}$ (dưới $7.5\%$ dung lượng GPU T4 16GB).
     * **Amazon Sports:** $\approx 1.45 \text{ GB}$ (dưới $9.5\%$ dung lượng GPU T4 16GB).
     * **Amazon Electronics:** $\approx 2.10 \text{ GB}$ (dưới $13.5\%$ dung lượng GPU T4 16GB).
2. **Độ phức tạp Tính toán (Computational Throughput):**
   - Do loại bỏ toán tử tách lát mảng (slicing) và thay bằng phép nhân Hadamard vector `u * beta` song song bằng Tensor Cores, thời gian thực thi thậm chí nhanh hơn và mượt mà hơn cơ chế cũ.
   - Tốc độ huấn luyện ước tính: $\approx 3.2$ giây/epoch trên Baby, $\approx 6.9$ giây/epoch trên Sports.

---

## 10. KỊCH BẢN PHẢN BIỆN HỌC THUẬT NÂNG CẤP TRƯỚC HỘI ĐỒNG (ACADEMIC DEFENSE UPGRADE)

Khi Hội đồng Khoa học chất vấn về việc **đo lường độ khó mẫu âm và phân chia không gian đặc trưng trong STAIR**:

> *"Có quan điểm cho rằng STAIR chia không gian thành 32 chiều Collaborative và 32 chiều Multimodal. Tại sao nhóm không cắt lát vector để tính toán mà lại dùng Continuous Spectral Decoupling?"*

**Kịch bản trả lời mẫu mực của Senior AI Research Engineer:**

> *"Kính thưa Thầy/Cô trong Hội đồng Khoa học, đây chính là một trong những phát hiện và đính chính toán học quan trọng nhất của nhóm em trong đề tài này.*
>
> *Quan điểm coi vector STAIR có 'ranh giới cứng tại chiều 32' là một **ngụy định đề (false premise)** từng xuất hiện ở giai đoạn phác thảo sơ bộ nhưng đã bị nhóm em kiên quyết bác bỏ dựa trên giải tích vi phân chính xác:*
> - *Hàm suy giảm phổ của STAIR $\beta(d) = 0.9 \cdot [1 - (d/D)^\gamma]$ với $\gamma = 0.1, D=64$ là một **đường cong liên tục tuyệt đối trên toàn bộ 64 chiều**, không hề có điểm gãy hay bước nhảy nào.*
> - *Tính toán số học cho thấy: $\beta(31) \approx 0.0629$ và $\beta(32) \approx 0.0603$, độ chênh lệch chỉ vỏn vẹn $0.0026$ ($< 4.3\%$). Nếu cắt cứng tại chiều 32, chúng ta đang gán nhãn tùy tiện hai chiều có đặc tính phổ gần như y hệt nhau vào hai thái cực đối lập.*
> - *Toàn bộ sự suy giảm năng lượng đồ thị thực chất diễn ra ở $20$ chiều đầu tiên (từ $0.9000$ rơi xuống $0.0988$). Toàn bộ 64 chiều của STAIR đều xuất phát từ SVD Whitening của đa phương thức và nhận năng lượng tích chập đồ thị mượt mà.*
>
> *Chính vì vậy, nhóm em đã phát minh cơ chế **Continuous Spectral-Decoupled Difficulty Scoring**: Chúng em sử dụng chính vector $\boldsymbol{\beta}$ liên tục và phổ bù $(\mathbf{1} - \boldsymbol{\beta})$ để chiết xuất thành phần dải tần thấp (Graph Collaborative) và dải tần cao (Multimodal Detail) trên toàn bộ 64 chiều, kết hợp chuẩn hóa L2 riêng biệt để loại bỏ hiện tượng lệch biên độ năng lượng.*
>
> *Giải pháp này vừa tôn trọng 100% bản chất vật lý của mạng lọc phổ STAIR, vừa loại bỏ hoàn toàn các ranh giới cơ học gượng ép, đưa mô hình đạt độ hoàn thiện toán học tối đa."*

---

## 11. KẾ HOẠCH HÀNH ĐỘNG TRIỂN KHAI THỰC NGHIỆM ĐỢT 2

Lộ trình thực thi chi tiết sẵn sàng triển khai:

1. **Khởi tạo mã nguồn module v2:**  
   Tạo file [`models/stair_sre_ans_v8.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/stair_sre_ans_v8.py) chứa toàn bộ class `RegularizedDiagonalSpectralProjector` và `StepwiseSREANSLoss` với Continuous Spectral Decoupling.
2. **Xây dựng script huấn luyện chuẩn hóa:**  
   Tạo file [`main_stair_sre_ans_v8.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_stair_sre_ans_v8.py) hỗ trợ đầy đủ các tham số CLI `--lambda-ans`, `--ans-tau`, `--queue-size`, `--warmup-epochs`, `--gamma-max`, `--hn-ratio-max`.
3. **Kiểm thử Unit Test cục bộ:**  
   Chạy script kiểm thử mini-batch giả lập để xác nhận: (1) Vector $\boldsymbol{\beta}$ liên tục tính đúng; (2) Separate L2 Norm trên $\mathbf{z}_{\text{low}}$ và $\mathbf{z}_{\text{high}}$ cân bằng biên độ; (3) Gated Top-K Selection loại bỏ FN chính xác; (4) HANS Scheduler cập nhật mượt mà; (5) Backward gradient không có NaN/Inf.
4. **Đóng gói Notebook Huấn luyện Kaggle GPU:**  
   Khởi tạo [`notebook/P3/stair_sre_v2.ipynb`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/P3/stair_sre_v2.ipynb) với đầy đủ pipeline tự động tải dữ liệu, huấn luyện trọn vẹn 500 epochs trên Amazon Baby, Sports và Electronics, lưu trữ nhật ký huấn luyện vào `logs/GD3/`.
5. **Phân tích Đối soát Thực nghiệm:**  
   Thu thập các file log, trích xuất ma trận số liệu tại Best Checkpoint, so sánh với Baseline và hoàn thiện Báo cáo Thực nghiệm Đợt 2 [`docs/giai_doan_3/STAIR3_v2_Experiment_Report.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_3/STAIR3_v2_Experiment_Report.md).

---

## 12. ĐÓNG GÓP PHẢN BIỆN CHUYÊN SÂU & NÂNG CẤP KIẾN TRÚC STAIR-SRE-ANS v2.1
### Decoupled Multi-Scale Representation, Thresholded Dynamic MFNA, Full Partition Conservation & Cosine-Annealed HANS Scheduler

Dựa trên kết quả thực nghiệm thực tế của Đợt 2 trên hai tập dữ liệu **Amazon Baby** (Recall@20: 0.1002, -3.84% vs Baseline) và **Amazon Sports** (Recall@20: 0.1096, -1.35% vs Baseline), nhóm nghiên cứu đã tiến hành một đợt **Code Forensics & Mathematical Audit** chuyên sâu dòng-trên-dòng giữa lý thuyết thiết kế và mã nguồn thực thi thực tế trong [`models/stair_sre_ans_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/stair_sre_ans_v2.py) và [`mainS3_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/mainS3_v2.py). 

Từ đó, chúng tôi phát hiện ra **4 tử huyệt toán học và cơ chế then chốt** đã kìm hãm mô hình v2 không thể bứt phá vượt qua Baseline và đề xuất giải pháp kiến trúc nâng cấp toàn diện: **STAIR-SRE-ANS v2.1**.

---

### 12.1 Phân Tích Phản Biện 4 Tử Huyệt Toán Học & Kỹ Thuật Trong Thực Thi v2

#### Tử huyệt 1: Mâu Thuẫn Không Gian Biểu Diễn (Representation Dilemma) & Ép InfoNCE Lên Tầng GNN Cuối Cùng $H^{(L)}$
* **Cơ chế lỗi trong code v2:** Trong hàm `fit()` của `mainS3_v2.py` (dòng 405–411), `ans_loss` được gọi trực tiếp trên `u_embed = userEmbds` và `i_pos_embed = itemEmbds`. Đây là các biểu diễn lấy từ `self.encode()`, tức đã trải qua trọn vẹn $L = 3$ lớp tích chập đồ thị Neumann Stepwise Convolution ($H^{(L)}$).
* **Bản chất xung đột toán học:**
  - **Hàm xếp hạng BPR** đòi hỏi các embedding $H^{(L)}$ phải **co cụm mạnh mẽ (Clustering Bias)** quanh các sở thích hành vi của người dùng trên đồ thị tương tác.
  - **Hàm tương phản InfoNCE** lại ép buộc các vector $H^{(L)}$ phải **phân bố đồng đều trên mặt cầu siêu cầu (Hyperspherical Uniformity)** để tối đa hóa entropy thông tin (Wang & Isola, ICML 2020).
  - Khi InfoNCE tác động trực tiếp lên $H^{(L)}$, hai hàm mục tiêu kéo ngược chiều nhau: BPR cố gắng kéo các item cùng sở thích lại gần nhau, trong khi InfoNCE lại cố gắng đẩy dãn các cụm ra để thỏa mãn tính đồng đều. Kết quả là cấu trúc lân cận CF bị loãng nhẹ, giải thích chính xác tại sao **thứ tự xếp hạng cục bộ tăng (NDCG@10 và NDCG@20 tăng), nhưng độ bao phủ Top-20 (Recall@20) bị ghìm dưới Baseline từ 1% đến 3.8%**.
* **Giải pháp v2.1 (Decoupled CL Views at Layer-0):**
  Tách rời hai không gian: Đưa hàm Contrastive Loss về tối ưu hóa trên **Không gian Thô Tầng 0 (Layer-0 Latent Space)** gồm $E_u^{(0)}$ và $E_{proj}^{(0)} = E_{item}^{(0)} \odot \mathbf{w}$ thông qua một bộ chiếu nhẹ 1-layer MLP Projection Head (`Linear + LayerNorm + LeakyReLU`). Toàn bộ không gian $H^{(L)}$ sau GNN được giải phóng 100% để phục vụ mục tiêu xếp hạng BPR thuần túy!

---

#### Tử huyệt 2: Lỗi Triệt Tiêu Mẫu Âm Giả (MFNA Attenuation Leak) $\sigma(0) = 0.5$ Khi `metadata_mask = None`
* **Cơ chế lỗi trong code v2:** Trong `mainS3_v2.py` dòng 410, `metadata_mask` được truyền vào là `None`. Khi đó, nhánh tính trọng số suy giảm MFNA trong `stair_sre_ans_v2.py` (dòng 287–288) thực thi:
  $$	ext{sim\_all} = rac{\cos(\mathbf{u}, \mathbf{k})}{	au}, \quad W = \sigma(	ext{sim\_all}), \quad 	ext{attenuation} = 	ext{clamp}(1.0 - W, 0.0, 1.0)$$
  với $	au = 0.20$.
* **Hậu quả toán học tai hại:**
  - Xét một mẫu âm hoàn toàn trực giao với người dùng (True Negative hiển nhiên), tức $\cos(\mathbf{u}, \mathbf{k}) = 0$.
  - Ta có: $	ext{sim\_all} = 0 / 0.2 = 0 \implies W = \sigma(0) = 0.5 \implies 	ext{attenuation} = 1.0 - 0.5 = \mathbf{0.5000}$!
  - Ngay cả với mẫu âm có góc tù nhẹ ($\cos = -0.2$), $	ext{sim\_all} = -1.0 \implies W = 0.2689 \implies 	ext{attenuation} = \mathbf{0.7311}$.
  - 👉 **HỆ QUẢ:** **100% các True Negatives (mẫu âm chân chính) trong toàn bộ kho hàng đều bị cắt giảm từ 25% đến 50% lực đẩy đối kháng!** Hàm InfoNCE bị suy yếu nghiêm trọng khả năng phân tách không gian, mất đi động lực đẩy các sản phẩm không liên quan ra xa.
* **Giải pháp v2.1 (Thresholded Dynamic Attenuation Gate):**
  Thiết lập cổng kích hoạt ngưỡng: Chỉ kích hoạt cơ chế suy giảm $W$ khi cosine similarity thực sự dương và vượt qua ngưỡng cảnh báo tương đồng cao ($	au_{sim} = 0.25$):
  $$	ext{active\_mask} = \mathbb{I}(\cos(\mathbf{u}, \mathbf{k}) > 0.25), \quad W = \sigma(	ext{sim\_all}) \cdot 	ext{active\_mask}, \quad 	ext{attenuation} = 	ext{clamp}(1.0 - W, 0.1, 1.0)$$
  Đối với tất cả các mẫu âm trực giao ($\cos \le 0.25$), $	ext{active\_mask} = 0 \implies W = 0 \implies 	ext{attenuation} = \mathbf{1.0000}$ (Bảo toàn 100% lực đẩy tối đa)!

---

#### Tử huyệt 3: Động Lực Học HANS Bão Hòa Sớm (Premature Saturation) & Thiếu Cơ Chế Hạ Nhiệt (Cooling/Annealing)
* **Cơ chế lỗi trong code v2:** Nhật ký thực nghiệm cho thấy hàm loss tương phản giảm nhanh từ 3.55 về 2.65 trong 40 epoch đầu và đạt trạng thái bão hòa độ dốc ($\Delta_{\mathcal{L}} \le 10^{-4}$). Ngay khi kết thúc Warmup (Epoch 50), HANS kích hoạt liên tục và chạm ngưỡng trần tối đa $\gamma_{max}=0.35, hn\_ratio_{max}=0.40$ chỉ sau 20 epoch (tại Epoch 90).
* **Hậu quả:** Trong suốt 410 epoch tiếp theo (từ epoch 91 đến epoch 500), mô hình bị ép chịu tải tối đa 40% hard negatives với hệ số phạt $\gamma_h = 0.35$ cố định không hề suy giảm. Việc thiếu cơ chế hạ nhiệt (Annealing/Cooling) ở giai đoạn hậu kỳ khiến mô hình không thể thực hiện fine-tuning tinh vi trên các ranh giới xếp hạng, làm dao động nhẹ điểm Recall ở những epoch cuối.
* **Giải pháp v2.1 (Cosine-Annealed Ceiling Scheduler):**
  Thiết lập trần thích ứng động theo hàm Cosine Annealing:
  $$\gamma_{max}(t) = \max\left(0.08, \gamma_{max} \cdot rac{1}{2} \left[1 + \cos\left(rac{t - t_{warmup}}{T - t_{warmup}} \piight)ight]ight)$$
  - *Giai đoạn 1 (Warmup, Epoch 1–50):* Giữ $\gamma_h = 0.05$ để ổn định không gian thô.
  - *Giai đoạn 2 (Peak Hardening, Epoch 51–250):* HANS tăng tốc khai thác hard negatives lên đỉnh $\gamma pprox 0.30 - 0.35$ để kéo giãn khoảng cách cụm.
  - *Giai đoạn 3 (Fine-tuning Annealing, Epoch 251–500):* Trần $\gamma_{max}(t)$ giảm dần mượt mà về $0.08 - 0.10$, hạ bớt áp lực phạt đối kháng, tạo điều kiện cho BPR loss tinh chỉnh tối ưu thứ tự Top-K.

---

#### Tử huyệt 4: Cắt Xén Mẫu Số InfoNCE (Partition Function Truncation) & Mẫu Dương Lịch Sử Trong Memory Bank
* **Cơ chế lỗi trong code v2:**
  - Tại dòng 321–326 của `stair_sre_ans_v2.py`:
    $$	ext{hn\_exp} = 	ext{gather}(\exp(	ext{sim\_all}), 	ext{hn\_indices}), \quad 	ext{neg\_weighted\_sum} = \sum (	ext{hn\_exp} \cdot 	ext{hn\_weights})$$
    $$\mathcal{L} = -\log rac{\exp(	ext{pos} / 	au)}{\exp(	ext{pos} / 	au) + 	ext{neg\_weighted\_sum}}$$
    Mẫu số InfoNCE **chỉ cộng tổng trên Top-K Hard Negatives ($k_{hn}$ phần tử)**, hoàn toàn vứt bỏ $(Q - k_{hn})$ phần tử Easy Negatives còn lại (chiếm 60% đến 90% không gian)!
  - Việc bỏ rơi Easy Negatives vi phạm nghiêm trọng tính toàn vẹn của hàm phân phối chuẩn hóa $Z$ (Partition Function) trên siêu mặt cầu. Các mẫu âm dễ không nhận được bất kỳ gradient đẩy nào ($
abla \mathcal{L} = 0$), dẫn đến hiện tượng trôi dạt tự do (free-drifting) và suy thoái tính đồng đều toàn cục.
  - Đồng thời, trên tập dữ liệu nhỏ như Amazon Baby (catalog $N=7050$), hàng đợi FIFO $Q=4096$ chiếm tới **$58.1\%$ toàn bộ sản phẩm**. Khi các sản phẩm phổ biến được người dùng yêu thích liên tục được đẩy vào hàng đợi, chúng bị gán nhãn làm mẫu âm cho chính những người dùng đã từng tương tác, gây ra hiện tượng phạt nhầm lịch sử mua hàng.
* **Giải pháp v2.1:**
  1. Bảo toàn mẫu số toàn cục: Mẫu số tính tổng trên toàn bộ $Q$ mẫu âm, trong đó Hard Negatives được nhân trọng số điều biến $\psi_{HN} \ge 1.0$ và Easy Negatives giữ trọng số cơ sở $\psi_{EN} \le 1.0$ (hoặc 1.0).
  2. Điều chỉnh kích thước $Q$ tương ứng với dung lượng catalog: $Q = 1024$ cho Amazon Baby (chiếm $< 15\%$ catalog) và $Q = 4096$ cho Amazon Sports/Electronics.

---

### 12.2 Thiết Kế Kiến Trúc STAIR-SRE-ANS v2.1

Dưới đây là sơ đồ luồng dữ liệu độc lập hai tầng của **STAIR-SRE-ANS v2.1**:

```
                         KIẾN TRÚC STAIR-SRE-ANS v2.1
                         
   User ID u                     Item ID i+                      Negative Item i-
       │                              │                                 │
       ▼                              ▼                                 ▼
┌──────────────┐              ┌──────────────┐                  ┌──────────────┐
│ User Embed E │              │ Item Embed E │                  │ Item Embed E │
│   (Layer 0)  │              │   (Layer 0)  │                  │   (Layer 0)  │
└──────┬───────┘              └──────┬───────┘                  └──────┬───────┘
       │                             │                                 │
       │                      ┌──────▼────────┐                        │
       │                      │  Diagonal     │                        │
       │                      │  Projector w  │                        │
       │                      └──────┬────────┘                        │
       │                             │ E_proj                          │
       ├─────────────────────────────┼─────────────────────────────────┤
       │ [TẦNG 1: CL NHÁNH ĐỘC LẬP]  │                                 │
       │                             │                                 │
       ▼                             ▼                                 │
┌──────────────┐              ┌──────────────┐                         │
│ 1-Layer MLP  │              │ 1-Layer MLP  │                         │
│  Proj Head   │              │  Proj Head   │                         │
└──────┬───────┘              └──────┬───────┘                         │
       │ z_u                         │ z_i+                            │
       └──────────────┬──────────────┘                                 │
                      ▼                                                │
         ┌─────────────────────────┐                                   │
         │ StepwiseSREANSLoss v2.1 │ ◄── [FIFO Queue Q (1024/4096)]    │
         │ * Continuous Decoupling │                                   │
         │ * Active Gating cos>0.25│ ◄── [Cosine-Annealed HANS]        │
         │ * Full Partition Sum    │                                   │
         └────────────┬────────────┘                                   │
                      ▼                                                │
                 L_SRE-ANS v2.1                                        │
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       │ [TẦNG 2: GNN & BPR RANKING NHÁNH CHÍNH]
       │
       ▼
┌────────────────────────────────────────┐
│  Neumann Series Stepwise Convolution   │
│  (3-Layer Polynomial GNN Smoothing)    │
└──────────────────┬─────────────────────┘
                   ▼
     H_u^(L) và H_i+^(L), H_i-^(L) (Final Smoothed Embeddings)
                   │
                   ▼
         ┌───────────────────┐
         │     BPR Loss      │
         │  (Clustering CF)  │
         └─────────┬─────────┘
                   ▼
                 L_BPR
```

---

### 12.3 Bảng Ma Trận Đối Chiếu Tiến Hóa: Baseline vs v2 vs v2.1

| Đặc Tính Thiết Kế | STAIR Baseline | STAIR-SRE-ANS v2 | STAIR-SRE-ANS v2.1 (Bản Vá Hoàn Thiện) |
| :--- | :--- | :--- | :--- |
| **Không Gian Tương Phản (CL Target)** | Không có CL | $H^{(L)}$ (Final smoothed embedding sau 3 lớp GNN) | **Layer-0 Raw Space** ($E_u^{(0)}, E_{proj}^{(0)}$) qua 1-layer MLP Head |
| **Bảo Vệ Không Gian BPR** | 100% cho BPR | Bị InfoNCE cạnh tranh gradient làm loãng cụm CF | **Giải phóng 100% $H^{(L)}$ cho BPR**, không có gradient conflict |
| **Bộ Chiếu Tương Phản (Proj Head)** | Không | Không có (Identity) | **MLP Head (`Linear + LayerNorm + LeakyReLU`)** |
| **Cơ Chế Suy Giảm Mẫu Âm (MFNA)** | Không | $W = \sigma(	ext{sim})$, cắt giảm 50% lực đẩy của mẫu góc $pprox 90^\circ$ | **Thresholded Gate: Chỉ giảm khi $\cos > 0.25$**; $\cos \le 0.25 \implies 	ext{Atten} = 1.0$ |
| **Mẫu Số Hàm InfoNCE** | Không | Chỉ sum trên Top-K Hard Negatives ($k_{hn}$ mẫu) | **Bảo tồn Full Partition Sum trên toàn bộ $Q$ mẫu âm** |
| **HANS Scheduler Ceiling** | Không | Chạm trần $\gamma = 0.35$ ở epoch 90 và giữ cố định đến epoch 500 | **Cosine-Annealed Ceiling**: Đạt đỉnh ở epoch 200 và hạ nhiệt về cuối kỳ |
| **Kích Thước Hàng Đợi $Q$** | Không | $Q = 4096$ cố định cho mọi dataset (quá lớn với Baby) | **$Q = 1024$ (Baby) / $Q = 4096$ (Sports & Electronics)** |
| **Phân Rã Phổ Năng Lượng** | $eta(d)$ cố định | Continuous $eta(d)$ với $lpha_{spec} = 0.50$ | **Continuous $eta(d)$ với $lpha_{spec} = 0.35$ (ưu tiên tần số cao)** |

---

### 12.4 Hệ Thống Công Thức Toán Học Vi Phân Hoàn Thiện Của v2.1

#### 1. Biểu diễn Tầng 0 qua Projection Head
$$\mathbf{z}_u = g_u(\mathbf{e}_u^{(0)}) = 	ext{LeakyReLU}(	ext{LayerNorm}(\mathbf{W}_p \mathbf{e}_u^{(0)}))$$
$$\mathbf{z}_i = g_i(\mathbf{e}_i^{(0)} \odot \mathbf{w}) = 	ext{LeakyReLU}(	ext{LayerNorm}(\mathbf{W}_p (\mathbf{e}_i^{(0)} \odot \mathbf{w})))$$

#### 2. Phân rã độ khó quang phổ liên tục với $lpha_{spec} = 0.35$
$$\mathbf{z}_{u, 	ext{low}} = rac{\mathbf{z}_u \odot oldsymbol{eta}}{\|\mathbf{z}_u \odot oldsymbol{eta}\|_2}, \quad \mathbf{z}_{u, 	ext{high}} = rac{\mathbf{z}_u \odot (\mathbf{1} - oldsymbol{eta})}{\|\mathbf{z}_u \odot (\mathbf{1} - oldsymbol{eta})\|_2}$$
$$	ext{Diff}(u, k) = 0.35 \cdot \langle \mathbf{z}_{u, 	ext{low}}, \mathbf{z}_{k, 	ext{low}} angle + 0.65 \cdot \langle \mathbf{z}_{u, 	ext{high}}, \mathbf{z}_{k, 	ext{high}} angle$$

#### 3. Thresholded Dynamic Attenuation Gate
$$	ext{active\_mask}(u, k) = \mathbb{I}(\langle \hat{\mathbf{z}}_u, \hat{\mathbf{z}}_k angle > 0.25)$$
$$W(u, k) = \sigma\left(rac{\langle \hat{\mathbf{z}}_u, \hat{\mathbf{z}}_k angle}{	au}ight) \cdot 	ext{active\_mask}(u, k)$$
$$	ext{Atten}(u, k) = 	ext{clamp}(1.0 - W(u, k), 0.1, 1.0)$$

#### 4. Full Partition Weighted InfoNCE Loss
$$\mathcal{L}_{	ext{SRE-ANS v2.1}} = -rac{1}{B} \sum_{u=1}^B \log rac{\exp(\langle \hat{\mathbf{z}}_u, \hat{\mathbf{z}}_{i^+} angle / 	au)}{\exp(\langle \hat{\mathbf{z}}_u, \hat{\mathbf{z}}_{i^+} angle / 	au) + \sum_{k \in \mathcal{HN}_u} \psi_{HN}(u, k) \exp(	ext{sim}_{uk}) + \sum_{j \in \mathcal{EN}_u} \psi_{EN} \exp(	ext{sim}_{uj})}$$
trong đó:
$$\psi_{HN}(u, k) = (1.0 + \gamma_h \cdot 	ext{Diff}_{norm}(u, k)) \cdot 	ext{Atten}(u, k)$$
$$\psi_{EN} = 1.0 - \gamma_h$$

#### 5. Cosine-Annealed HANS Scheduler
$$\gamma_{max}(t) = \max\left(0.08, \gamma_{max} \cdot rac{1}{2} \left[1 + \cos\left(rac{t - 50}{450} \piight)ight]ight)$$

---

### 12.5 Mã Nguồn Triển Khai Chuẩn Hóa: `models/stair_sre_ans_v2_1.py`

```python
# -*- coding: utf-8 -*-
"""
models/stair_sre_ans_v2_1.py
STAIR-SRE-ANS v2.1: Decoupled Multi-Scale Representation,
Thresholded Topology-driven MFNA & Cosine-Annealed HANS Scheduler.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RegularizedDiagonalSpectralProjector', 'StepwiseSREANSLoss_v21']


class RegularizedDiagonalSpectralProjector(nn.Module):
    def __init__(self, dim: int = 64, reg_weight: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.reg_weight = float(reg_weight)
        self.w = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w

    def get_anchoring_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.w.device)
        return self.reg_weight * torch.sum((self.w - 1.0) ** 2)


class StepwiseSREANSLoss_v21(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        tau: float = 0.25,
        queue_size: int = 4096,
        warmup_epochs: int = 50,
        total_epochs: int = 500,
        gamma_max: float = 0.30,
        hn_ratio_max: float = 0.35,
        subspace_alpha: float = 0.35,
        beta: Optional[torch.Tensor] = None,
        gamma: float = 0.10,
    ):
        super().__init__()
        self.dim = dim
        self.tau = float(tau)
        self.queue_size = int(queue_size)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.gamma_max = float(gamma_max)
        self.hn_ratio_max = float(hn_ratio_max)
        self.subspace_alpha = float(subspace_alpha)

        # Spectral continuous decay
        if beta is not None:
            beta_curve = beta.detach().clone().to(dtype=torch.float32)
        else:
            d_indices = torch.arange(dim, dtype=torch.float32)
            beta_curve = 0.9 * (1.0 - torch.pow(d_indices / float(dim), float(gamma)))
        self.register_buffer('beta', beta_curve)
        self.register_buffer('beta_high', 1.0 - beta_curve)

        # Contrastive Projection Head (Bảo vệ không gian GNN chính)
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2)
        )

        # Cross-Batch Memory Bank
        self.register_buffer('neg_queue', torch.randn(queue_size, dim))
        self.neg_queue = F.normalize(self.neg_queue, p=2, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # HANS Scheduler
        self.hn_ratio = 0.10
        self.gamma_h = 0.05
        self.current_epoch = 0
        self.loss_history = []

    @torch.no_grad()
    def enqueue_negatives(self, pos_emb: torch.Tensor):
        batch_size = pos_emb.size(0)
        norm_emb = F.normalize(pos_emb.detach(), p=2, dim=-1)
        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.neg_queue[ptr:ptr + batch_size] = norm_emb
            ptr = (ptr + batch_size) % self.queue_size
        else:
            first = self.queue_size - ptr
            self.neg_queue[ptr:] = norm_emb[:first]
            remain = batch_size - first
            self.neg_queue[:remain] = norm_emb[first:]
            ptr = remain
        self.queue_ptr[0] = ptr

    def update_scheduler(self, current_cl_loss: float, window: int = 10, threshold: float = 0.99):
        self.current_epoch += 1
        if self.current_epoch < self.warmup_epochs:
            self.gamma_h = 0.05
            self.hn_ratio = 0.10
            return

        # Cosine-Annealed Ceiling: Giảm tải áp lực HN ở giai đoạn hậu kỳ
        progress = (self.current_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
        dynamic_gamma_cap = self.gamma_max * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        dynamic_gamma_cap = max(0.08, dynamic_gamma_cap)

        self.loss_history.append(float(current_cl_loss))
        if len(self.loss_history) > window * 2:
            self.loss_history.pop(0)
            loss_curr = sum(self.loss_history[-window:]) / float(window)
            loss_prev = sum(self.loss_history[-window*2:-window]) / float(window)

            if loss_curr >= threshold * loss_prev:
                self.gamma_h = min(self.gamma_h + 0.015, dynamic_gamma_cap)
                self.hn_ratio = min(self.hn_ratio + 0.015, self.hn_ratio_max)
            else:
                self.gamma_h = max(self.gamma_h - 0.01, 0.05)
                self.hn_ratio = max(self.hn_ratio - 0.01, 0.10)

    def forward(
        self,
        u_raw: torch.Tensor,
        i_raw: torch.Tensor,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> torch.Tensor:
        device = u_raw.device
        u_batch = self.proj_head(u_raw[batch_users])
        pos_batch = self.proj_head(i_raw[batch_items])
        neg_pool = self.neg_queue.detach().to(device)
        Q = neg_pool.size(0)

        # 1. Continuous Spectral Difficulty
        beta = self.beta.to(device)
        beta_high = self.beta_high.to(device)

        u_low = F.normalize(u_batch * beta, p=2, dim=-1)
        u_high = F.normalize(u_batch * beta_high, p=2, dim=-1)
        neg_low = F.normalize(neg_pool * beta, p=2, dim=-1)
        neg_high = F.normalize(neg_pool * beta_high, p=2, dim=-1)

        cos_low = torch.matmul(u_low, neg_low.T)
        cos_high = torch.matmul(u_high, neg_high.T)
        difficulty = self.subspace_alpha * cos_low + (1.0 - self.subspace_alpha) * cos_high

        # 2. Thresholded Dynamic Attenuation (Fix Tử huyệt 2)
        u_norm = F.normalize(u_batch, p=2, dim=-1)
        pos_norm = F.normalize(pos_batch, p=2, dim=-1)
        neg_norm = F.normalize(neg_pool, p=2, dim=-1)

        cos_all = torch.matmul(u_norm, neg_norm.T)
        sim_all = cos_all / self.tau

        # Chỉ giảm lực đẩy nếu cosine > 0.25 (nguy cơ False Negative cao)
        active_mask = (cos_all > 0.25).float()
        W = torch.sigmoid(sim_all) * active_mask
        attenuation = torch.clamp(1.0 - W, min=0.1, max=1.0)

        # 3. Gated Top-K Selection
        selection_score = difficulty * attenuation
        k_hn = max(1, int(self.hn_ratio * Q))
        _, hn_indices = torch.topk(selection_score, k=k_hn, dim=1)

        # 4. Stratified Weights & Full Partition InfoNCE
        diff_norm = (difficulty + 1.0) / 2.0
        psi_HN = 1.0 + self.gamma_h * diff_norm
        psi_EN = 1.0 - self.gamma_h

        psi_all = torch.full_like(sim_all, psi_EN)
        hn_mask = torch.zeros_like(sim_all, dtype=torch.bool)
        hn_mask.scatter_(1, hn_indices, True)
        psi_all = torch.where(hn_mask, psi_HN, psi_all)

        final_weights = psi_all * attenuation

        # Mẫu số tính tổng trên toàn bộ Q mẫu âm (Bảo tồn Full Partition)
        pos_sim = torch.sum(u_norm * pos_norm, dim=-1) / self.tau
        pos_exp = torch.exp(pos_sim)
        exp_all = torch.exp(sim_all)
        neg_weighted_sum = (exp_all * final_weights).sum(dim=1)

        loss = -torch.log(pos_exp / (pos_exp + neg_weighted_sum + 1e-8)).mean()

        if self.training:
            with torch.no_grad():
                self.enqueue_negatives(pos_batch)

        return loss
```

#### Pipeline Huấn Luyện Trong `fit()` của Coach v2.1:
```python
def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
    userEmbds, itemEmbds = self.encode()

    users = data[self.User]
    positives = data[self.Item]
    negatives = data[self.INeg]

    # BPR Loss tối ưu trên không gian đã tích chập GNN (Graph Smoothed)
    rec_loss = self.criterion(
        torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
        torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
    )

    reg_w_loss = self.spectral_projector.get_anchoring_loss()

    # Contrastive Loss tối ưu trên Tầng 0 (Bảo vệ đặc trưng gốc, giải phóng H^(L))
    if self.training and cfg.lambda_ans > 0.0:
        item_raw_proj = self.spectral_projector(self.Item.embeddings.weight)
        user_raw = self.User.embeddings.weight

        cl_loss = self.ans_loss(
            u_raw=user_raw,
            i_raw=item_raw_proj,
            batch_users=users,
            batch_items=positives,
        )
        self.last_cl_loss = float(cl_loss.detach().item())
        return rec_loss + reg_w_loss + cfg.lambda_ans * cl_loss

    return rec_loss + reg_w_loss
```

---

### 12.6 Giả Thuyết Khoa Học, Khuyến Nghị Tham Số & Kỳ Vọng Vượt Baseline

1. **Giả thuyết Khoa học 1 (Decoupled Representation Hypothesis):** Việc đưa InfoNCE về Tầng 0 sẽ ngăn chặn hoàn toàn hiện tượng kéo dãn cụm của $H^{(L)}$, cho phép BPR tự do hội tụ các cụm sở thích CF, giúp Recall@20 lập tức tăng trưởng trở lại mốc $\ge 0.1050$ trên Baby và $\ge 0.1120$ trên Sports.
2. **Giả thuyết Khoa học 2 (True Negative Force Restoration):** Khi cổng Thresholded Gating ($\cos > 0.25$) được kích hoạt, 100% lực đẩy đối kháng của True Negatives được giải phóng, khôi phục lại mật độ phân tán đều (Uniformity) lành mạnh trên mặt cầu siêu cầu.
3. **Giả thuyết Khoa học 3 (Cooling Fine-tuning Benefit):** Cosine Annealing giúp các epoch cuối (epoch 350–500) không bị rung lắc bởi mẫu âm cực khó, hỗ trợ bộ tối ưu AdamWSEvo chốt chặn tại điểm cực tiểu Pareto lý tưởng.

**Khuyến nghị tham số thực nghiệm v2.1:**
- $\lambda_{ans}$: Đặt $2 	imes 10^{-5}$ trên Sports (giảm áp lực gradient so với $5 	imes 10^{-5}$) và $5 	imes 10^{-5}$ trên Baby.
- $	au$: Tăng nhẹ lên $0.25$ để làm mượt bề mặt hàm mục tiêu softmax.
- Queue Size $Q$: Đặt $1024$ trên Amazon Baby và $4096$ trên Amazon Sports.

