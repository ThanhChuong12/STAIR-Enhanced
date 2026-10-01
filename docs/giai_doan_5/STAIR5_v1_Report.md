# BÁO CÁO THIẾT KẾ VÀ PHẢN BIỆN KIẾN TRÚC STAIR5-v1 (REVISED v2)
## STAIR-LHC: STAIR WITH LORENTZ HIDDEN CONTRASTIVE REGULARIZATION
### Tích Hợp Đánh Giá Peer Review (Vòng 1 & Vòng 2), Phản Biện Ngược Về Khoảng Cách Bình Phương (Squared Distance), Cơ Chế Triệt Tiêu Self-Return (Degree-1 Safety), Bù Suy Giảm Phổ Thích Ứng (Adaptive Spectral Re-weighting) & Thẩm Định Độc Lập Hình Học Lorentz (Constant-Radius Control)

---

**Đề tài:** Recommender Systems using Graph Representation: Multi-modal  
**Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Sinh viên thực hiện:**  
- Lê Hà Thanh Chương (MSSV: 23120195)  
- Bùi Trung Hiếu (MSSV: 23120257)  
**Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo  
**Mã nguồn triển khai:** [`ThanhChuong12/STAIR-Enhanced`](https://github.com/ThanhChuong12/STAIR-Enhanced)  
**Tên đề xuất kiến trúc:** **STAIR-LHC v1 (REVISED)** — *STAIR with Lorentz Hidden Contrastive Regularization*  
**Ngày cập nhật:** 30/09/2026 (Revised v2 — Tích hợp Peer Review Vòng 2)  
**Trạng thái nghiên cứu:** Thiết kế phương pháp luận & đặc tả toán học kiểm chứng nghiêm ngặt; **Bước tiếp theo: triển khai mã nguồn kiến trúc đề xuất chính (B0 + H0) trước, chạy thử nghiệm xác nhận trước khi bắt đầu ablation study.**

---

## MỤC LỤC

1. [TỔNG QUAN QUẢN TRỊ & ĐÁNH GIÁ PEER REVIEW](#1-tổng-quan-quản-trị--đánh-giá-peer-review)
   - 1.1. Quyết định kiến trúc cốt lõi: Chọn hướng STAIR-LHC, dứt khoát từ chối DualHypCL
   - 1.2. Bảng tổng hợp đánh giá Peer Review và nhận diện các lỗ hổng bị bỏ sót
   - 1.3. Ma trận quyết định kiến trúc STAIR-LHC v1 (Revised)
2. [MATERIAL PASSPORT & ĐỐI SOÁT BẰNG CHỨNG LỊCH SỬ](#2-material-passport--đối-soát-bằng-chứng-lịch-sử)
   - 2.1. Nguồn gốc tài liệu và phạm vi kiểm định mã nguồn
   - 2.2. Phân định rạch ròi các thế hệ nghiên cứu (GĐ2, GĐ3, GĐ4 v1-v5 và GĐ5)
   - 2.3. Tính toán lại tăng trưởng thực chất của tiền thân NE-NLGCL
   - 2.4. Đối soát thực nghiệm sơ bộ: STAIR-LHC v1 so với STAIR Baseline tái lập (03_stair.tex)
3. [PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU & PHẢN BIỆN NGƯỢC ĐỐI VỚI PEER REVIEW](#3-phân-tích-toán-học-chuyên-sâu--phản-biện-ngược-đối-với-peer-review)
   - 3.1. Phản biện ngược Peer Review về "Squared Distance": Đúng số học nhưng tiềm ẩn 3 rủi ro lớn & Giải pháp Hybrid Kernel
   - 3.2. Xử lý triệt để hiện tượng suy giảm phổ $\beta$ decay trong tầng $H^{(1)}$: Ba phương án & Lựa chọn Spectral Re-weighting
   - 3.3. Vấn đề cốt tử Peer Review bỏ sót #1: Triệt tiêu Self-Return Shortcut trong Positive Pairs
   - 3.4. Vấn đề cốt tử Peer Review bỏ sót #2: Tách biệt Hình học Hyperbolic khỏi Hiệu ứng Kernel (Constant-Radius Control - HC Arm)
   - 3.5. Vấn đề cốt tử Peer Review bỏ sót #3: Cách ly nhóm Optimizer và xung đột với BSC Smoother
   - 3.6. Bác bỏ giả thuyết "Long-tail đồng nghĩa với Hyperbolic Hierarchy" trên đồ thị E-Commerce
   - 3.7. Vấn đề triệt tiêu toán học của DualHypCL (Curvature Vanishing Phenomenon)
   - 3.8. Phân tích rủi ro của Semantic Filtering đối với False Negatives theo cảnh báo EStair
4. [ĐỐI CHIẾU TÀI LIỆU HỌC THUẬT QUỐC TẾ (LITERATURE BENCHMARK)](#4-đối-chiếu-tài-liệu-học-thuật-quốc-tế-literature-benchmark)
5. [ĐẶC TẢ KIẾN TRÚC HOÀN CHỈNH: STAIR-LHC v1 (REVISED)](#5-đặc-tả-kiến-trúc-hoàn-chỉnh-stair-lhc-v1-revised)
   - 5.1. Sơ đồ luồng dữ liệu tổng thể (Pipeline Flowchart)
   - 5.2. Hợp đồng tầng biểu diễn (Layer Contract), Self-Return Removal & Spectral Re-weighting
   - 5.3. Chuẩn hóa thang đo cố định ($q_{t,\ell}$) và Chiếu bán kính biến thiên có chặn ($R=2$)
   - 5.4. Ánh xạ Lorentz (Lorentz Exponential Map) và Tiệm cận Euclidean khi $\kappa \to 0$
   - 5.5. Cấu trúc Numerical Kernel, Khai triển chuỗi Taylor $t \to 0$ và Hàm khoảng cách Hybrid
   - 5.6. Ma trận Train-Positive đa mẫu (Multi-Positive InfoNCE) 2 hướng ($U_0 \leftrightarrow I_1$, $I_0 \leftrightarrow U_1$)
   - 5.7. Tổng hợp hàm mất mát, Lịch trình Warmup $\lambda(t)$ và Phân nhóm Optimizer
   - 5.8. Thuật toán và Mã giả triển khai các module cốt lõi
6. [PHÂN TÍCH CHI PHÍ TÍNH TOÁN & HỒ SƠ BỘ NHỚ VRAM](#6-phân-tích-chi-phí-tính-toán--hồ-sơ-bộ-nhớ-vram)
7. [GIAO THỨC THỰC NGHIỆM, MA TRẬN ABLATION & TIÊU CHÍ DỪNG](#7-giao-thức-thực-nghiệm-ma-trận-ablation--tiêu-chí-dừng)
   - 7.1. Ma trận Ablation đa nhánh có phân cấp ưu tiên (P0, P1, P2)
   - 7.2. Bộ tiêu chuẩn quyết định Go / No-Go định lượng nghiêm ngặt
   - 7.3. Cấu hình siêu tham số và ngân sách thực nghiệm (Sports, Baby, Electronics)
   - 7.4. Quy tắc lựa chọn checkpoint, đánh giá kiểm định và xử lý độ bất định
8. [KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN 5 BƯỚC (5-STEP IMPLEMENTATION ROADMAP)](#8-kế-hoạch-triển-khai-mã-nguồn-5-bước-5-step-implementation-roadmap)
9. [BẢNG TỔNG HỢP RỦI RO & CHIẾN LƯỢC GIẢM THIỂU (RISK & MITIGATION MATRIX)](#9-bảng-tổng-hợp-rủi-ro--chiến-lược-giảm-thiểu-risk--mitigation-matrix)
10. [KẾT LUẬN KHOA HỌC & ĐIỀU KIỆN TUYÊN BỐ ĐÓNG GÓP](#10-kết-luận-khoa-học--điều-kiện-tuyên-bố-đóng-góp)
11. [NGUỒN THAM KHẢO & MỨC ĐỘ ĐỐI SOÁT](#11-nguồn-tham-khảo--mức-độ-đối-soát)

---

## 1. TỔNG QUAN QUẢN TRỊ & ĐÁNH GIÁ PEER REVIEW

### 1.1. Quyết định kiến trúc cốt lõi: Chọn hướng STAIR-LHC, dứt khoát từ chối DualHypCL

Trong quá trình khởi động **Giai đoạn 5 (STAIR5)**, nhóm nghiên cứu đã tiến hành đánh giá toàn diện hai hướng tiếp cận hình học phi Euclidean:
- **Hướng 1 (LHyperNLGCL / STAIR-LHC):** Giữ nguyên vẹn 100% backbone Euclidean STAIR (khởi tạo SVD Whitening, tích chập phổ FSC, toán tử làm mịn BSC trong optimizer AdamWSEvo, hàm mất mát BPR và bộ chấm điểm tích vô hướng Euclidean). Bổ sung một nhánh điều hòa tương phản phụ (Auxiliary Contrastive Regularization) ánh xạ các biểu diễn ẩn qua đa tạp Lorentz với độ cong cố định $\kappa > 0$.
- **Hướng 2 (DualHypCL):** Chiếu toàn bộ cả hai nhánh Collaborative Filtering (CF) và Đa phương thức (MM) qua hai không gian hyperbolic với hai độ cong độc lập $\kappa_{\text{CF}}$ và $\kappa_{\text{MM}}$, sau đó ánh xạ ngược về tangent space để tính loss tương phản.

**Quyết định dứt khoát của nhóm nghiên cứu:**  
1. **Tuyệt đối không triển khai DualHypCL nguyên trạng:** Phân tích toán học tại §3.7 chỉ ra rằng việc áp dụng ánh xạ mũ $\exp_0^\kappa$ rồi ngay lập tức áp dụng ánh xạ logarit $\log_0^\kappa$ tại cùng một gốc tọa độ với cùng độ cong sẽ tạo ra một ánh xạ đồng nhất (identity mapping: $\log_0^\kappa(\exp_0^\kappa(v)) = v$). Điều này khiến độ cong hoàn toàn bị triệt tiêu khỏi đạo hàm của hàm mất mát ($\partial \mathcal{L} / \partial \kappa = 0$). Việc gán ghép các nhãn "học được độ cong ngữ nghĩa" trong trường hợp này là ảo tưởng toán học do sai số dấu phẩy động sinh ra.
2. **Chọn và hoàn thiện STAIR-LHC v1 (REVISED):** Triển khai nhánh điều hòa Lorentz có đối chứng Euclidean tương ứng hoàn hảo ($E_0$). Giữ suy luận ở không gian Euclidean thuần nhất để bảo đảm độ trễ $O(1)$ và không làm xáo trộn cấu trúc xếp hạng đã được chứng minh của STAIR.

---

### 1.2. Bảng tổng hợp đánh giá Peer Review và nhận diện các lỗ hổng bị bỏ sót

Sau khi công bố bản dự thảo thiết kế ban đầu, báo cáo đã nhận được bản phản biện chuyên gia (Peer Review Critique). Đánh giá tổng thể: **Báo cáo gốc đạt chất lượng phương pháp luận xuất sắc**, đặc biệt ở sự tỉnh táo từ chối DualHypCL và bảo lưu suy luận Euclidean. Peer Review đã chỉ ra 4 điểm phản biện rất sắc sảo, tuy nhiên qua phân tích sâu của Senior Researcher, vẫn còn **3 vấn đề kỹ thuật cốt tử chưa được Peer Review phát hiện**, đồng thời có **1 khuyến nghị của Peer Review cần phải phản biện ngược** để tránh nguy cơ sụp đổ biểu diễn (representation collapse).

#### Bảng 1.1: Đánh giá chi tiết các điểm phản biện của Peer Review

| Phản Biện Của Peer Review | Đánh Giá Của Senior Researcher | Hiện Trạng & Hành Động Kỹ Thuật |
| :--- | :---: | :--- |
| **A. Giới hạn bán kính (Radius Capping) để chống tràn số $\cosh/\sinh$** | ✅ **ĐÚNG 100%** | Bắt buộc phải áp dụng hàm chặn bão hòa $R \tanh(\|a\|/R)$ trước khi đưa vào ánh xạ mũ Lorentz $\exp_0$. Đã chốt $R=2$. |
| **B. Dùng bình phương khoảng cách Lorentz ($d_L^2$) thay vì $d_L$** | ⚠️ **ĐÚNG SỐ HỌC NHƯNG CẦN PHẢN BIỆN NGƯỢC** | Đúng về mặt tránh đạo hàm kỳ dị $\text{arcosh}'(1)=\infty$, nhưng tạo ra 3 hệ quả tai hại về mặt Metric learning và Kernel weighting. Cần giải pháp **Hybrid Distance Kernel** (Xem §3.1). |
| **C. Tầng $H^{(1)}$ bị suy giảm phổ $\beta$ ở các chiều cao (High-dims decay)** | ✅ **ĐÚNG 100%** | Hệ số suy giảm phổ $\beta_j$ triệt tiêu năng lượng các chiều cao nơi chứa thông tin ngữ nghĩa đa phương thức. Cần áp dụng **Spectral Re-weighting** trước ánh xạ Lorentz (Xem §3.2). |
| **D. Bắt buộc phải có Euclidean Control ($E_0$) đối chứng song song** | ✅ **ĐÚNG 100%** | Đã được thiết lập làm điều kiện tiên quyết trong bảng Ablation Matrix. Mọi khẳng định về độ cong $\kappa>0$ đều vô giá trị nếu không thắng được $E_0$ cùng ngân sách. |

#### Bảng 1.2: Ba vấn đề kỹ thuật cốt tử Peer Review đã bỏ sót

| Vấn Đề Bị Bỏ Sót | Bản Chất Kỹ Thuật | Giải Pháp Đột Phá Bổ Sung Trong STAIR-LHC v1 (Revised) |
| :--- | :--- | :--- |
| **1. Self-Return Shortcut trong Positive Pairs** | Khi tính $H_i^{(1)}$ cho cặp dương $(u,i) \in \mathcal{E}_{\text{train}}$, trong tổng lân cận của node $i$ đã chứa sẵn embedding $E_u$. Mô hình có thể "gian lận" bằng cách học identity mapping để copy $E_u$ vào $H_i^{(1)}$ mà không học cấu trúc đồ thị. | **Triệt tiêu Self-Return có điều kiện:** Trừ bỏ trực tiếp thành phần $E_u / \sqrt{d_i d_u} \odot \beta$ khỏi key $H_i^{(1)}$ trên từng cặp dương trong batch (Xem §3.3). |
| **2. Tách biệt Hình học Hyperbolic khỏi Hiệu ứng Kernel** | Nếu chuẩn vector của tất cả các thực thể bằng nhau ($\|v\|=\|w\|=r$), khoảng cách Lorentz là hàm đơn điệu của tích vô hướng Cosine. Khi đó $\kappa$ chỉ đóng vai trò như một hàm biến đổi phi tuyến (kernel trick) chứ không đem lại dung lượng hình học thực sự. | **Thiết lập nhánh kiểm định Constant-Radius Control (HC arm):** Chuẩn hóa L2 $\|v\|=1$ trước khi đưa qua exp map để phân biệt rạch ròi giữa Geometric capacity và Kernel effect (Xem §3.4). |
| **3. Xung đột Gradient giữa CL Loss và Toán tử BSC Smoother** | Gradient từ CL loss chảy ngược vào `Item.embeddings.weight` sẽ bị toán tử BSC Smoother trong optimizer AdamWSEvo làm mịn phổ, vô tình áp đặt lại cấu trúc suy giảm phổ lên gradient Lorentz. | **Cách ly nhóm tham số (Optimizer Group Disjointness):** Theo dõi góc cosine gradient giữa BPR và CL; thiết lập cơ chế gradient isolation nếu phát hiện triệt tiêu lẫn nhau (Xem §3.5). |

---

### 1.3. Ma trận quyết định kiến trúc STAIR-LHC v1 (Revised)

| Quyết Định | Nội Dung Triển Khai |
| :--- | :--- |
| **BẢO TỒN NGUYÊN VẸN** | Khởi tạo MI Whitening + SVD; Tích chập phổ FSC; Toán tử làm mịn BSC trong AdamWSEvo; Hàm mất mát BPR; Bộ chấm điểm xếp hạng tích vô hướng Euclidean; Giao thức chọn checkpoint theo Validation NDCG@20. |
| **BỔ SUNG VÀO TRAINING** | Nhánh điều hòa tương phản phụ Multi-Positive InfoNCE 2 chiều ($U_0 \leftrightarrow I_1$ và $I_0 \leftrightarrow U_1$); Bù suy giảm phổ Spectral Re-weighting trên $H^{(1)}$; Triệt tiêu Self-Return trên positive keys; Ánh xạ Lorentz với $\kappa$ cố định; Hàm khoảng cách an toàn số học Hybrid Distance Kernel. |
| **THIẾT KẾ ĐỐI CHỨNG CÔNG BẰNG** | Nhánh Euclidean Control ($E_0$) bắt buộc phải sử dụng chính xác cùng hàm bình phương khoảng cách $d_E^2 = \|v-w\|_2^2$, cùng trần bán kính $R$, cùng chuẩn hóa thang đo $q$, cùng ma trận positive $P$ và cùng cơ chế triệt tiêu self-return. |
| **LOẠI BỎ KHỎI TUYÊN BỐ** | Không tuyên bố "ổn định số học tuyệt đối", không tuyên bố "chữa dứt điểm long-tail", không tuyên bố "học được độ cong cấu trúc cây hoàn hảo", không hứa hẹn "chắc chắn đạt SOTA". Mọi kết luận đều phải dựa trên kiểm định thống kê paired multi-seed. |

---

## 2. MATERIAL PASSPORT & ĐỐI SOÁT BẰNG CHỨNG LỊCH SỬ

### 2.1. Nguồn gốc tài liệu và phạm vi kiểm định mã nguồn

- **Đầu vào thiết kế:** Bản ý tưởng hai hướng hình học phi Euclidean đính kèm, tài liệu [EStair.md](../EStair.md), mã nguồn các mô hình tiền thân [`models/stair_ne_nlgcl.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/stair_ne_nlgcl.py) và launcher [`main_stair_ne_nlgcl_v5.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_stair_ne_nlgcl_v5.py).
- **Nhật ký thực nghiệm đối soát:** Báo cáo thực nghiệm Giai đoạn 4 v4 ([`STAIR4_v4_Experiment_Report.md`](../giai_doan_4/STAIR4_v4_Experiment_Report.md)) và Giai đoạn 4 v5 ([`STAIR4_v5_Experiment_Report.md`](../giai_doan_4/STAIR4_v5_Experiment_Report.md)).
- **Nguyên tắc liêm chính học thuật:** Phân định rõ ràng giữa:
  1. *Quan sát thực tế trong mã nguồn* (Code observations).
  2. *Số liệu thực nghiệm đã kiểm chứng và lưu trữ nhật ký* (Empirical telemetry).
  3. *Suy diễn toán học thuần túy* (Mathematical derivations).
  4. *Giả thuyết khoa học chưa qua thực nghiệm* (Unverified hypotheses).

---

### 2.2. Phân định rạch ròi các thế hệ nghiên cứu

Để tránh việc nhập nhằng giữa các phiên bản cùng mang tên gọi "v5" trong lịch sử đề tài:

| Thế Hệ Nghiên Cứu | Tên Định Danh | Bản Chất Cơ Chế Thuật Toán | Trạng Thái Học Thuật |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 2** | `NE-NLGCL v5` | Bổ sung nhiễu có trọng số theo tọa độ + Tương phản đa tầng (Cross-entity CL) trong không gian Euclidean. | Tiền thân trực tiếp của ý tưởng tương phản đa tầng. |
| **Giai đoạn 3** | `STAIR3-v5` | Can thiệp phổ đồ thị động (SPSD/SSB, modal_only/full_ssb). | Can thiệp cấu trúc đồ thị, không liên quan đến hình học Lorentz. |
| **Giai đoạn 4** | `STAIR-RAM v5` | Quy trình hai giai đoạn tách rời: Đóng băng Teacher Stage A + Học Residual Head Stage B bằng Candidate Cross-Entropy. | Đã hoàn tất thực nghiệm kiểm định. Safe-Fallback Invariant bảo toàn 100% Teacher. |
| **Giai đoạn 5 (Hiện tại)** | **STAIR-LHC v1 (Revised)** | **Huấn luyện đồng thời (Joint Training) với nhánh điều hòa phụ Lorentz Contrastive trên không gian ẩn, giữ nguyên vẹn suy luận Euclidean.** | **Thiết kế mới hoàn toàn; quy định mã nguồn, cấu hình và Run ID độc lập.** |

---

### 2.3. Tính toán lại tăng trưởng thực chất của tiền thân NE-NLGCL

Bảng 2.1 trích xuất số liệu đối soát từ chương 3 báo cáo Giai đoạn 2 (`report/chapters_v2/03_stair.tex`), tính toán lại mức tăng trưởng tương đối thực chất:
$$\%\,\text{Thay đổi} = \left(\frac{M_{\text{variant}}}{M_{\text{baseline}}} - 1\right) \times 100\%$$

#### Bảng 2.1: Đối soát mức tăng trưởng thực tế của NE-NLGCL v5 so với STAIR Baseline

| Tập Dữ Liệu | Chỉ Số Metric | STAIR Baseline | NE-NLGCL v5 | Thay Đổi Tuyệt Đối | Thay Đổi Tương Đối | Đánh Giá Bản Chất |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Amazon Baby** | Recall@10 | 0.0674 | 0.0666 | -0.0008 | **-1.19%** 🔴 | Suy giảm nhẹ |
| *(19.4K Users,* | Recall@20 | 0.1042 | 0.1022 | -0.0020 | **-1.92%** 🔴 | Suy giảm nhẹ |
| *7.05K Items)* | NDCG@10 | 0.0359 | 0.0361 | +0.0002 | **+0.56%** 🟡 | Tăng không đáng kể |
| | NDCG@20 | 0.0454 | 0.0452 | -0.0002 | **-0.44%** 🔴 | Suy giảm nhẹ |
| **Amazon Sports** | Recall@10 | 0.0743 | 0.0753 | +0.0010 | **+1.35%** 🟢 | Tăng trưởng thực tế |
| *(35.6K Users,* | Recall@20 | 0.1111 | 0.1113 | **+0.0002** | **+0.18%** 🟡 | **Chỉ tăng 0.02 điểm phần trăm** |
| *18.4K Items)* | NDCG@10 | 0.0405 | 0.0415 | +0.0010 | **+2.47%** 🟢 | Cải thiện xếp hạng đầu danh sách |
| | NDCG@20 | 0.0500 | 0.0508 | +0.0008 | **+1.60%** 🟢 | Tăng trưởng có ý nghĩa |

*Bài học thực nghiệm sâu sắc:*  
Mức tăng Recall@20 trên Amazon Sports thực chất chỉ là **+0.0002 tuyệt đối** (từ 0.1111 lên 0.1113), tương đương **0.18% tương đối**, không phải 18%. Trên Amazon Baby, mô hình bị thoái hóa trên hầu hết các chỉ số. Do đó, **chưa hề có bằng chứng thực nghiệm nào khẳng định không gian Euclidean 64 chiều đã bị bão hòa dung lượng** hay hình học hyperbolic chắc chắn sẽ mang lại bước nhảy vọt. Mọi giả thuyết cần được kiểm chứng bằng thái độ hoài nghi khoa học nghiêm ngặt nhất.

---

### 2.4. Đối soát thực nghiệm sơ bộ: STAIR-LHC v1 so với STAIR Baseline tái lập (`03_stair.tex`)

Sau khi hoàn tất đợt chạy 500 epochs trên Amazon Sports (Run ID: `0930090605`) và Amazon Baby (Run ID: `0930132222`), nhóm nghiên cứu tiến hành đối chuẩn trực tiếp STAIR-LHC v1 với **Kết quả tái lập thực nghiệm gốc** công bố tại Mục 3.2 và Bảng 3.7 trong `report/chapters_v2/03_stair.tex`. Checkpoint được lựa chọn nghiêm ngặt theo **Validation NDCG@20 cao nhất**, và toàn bộ số liệu công bố là kết quả trên tập **Test** tại đúng checkpoint này.

#### Bảng 2.2: Đối soát kết quả TEST SET của STAIR-LHC v1 so với STAIR Baseline tái lập và Paper gốc

| Tập Dữ Liệu | Chỉ Số Metric | Paper Gốc (Table 2) | STAIR Baseline (Tái Lập) | STAIR-LHC v1 (H0) | Thay Đổi vs Tái Lập | Thay Đổi vs Paper | Đánh Giá Khoa Học |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Amazon Sports** | Recall@10 | 0.0743 | 0.0743 | **0.0747** | **+0.54%** (+0.0004) | **+0.54%** | Cải thiện ổn định |
| *(Best Epoch 500/500)* | Recall@20 | 0.1117 | 0.1111 | **0.1133** | **+1.98%** (+0.0022) 🚀 | **+1.43%** (+0.0016) 🚀 | **Kỷ lục mới, phá vỡ trần v5 (0.1113)** |
| | NDCG@10 | 0.0407 | 0.0405 | **0.0407** | **+0.49%** (+0.0002) | -0.03% (Ngang bằng) | Bảo toàn Top-10 |
| | NDCG@20 | 0.0503 | 0.0500 | **0.0506** | **+1.20%** (+0.0006) ✅ | **+0.62%** (+0.0003) | **Vượt cả Paper và Baseline** |
| **Amazon Baby** | Recall@10 | 0.0674 | 0.0674 | 0.0660 | **-2.08%** (-0.0014) 🔴 | **-2.08%** | Giảm nhẹ |
| *(Best Epoch 215/500)* | Recall@20 | 0.1042 | 0.1042 | 0.1030 | **-1.15%** (-0.0012) 🟡 | **-1.15%** | Thấp hơn Baseline, nhưng cao hơn v5 (0.1022) |
| *(Ep. 500: R20=0.1039,* | NDCG@10 | 0.0359 | 0.0359 | 0.0352 | **-1.95%** (-0.0007) 🔴 | **-1.95%** | Giảm nhẹ |
| *NDCG20=0.0454)* | NDCG@20 | 0.0453 | 0.0454 | 0.0447 | **-1.54%** (-0.0007) 🟡 | **-1.32%** | Thấp hơn Baseline, nhưng cao hơn GĐ4 (0.0440) |

*Đánh giá trung thực khoa học:*  
- **Amazon Sports:** Nhánh Lorentz H0 đạt bước tiến thực chất trên toàn bộ 4 chỉ số, đặc biệt Recall@20 lập kỷ lục mới **0.1133 (+1.98% vs Baseline, +1.43% vs Paper)** và NDCG@20 đạt **0.0506 (+1.20% vs Baseline, +0.62% vs Paper)**. Điều này thỏa mãn điều kiện Tầng 1 Go/No-Go ($\Delta \text{NDCG@20} \ge +0.5\%$).
- **Amazon Baby:** Tại checkpoint tốt nhất trên validation (Epoch 215), mô hình ghi nhận mức suy giảm nhẹ (-1.15% đến -2.08%) so với Baseline tái lập chuẩn, dù vẫn cao hơn các thế hệ trước (v5 đạt 0.1022, GĐ4 đạt 0.1010). Điều này cho thấy đồ thị dày hơn (Baby) ít hưởng lợi từ không gian Lorentz ở cấu hình $\lambda=3\times 10^{-4}, wd=0.3$ mặc định, và cần được tinh chỉnh siêu tham số khắt khe hơn.

---

## 3. PHÂN TÍCH TOÁN HỌC CHUYÊN SÂU & PHẢN BIỆN NGƯỢC ĐỐI VỚI PEER REVIEW

### 3.1. Phản biện ngược Peer Review về "Squared Distance": Đúng số học nhưng tiềm ẩn 3 rủi ro lớn & Giải pháp Hybrid Kernel

Peer Review khuyến nghị thay thế khoảng cách trắc địa chuẩn (geodesic distance) $d_L(x, y) = \frac{1}{\sqrt{\kappa}} \text{arcosh}(-\kappa \langle x, y \rangle_L)$ bằng bình phương khoảng cách $d_L^2(x, y)$ để loại bỏ điểm kỳ dị tại biên khi $x \to y$, nơi đạo hàm $\text{arcosh}'(1) = \frac{1}{\sqrt{1^2 - 1}} = \infty$.

**Đánh giá của Senior Researcher:** Về mặt số học thuần túy, khuyến nghị này hoàn toàn chính xác. Tuy nhiên, việc áp dụng máy móc bình phương khoảng cách vào hàm mất mát InfoNCE sẽ vấp phải **3 vấn đề lý thuyết nghiêm trọng** mà Peer Review chưa lường trước:

#### Vấn đề 1: Đánh mất tính chất Metric chuẩn (Vi phạm Bất đẳng thức Tam giác)
Hàm mất mát InfoNCE tiêu chuẩn vận hành dựa trên logits khoảng cách $z_{ij} = -d(x_i, y_j) / \tau$. Bản chất của việc học biểu diễn metric (Metric Learning) dựa trên **bất đẳng thức tam giác** ($d(x, z) \le d(x, y) + d(y, z)$) để ép các điểm tương đồng co cụm trong một quả cầu metric và đẩy các mẫu âm ra xa một cách có trật tự hình học. Khi lấy bình phương $d^2$, bất đẳng thức tam giác bị phá vỡ hoàn toàn ($d^2$ không còn là một metric). Nhiệt độ $\tau$ khi đó sẽ mang một ý nghĩa phân phối hoàn toàn khác so với nhánh đối chứng Euclidean.

> **Giải pháp bắt buộc #1:** Nếu sử dụng $d_L^2$ cho nhánh Lorentz, **bắt buộc nhánh Euclidean Control ($E_0$) cũng phải sử dụng bình phương khoảng cách Euclidean**:
> $$D_E(v, w) = \|v - w\|_2^2 = \sum_{j=1}^D (v_j - w_j)^2$$
> Tuyệt đối không được so sánh một bên dùng $d_L^2$ với một bên dùng khoảng cách Euclidean chuẩn $\|v-w\|_2$.

#### Vấn đề 2: Bình phương khoảng cách không có chặn trên (Unbounded Growth)
Với trần bán kính $R=2$ và $\kappa=1$, khoảng cách cực đại $d_{L,\max} \approx 4 \implies d_L^2 \approx 16$. Khi độ cong $\kappa \to 0$ (tiệm cận Euclidean), khoảng cách giữa hai vector đối cực trong quả cầu bán kính $R$ có thể lên tới $\|v - w\|_2^2 \le (2R)^2 = 16$. Độ lớn của $d^2$ tăng theo hàm bậc hai khiến logits bị dạt về vùng bão hòa của hàm Softmax, làm gradient bị triệt tiêu đối với các mẫu âm ở xa.

> **Giải pháp bắt buộc #2:** Chuẩn hóa logits bằng hằng số tỷ lệ an toàn $S = \frac{1}{2 R^2}$ chia đều cho cả nhánh Lorentz và Euclidean:
> $$z_{ij} = -\frac{d^2(x_i, y_j)}{2 R^2 \cdot \tau}$$
> nhằm giữ cho giá trị logits luôn nằm trong khoảng ổn định $[-1/\tau, 0]$.

#### Vấn đề 3: Sự biến dạng của Kernel Weighting và nguy cơ sụp đổ biểu diễn (Representation Collapse)
Đây là điểm mấu chốt sâu sắc nhất về mặt toán học tối ưu:
- **Với Geodesic Distance $d_L$:** Đạo hàm theo khoảng cách thỏa mãn:
  $$\frac{\partial z_{ij}}{\partial d_L} = -\frac{1}{\tau} \implies \text{Gradient theo tọa độ tỉ lệ với } \frac{1}{\sqrt{\delta^2 - 1}}$$
  Hàm này có độ dốc cực lớn ở vùng lân cận điểm trùng nhau ($\delta \to 1^+$). Nó tạo ra một lực đẩy/kéo cực kỳ sắc nhọn (sharp penalty), buộc mô hình phải phân tách rạch ròi các cặp thực thể dù chúng nằm rất gần nhau.
- **Với Squared Distance $d_L^2$:** Vì $d_L^2 \approx 2(\delta - 1)$ khi $\delta \to 1^+$, đạo hàm trở thành hằng số:
  $$\frac{\partial z_{ij}}{\partial \delta} \approx -\frac{2}{\tau} = \text{const}$$
  Gradient phẳng đều trên toàn bộ không gian. Khi các cặp dương (positive pairs) đã tương đối gần nhau, lực kéo của hàm mất mát bị suy giảm nghiêm trọng. Hậu quả là mô hình **không phạt đủ mạnh các sai lệch nhỏ**, dẫn tới hiện tượng biểu diễn bị co cụm lỏng lẻo (representation collapse) hoặc mất độ sắc nét trong xếp hạng Top-K.

#### Giải pháp Đột phá: Hàm Khoảng Cách Hỗn Hợp (Hybrid Distance Kernel)
Để dung hòa hoàn hảo giữa **tính an toàn số học tuyệt đối** (không bao giờ gặp lỗi $\infty$ của arcosh) và **độ nhọn gradient cần thiết** để tối ưu hóa Top-K, chúng tôi đề xuất hàm khoảng cách hỗn hợp điều khiển qua tham số $w \in [0, 1]$:

$$D_{\text{hybrid}}(x, y) = (1 - w) \cdot D_\kappa(x, y) + w \cdot \log\left( 1 + D_\kappa(x, y) \right)$$

với $D_\kappa(x, y)$ là bình phương khoảng cách Lorentz an toàn số học.
- Khi $w = 0$: Trở về thuần túy bình phương khoảng cách an toàn ($d_L^2$).
- Khi $w > 0$: Thành phần $\log(1 + d_L^2)$ đóng vai trò như một hàm nhân bão hòa (saturating kernel), tăng cường độ nhạy gradient ở vùng cự ly gần ($d \to 0$ thì $\frac{d}{d(d^2)}\log(1+d^2) \to 1$) và nén nhẹ gradient ở cự ly xa, triệt tiêu hoàn toàn nguy cơ bùng nổ số học.
- **Kế hoạch thực nghiệm:** Chạy pilot so sánh trực tiếp giữa hai cấu hình $w = 0.0$ và $w = 0.5$ trên Amazon Baby để xác định hàm nhân tối ưu trước khi triển khai toàn diện.

---

### 3.2. Xử lý triệt để hiện tượng suy giảm phổ $\beta$ decay trong tầng $H^{(1)}$: Ba phương án & Lựa chọn Spectral Re-weighting

Phản biện C của Peer Review đã chỉ ra một điểm xung đột kiến trúc tinh vi trong STAIR:
Trong bộ lọc phổ FSC của STAIR, biểu diễn tầng 1 được tính toán thông qua tích chập chuẩn hóa kèm hệ số suy giảm phổ tọa độ $\beta$:
$$H^{(1)} = (\widehat{A} H^{(0)}) \odot \beta, \quad \beta_j = 1 - \beta_{3, j} = 0.9 - 0.9 \left(\frac{j}{D}\right)^\gamma, \quad j \in [0, D-1]$$
Với số chiều $D=64$, khi chỉ số tọa độ $j$ tăng dần ($j \to 63$), hệ số $\beta_j$ suy giảm dần về $0$. Điều này đồng nghĩa với việc $H^{(1)}$ bị suy giảm năng lượng rất mạnh (mất tới $50\% - 90\%$ biên độ) ở các chiều tọa độ cao.

Tuy nhiên, theo cơ chế tiền xử lý SVD Whitening của STAIR, **các chiều tọa độ cao chính là nơi lưu trữ các biến thiên ngữ nghĩa chi tiết của đặc trưng văn bản và hình ảnh** (sau khi các chiều đầu tiên đã nắm giữ cấu trúc tương tác collaborative bậc thấp). Nếu đưa trực tiếp $H^{(1)}$ bị suy giảm phổ vào ánh xạ mũ Lorentz $\exp_0^\kappa$:
1. Chuẩn vector $\|H^{(1)}\|$ bị thu nhỏ nhân tạo, khiến các điểm bị kéo tụt về gần gốc tọa độ $\mathbf{o}$ của không gian hyperbolic.
2. Dung lượng biểu diễn hình học của không gian Lorentz ở các chiều ngữ nghĩa cao bị triệt tiêu lãng phí.

#### Ba phương án kỹ thuật xử lý:

- **Phương án A (Bảo thủ - Baseline Fallback):** Không dùng $H^{(1)}$, chỉ thực hiện Contrastive Learning trực tiếp trên tầng $H^{(0)}$ (Same-layer CL: $U_0 \leftrightarrow I_0$).  
  *Ưu điểm:* Hoàn toàn miễn nhiễm với suy giảm phổ $\beta$.  
  *Nhược điểm:* Đánh mất bản chất học biểu diễn xuyên tầng (Cross-layer contrastive) vốn là linh hồn của trường phái NLGCL/XSimGCL.
- **Phương án B (Khuyến nghị của Senior Researcher - Spectral Re-weighting Correction):**  
  Thực hiện khôi phục phổ (Spectral Re-weighting) cho $H^{(1)}$ trước khi đưa vào bộ chuẩn hóa tỷ lệ $q$ và ánh xạ Lorentz:
  $$\widetilde{H}_j^{(1)} = \frac{H_j^{(1)}}{\max(\beta_j, \epsilon_\beta)}, \quad \epsilon_\beta = 0.05$$
  Phép toán này bù đắp chính xác lượng năng lượng đã bị toán tử FSC làm suy giảm, đưa $H^{(1)}$ trở về cùng mặt bằng năng lượng phổ với $H^{(0)}$, bảo toàn trọn vẹn thông tin đa phương thái ở các chiều cao. Để phòng ngừa hiện tượng khuếch đại nhiễu số học tại các chiều có $\beta_j \approx \epsilon_\beta$, bổ sung thêm bước chặn chuẩn vector **thích ứng theo phân vị (Quantile-based Adaptive Norm Capping)**:
  $$M_{\text{norm}} = \text{Quantile}_{95\%}\left(\|H^{(0)}\|_2 \text{ trên toàn bộ tập train tại epoch 0}\right)$$
  $$\widetilde{H}^{(1)} = \widetilde{H}^{(1)} \cdot \min\left(1.0, \; \frac{M_{\text{norm}}}{\|\widetilde{H}^{(1)}\|_2 + 10^{-8}}\right)$$
  > **[Peer Review Vòng 2 — Sửa lỗi]:** Phiên bản trước sử dụng giá trị cứng $M_{\text{norm}} = 5.0$ là tùy ý, không có cơ sở thực nghiệm. Giá trị thích ứng theo phân vị 95% của phân phối chuẩn $H^{(0)}$ tại epoch khởi tạo đảm bảo ngưỡng chặn phù hợp với từng tập dữ liệu cụ thể (Baby, Sports, Electronics sẽ có $M_{\text{norm}}$ khác nhau). Giá trị này được đóng băng thành `torch.nn.Buffer` cùng với $q_{t,\ell}$.
- **Phương án C (Đa góc nhìn - Multi-view CL):** Thiết lập hàm mất mát tương phản trên cả 3 cặp tầng: $(H^{(0)}, H^{(1)})$, $(H^{(0)}, H^{(2)})$ và $(H^{(1)}, H^{(2)})$.  
  *Đánh giá:* Quá tốn kém bộ nhớ VRAM (tăng gấp 3 lần chi phí tính ma trận tương phản $4096 \times 4096$) và làm phức tạp hóa quá trình phân tích nguyên nhân - kết quả.

> **Quyết định chốt:** Áp dụng **Phương án B (Spectral Re-weighting)** làm cấu hình chuẩn của STAIR-LHC v1 (Revised). Đưa cấu hình không bù phổ (`H0-reweight`) vào nhánh Ablation P1 để đo đạc chính xác tác động thực tế của cơ chế bù phổ này.

---

### 3.3. Vấn đề cốt tử Peer Review bỏ sót #1: Triệt tiêu Self-Return Shortcut trong Positive Pairs

Đây là lỗ hổng phương pháp luận lớn nhất mà cả bản dự thảo ban đầu lẫn Peer Review đều chưa giải quyết triệt để.

Xét quá trình tạo mẫu tương phản dương giữa User $u$ và Item $i$ trên một cạnh tương tác có thực trong tập huấn luyện ($(u, i) \in \mathcal{E}_{\text{train}}$):
- Biểu diễn của User tại tầng 0: $H_u^{(0)} = E_u \in \mathbb{R}^D$.
- Biểu diễn của Item tại tầng 1 (Key):
  $$H_i^{(1)} = \sum_{v \in \mathcal{N}(i)} \frac{E_v}{\sqrt{d_i \cdot d_v}} \odot \beta$$
Vì $(u, i) \in \mathcal{E}_{\text{train}}$, chắc chắn rằng người dùng $u$ nằm trong tập lân cận $\mathcal{N}(i)$ của sản phẩm $i$! Do đó, ta có thể phân rã tường minh:
$$H_i^{(1)} = \underbrace{\frac{E_u}{\sqrt{d_i \cdot d_u}} \odot \beta}_{\text{Thành phần Self-Return từ chính User } u} + \sum_{v \in \mathcal{N}(i) \setminus \{u\}} \frac{E_v}{\sqrt{d_i \cdot d_v}} \odot \beta$$

#### Hệ quả tai hại của "Self-Return Shortcut":
Hàm mất mát InfoNCE sẽ tìm con đường ngắn nhất (lười biếng nhất) để cực tiểu hóa khoảng cách giữa $H_u^{(0)}$ và $H_i^{(1)}$. Do $H_i^{(1)}$ đã chứa sẵn một phần của $E_u$, mạng nơ-ron chỉ cần học cách **tăng biên độ của $E_u$ hoặc học một phép đồng nhất (identity mapping)** để bắt cặp thành phần $E_u$ này, thay vì phải nỗ lực học sự tương đồng cấu trúc đồ thị và đặc trưng đa phương thái giữa người dùng $u$ và các sản phẩm lân cận khác trong $\mathcal{N}(i) \setminus \{u\}$. Hiện tượng này đặc biệt nghiêm trọng đối với các sản phẩm có bậc nhỏ ($d_i$ bé), nơi thành phần $E_u / \sqrt{d_i d_u}$ chiếm ưu thế áp đảo trong tổng vector!

```
[MÔ HÌNH BỊ LỖ HỔNG SHORTCUT]
Query: H_u^(0) = E_u ────────┐ (Khoảng cách cực tiểu nhân tạo vì E_u có sẵn trong Key!)
                             ├──► Loss giảm ảo nhưng KHÔNG học cấu trúc đồ thị!
Key:   H_i^(1) = [E_u/√d_i d_u + ...] ┘

[CƠ CHẾ TRIỆT TIÊU SELF-RETURN (PROPOSED)]
Query: H_u^(0) = E_u ────────┐ (Bắt buộc phải so khớp với ngữ cảnh lân cận thực sự!)
                             ├──► Ép buộc mô hình học quan hệ liên kết đồ thị & đa phương thái!
Key:   H_i^(1) - E_u/√d_i d_u ┘
```

#### Giải pháp Triệt tiêu Self-Return có điều kiện (Conditional Self-Return Removal):
Trong quá trình gom batch và tính toán positive keys cho hàm mất mát InfoNCE, với mỗi cặp tương tác dương $(u, i)$, ta thực hiện phép trừ cục bộ:

$$\widetilde{H}_{i \mid u}^{(1)} = H_i^{(1)} - \mathbb{1}\left[(u, i) \in \mathcal{E}_{\text{train}}\right] \cdot \frac{E_u}{\sqrt{\widetilde{d}_i \cdot \widetilde{d}_u}} \odot \beta$$

trong đó $\widetilde{d}_i = \max(1, d_i - 1)$ và $\widetilde{d}_u = \max(1, d_u - 1)$ là bậc hiệu chỉnh sau khi tạm thời ngắt cạnh tương tác $(u, i)$.

#### Xử lý trường hợp biên: Items bậc 1 (Degree-1 Safety)

> **[Peer Review Vòng 2 — Bổ sung quan trọng]:** Khi sản phẩm $i$ chỉ có đúng một lượt tương tác ($d_i = 1$, tương đương chỉ có duy nhất cạnh $(u, i)$), phép trừ Self-Return sẽ triệt tiêu toàn bộ $H_i^{(1)}$, để lại vector gần-zero hoặc chính xác zero. Đây là trường hợp biên nguy hiểm vì:
> 1. Ánh xạ Lorentz $\exp_0^\kappa(\mathbf{0})$ sẽ đặt điểm tại gốc tọa độ hyperbolic $\mathbf{o}$, tạo ra **điểm thu hút giả (spurious attractor)** trong không gian Lorentz.
> 2. Gradient tại gốc tọa độ có thể gây ra hiện tượng co cụm (gravitational collapse) cho các mẫu âm lân cận.

**Giải pháp Norm Protection (Bảo vệ chuẩn vector):**
Sau phép trừ Self-Return, áp dụng bước bảo vệ chuẩn tối thiểu:
$$\widetilde{H}_{i \mid u}^{(1)} \leftarrow \begin{cases}
\widetilde{H}_{i \mid u}^{(1)} & \text{nếu } \|\widetilde{H}_{i \mid u}^{(1)}\|_2 \ge \epsilon_{\text{floor}} \\
\epsilon_{\text{floor}} \cdot \frac{\widetilde{H}_{i \mid u}^{(1)}}{\|\widetilde{H}_{i \mid u}^{(1)}\|_2 + 10^{-8}} & \text{nếu } \|\widetilde{H}_{i \mid u}^{(1)}\|_2 < \epsilon_{\text{floor}}
\end{cases}$$
với $\epsilon_{\text{floor}} = 10^{-2}$. Phép toán này giữ nguyên **hướng vector** (direction) nhưng đảm bảo chuẩn không bao giờ rơi xuống dưới ngưỡng an toàn, tránh tạo điểm thu hút giả tại gốc hyperbolic.

*Quy tắc triển khai bất biến:*
1. Phép loại trừ này **chỉ áp dụng đối với Positive Keys** trong batch tính loss CL.
2. Không áp dụng cho Negative Keys (đối với mẫu âm $j \neq i$, sản phẩm $j$ không có tương tác dương đang xét với $u$ nên không tồn tại shortcut này).
3. Không làm thay đổi ma trận kề $\widehat{A}$ của nhánh tính toán BPR chính thống.
4. **[Mới]** Áp dụng Norm Protection sau mỗi phép trừ Self-Return để bảo vệ items bậc 1.

---

### 3.4. Vấn đề cốt tử Peer Review bỏ sót #2: Tách biệt Hình học Hyperbolic khỏi Hiệu ứng Kernel (Constant-Radius Control - HC Arm)

Một luận điểm phản biện mang tính hủy diệt đối với các nghiên cứu áp dụng không gian Hyperbolic trong hệ khuyến nghị là:  
*Liệu mức tăng trưởng (nếu có) thực sự đến từ "Dung lượng hình học mở rộng theo hàm mũ của không gian Hyperbolic", hay thực chất chỉ là một hiệu ứng biến đổi phi tuyến (Kernel Trick) tình cờ làm thay đổi trọng số phân bổ mẫu âm trong hàm Softmax?*

#### Chứng minh toán học về tính suy biến trên mặt cầu đồng bán kính:
Xét hai điểm $v, w \in \mathbb{R}^D$ trong không gian tiếp xúc Euclidean. Khi đưa qua ánh xạ mũ Lorentz $\exp_0^\kappa$:
$$x = \phi_\kappa(v) = \left[ \frac{\cosh(\sqrt{\kappa}\|v\|)}{\sqrt{\kappa}}, \; \text{sinhc}(\sqrt{\kappa}\|v\|) v \right]^\top$$
$$y = \phi_\kappa(w) = \left[ \frac{\cosh(\sqrt{\kappa}\|w\|)}{\sqrt{\kappa}}, \; \text{sinhc}(\sqrt{\kappa}\|w\|) w \right]^\top$$

Tích vô hướng Lorentz có dạng:
$$-\langle x, y \rangle_L = x_0 y_0 - x_s^\top y_s = \frac{\cosh(\sqrt{\kappa}\|v\|)\cosh(\sqrt{\kappa}\|w\|)}{\kappa} - \text{sinhc}(\sqrt{\kappa}\|v\|)\text{sinhc}(\sqrt{\kappa}\|w\|) (v^\top w)$$

**Giả sử tất cả các vector đều có cùng chuẩn Euclidean (cùng bán kính $\|v\| = \|w\| = r$):**  
Đặt $a = \sqrt{\kappa} r$, ta có:
$$-\kappa \langle x, y \rangle_L = \cosh^2(a) - \sinh^2(a) \cos(\theta_{v, w})$$
trong đó $\cos(\theta_{v, w}) = \frac{v^\top w}{\|v\|\|w\|}$ chính là độ tương đồng Cosine chuẩn trong không gian Euclidean!

Khi đó, khoảng cách Lorentz trở thành:
$$d_L(x, y) = \frac{1}{\sqrt{\kappa}} \text{arcosh}\left( \cosh^2(a) - \sinh^2(a) \cos(\theta_{v, w}) \right)$$

Vì hàm $\text{arcosh}$ và hàm $-\cos(\theta)$ đều là các **hàm đơn điệu tăng**, thứ tự khoảng cách giữa bất kỳ cặp vector nào trên mặt cầu đồng bán kính trong không gian Lorentz **hoàn toàn đồng nhất $100\%$ với thứ tự khoảng cách góc Cosine trong không gian Euclidean**!
Không hề có bất kỳ một sự đảo trật tự xếp hạng lân cận nào được tạo ra bởi độ cong $\kappa$. Khác biệt duy nhất chỉ là hàm khoảng cách bị uốn cong phi tuyến, tương đương với việc áp dụng một hàm hạt nhân phi tuyến $f(\cos\theta)$ lên không gian Euclidean.

#### Thiết kế phép thử chẩn đoán quyết định: Nhánh HC (Constant-Radius Control Arm)
Để kiểm chứng dứt khoát giả thuyết về tính ưu việt của dung lượng hình học Hyperbolic:
1. **Quy trình nhánh HC:**
   - Chuẩn hóa L2 toàn bộ embeddings về mặt cầu đơn vị trước khi đưa vào ánh xạ Lorentz: $v_{\text{norm}} = v / (\|v\|_2 + 10^{-8})$.
   - Cố định bán kính $r = 1.0$ cho mọi thực thể.
   - Huấn luyện STAIR-LHC với nhánh đối chứng này.
2. **Tiêu chuẩn kết luận khoa học:**
   - **Trường hợp 1 (HC $\approx$ H0):** Nếu nhánh bán kính cố định HC đạt hiệu năng tương đương nhánh bán kính tự do H0, ta có bằng chứng toán học vững chắc để kết luận: **Độ cong Lorentz trong mô hình chỉ đóng vai trò như một hàm nhân phi tuyến (Kernel effect)**, việc tuyên bố mô hình học được "phân tầng hình học theo bán kính" (radial hierarchy) là sai sự thật.
   - **Trường hợp 2 (H0 vượt trội rõ rệt so với HC và E0):** Khi và chỉ khi H0 thắng cả E0 và HC với khoảng cách có ý nghĩa thống kê ($p < 0.05$), ta mới có đủ cơ sở khoa học để khẳng định: **Dung lượng biểu diễn hình học theo bán kính của không gian Hyperbolic thực sự mang lại giá trị biểu diễn vượt trội.**

---

### 3.5. Vấn đề cốt tử Peer Review bỏ sót #3: Cách ly nhóm Optimizer và xung đột với BSC Smoother

Trong kiến trúc STAIR gốc, bộ tối ưu hóa `AdamWSEvo` tích hợp sẵn một toán tử làm mịn phổ hành vi sản phẩm mang tên **BSC Smoother (Behavioral Spectral Convolution Smoother)**. Toán tử này áp dụng trực tiếp lên ma trận nhúng của Item (`Item.embeddings.weight`) sau mỗi bước tính gradient:
$$G_{\text{item}} \leftarrow \text{BSC\_Smoother}(G_{\text{item}}, S)$$
trong đó $S$ là ma trận tương đồng đồ thị kề của sản phẩm.

#### Xung đột gradient tiềm ẩn:
Khi bổ sung hàm mất mát Lorentz Contrastive Learning ($\mathcal{L}_{\text{CL}}$), luồng gradient tổng hợp chảy vào Item embedding bao gồm hai thành phần:
$$G_{\text{total}} = \nabla_{\text{item}} \mathcal{L}_{\text{BPR}} + \lambda(t) \cdot \nabla_{\text{item}} \mathcal{L}_{\text{CL}}$$
Nếu đưa toàn bộ $G_{\text{total}}$ qua toán tử BSC Smoother:
1. Toán tử BSC Smoother sẽ tự động áp đặt bộ lọc suy giảm phổ đồ thị lên cả gradient của Lorentz CL.
2. Điều này vô tình triệt tiêu các thành phần cập nhật hình học phi Euclidean mà nhánh Lorentz vừa nhọc công tính toán, làm méo mó mục tiêu tối ưu hóa của không gian hyperbolic.

#### Giải pháp Cách ly và Giám sát Gradient (Gradient Isolation & Monitoring):
1. **Định đồng hóa nhóm tham số (Disjoint Parameter Groups):**
   - Tách biệt rõ ràng việc theo dõi gradient của từng nhánh.
   - Không áp dụng BSC Smoother lên các tham số riêng của nhánh phụ (như tham số $\theta$ của độ cong $\kappa$ hoặc ma trận chiếu projector nếu có ở các pha ablation sau).
2. **Cơ chế giám sát góc định hướng Gradient (Gradient Cosine Diagnostics) — Phương pháp Baseline-Calibrated:**
   - Định kỳ mỗi 10 epochs, tính toán độ tương đồng Cosine giữa vector gradient của nhiệm vụ chính BPR và nhiệm vụ phụ CL:
     $$\rho_{\text{grad}} = \frac{\langle \nabla \mathcal{L}_{\text{BPR}}, \; \nabla \mathcal{L}_{\text{CL}} \rangle}{\|\nabla \mathcal{L}_{\text{BPR}}\|_2 \cdot \|\nabla \mathcal{L}_{\text{CL}}\|_2 + 10^{-8}}$$
   > **[Peer Review Vòng 2 — Sửa lỗi]:** Phiên bản trước sử dụng ngưỡng cứng $\rho_{\text{grad}} < -0.3$ là tùy ý, không có cơ sở lý thuyết hay thực nghiệm. **Phương pháp mới: Baseline-Calibrated Threshold:**
   > 1. Trong 10 epoch đầu tiên (khi $\lambda = 0$, chỉ có BPR), ghi nhận phân phối $\rho_{\text{grad}}$ giữa gradient BPR thực tế và gradient CL "ảo" (forward-only, không backward) để xây dựng **baseline distribution** $\mathcal{D}_{\text{baseline}}$.
   > 2. Ngưỡng cảnh báo được thiết lập tự động: $\theta_{\text{alarm}} = \mu_{\text{baseline}} - 2\sigma_{\text{baseline}}$.
   > 3. Nếu $\rho_{\text{grad}} < \theta_{\text{alarm}}$ kéo dài **liên tục 3 lần đo liên tiếp** (30 epochs), hệ thống mới kích hoạt cảnh báo, tránh false alarm do nhiễu stochastic.
   - Khi cảnh báo được kích hoạt, giải pháp dự phòng theo thứ tự ưu tiên:
     - **(a) Giảm $\lambda_{\max}$ xuống $50\%$.**
     - **(b) Áp dụng kỹ thuật chiếu triệt tiêu gradient đối kháng (Gradient Surgery / PCGrad).**
     - **(c) Ngắt gradient từ nhánh CL chảy vào Item embedding (`detach()`),** chỉ cho phép nhánh CL cập nhật thông qua User embedding.

---

### 3.6. Bác bỏ giả thuyết "Long-tail đồng nghĩa với Hyperbolic Hierarchy" trên đồ thị E-Commerce

Một nhầm lẫn kinh điển trong cộng đồng nghiên cứu hệ khuyến nghị là đánh đồng:
$$\text{Đồ thị có phân phối bậc đuôi dài (Power-law degree)} \implies \text{Đồ thị có cấu trúc phân tầng dạng cây (Tree-like hierarchy)}$$

**Phân tích phản bác khoa học:**
1. **Phân phối bậc đuôi dài (Heavy-tailed distribution):** Chỉ phản ánh hiện tượng một số ít sản phẩm "hot" (head items) có rất nhiều lượt tương tác, trong khi đa số sản phẩm (tail items) có rất ít tương tác. Thuộc tính này xuất hiện ở hầu hết mọi mạng xã hội và đồ thị thương mại điện tử do cơ chế liên kết ưu tiên (preferential attachment).
2. **Độ hyperbol (Gromov's $\delta$-hyperbolicity):** Đòi hỏi mọi tam giác trắc địa trên đồ thị phải có tính chất "mảnh" ($\delta$-slim), nghĩa là đồ thị không được chứa nhiều chu trình ngắn (short cycles).
3. **Thực tế trên đồ thị người dùng - sản phẩm (Bipartite User-Item Graph):**  
   Đồ thị thương mại điện tử chứa vô số các chu trình bậc 4 ($u_1 \to i_1 \to u_2 \to i_2 \to u_1$) do hiện tượng đồng mua sắm (co-purchasing) giữa các nhóm người dùng có cùng gu sở thích. Các sản phẩm cùng danh mục thường có mối quan hệ cạnh tranh thay thế ngang hàng (horizontal substitutes) chứ không hề có quan hệ cha - con phân cấp nghiêm ngặt (strict taxonomy).
   
Do đó, việc tuyên bố không gian Hyperbolic chắc chắn vượt trội không gian Euclidean trên đồ thị e-commerce là một **tuyên bố phiến diện, thiếu căn cứ toán học**. Đây chính là lý do vì sao STAIR-LHC bắt buộc phải thiết lập nhánh đối chứng Euclidean $E_0$ để kiểm chứng một cách khách quan nhất.

---

### 3.7. Vấn đề triệt tiêu toán học của DualHypCL (Curvature Vanishing Phenomenon)

Nhóm nghiên cứu tái khẳng định nguyên nhân toán học dẫn đến việc từ chối mô hình DualHypCL:
Trong DualHypCL, vector biểu diện $h$ được ánh xạ qua không gian Hyperbolic rồi chiếu ngược về không gian tiếp xúc Euclidean:
$$z = \exp_0^\kappa(W h) \in \mathbb{H}_\kappa^D$$
$$u = \log_0^\kappa(z) = \log_0^\kappa\left(\exp_0^\kappa(W h)\right) = W h \in T_0 \mathbb{H}_\kappa^D = \mathbb{R}^D$$

Sau đó, hàm mất mát tương phản được tính bằng tích vô hướng Cosine trong không gian tiếp xúc Euclidean:
$$\mathcal{L}_{\text{CL}} = -\log \frac{\exp(u_1^\top u_2 / \tau)}{\sum \exp(u_1^\top u_j / \tau)}$$

**Bằng chứng triệt tiêu:**
Vì $\log_0^\kappa$ là ánh xạ ngược chính xác của $\exp_0^\kappa$ trên toàn bộ miền xác định, biểu thức $u$ hoàn toàn không còn chứa tham số độ cong $\kappa$. Do đó:
$$\frac{\partial \mathcal{L}_{\text{CL}}}{\partial \kappa} \equiv 0$$
Bất kỳ gradient khác 0 nào thu được trong quá trình huấn luyện mô hình DualHypCL thực chất chỉ là **nhiễu làm tròn dấu phẩy động (floating-point roundoff noise)** hoặc do các hàm chặn biên `clamp` vô tình cắt xén giá trị. Việc xây dựng một mô hình học sâu phức tạp với hai độ cong độc lập nhưng về mặt toán học độ cong lại bị triệt tiêu hoàn toàn là một sai lầm thiết kế nghiêm trọng cần phải loại bỏ.

---

### 3.8. Phân tích rủi ro của Semantic Filtering đối với False Negatives theo cảnh báo EStair

Tài liệu [EStair.md](../EStair.md) đã cảnh báo sâu sắc về nghịch lý: *"Tương đồng ngữ nghĩa đa phương thái không đồng nghĩa với khả năng tương thích hành vi (Semantic similarity does NOT equal preference compatibility)."*

Trong một số đề xuất trước đây, người ta thường áp dụng kỹ thuật lọc mẫu âm ngữ nghĩa (Semantic Hard Negative Filtering): tính độ tương đồng cosine giữa các sản phẩm dựa trên vector text/image, nếu độ tương đồng vượt một ngưỡng $\theta_{\text{sim}}$ thì loại bỏ mẫu đó khỏi tập mẫu âm của InfoNCE vì nghi ngờ nó là "mẫu âm giả" (false negative).

**Phân tích rủi ro của Senior Researcher:**
1. Việc hai chiếc áo cùng màu đỏ, cùng kiểu dáng (độ tương đồng ngữ nghĩa cực cao) không có nghĩa là người dùng sẽ mua cả hai. Người dùng thường chỉ chọn một trong hai (sản phẩm thay thế). Chiếc áo còn lại hoàn toàn có thể là một mẫu âm chân thực có giá trị thông tin cao (informative hard negative).
2. Việc tự ý dùng một ngưỡng cứng $\theta_{\text{sim}}$ để xóa bỏ mẫu âm sẽ vô tình loại bỏ đúng những mẫu âm khó nhất, làm phẳng mặt phẳng tối ưu và làm giảm độ sắc nét của không gian biểu diễn.
3. **Quyết định chốt:** Trong STAIR-LHC v1, **tắt hoàn toàn tính năng Semantic Hard Negative Filtering**. Chỉ sử dụng bộ lọc mẫu âm dựa trên ma trận tương tác thực tế của tập train (Train-Positive Matrix $P$) để loại trừ các tương tác dương đã biết của người dùng trong tập huấn luyện.

---

## 4. ĐỐI CHIẾU TÀI LIỆU HỌC THUẬT QUỐC TẾ (LITERATURE BENCHMARK)

#### Bảng 4.1: Ma trận đối chiếu các công trình liên quan và bài học kế thừa

| Tài Liệu Tham Khảo | Đóng Góp Cốt Lõi | Giới Hạn Khi Áp Dụng Vào STAIR | Bài Học Kế Thừa Vào STAIR-LHC v1 |
| :--- | :--- | :--- | :--- |
| **STAIR (arXiv 2024)** [R1] | Khởi tạo MI Whitening, tích chập phổ FSC, toán tử làm mịn BSC trong AdamWSEvo. | Chưa khai thác cấu trúc hình học phi Euclidean. | Giữ nguyên 100% làm nền tảng Backbone vững chắc. |
| **Nickel & Kiela (ICML 2018)** [R2] | Mô hình Lorentz biểu diễn cấu trúc cây phân cấp liên tục với độ ổn định số học cao. | Áp dụng trên đồ thị tri thức đơn phương thái WordNet, không phải hệ khuyến nghị đa phương thái. | Kế thừa công thức ánh xạ mũ Lorentz $\exp_0$ và không gian Minkowski. |
| **HGCN (NeurIPS 2019)** [R3] | Phép toán tích chập đồ thị trực tiếp trên đa tạp Hyperbolic. | Chi phí tính toán cực lớn, dễ gây tràn số khi áp dụng trên đồ thị quy mô lớn. | Chỉ sử dụng Lorentz ở nhánh điều hòa phụ (Auxiliary loss), giữ suy luận Euclidean. |
| **Mishne et al. (ICML 2023)** [R4] | Phân tích giới hạn số học và tính bất ổn định của biểu diễn Hyperbolic trong học máy. | Phản bác luận điểm "Lorentz tự động ổn định tuyệt đối". | Bắt buộc phải áp dụng Radius Capping $R=2$, FP32 ngoài autocast và khai triển chuỗi Taylor gần 0. |
| **XSimGCL (arXiv 2022)** [R5] | Nhánh tương phản đa tầng cực kỳ tinh gọn, không cần augment đồ thị phức tạp. | Chỉ khảo sát trên không gian Euclidean thuần túy. | Kế thừa cấu trúc đối chứng xuyên tầng $H^{(0)} \leftrightarrow H^{(1)}$ và rút trích Unique ID. |
| **HCMKR (ECML-PKDD 2024)** [R6] | Tương phản Hyperbolic kết hợp tri thức đa phương thái trên đồ thị tri thức (KG). | Phụ thuộc vào đồ thị tri thức ngoài; chưa xử lý triệt tiêu self-return. | Kế thừa kỹ thuật chuẩn hóa thang đo thực nghiệm $q_{t,\ell}$. |
| **HyperCL (IEEE TKDE 2024)** [R7] | Học tương phản Hyperbolic cho lọc cộng tác. | Kiến trúc phức tạp, thời gian huấn luyện kéo dài. | Tinh gọn hóa thành một module điều hòa phụ có thể bật/tắt linh hoạt. |

---

## 5. ĐẶC TẢ KIẾN TRÚC HOÀN CHỈNH: STAIR-LHC v1 (REVISED)

### 5.1. Sơ đồ luồng dữ liệu tổng thể (Pipeline Flowchart)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     STAIR-LHC v1 (REVISED) PIPELINE                                    │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                        │
│   Đầu vào: Đặc trưng Text (384D), Visual (4096D), Đồ thị tương tác User-Item R_train                   │
│                                           │                                                            │
│                                           ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ BACKBONE STAIR CHUẨN (KHÔNG THAY ĐỔI TOÁN TỬ CỐT LÕI)                                          │   │
│   │ • Khởi tạo: SVD Whitening + Modal Interaction ──► E_u^(0) ∈ R^64, E_i^(0) ∈ R^64               │   │
│   │ • Tích chập phổ: FSC 3 tầng với hệ số suy giảm phổ β_3                                         │   │
│   │ • Đánh giá & Xếp hạng: s(u, i) = <H_u_bar, H_i_bar> (Tích vô hướng Euclidean 100%)             │   │
│   │ • Tối ưu hóa: AdamWSEvo + Toán tử BSC Smoother cho Item Embeddings                             │   │
│   └───────────────────────────────────────┬────────────────────────────────────────────────────────┘   │
│                                           │                                                            │
│                    ┌──────────────────────┴─────────────────────────────────┐                          │
│                    ▼                                                        ▼                          │
│          [NHÁNH CHÍNH: BPR LOSS]                             [NHÁNH PHỤ: LORENTZ CONTRASTIVE]          │
│                    │                                                        │                          │
│                    │                               ┌────────────────────────┴───────────────────────┐  │
│                    │                               │                                                │  │
│                    │                         H^(0) = E(t)                      H^(1) = (Â E) ⊙ β    │  │
│                    │                       (Euclidean, D=64)                   (Euclidean, D=64)    │  │
│                    │                               │                                │               │  │
│                    │                               │               ┌────────────────┴─────────────┐ │  │
│                    │                               │               │ BƯỚC 1: SELF-RETURN REMOVAL  │ │  │
│                    │                               │               │ H~(1) = H(1) - E_u/√d_i d_u  │ │  │
│                    │                               │               └────────────────┬─────────────┘ │  │
│                    │                               │                                │               │  │
│                    │                               │               ┌────────────────┴─────────────┐ │  │
│                    │                               │               │ BƯỚC 2: SPECTRAL RE-WEIGHT   │ │  │
│                    │                               │               │ H~(1) / max(β_j, 0.05)       │ │  │
│                    │                               │               └────────────────┬─────────────┘ │  │
│                    │                               │                                │               │  │
│                    │                               └────────────────┬───────────────┘               │  │
│                    │                                                │                               │  │
│                    │                                ┌───────────────┴──────────────┐                │  │
│                    │                                │ CHUẨN HÓA THANG ĐO & BÁN KÍNH│                │  │
│                    │                                │ q = median ||H||; a = H / q  │                │  │
│                    │                                │ v = R · tanh(||a||/R) · a/||a│                │  │
│                    │                                └───────────────┬──────────────┘                │  │
│                    │                                                │                               │  │
│                    │                                ┌───────────────┴──────────────┐                │  │
│                    │                                │ ÁNH XẠ LORENTZ EXP0 (FP32)   │                │  │
│                    │                                │ x = [cosh(√κ r)/√κ, sinhc·v] │                │  │
│                    │                                └───────────────┬──────────────┘                │  │
│                    │                                                │                               │  │
│                    │                                ┌───────────────┴──────────────┐                │  │
│                    │                                │ HYBRID DISTANCE KERNEL       │                │  │
│                    │                                │ z = -(1-w)·d_L² - w·log(1+d²)│                │  │
│                    │                                └───────────────┬──────────────┘                │  │
│                    │                                                │                               │  │
│                    │                                ┌───────────────┴──────────────┐                │  │
│                    │                                │ MULTI-POSITIVE INFONCE       │                │  │
│                    │                                │ 2 Chiều: U0 ↔ I1, I0 ↔ U1    │                │  │
│                    │                                └───────────────┬──────────────┘                │  │
│                    │                                                │                               │  │
│                    └───────────────────────┬────────────────────────┘                               │  │
│                                            ▼                                                         │  │
│                    TỔNG HỢP HÀM MẤT MÁT: L_total = L_BPR + λ(t) · L_LHC                              │  │
│                    (Lịch trình Warmup: 20 epoch đầu λ=0, tăng dần tới epoch 50)                      │  │
│                                                                                                        │
│   GIAI ĐOẠN SUY LUẬN (INFERENCE): 100% EUCLIDEAN DOT PRODUCT (HOÀN TOÀN TẮT NHÁNH LORENTZ)             │
│                                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 5.2. Hợp đồng tầng biểu diễn (Layer Contract), Self-Return Removal & Spectral Re-weighting

Hợp đồng tính toán giữa các tầng biểu diễn tuân thủ nghiêm ngặt thứ tự số học của STAIR:
$$H^{(0)} = E \in \mathbb{R}^{(|U| + |I|) \times D}$$
$$H^{(\ell+1)} = (\widehat{A} H^{(\ell)}) \odot \beta, \quad \ell \in \{0, 1, 2\}$$
$$\bar{H} = \left(\sum_{\ell=0}^L H^{(\ell)}\right) \odot \frac{1 - \beta}{1 - \beta^{L+1}}$$

Nhánh điều hòa Lorentz CL chỉ trích xuất hai tầng: tầng gốc $H^{(0)}$ và tầng tích chập thứ nhất $H^{(1)}$.
Trước khi đưa vào không gian Lorentz, $H^{(1)}$ phải trải qua hai bộ biến đổi bắt buộc:
1. **Triệt tiêu Self-Return trên Positive Keys:**
   $$\widetilde{H}_{i \mid u}^{(1)} = H_i^{(1)} - \frac{E_u}{\sqrt{\widetilde{d}_i \cdot \widetilde{d}_u}} \odot \beta$$
2. **Khôi phục phổ (Spectral Re-weighting):**
   $$\widehat{H}_j^{(1)} = \frac{\widetilde{H}_j^{(1)}}{\max(\beta_j, 0.05)}, \quad j \in [0, D-1]$$

---

### 5.3. Chuẩn hóa thang đo cố định ($q_{t,\ell}$) và Chiếu bán kính biến thiên có chặn ($R=2$)

Tại bước khởi tạo (`epoch 0`), trước khi bắt đầu tối ưu hóa, mô hình rút trích độ lớn chuẩn Euclidean trung vị cho từng loại thực thể $t \in \{\text{user}, \text{item}\}$ và từng tầng $\ell \in \{0, 1\}$ trên tập huấn luyện:
$$q_{t, \ell} = \max\left(10^{-3}, \; \text{Median}_{n \in V_t^{\text{train}}} \|H_n^{(\ell)}(0)\|_2\right)$$
Giá trị $q_{t, \ell}$ được đóng băng vĩnh viễn thành các hằng số đệm (`torch.nn.Buffer`), hoàn toàn không tham gia cập nhật gradient.

Với mọi vector ẩn $H \in \mathbb{R}^D$, bước chuẩn hóa thang đo được thực hiện:
$$a = \frac{H}{q_{t, \ell}} \in \mathbb{R}^D$$

Để triệt tiêu hoàn toàn nguy cơ tràn số (overflow) khi tính toán các hàm mũ $\cosh$ và $\sinh$ trong không gian Lorentz, áp dụng hàm chặn bán kính hyperbolic bão hòa trơn với bán kính cực đại $R = 2.0$:
$$v = \text{cap}_R(a) = R \cdot \tanh\left(\frac{\|a\|_2}{R}\right) \cdot \frac{a}{\|a\|_2 + 10^{-8}}$$

*Bảo đảm toán học:*  
$\|v\|_2 < R = 2.0$ với mọi vector đầu vào. Do đó, đối số của các hàm hyperbolic không bao giờ vượt quá $2.0 \sqrt{\kappa}$. Với $\kappa = 1.0$, giá trị cực đại của $\cosh(2) \approx 3.762$ và $\sinh(2) \approx 3.627$, loại bỏ $100\%$ nguy cơ tràn số dấu phẩy động trên GPU.

---

### 5.4. Ánh xạ Lorentz (Lorentz Exponential Map) và Tiệm cận Euclidean khi $\kappa \to 0$

Không gian Lorentz $D$ chiều $\mathbb{H}_\kappa^D$ với độ cong $-\kappa$ ($\kappa > 0$) được định nghĩa là một đa tạp Riemann nhúng trong không gian Minkowski $\mathbb{R}^{D+1}$:
$$\mathbb{H}_\kappa^D = \left\{ x = [x_0, x_s]^\top \in \mathbb{R}^{D+1} : \langle x, x \rangle_L = -x_0^2 + \|x_s\|_2^2 = -\frac{1}{\kappa}, \; x_0 > 0 \right\}$$

Ánh xạ mũ từ gốc tọa độ $\mathbf{o} = [1/\sqrt{\kappa}, \mathbf{0}_D]^\top$ đưa một vector tiếp xúc $v \in T_{\mathbf{o}} \mathbb{H}_\kappa^D \cong \mathbb{R}^D$ lên đa tạp Lorentz được tính toán thông qua hàm $\text{sinhc}(z) = \sinh(z)/z$:
$$\phi_\kappa(v) = \exp_{\mathbf{o}}^\kappa(v) = \left[ \frac{\cosh(\sqrt{\kappa} r)}{\sqrt{\kappa}}, \; \text{sinhc}(\sqrt{\kappa} r) \cdot v \right]^\top \in \mathbb{H}_\kappa^D$$
với $r = \|v\|_2$.

Khoảng cách bình phương Lorentz giữa hai điểm $x, y \in \mathbb{H}_\kappa^D$ là:
$$D_\kappa(x, y) = \frac{1}{\kappa} \left[ \text{arcosh}\left(-\kappa \langle x, y \rangle_L\right) \right]^2$$

*Tính chất tiệm cận bảo toàn (Euclidean Limit):*  
Khi $\kappa \to 0^+$, theo khai triển Taylor của hàm $\cosh$ và $\text{arcosh}$, ta có bảo đảm giải tích:
$$\lim_{\kappa \to 0^+} D_\kappa(\phi_\kappa(v), \phi_\kappa(w)) = \|v - w\|_2^2$$
Nhờ tính chất này, cấu hình đối chứng Euclidean $E_0$ được định nghĩa chính xác là $D_0(v, w) = \|v - w\|_2^2$, bảo đảm tính liên tục và công bằng toán học tuyệt đối giữa các nhánh đối chứng.

---

### 5.5. Cấu trúc Numerical Kernel, Khai triển chuỗi Taylor $t \to 0$ và Hàm khoảng cách Hybrid

Đặt đối số của hàm khoảng cách là $\delta = -\kappa \langle x, y \rangle_L = 1 + t$ với $t \ge 0$.  
Khi hai điểm nằm rất gần nhau ($t \to 0^+$), hàm $\text{arcosh}(1 + t)$ có đạo hàm tiến tới vô cùng, gây ra lỗi $\text{NaN}$ trong quá trình lan truyền ngược gradient.

STAIR-LHC v1 giải quyết triệt để vấn đề này bằng cách kết hợp **khai triển chuỗi Taylor bậc 4** quanh điểm $t = 0$:
$$\text{arcosh}(1 + t)^2 = 2t - \frac{t^2}{3} + \frac{4t^3}{45} - \frac{t^4}{35} + \mathcal{O}(t^5)$$

Cơ chế chuyển mạch kép an toàn số học:
$$D_\kappa(x, y) = \begin{cases} 
\frac{1}{\kappa} \left( 2t - \frac{t^2}{3} + \frac{4t^3}{45} \right), & \text{nếu } 0 \le t < 10^{-4} \\ 
\frac{1}{\kappa} \left[ \text{arcosh}\left(\max(1 + 10^{-7}, \; 1 + t)\right) \right]^2, & \text{nếu } t \ge 10^{-4} 
\end{cases}$$

Kết hợp với hàm khoảng cách hỗn hợp (Hybrid Distance Kernel) đã phân tích tại §3.1:
$$D_{\text{hybrid}}(x, y) = (1 - w) \cdot D_\kappa(x, y) + w \cdot \log\left(1 + D_\kappa(x, y)\right)$$
Toàn bộ ma trận khoảng cách giữa tập User và Item trong batch được tính toán thông qua một phép nhân ma trận khối GEMM duy nhất:
$$-\kappa \langle X, Y \rangle_L = \kappa \left( X_0 Y_0^\top - X_s Y_s^\top \right) \in \mathbb{R}^{B_u \times B_i}$$
loại bỏ hoàn toàn việc tạo tensor trung gian 3 chiều $B \times B \times D$, tiết kiệm tối đa bộ nhớ VRAM.

---

### 5.6. Ma trận Train-Positive đa mẫu (Multi-Positive InfoNCE) 2 hướng

Từ batch tương tác huấn luyện, trích xuất tập các người dùng duy nhất $\mathcal{U}_B$ (kích thước $B_u$) và tập các sản phẩm duy nhất $\mathcal{I}_B$ (kích thước $B_i$).

Xây dựng ma trận nhãn dương đa mẫu $P \in \{0, 1\}^{B_u \times B_i}$ bằng cách tra cứu trực tiếp cấu trúc ma trận kề thưa của tập huấn luyện:
$$P_{u, i} = \mathbb{1}\left[(u, i) \in \mathcal{E}_{\text{train}}\right]$$
*Tuyệt đối không truy cập hoặc đọc thông tin từ tập Validation hay Test.*

Hàm mất mát tương phản được tính toán đối xứng theo **hai chiều độc lập**:
1. **Chiều 1 ($U_0 \to I_1$):** User tầng 0 tìm kiếm Item tầng 1:
   $$\ell_u^{(1)} = -\frac{1}{|P_u|} \sum_{i \in P_u} \left[ z_{u, i}^{(1)} - \text{logsumexp}_{j \in \mathcal{I}_B}\left(z_{u, j}^{(1)}\right) \right], \quad z_{u, i}^{(1)} = -\frac{D_{\text{hybrid}}\left(\phi_\kappa(v_{u, 0}), \; \phi_\kappa(\widetilde{v}_{i \mid u, 1})\right)}{2 R^2 \cdot \tau}$$
2. **Chiều 2 ($I_0 \to U_1$):** Item tầng 0 tìm kiếm User tầng 1:
   $$\ell_i^{(2)} = -\frac{1}{|P_i^\top|} \sum_{u \in P_i^\top} \left[ z_{i, u}^{(2)} - \text{logsumexp}_{v \in \mathcal{U}_B}\left(z_{i, v}^{(2)}\right) \right], \quad z_{i, u}^{(2)} = -\frac{D_{\text{hybrid}}\left(\phi_\kappa(v_{i, 0}), \; \phi_\kappa(v_{u, 1})\right)}{2 R^2 \cdot \tau}$$

Hàm mất mát tương phản tổng hợp:
$$\mathcal{L}_{\text{LHC}} = \frac{1}{2 \cdot |A_u|} \sum_{u \in A_u} \ell_u^{(1)} + \frac{1}{2 \cdot |A_i|} \sum_{i \in A_i} \ell_i^{(2)}$$
trong đó $A_u$ và $A_i$ là tập các hàng/cột hợp lệ có ít nhất một mẫu dương và một mẫu âm trong batch.

---

### 5.7. Tổng hợp hàm mất mát, Lịch trình Warmup $\lambda(t)$ và Phân nhóm Optimizer

Hàm mất mát toàn cục:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BPR}} + \lambda(t) \cdot \mathcal{L}_{\text{LHC}}$$

Để bảo đảm cấu trúc đa tạp Euclidean của Backbone STAIR được định hình vững chắc trong giai đoạn đầu, hệ số điều hòa $\lambda(t)$ tuân thủ lịch trình Warmup tuyến tính:
$$\lambda(t) = \lambda_{\max} \cdot \text{clip}\left(\frac{t - 20}{30}, \; 0.0, \; 1.0\right)$$
- **Từ Epoch 0 đến Epoch 20:** $\lambda(t) = 0.0$ (Huấn luyện $100\%$ BPR thuần túy để ổn định không gian nhúng).
- **Từ Epoch 21 đến Epoch 49:** $\lambda(t)$ tăng tuyến tính từ $0$ lên $\lambda_{\max}$.
- **Từ Epoch 50 trở đi:** $\lambda(t) = \lambda_{\max}$ cố định.

#### Bảng 5.1: Hợp đồng phân nhóm tham số trong Optimizer

| Nhóm Tham Số (Parameter Group) | Tốc Độ Học (Learning Rate) | Trọng Số Suy Giảm (Weight Decay) | Toán Tử Làm Mịn (Smoother) |
| :--- | :---: | :---: | :---: |
| `User.embeddings.weight` | Baseline $lr$ (e.g., $10^{-3}$) | Baseline $wd$ (e.g., $10^{-4}$) | `None` (Không áp dụng) |
| `Item.embeddings.weight` | Baseline $lr$ (e.g., $10^{-3}$) | Baseline $wd$ (e.g., $10^{-4}$) | **BSC Smoother** (Chỉ làm mịn gradient cập nhật) |
| Hằng số đệm ($q, \kappa, \beta$) | $0.0$ | $0.0$ | `None` (Đóng băng không cập nhật) |

---

### 5.8. Thuật toán và Mã giả triển khai các module cốt lõi

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LorentzGeometryModule(nn.Module):
    """
    Module hình học Lorentz an toàn số học tuyệt đối.
    Hỗ trợ: Bounded Radius Capping, Exp Map, Taylor Series acosh^2, Hybrid Kernel.
    """
    def __init__(self, kappa=1.0, radius_cap=2.0, w_hybrid=0.0, eps_taylor=1e-4):
        super().__init__()
        self.kappa = kappa
        self.radius_cap = radius_cap
        self.w_hybrid = w_hybrid
        self.eps_taylor = eps_taylor

    def cap_radius(self, a):
        """Chiếu vector vào quả cầu bán kính R thông qua hàm tanh bão hòa trơn."""
        norm = torch.norm(a, p=2, dim=-1, keepdim=True)
        scale = torch.tanh(norm / self.radius_cap) / (norm + 1e-8)
        return self.radius_cap * scale * a

    def lorentz_exp0(self, v):
        """Ánh xạ mũ Lorentz từ gốc tọa độ: R^D -> H_kappa^D (Minkowski R^{D+1})."""
        r = torch.norm(v, p=2, dim=-1, keepdim=True)
        sqrt_k = math.sqrt(self.kappa)
        kr = sqrt_k * r
        # sinhc(kr) = sinh(kr) / kr; an toàn tại 0
        sinhc = torch.where(kr < 1e-5, 1.0 + (kr**2) / 6.0, torch.sinh(kr) / (kr + 1e-8))
        x0 = torch.cosh(kr) / sqrt_k
        xs = sinhc * v
        return torch.cat([x0, xs], dim=-1)

    def pairwise_lorentz_distance_squared(self, x, y):
        """
        Tính ma trận bình phương khoảng cách Lorentz bằng GEMM.
        x: [B_u, D+1], y: [B_i, D+1]
        """
        # -<x, y>_L = x0 y0^T - xs ys^T
        inner_L = x[:, 0:1] @ y[:, 0:1].t() - x[:, 1:] @ y[:, 1:].t()
        # Đối số: delta = -kappa * <x, y>_L = 1 + t
        delta = self.kappa * inner_L
        delta = torch.clamp(delta, min=1.0)
        t = delta - 1.0

        # Khai triển chuỗi Taylor gần t = 0 để tránh kỳ dị đạo hàm
        taylor_series = 2.0 * t - (t**2) / 3.0 + (4.0 * (t**3)) / 45.0
        # Nhánh chuẩn dùng acosh an toàn với clamp
        acosh_branch = torch.acosh(torch.clamp(delta, min=1.0 + 1e-7)).pow(2)
        d_squared = torch.where(t < self.eps_taylor, taylor_series, acosh_branch) / self.kappa
        return d_squared

    def compute_hybrid_distance(self, x, y):
        """Hàm khoảng cách hỗn hợp (Hybrid Distance Kernel)."""
        d2 = self.pairwise_lorentz_distance_squared(x, y)
        if self.w_hybrid == 0.0:
            return d2
        return (1.0 - self.w_hybrid) * d2 + self.w_hybrid * torch.log1p(d2)

def remove_self_return_positive_keys(H1_item, E_user, pos_u_idx, pos_i_idx, deg_u, deg_i, beta, eps_floor=1e-2):
    """
    Loại bỏ thành phần self-return E_u / sqrt(d_i * d_u) * beta trên positive keys.
    [Revised v2] Bổ sung Norm Protection cho items bậc 1 (degree-1 safety).
    """
    H1_pos = H1_item[pos_i_idx].clone()
    d_u_eff = torch.clamp(deg_u[pos_u_idx] - 1.0, min=1.0)
    d_i_eff = torch.clamp(deg_i[pos_i_idx] - 1.0, min=1.0)
    norm_factor = torch.sqrt(d_u_eff * d_i_eff).unsqueeze(-1)
    self_return = (E_user[pos_u_idx] / norm_factor) * beta.unsqueeze(0)
    H1_pos_corrected = H1_pos - self_return
    
    # [Peer Review Vòng 2] Norm Protection: bảo vệ items bậc 1 khỏi vector zero
    corrected_norm = torch.norm(H1_pos_corrected, p=2, dim=-1, keepdim=True)
    need_protection = (corrected_norm < eps_floor).float()
    safe_direction = H1_pos_corrected / (corrected_norm + 1e-8)
    H1_pos_corrected = (1 - need_protection) * H1_pos_corrected + need_protection * (eps_floor * safe_direction)
    
    return H1_pos_corrected

def spectral_reweight(H1, beta, eps_beta=0.05, max_norm=None):
    """
    Bù suy giảm phổ cho tầng H^(1).
    [Revised v2] max_norm giờ là tham số thích ứng theo quantile 95% của ||H^(0)||,
    được tính một lần tại epoch 0 và đóng băng thành buffer.
    Nếu max_norm=None, chỉ thực hiện chia beta mà không chặn chuẩn.
    """
    beta_safe = torch.clamp(beta, min=eps_beta).unsqueeze(0)
    H1_reweighted = H1 / beta_safe
    if max_norm is not None:
        norm = torch.norm(H1_reweighted, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        H1_reweighted = H1_reweighted * scale
    return H1_reweighted
```

---

## 6. PHÂN TÍCH CHI PHÍ TÍNH TOÁN & HỒ SƠ BỘ NHỚ VRAM

1. **Bộ nhớ ma trận Logits tương phản:**  
   Với kích thước batch tối đa 1024 cặp tương tác huấn luyện, số lượng thực thể duy nhất sau khi lọc trùng (deduplication) thỏa mãn: $|\mathcal{U}_B| \le 1024$ và $|\mathcal{I}_B| \le 1024$.
   Kích thước ma trận khoảng cách tương phản:
   $$\text{Matrix Size} = 1024 \times 1024 \times 4\text{ bytes} \approx \mathbf{4.0\text{ MiB}}$$
   Ngay cả khi mở rộng lên quy mô $4096 \times 4096$ trên Amazon Electronics, bộ nhớ ma trận khoảng cách cũng chỉ chiếm **$64.0\text{ MiB}$**, hoàn toàn nằm gọn trong dung lượng 16GB của GPU Tesla T4.
2. **Kỹ thuật Micro-Chunking bảo toàn mẫu số InfoNCE:**  
   Trong trường hợp mở rộng batch lớn trên Electronics, quá trình tính toán loss InfoNCE được chia thành các khối nhỏ gồm 512 người dùng (`Chunk Size = 512`). Lưu ý sống còn: **Mẫu số của hàm Softmax vẫn phải giữ nguyên toàn bộ tập ứng viên $|\mathcal{I}_B|$**, tuyệt đối không chia cắt không gian mẫu âm làm biến dạng hàm mục tiêu InfoNCE.
3. **Độ phức tạp tính toán:**  
   Toàn bộ nhánh Lorentz CL chỉ tốn đúng hai phép nhân ma trận dày $B_u \times (D+1) \times B_i$. Với $D=64$, chi phí này chỉ chiếm chưa đầy $12\%$ tổng thời gian huấn luyện mỗi epoch của STAIR backbone.
4. **Không phát sinh độ trễ khi suy luận:**  
   Toàn bộ nhánh hình học Lorentz hoàn toàn bị vô hiệu hóa khi đánh giá xếp hạng trên tập Validation và Test. Độ trễ suy luận và lượng tiêu thụ VRAM khi chạy thực tế là **Zero Overhead ($0\%$)**.
5. **[Peer Review Vòng 2] Ước tính VRAM chi tiết theo tập dữ liệu:**

   | Thành phần | Amazon Baby (B=1024) | Amazon Sports (B=1024) | Amazon Electronics (B=4096) |
   | :--- | :---: | :---: | :---: |
   | Lorentz Embeddings ($2B \times 65 \times 4$ bytes) | ~0.5 MiB | ~0.5 MiB | ~2.0 MiB |
   | Ma trận khoảng cách ($B_u \times B_i \times 4$ bytes) | ~4.0 MiB | ~4.0 MiB | ~64.0 MiB |
   | Gradient + Autograd Graph (\~2x forward) | ~9.0 MiB | ~9.0 MiB | ~132.0 MiB |
   | **Tổng overhead Lorentz CL** | **~13.5 MiB** | **~13.5 MiB** | **~198.0 MiB** |
   | STAIR Backbone (tham khảo V5) | ~2.8 GiB | ~4.2 GiB | ~8.5 GiB |
   | **% overhead so với Backbone** | **~0.5%** | **~0.3%** | **~2.3%** |
   
   > **Kết luận:** Chi phí bộ nhớ VRAM của nhánh Lorentz CL là không đáng kể ($< 3\%$) trên mọi tập dữ liệu, hoàn toàn nằm trong ngân sách 16GB của GPU Tesla T4.

---

## 7. GIAO THỨC THỰC NGHIỆM, MA TRẬN ABLATION & TIÊU CHÍ DỪNG

> **⚠️ LƯU Ý CHIẾN LƯỢC THỰC THI (Peer Review Vòng 2):**  
> Theo nguyên tắc "Fail-fast", **bước đầu tiên là triển khai và chạy kiến trúc đề xuất chính (B0 + H0)** trên Amazon Sports để xác nhận nhánh Lorentz CL có mang lại cải thiện hay không. **Chỉ khi H0 vượt B0 ít nhất +0.5% NDCG@20, mới tiếp tục triển khai các nhánh ablation E0, HC, v.v.** Cách tiếp cận này tiết kiệm đáng kể ngân sách GPU (~25h thay vì ~140h cho toàn bộ ma trận).

### 7.1. Chiến lược Ablation 3 tầng phân cấp (3-Tier Fail-Fast Ablation)

Để thẩm định khoa học từng thành phần kiến trúc với ngân sách tối ưu, toàn bộ các cấu hình được tổ chức thành **3 tầng xét duyệt liên tiếp** — mỗi tầng chỉ được kích hoạt khi tầng trước vượt qua điều kiện Go:

#### Tầng 1: PILOT — Kiểm chứng nhanh (Ước tính: ~8h GPU)
- **Tập dữ liệu:** Amazon Sports (trung bình kích thước, đủ diversity).
- **Số epoch:** 50 epochs (đủ để quan sát xu hướng hội tụ).
- **Seed:** 1 seed duy nhất.
- **Các nhánh:** B0, H0.
- **Điều kiện Go → Tầng 2:** H0 phải vượt B0 ít nhất +0.5% NDCG@20 trên Validation.
- **Điều kiện No-Go:** Nếu H0 ≤ B0, dừng dự án Lorentz CL, báo cáo trung thực kết quả âm.

#### Tầng 2: CONFIRMATORY — Xác nhận khoa học (Ước tính: ~12h GPU, chỉ chạy nếu Tầng 1 Go)
- **Tập dữ liệu:** Amazon Sports + Amazon Baby.
- **Số epoch:** 500 epochs (full training).
- **Seed:** 5 seeds paired.
- **Các nhánh bổ sung:** E0 (Euclidean control), HC (Constant-Radius control).
- **Mục tiêu:** Khẳng định H0 > E0 (Lorentz vượt trội Euclidean), và phân biệt H0 vs HC (Geometry vs Kernel).

#### Tầng 3: EXTENSION — Mở rộng & Ablation chi tiết (Ước tính: ~5h GPU, chỉ chạy nếu Tầng 2 Go)
- **Tập dữ liệu:** Tri-dataset (Sports + Baby + Electronics).
- **Các nhánh bổ sung:** H0w5, H0-reweight, H0-noself, HN, κ-grid.

#### Bảng 7.1: Ma trận Ablation STAIR-LHC v1 (Revised v2)

| Nhánh (Arm) | Mô Tả Cấu Hình Thuật Toán | Mục Tiêu Khoa Học & Câu Hỏi Kiểm Chứng | Tầng |
| :--- | :--- | :--- | :---: |
| **B0** | STAIR Baseline chuẩn ($\lambda=0.0$) | Thiết lập đường cơ sở (Baseline Parity Control) | 🔴 **Tầng 1** |
| **H0** | Lorentz CL ($D_\kappa$), $\kappa=1.0$, $w=0$, đầy đủ Self-Return Removal (Norm Protection) & Adaptive Spectral Re-weight | **Giả thuyết trung tâm:** Nhánh Lorentz CL có mang lại cải thiện hay không? | 🔴 **Tầng 1** |
| **E0** | Euclidean CL ($D_E = \|v-w\|_2^2$), $\kappa=0$, có Self-Return Removal & Spectral Re-weight | **Đối chứng chuẩn:** Đo lường đóng góp CL khi chưa có độ cong Lorentz | 🟡 **Tầng 2** |
| **HC** | Lorentz CL với **Constant-Radius** ($\|v\|=1.0$) | **Bác bỏ giả thuyết:** Phân biệt rạch ròi Geometric Capacity vs Kernel Effect | 🟡 **Tầng 2** |
| **H0w5** | Lorentz CL với **Hybrid Kernel ($w=0.5$)** | Đánh giá vai trò của độ nhọn gradient vùng cự ly gần | 🟢 **Tầng 3** |
| **H0-reweight**| Lorentz CL **không có Spectral Re-weighting** | Kiểm chứng tác động của việc suy giảm năng lượng phổ $\beta$ | 🟢 **Tầng 3** |
| **H0-noself** | Lorentz CL **không triệt tiêu Self-Return** | Đo lường mức độ "học vẹt shortcut" của mô hình | 🟢 **Tầng 3** |
| **HN** | Lorentz CL + Coordinate Noise $\varepsilon = 0.05$ | Khảo sát tác động tương hỗ của nhiễu tọa độ | 🟢 **Tầng 3** |
| **$\kappa$-grid** | Quét độ cong $\kappa \in \{0.05, 0.1, 0.5, 1.0, 2.0\}$ | Phân tích độ nhạy của mô hình theo độ cong không gian | 🟢 **Tầng 3** |

---

### 7.2. Bộ tiêu chuẩn quyết định Go / No-Go định lượng nghiêm ngặt (Đồng bộ 3 tầng)

Nhóm nghiên cứu thiết lập trước các ngưỡng quyết định định lượng (Pre-registered Decision Thresholds) để bảo đảm tính liêm chính, không diễn giải số liệu cảm tính sau khi chạy:

#### Tầng 1 — Quyết định Go/No-Go đầu tiên (Quan trọng nhất):
1. **Kiểm định B0 vs H0 (Lực đẩy tổng hợp của Lorentz CL):**  
   - *Điều kiện Go:* Nhánh $H_0$ phải đạt mức tăng tối thiểu $\Delta \text{NDCG@20} \ge +0.5\%$ so với $B_0$ trên tập Validation của Amazon Sports sau 50 epochs.
   - *Điều kiện No-Go:* Nếu $H_0 \le B_0$ hoặc tăng < +0.5%, **dừng toàn bộ dự án Lorentz CL.** Báo cáo kết quả âm một cách trung thực.
   - *Lưu ý:* Ở tầng 1, chưa cần phân biệt giữa đóng góp của CL loss vs Lorentz geometry (đó là mục tiêu Tầng 2).

> **[Peer Review Vòng 2] Bổ sung kiểm tra B0 Baseline Parity:**  
> Trước khi so sánh H0 vs B0, **bắt buộc xác nhận B0 tái tạo được kết quả Baseline STAIR gốc** (NDCG@20 sai lệch < 1% so với kết quả đã báo cáo tại V5). Nếu B0 bị suy giảm bất thường (do bug code, sai cấu hình, hoặc thay đổi preprocessing), toàn bộ so sánh downstream đều vô giá trị.

#### Tầng 2 — Phân tách đóng góp (Chỉ chạy nếu Tầng 1 Go):
2. **Kiểm định B0 vs E0 (Lực đẩy thuần của nhánh CL, không có Lorentz):**  
   - *Điều kiện:* Nhánh $E_0$ phải đạt mức tăng tối thiểu $\Delta \text{NDCG@20} \ge +0.5\%$ so với $B_0$.
   - *Hành động nếu không đạt:* Nếu $E_0 \le B_0$ nhưng $H_0 > B_0$, kết luận rằng đóng góp đến từ Lorentz geometry chứ không phải CL loss.
3. **Kiểm định E0 vs H0 (Hiệu quả của Độ cong Lorentz):**  
   - *Điều kiện:* Nhánh $H_0$ phải vượt $E_0$ ít nhất $+1.0\%$ tương đối về NDCG@20 trên Amazon Sports, và không làm suy giảm quá $0.5\%$ trên Amazon Baby.  
   - *Hành động nếu không đạt:* Nếu $H_0 \approx E_0$, bác bỏ giả thuyết về ưu thế của không gian Lorentz; kết luận không gian Euclidean 64D là hoàn toàn đủ năng lực biểu diễn.
4. **Kiểm định HC vs H0 (Bản chất Hình học vs Kernel Trick):**  
   - *Điều kiện:* Nếu $H_0$ vượt $HC$ với mức chênh lệch có ý nghĩa thống kê ($p < 0.05$), xác nhận cấu trúc phân tầng bán kính hyperbolic phát huy tác dụng.  
   - *Hành động nếu ngược lại:* Nếu $HC \approx H_0$, báo cáo trung thực rằng độ cong Lorentz chỉ đóng vai trò như một hàm nhân phi tuyến.

#### Tầng 3 — Ablation chi tiết (Chỉ chạy nếu Tầng 2 xác nhận H0 vượt trội):
5. **Kiểm định H0 vs H0-noself (Xác nhận vai trò của Self-Return Removal):**  
   - Nếu `H0-noself` có Loss CL giảm nhanh hơn nhưng Validation NDCG@20 lại thấp hơn `H0`, đây là bằng chứng thực nghiệm khẳng định Self-Return Shortcut gây hại cho khả năng khái quát hóa.
6. **Kiểm định H0 vs H0-reweight (Vai trò Spectral Re-weighting).**
7. **Kiểm định H0 vs H0w5 (Hybrid Kernel).**

### 7.2.1. [Peer Review Vòng 2] Bộ chẩn đoán CL Loss Dynamics bắt buộc

Ngoài các chỉ số ranking (NDCG, Recall), **bắt buộc giám sát 3 tín hiệu chẩn đoán sức khỏe** của nhánh CL để phát hiện sớm các lỗi ẩn:

| Tín Hiệu Chẩn Đoán | Hành vi Mong Đợi (Healthy) | Cảnh Báo Đỏ (Unhealthy) | Hành Động |
| :--- | :--- | :--- | :--- |
| **CL Loss ($\mathcal{L}_{\text{LHC}}$)** | Giảm dần, hội tụ ổn định | Giảm đến ~0 rất nhanh (< 20 epochs sau warmup) | Nghi ngờ Self-Return chưa triệt tiêu hoặc positive leaking |
| **Alignment (trung bình $d_{\text{pos}}$)** | Giảm dần (cặp dương co cụm) | Giảm cùng tốc với $d_{\text{neg}}$ | Representation Collapse — giảm $\lambda$ |
| **Uniformity ($\log \mathbb{E}[e^{-d_{\text{neg}}}]$)** | Duy trì cao (mẫu âm phân tán đều) | Giảm nhanh (mẫu âm co cụm) | Loss CL đang kéo sập không gian — dừng sớm |

---

### 7.3. Cấu hình siêu tham số và ngân sách thực nghiệm

#### Bảng 7.2: Đặc tả cấu hình siêu tham số STAIR-LHC v1

| Tham Số / Cấu Hình | Amazon Baby | Amazon Sports | Amazon Electronics | Ghi Chú Thiết Kế |
| :--- | :---: | :---: | :---: | :--- |
| **Số chiều tiềm ẩn ($D$)** | 64 | 64 | 64 | Cố định chuẩn STAIR |
| **Số tầng tích chập ($L$)** | 3 | 3 | 3 | Bộ lọc phổ FSC |
| **Batch Size ($B$)** | 1024 | 1024 | 4096 | Micro-batching 512 trên Elec |
| **Tốc độ học ($lr$)** | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | Optimizer AdamWSEvo |
| **Trọng số điều hòa $\lambda_{\max}$**| $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $1 \times 10^{-4}$ | Lưới quét $\{10^{-4}, 3 \times 10^{-4}, 10^{-3}\}$ |
| **Nhiệt độ tương phản $\tau$** | 0.3 | 0.3 | 0.3 | Lưới quét $\{0.1, 0.3, 1.0\}$ |
| **Bán kính chặn ($R$)** | 2.0 | 2.0 | 2.0 | Bão hòa qua tanh |
| **Độ cong Lorentz ($\kappa$)** | 1.0 | 1.0 | 1.0 | Nhánh P0 chính |
| **Trọng số Hybrid ($w$)** | 0.0 / 0.5 | 0.0 / 0.5 | Theo kết quả Pilot | Đo đạc độ nhọn gradient |
| **Ngân sách huấn luyện** | 500 Epochs | 500 Epochs | 500 Epochs | Đồng nhất mọi nhánh đối chứng |

---

### 7.4. Quy tắc lựa chọn checkpoint, đánh giá kiểm định và xử lý độ bất định

- **Nguyên tắc bất biến chọn mô hình:** Điểm checkpoint triển khai tối ưu được chọn duy nhất dựa trên chỉ số **NDCG@20 trên tập Validation**.
- **Đánh giá tập Test:** Chỉ chạy đánh giá duy nhất một lần trên tập Test tại checkpoint đã chọn; tuyệt đối không quét max Test qua các epochs.
- **Xử lý độ bất định thống kê:** Mọi cấu hình vượt qua vòng Pilot đều phải được huấn luyện trên **5 hạt giống ngẫu nhiên ghép cặp (5 Paired Seeds)**: `seed ∈ {1, 2, 3, 4, 5}`. Báo cáo đầy đủ giá trị trung bình $\mu$, độ lệch chuẩn $\sigma$ và kết quả kiểm định Paired t-test ($p$-value).

---

## 8. KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN 5 BƯỚC (5-STEP IMPLEMENTATION ROADMAP)

> **⚠️ NGUYÊN TẮC THỰC THI:**  
> **Bước 1-3:** Triển khai mã nguồn kiến trúc đề xuất chính (H0).
> **Bước 4:** Chạy Pilot B0 + H0 trên Amazon Sports (50 epochs, 1 seed). **KHÔNG chạy ablation ở bước này.**  
> **Bước 5:** Chỉ khi Pilot Go, mới mở rộng ra full training và ablation.
> 
> *"Tạm chỉ để phân tích ablation vào file thôi chứ khi code tôi sẽ tạm thời chưa chạy ablation study mà chạy bản kiến trúc đề xuất chỉnh để xem có cải thiện hay không thì mới tính tiếp tới ablation."*

Quá trình chuyển hóa từ tài liệu thiết kế sang mã nguồn thực thi được kiểm soát qua quy trình 5 bước nghiêm ngặt:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        KẾ HOẠCH TRIỂN KHAI 5 BƯỚC (5-STEP ROADMAP)                     │
│                      [Revised v2 — Fail-Fast, Run Architecture First]                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [BƯỚC 1: SETUP GEOMETRY MODULE & UNIT TESTS]                                         │
│   • Tạo `models/stair5_v1_geometry.py`                                                 │
│   • Viết bộ kiểm thử `tests/test_stair5_v1_geometry.py`                                │
│   • Kiểm tra: Self-distance = 0, Gradient finite tại cự ly 0,                         │
│     Bán kính cap_R <= 2.0, Tiệm cận Euclidean khi κ -> 0.                              │
│   • [v2] Kiểm tra Norm Protection: items bậc 1 sau Self-Return Removal                │
│     phải có ||H|| >= eps_floor = 1e-2.                                                  │
│                                           │                                            │
│                                           ▼                                            │
│   [BƯỚC 2: SETUP OBJECTIVES & POSITIVE HANDLING MODULE]                                │
│   • Tạo `models/stair5_v1_objectives.py`                                               │
│   • Triển khai Self-Return Removal + Norm Protection (degree-1 safety)                 │
│   • Triển khai Adaptive Spectral Re-weighting (quantile-based M_norm)                  │
│   • Xây dựng ma trận Train-Positive đa mẫu & InfoNCE 2 chiều đối xứng                  │
│   • [v2] Tính M_norm = Q95(||H^(0)||) tại epoch 0 và đóng băng thành buffer            │
│                                           │                                            │
│                                           ▼                                            │
│   [BƯỚC 3: TÍCH HỢP VÀO STAIR BACKBONE]                                                │
│   • Tạo `models/stair5_v1.py` kế thừa từ backbone STAIR                                │
│   • Định nghĩa các Parameter Groups riêng biệt trong AdamWSEvo                         │
│   • Thiết lập lịch trình Warmup λ(t)                                                   │
│   • [v2] Tích hợp Gradient Cosine Diagnostics (Baseline-Calibrated)                    │
│   • [v2] Tích hợp CL Loss Diagnostics (Alignment + Uniformity monitoring)              │
│   • [v2] Kiểm tra B0 Baseline Parity trước khi so sánh                                │
│                                           │                                            │
│                                           ▼                                            │
│   [BƯỚC 4: PILOT — B0 + H0 trên Amazon Sports (50 epochs, 1 seed)]                     │
│   ★ CHẠY KIẾN TRÚC ĐỀ XUẤT CHÍNH TRƯỚC — KHÔNG CHẠY ABLATION ★                        │
│   • Khởi chạy 2 cấu hình: B0 (baseline), H0 (đề xuất chính)                            │
│   • Giám sát: CL Loss dynamics, Alignment/Uniformity, VRAM peak, Grad cosine           │
│   • Fail-fast: NaN → dừng ngay; H0 ≤ B0 sau 50ep → dừng dự án                         │
│                                           │                                            │
│                                           ▼                                            │
│   [BƯỚC 5: CONFIRMATORY + ABLATION (chỉ nếu Bước 4 Go)]                                │
│   • Tầng 2: E0 + HC trên Sports + Baby (500ep, 5 seeds)                                │
│   • Tầng 3: H0w5, H0-reweight, H0-noself, HN, κ-grid (nếu Tầng 2 Go)                  │
│   • Xuất báo cáo thực nghiệm và bản đối soát SHA-256                                   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. BẢNG TỔNG HỢP RỦI RO & CHIẾN LƯỢC GIẢM THIỂU (RISK & MITIGATION MATRIX)

#### Bảng 9.1: Ma trận quản trị rủi ro kỹ thuật

| Rủi Ro Nhận Diện | Xác Suất | Mức Độ Tác Động | Chiến Lược Phòng Ngừa & Giảm Thiểu Cụ Thể |
| :--- | :---: | :---: | :--- |
| **1. Lỗi NaN Gradient tại điểm trùng ($x \to y$)** | Trung bình | Rất cao | Áp dụng khai triển chuỗi Taylor bậc 4 cho $t < 10^{-4}$; kẹp biên dưới $\text{clamp}(1 + 10^{-7})$; dùng bình phương khoảng cách $d_L^2$. |
| **2. Tràn số (Overflow) các hàm $\cosh / \sinh$** | Thấp | Rất cao | Khóa cứng trần bán kính $R = 2.0$ thông qua hàm $\tanh$ bão hòa; chạy tính toán hình học ở độ chính xác FP32 (tắt autocast FP16 cho hình học). |
| **3. CL Loss giảm nhưng Validation NDCG giảm** | Cao | Cao | Triệt tiêu Self-Return trên positive keys (bao gồm Norm Protection cho degree-1); điều chỉnh giảm $\lambda_{\max}$; giám sát Alignment/Uniformity diagnostics; thiết lập cơ chế Early Stopping fallback về Epoch 0. |
| **4. Hiện tượng Over-smoothing trên Amazon Baby** | Trung bình | Trung bình | Áp dụng bù phổ Adaptive Spectral Re-weighting; rút ngắn giai đoạn Warmup; kiểm soát trần chuẩn vector $M_{\text{norm}}$ thích ứng theo quantile 95% (không dùng giá trị cứng). |
| **5. Nghi ngờ kết quả chỉ là Kernel Trick** | Cao | Trung bình | Bắt buộc triển khai nhánh đối chứng Constant-Radius ($HC$); công bố trung thực kết quả so sánh giữa $HC$ và $H0$. |
| **6. Xung đột Gradient giữa CL Loss và BPR** | Trung bình | Cao | Giám sát góc gradient $\rho_{\text{grad}}$; cô lập nhóm tham số trong optimizer; áp dụng cơ chế ngắt gradient (`detach`) nếu phát hiện đối kháng. |

---

## 10. KẾT LUẬN KHOA HỌC & ĐIỀU KIỆN TUYÊN BỐ ĐÓNG GÓP

Thiết kế **STAIR-LHC v1 (REVISED v2)** đại diện cho một bước tiến vững chắc về mặt phương pháp luận nghiên cứu, đã tích hợp đầy đủ 2 vòng phản biện Peer Review:
1. Bản thiết kế đã khắc phục toàn diện các hạn chế số học của hình học phi Euclidean bằng các giải pháp toán học nghiêm cẩn (Radius Capping, Taylor Series, Hybrid Distance Kernel).
2. Giải quyết triệt để hai bài toán cốt lõi bị bỏ sót: **Triệt tiêu Self-Return Shortcut** (bao gồm Norm Protection cho items bậc 1 — Peer Review Vòng 2) để ép mô hình học cấu trúc đồ thị thực chất, và **Bù suy giảm phổ Adaptive Spectral Re-weighting** (quantile-based $M_{\text{norm}}$ thay vì giá trị cứng — Peer Review Vòng 2) để bảo toàn thông tin đa phương thái ở các chiều cao.
3. Thiết lập hệ thống kiểm định độc lập **Constant-Radius Control (HC arm)** để phân biệt rạch ròi giữa dung lượng hình học thực sự và hiệu ứng hàm nhân phi tuyến.
4. **[Mới — Peer Review Vòng 2]** Áp dụng chiến lược thực thi **3-Tier Fail-Fast Ablation** (Pilot → Confirmatory → Extension) tiết kiệm ngân sách GPU từ ~140h xuống ~25h, ưu tiên chạy kiến trúc đề xuất chính (B0 + H0) trước khi mở rộng ablation.
5. **[Mới — Peer Review Vòng 2]** Tích hợp bộ chẩn đoán CL Loss Dynamics (Alignment + Uniformity), Baseline-Calibrated Gradient Cosine Threshold, và yêu cầu B0 Baseline Parity verification.

**Tuyên ngôn về Liêm chính Học thuật:**  
Nếu sau khi thực hiện đầy đủ các bước kiểm định thực nghiệm nghiêm ngặt, nhánh Lorentz $H0$ không vượt qua được nhánh Euclidean $E0$ hoặc tiệm cận nhánh $HC$, nhóm nghiên cứu sẽ **báo cáo kết quả trung hòa một cách minh bạch, trung thực**. Trong khoa học, việc bác bỏ một giả thuyết thời thượng (Hyping of Hyperbolic Geometry) bằng các bằng chứng thực nghiệm chặt chẽ có đối chứng là một đóng góp học thuật vô cùng giá trị, khẳng định tính đúng đắn và độ bền vững của kiến trúc STAIR gốc.

---

## 11. NGUỒN THAM KHẢO & MỨC ĐỘ ĐỐI SOÁT

- **[R1] STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation.** [arXiv:2412.11729](https://arxiv.org/html/2412.11729v1), [GitHub](https://github.com/yhhe2004/STAIR). Đối soát cấu trúc mã nguồn nền tảng.
- **[R2] Nickel, M., & Kiela, D. (2018). Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.** *ICML 2018*, PMLR 80, 3779–3788. [PMLR](https://proceedings.mlr.press/v80/nickel18a.html). Cơ sở toán học của mô hình Lorentz và ánh xạ mũ.
- **[R3] Chami, I., Ying, R., Ré, C., & Leskovec, J. (2019). Hyperbolic Graph Convolutional Neural Networks.** *NeurIPS 2019*. [arXiv:1910.12933](https://arxiv.org/abs/1910.12933). Tham chiếu về tích chập đồ thị hyperbolic và độ cong khả vi.
- **[R4] Mishne, G., Wan, Z., Wang, Y., & Yang, S. (2023). The Numerical Stability of Hyperbolic Representation Learning.** *ICML 2023*, PMLR 202, 24925–24949. [PMLR](https://proceedings.mlr.press/v202/mishne23a.html). Bằng chứng thực nghiệm về các giới hạn số học và điểm kỳ dị đạo hàm trong không gian hyperbolic.
- **[R5] Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2022). XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation.** [arXiv:2209.02544](https://arxiv.org/abs/2209.02544). Cơ chế tương phản đa tầng và xử lý mẫu âm trong batch.
- **[R6] Sun, S., & Ma, C. (2024). Hyperbolic Contrastive Learning with Model-Augmentation for Knowledge-Aware Recommendation.** *ECML-PKDD 2024*. [arXiv:2505.08157](https://arxiv.org/html/2505.08157v1), [GitHub](https://github.com/sunshy-1/HCMKR). Tham khảo chuẩn hóa thang đo thực nghiệm $q_{t,\ell}$.
- **[R7] Hyperbolic Graph Contrastive Learning for Collaborative Filtering.** *IEEE TKDE 2024*. [DOI: 10.1109/TKDE.2024.3522960](https://doi.org/10.1109/TKDE.2024.3522960). Khảo sát hướng nghiên cứu Hyperbolic CF.
- **[R8] Fang, S., Wang, J., & Chen, F. (2025). A Hyperbolic Graph Neural Network Model with Contrastive Learning for Rating–Review Recommendation.** *Entropy*, 27(8), 886. [MDPI](https://www.mdpi.com/1099-4300/27/8/886).
- **[R9] Zhang, Q., et al. (2026). HMamba: Hyperbolic Mamba for Sequential Recommendation.** *ACM TOIS 2026*. [ACM DL](https://doi.org/10.1145/3811405). Khảo sát định nghĩa hình học Lorentz trong bài toán gợi ý tuần tự.
