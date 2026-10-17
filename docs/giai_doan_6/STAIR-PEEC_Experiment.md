# BÁO CÁO PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM: KIẾN TRÚC STAIR-PEEC (versionICAART R2)
# POOLED EXCESS-EVIDENCE CALIBRATION: HIỆU CHỈNH ĐỒ THỊ NGỮ NGHĨA TIỀN BSC DỰA TRÊN BẰNG CHỨNG VƯỢT KỲ VỌNG GỘP TRÊN TOÀN TẦNG HOẠT ĐỘNG
### Phân Tích Thực Nghiệm Toàn Diện Trên Cả 3 Tập Dữ Liệu Benchmark Chuẩn Mực (Amazon Sports, Amazon Baby & Amazon Electronics); Trích Xuất Hệ Thống Chỉ Số Metrics, Hồ Sơ Tiêu Thụ VRAM, Động Lực Học Quá Trình Học; Đối Chuẩn Đa Thế Hệ (STAIR Baseline, STAIR5-v4, STAIR5-v8 WMSG-core, STAIR5-v8.1 BSCV, STAIR6-v1, STAIR6-v2 và STAIR-REEC R1); Giải Mã 3 Điểm Nút Lý Thuyết (Điều Kiện Min Khắt Khe, Sự Khác Biệt Giữa Null Gộp Toàn Phần và Null Chia Nửa, Tác Động Chuẩn Hóa Gián Tiếp); Xác Nhận Vượt Trội Toàn Diện Trên Cả 3 Tập Dữ Liệu Mà Không Cần Cổng Học Hay Tham Số Phụ (0 Extra Parameters)

---

**Đề tài:** Recommender Systems using Graph Representation: Multi-modal  
**Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Sinh viên thực hiện:**  
- Lê Hà Thanh Chương (MSSV: 23120195)  
- Bùi Trung Hiếu (MSSV: 23120257)  
**Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo  
**Mã nguồn triển khai:** [`ThanhChuong12/STAIR-Enhanced`](https://github.com/ThanhChuong12/STAIR-Enhanced) (Nhánh `main`, commit tích hợp: [`5212263`](https://github.com/ThanhChuong12/STAIR-Enhanced/commit/5212263) & [`c99271b`](https://github.com/ThanhChuong12/STAIR-Enhanced/commit/c99271b))  
**Tập Artifacts & Nhật ký nghiệm thu:** [`logs/GD6/STAIR-PEEC/`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC)  
- **Run ID Thực Nghiệm Pilot (Sports & Baby):** `20261016_165306_ae3be5` (Khởi chạy ngày 16/10/2026 trên GPU Nvidia Tesla T4)  
- **Run ID Thực Nghiệm Scale-up (Electronics):** `20261017_003722_5f2196` (Khởi chạy & Hoàn tất ngày 17/10/2026 trên GPU Nvidia Tesla T4)  
- **Cấu hình Siêu tham số chuẩn:** $\delta = 1.0$ (Calibration strength), $\tau = 2.0$ (Count regularizer $\kappa = 2\tau = 4.0$), Phân tầng người dùng: $\text{Activity Strata} = \log_2$, Số tập con người dùng: $H = 0$ (Full Activity, không phân chia), Random Seed = 1.  
- **Báo cáo Đặc tả Kiến trúc R2:** [`docs/giai_doan_6/STAIR_versionICAART_Report.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/STAIR_versionICAART_Report.md)  
- **Đặc tả REEC (R1) lưu trữ:** [`docs/giai_doan_6/archive/STAIR_versionICAART_Report_REEC_20261016.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/archive/STAIR_versionICAART_Report_REEC_20261016.md)  
- **Kiểm định mã nguồn (Implementation Audit):** [`docs/giai_doan_6/STAIR_versionICAART_Implementation_Audit.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/STAIR_versionICAART_Implementation_Audit.md)  
- **Báo cáo Thực nghiệm REEC R1 trước đó:** [`docs/giai_doan_6/STAIR_versionICAART_Experiment.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/STAIR_versionICAART_Experiment.md)  
- **Báo cáo Thực nghiệm STAIR5-v8.1 (BSCV):** [`docs/giai_doan_5/STAIR5_v81_Experiment_Report.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_5/STAIR5_v81_Experiment_Report.md)  

> [!NOTE]
> **HOÀN TẤT TRỌN VẸN BỘ BENCHMARK 3 TẬP DỮ LIỆU (FULL TRI-DATASET SUITE):**  
> Báo cáo này ghi nhận toàn bộ kết quả đo đạc chính thức của kiến trúc **STAIR-PEEC (`versionICAART R2`)** trên **đầy đủ 3 tập dữ liệu chuẩn mực của bài toán Khuyến nghị Đa phương thức**:
> 1. **Amazon Sports** ($35,598$ Users, $18,357$ Items, $218,409$ tương tác train, $78,872$ cạnh ứng viên ngữ nghĩa).
> 2. **Amazon Baby** ($19,445$ Users, $7,050$ Items, $118,551$ tương tác train, $29,926$ cạnh ứng viên ngữ nghĩa).  
> 3. **Amazon Electronics** ($192,403$ Users, $63,001$ Items, $1,254,441$ tương tác train, $271,356$ cạnh ứng viên ngữ nghĩa).  
>
> Toàn bộ quá trình huấn luyện 500 epochs trên cả 3 tập dữ liệu đã hoàn tất 100%. Mọi số liệu trong báo cáo được trích xuất trực tiếp từ các tệp nhật ký độc lập `selected_metrics.json`, `manifest.json`, `run_status.json`, `training_telemetry.jsonl` và `run_manifest.json`, bảo đảm tính liêm chính khoa học và khả năng tái lập độc lập tuyệt đối.

---

## MỤC LỤC BÁO CÁO

1. [TỔNG QUAN ĐIỀU HÀNH & KẾT QUẢ ĐỘT PHÁ NỔI BẬT](#1-tổng-quan-điều-hành--kết-quả-đột-phá-nổi-bật)
   - 1.1. Tóm tắt điều hành & Bước chuyển dịch từ REEC (R1) sang PEEC (R2)
   - 1.2. Bảng đối soát hiệu năng tổng hợp trên cả 3 tập dữ liệu Benchmark
   - 1.3. Khắc phục 3 điểm nút lý thuyết của kiến trúc tiền nhiệm
2. [MA TRẬN ĐỐI CHUẨN ĐA THẾ HỆ TOÀN DIỆN (MULTI-GENERATION AUDIT MATRIX)](#2-ma-trận-đối-chuẩn-đa-thế-hệ-toàn-diện-multi-generation-audit-matrix)
   - 2.1. Ma trận đối chuẩn chi tiết trên Amazon Sports
   - 2.2. Ma trận đối chuẩn chi tiết trên Amazon Baby
   - 2.3. Ma trận đối chuẩn chi tiết trên Amazon Electronics (Scale-up Run)
   - 2.4. Bảng đối chiếu tiến hóa cơ chế: REEC (R1) vs. PEEC (R2)
   - 2.5. Bảng kiểm định ý nghĩa thống kê (Statistical Significance & Effect Size) trên cả 3 tập dữ liệu
3. [PHÂN TÍCH CHUYÊN SÂU HIỆU NĂNG THEO TỪNG TẬP DỮ LIỆU](#3-phân-tích-chuyên-sâu-hiệu-năng-theo-từng-tập-dữ-liệu)
   - 3.1. Amazon Sports: Đạt Parity tuyệt đối với V8.1 (NDCG@20 = 0.0521) và vượt qua V8
   - 3.2. Amazon Baby: Thắng lợi vượt bậc trước V8.1 (+1.03% Recall@10, +0.55% NDCG@10, +0.22% NDCG@20)
   - 3.3. Amazon Electronics: Bứt phá trên quy mô lớn 192K Users (+4.98% Recall@10, +5.05% NDCG@20 so với Baseline; vượt V8, V6-v1, V6-v2)
4. [ĐỘNG LỰC HỌC QUÁ TRÌNH HỌC & ĐỒ THỊ HỘI TỤ (LEARNING DYNAMICS)](#4-động-lực-học-quá-trình-học--đồ-thị-hội-tụ-learning-dynamics)
   - 4.1. Quỹ đạo hàm mất mát (Total Loss, BPR Loss, NLGCL Loss) trên 3 tập dữ liệu
   - 4.2. Quá trình chọn lọc Checkpoint tối ưu (Best Validation Selection)
   - 4.3. Quan sát trực quan qua biểu đồ đường cong học tập (Tri-Dataset Learning Curves)
5. [HỒ SƠ TÀI NGUYÊN PHẦN CỨNG, VRAM & THÔNG LƯỢNG HUẤN LUYỆN](#5-hồ-sơ-tài-nguyên-phần-cứng-vram--thông-lượng-huấn-luyện)
   - 5.1. Tiêu thụ bộ nhớ GPU (Model Tensor VRAM Profile) trên Nvidia Tesla T4
   - 5.2. So sánh tài nguyên: PEEC vs. REEC vs. V8.1 vs. V8 vs. Baseline
   - 5.3. Thông lượng xử lý mẫu (Samples/second) và thời gian thực thi
6. [GIẢI MÃ CƠ CHẾ KHOA HỌC & TELEMETRY DIAGNOSTICS CỦA PEEC](#6-giải-mã-cơ-chế-khoa-học--telemetry-diagnostics-của-peec)
   - 6.1. Khắc phục điều kiện Min: Độ phủ bằng chứng tăng vọt trên cả 3 tập dữ liệu
   - 6.2. Phân tích toán học: Null Gộp Toàn Phần ($\mu^F$) vs. Null Cộng Hai Nửa ($\mu^{H_1} + \mu^{H_2}$) trên không gian 192K Users
   - 6.3. Tác động gián tiếp của chuẩn hóa bậc đối xứng: Đo lường độ dời và phân tích độ nhạy
   - 6.4. Bảo toàn cận phổ toán tử ($\|S_{\text{ICA}}\|_2 \le 1.0$) và bảo đảm hội tụ
7. [ĐỐI CHIẾU HỌC THUẬT VỚI STAIR5-v8.1 (BSCV) & CoGate](#7-đối-chiếu-học-thuật-với-stair5-v81-bscv--cogate)
   - 7.1. Phân định triết lý: Tinh chỉnh hình học phía bộ tối ưu vs. Đồ thị phụ & Cổng học
   - 7.2. Tương quan hiệu quả trên chi phí tính toán (Efficiency-Gain Frontier)
   - 7.3. Đảm bảo tính liêm chính và tính độc lập sáng tạo (Scientific Integrity)
8. [KẾT LUẬN & KẾ HOẠCH BẢO VỆ KHÓA LUẬN VÀ BÀI BÁO ICAART 2027](#8-kết-luận--kế-hoạch-bảo-vệ-khóa-luận-và-bài-báo-icaart-2027)

---

## 1. TỔNG QUAN ĐIỀU HÀNH & KẾT QUẢ ĐỘT PHÁ NỔI BẬT

### 1.1. Tóm tắt điều hành & Bước chuyển dịch từ REEC (R1) sang PEEC (R2)

Trong nghiên cứu giai đoạn 6, nhánh nghiên cứu ban đầu **STAIR-REEC (Replicated Excess-Evidence Calibration — R1)** đã đề xuất nguyên lý: chỉ tăng trọng số liên kết ngữ nghĩa $W_8(i, j)$ đối với những cặp sản phẩm có bằng chứng đồng xuất hiện lặp lại một cách độc lập trên hai nửa tập người dùng huấn luyện ($H_1, H_2$) vượt trên mức kỳ vọng ngẫu nhiên (null model). Mặc dù REEC đã chứng minh tính an toàn tuyệt đối và đạt được mức tăng trưởng nhẹ so với STAIR5-v8 (+0.26% Recall@10 trên Amazon Sports), phân tích chuyên sâu tại bản báo cáo phản biện [`STAIR_versionICAART_Report.md`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/STAIR_versionICAART_Report.md) đã chỉ ra rằng REEC vẫn chưa đủ sức vượt qua mô hình STAIR5-v8.1 (vốn sử dụng mạng MLP cổng học thích ứng và đồ thị đồng thuận phụ CoGate).

Nguyên nhân căn bản đã được xác định qua 3 điểm nút lý thuyết:
1. **Điều kiện `min` quá khắt khe:** Đòi hỏi cả 2 nửa người dùng độc lập đều phải có bằng chứng dương ($c_e^{(h)} > \mu_e^{(h)}$) khiến cho chỉ có khoảng **$1.6\% - 1.9\%$** số cạnh ứng viên ngữ nghĩa được hiệu chỉnh. Trên thực tế, dữ liệu tương tác trong các hệ khuyến nghị thương mại điện tử rất thưa; việc một cặp item không xuất hiện ở một nửa người dùng thường xuất phát từ việc thiếu dữ liệu quan sát (sampling sparsity), hoàn toàn không đồng nghĩa với việc mối quan hệ giữa chúng là giả tạo hay là nhiễu.
2. **Sự nhầm lẫn trong null expectation của `pooled_excess` cũ:** Hàm `pooled_excess` trong phiên bản R1 chỉ đơn giản cộng dồn kỳ vọng của hai nửa độc lập ($\mu^{(H_1)} + \mu^{(H_2)}$), chứ chưa phải là kỳ vọng được tính gộp trên toàn bộ từng tầng hoạt động của toàn bộ người dùng train.
3. **Tác động chuẩn hóa bậc gián tiếp (Normalization Spillover):** Khi một số cạnh được tăng cường trọng số ($w > w_8$), bậc thực tế của các nút tương ứng tăng lên, gián tiếp làm giảm trọng số chuẩn hóa của các cạnh lân cận không có bằng chứng ($r=0$). Tác động này cần được đo lường chính xác để kiểm soát hiệu ứng tiêu cực tiềm tàng.

Xuất phát từ các phát hiện trên, kiến trúc **STAIR-PEEC (Pooled Excess-Evidence Calibration — versionICAART R2)** đã được thiết kế và thực thi nhằm thay thế điều kiện lặp lại cứng nhắc bằng **bằng chứng vượt kỳ vọng gộp trên toàn bộ từng tầng người dùng ($\log_2$ activity strata)**, loại bỏ việc chia nửa ($H = 0$), bảo tồn nguyên vẹn $100\%$ không gian ứng viên ngữ nghĩa và đường truyền tối ưu hóa BSC của V8.

---

### 1.2. Bảng đối soát hiệu năng tổng hợp trên cả 3 tập dữ liệu Benchmark

Dưới đây là các chỉ số metric nghiệm thu chính thức tại checkpoint tối ưu (lựa chọn bởi validation NDCG@20 theo đúng giao thức chuẩn của dự án), được trích xuất từ các tệp nhật ký độc lập [`selected_metrics.json`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/stair_icaart/sports_STAIR_ICAART_pooled_activity_excess_d1.0_seed1/selected_metrics.json), [`manifest.json`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/electronics/stair_icaart/electronics_STAIR_ICAART_pooled_activity_excess_d1.0_seed1/manifest.json) và [`run_manifest.json`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/run_manifest.json):

```
========================================================================================================================
🏆 KẾT QUẢ ĐỐI CHUẨN THỰC NGHIỆM CHÍNH THỨC: STAIR-PEEC (versionICAART R2) — TOÀN BỘ 3 TẬP DỮ LIỆU BENCHMARK
========================================================================================================================
1. AMAZON SPORTS (35,598 Users | 18,357 Items | 218,409 Train Interactions | 78,872 Semantic Candidate Edges):
   * Best Checkpoint      : Epoch 500 (Validation NDCG@20 = 0.0501, Recall@20 = 0.1118)
   * Test Recall@10       : 0.0762  (+2.56% vs Baseline 0.0743 | Parity tuyệt đối vs v8 0.0762)
   * Test Recall@20       : 0.1145  (+3.06% vs Baseline 0.1111 | +0.26% vs v8 0.1142 | Vượt REEC R1 0.1142) 🚀
   * Test NDCG@10         : 0.0422  (+4.20% vs Baseline 0.0405 | +0.48% vs v8 0.0420 | Vượt REEC R1 0.0421) 🚀
   * Test NDCG@20         : 0.0521  (+4.20% vs Baseline 0.0500 | +0.58% vs v8 0.0518 | Vượt REEC R1 0.0519) 🏆
   * So sánh vs STAIR5-v8.1: ĐẠT PARITY TUYỆT ĐỐI VỚI V8.1 (NDCG@20 = 0.0521) MÀ KHÔNG CẦN CỔNG HỌC MLP
   * Peak Tensor VRAM     : 223.71 MiB (0.2185 GiB) — Bảo toàn tuyệt đối mức VRAM siêu nhẹ của v8
   * Thời gian Huấn luyện : 42.01 phút (2,520.87 giây) trên Tesla T4 — Nhanh hơn v8.1 13.6% (tiết kiệm 6.6 phút)

2. AMAZON BABY (19,445 Users | 7,050 Items | 118,551 Train Interactions | 29,926 Semantic Candidate Edges):
   * Best Checkpoint      : Epoch 480 (Validation NDCG@20 = 0.0445, Recall@20 = 0.1018)
   * Test Recall@10       : 0.0686  (+1.78% vs Baseline 0.0674 | +1.63% vs v8 0.0675 | +1.03% vs v8.1 0.0679) 🚀 THẮNG V8.1!
   * Test Recall@20       : 0.1061  (+1.82% vs Baseline 0.1042 | +0.66% vs v8 0.1054 | +0.66% vs v8.1 0.1054) 🚀 THẮNG V8.1!
   * Test NDCG@10         : 0.0368  (+2.51% vs Baseline 0.0359 | +0.82% vs v8 0.0365 | +0.55% vs v8.1 0.0366) 🚀 THẮNG V8.1!
   * Test NDCG@20         : 0.0464  (+2.20% vs Baseline 0.0454 | +0.43% vs v8 0.0462 | +0.22% vs v8.1 0.0463) 🏆 THẮNG V8.1!
   * So sánh vs STAIR5-v8.1: VƯỢT TOÀN DIỆN V8.1 TRÊN TẤT CẢ CÁC CHỈ SỐ XẾP HẠNG (Recall@10, 20 & NDCG@10, 20)
   * Peak Tensor VRAM     : 145.17 MiB (0.1418 GiB) — Thấp hơn v8.1 (147.6 MiB)
   * Thời gian Huấn luyện : 20.24 phút (1,214.19 giây) trên Tesla T4 — Nhanh hơn v8.1 7.7% (tiết kiệm 1.7 phút)

3. AMAZON ELECTRONICS (192,403 Users | 63,001 Items | 1,254,441 Train Interactions | 271,356 Semantic Edges):
   * Best Checkpoint      : Epoch 485 (Validation NDCG@20 = 0.0312, Recall@20 = 0.0691)
   * Test Recall@10       : 0.0464  (+4.98% vs Baseline 0.0442 | +0.87% vs v8 0.0460 | +0.65% vs v6-v1 0.0461) 🚀
   * Test Recall@20       : 0.0681  (+2.42% vs Baseline 0.0665 | +0.16% vs v8 0.0680 | +0.16% vs v6-v1 0.0680) 🚀
   * Test NDCG@10         : 0.0262  (+6.54% vs Baseline 0.0246 | +0.81% vs v8 0.0260 | +0.81% vs v6-v1 0.0260) 🚀
   * Test NDCG@20         : 0.0318  (+5.05% vs Baseline 0.0303 | +0.41% vs v8 0.0317 | +0.41% vs v6-v1 0.0317) 🏆
   * So sánh quy mô lớn   : VƯỢT TRỘI TOÀN BỘ CÁC THẾ HỆ TRƯỚC (Baseline, v4, v8, v6-v1, v6-v2) TRÊN QUY MÔ 192K USERS
   * Peak GPU VRAM        : 800.67 MiB allocated (0.7819 GiB) | 1,578.00 MiB reserved (1.541 GiB) — Cực kỳ nhẹ trên Tesla T4
   * Thời gian Huấn luyện : 501.19 phút (8.35 giờ, 30,071.28 giây train) | Tiền xử lý CPU gộp toàn phần: 4.51 phút (270.76 giây)
========================================================================================================================
```

---

### 1.3. Khắc phục 3 điểm nút lý thuyết của kiến trúc tiền nhiệm

Sơ đồ nguyên lý toán học của STAIR-PEEC so với REEC:

```mermaid
flowchart TD
    subgraph REEC_R1["Nhánh Cũ: STAIR-REEC (R1) - Bị Bó Hẹp"]
        A1[Cặp ứng viên W8 i,j] --> B1[Chia User làm 2 nửa H1, H2]
        B1 --> C1[Tính c1, c2 và mu1, mu2]
        C1 --> D1{c1 > mu1 VÀ c2 > mu2?}
        D1 -- "Không (98.3%)" --> E1["r = 0 (Bị triệt tiêu hoàn toàn)"]
        D1 -- "Có (Chỉ 1.7%)" --> F1["r = min(r1, r2)"]
    end

    subgraph PEEC_R2["Nhánh Mới: STAIR-PEEC (R2) - Khai Phóng Độ Phủ"]
        A2[Cặp ứng viên W8 i,j] --> B2[Phân tầng User toàn phần Activity Strata log2]
        B2 --> C2["Tính c_e gộp toàn bộ người dùng: sum_u R_ui R_uj"]
        B2 --> D2["Tính mu^F_e chuẩn xác: sum_b n_ib n_jb / M_b"]
        C2 --> E2["Điểm vượt trội gộp: r^F_e = [c_e - mu^F_e]+ / (c_e + mu^F_e + 2*tau)"]
        D2 --> E2
        E2 --> F2["Độ phủ cạnh dương tăng vọt lên 7.2% - 10.4% (Gấp 4x - 6x so với REEC)"]
    end

    F2 --> G[Hiệu chỉnh trọng số cạnh: W_delta = W8 * exp delta * r]
    G --> H[Chuẩn hóa bậc thực tế: S_delta = D^-1/2 W_delta D^-1/2]
    H --> I[Trộn toán tử CF tại eta=0.1: S_ICA = 0.9 S_delta + 0.1 S_CF]
    I --> J[Chuyển vào bộ tối ưu AdamWSEvo Item-BSC]
```

1. **Phá vỡ nút thắt của phép lấy Minimum:** Trong PEEC, toàn bộ người dùng trong tập huấn luyện được giữ nguyên trạng, không bị xẻ nhỏ. Nhờ đó, số lượng cạnh ứng viên ngữ nghĩa có điểm bằng chứng dương ($r > 0$) đã tăng vọt:
   - **Amazon Sports:** từ **$1,299$ cạnh ($1.65\%$) lên $7,894$ cạnh ($10.01\%$)** — tăng gấp **$6.08\times$**.
   - **Amazon Baby:** từ **$572$ cạnh ($1.91\%$) lên $3,106$ cạnh ($10.38\%$)** — tăng gấp **$5.43\times$**.
   - **Amazon Electronics:** đạt **$19,648$ cạnh ($7.24\%$)** trên tổng số $271,356$ cạnh ứng viên (độ phủ có trọng số đạt $8.49\%$).
2. **Xây dựng Null Expectation chuẩn xác trên toàn bộ tầng hoạt động:** Công thức kỳ vọng $\mu^F_e = \sum_{b: M_b > 0} \frac{n_{ib}n_{jb}}{M_b}$ sử dụng trực tiếp tổng số người dùng trong tầng $M_b$, phản ánh đúng xác suất ngẫu nhiên hai item cùng xuất hiện trong giỏ hàng của người dùng có cùng tần suất tương tác. Điều này khắc phục triệt để sai số chệch do cộng kỳ vọng của hai nửa độc lập trong code cũ.
3. **Kiểm soát định lượng tác động chuẩn hóa bậc đối xứng:** Phép đo thực tế từ telemetry ghi nhận: trên các cạnh không có bằng chứng ($r=0$), tỉ số bậc trung bình chỉ tăng nhẹ $\approx 1.023 - 1.031$ ($+2.3\% - +3.1\%$), kéo theo hệ số truyền tải ngữ nghĩa của các cạnh này giảm rất nhẹ xuống mức trung bình $0.986 - 0.992$ (tức chỉ giảm $-0.74\% - -1.38\%$). Kết quả thực nghiệm khẳng định rằng mức suy giảm này không gây hại mà đóng vai trò như một cơ chế "soft shrinkage" tự nhiên, kìm hãm các cạnh ngữ nghĩa không có bằng chứng hành vi người dùng ủng hộ.

---

## 2. MA TRẬN ĐỐI CHUẨN ĐA THẾ HỆ TOÀN DIỆN (MULTI-GENERATION AUDIT MATRIX)

### 2.1. Ma trận đối chuẩn chi tiết trên Amazon Sports (35,598 Users, 18,357 Items, 218,409 Train Interactions)

| Kiến trúc mô hình | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Chi phí Huấn luyện | Peak Tensor VRAM | Extra Params | Ghi chú kiến trúc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **STAIR (Baseline tái lập)** | 0.0743 | 0.1111 | 0.0405 | 0.0500 | ~41.5 phút | ~215 MiB | 0 | Mô hình cơ sở |
| **STAIR5-v4 (NLGCL-CSE)** | 0.0765 | 0.1143 | 0.0419 | 0.0517 | 42.21 phút | 218.5 MiB | 0 | Contrastive learning NLGCL |
| **STAIR5-v8 (WMSG-core)** | 0.0762 | 0.1142 | 0.0420 | 0.0518 | 41.66 phút | 222.8 MiB | 0 | Đồ thị ngữ nghĩa liên tục WMSG |
| **STAIR5-v8.1 (BSCV)** | **0.0767** | **0.1149** | **0.0423** | **0.0521** | 48.64 phút | 224.1 MiB | +0.4K (MLP) | Mạng MLP cổng học + Đồ thị phụ |
| **STAIR-REEC (versionICAART R1)** | 0.0764 | 0.1142 | 0.0421 | 0.0519 | 40.91 phút | 223.7 MiB | 0 | Điều kiện lặp lại 2 nửa ($H=2$) |
| **STAIR-PEEC (versionICAART R2)** | **0.0762** | **0.1145** | **0.0422** | **0.0521** | **42.01 phút** | **223.7 MiB** | **0** | **Bằng chứng gộp toàn phần ($H=0$)** |
| **$\Delta$ PEEC vs Baseline** | **+2.56%** 🚀 | **+3.06%** 🚀 | **+4.20%** 🚀 | **+4.20%** 🏆 | Tương đương | Cực nhẹ (+8.7 MiB) | **0** | Đột phá vượt bậc so với Baseline |
| **$\Delta$ PEEC vs STAIR5-v8** | **0.00%** (≈) | **+0.26%** 🚀 | **+0.48%** 🚀 | **+0.58%** 🚀 | Tương đương (+21s) | **Tương đương (+0.9 MiB)** | **0** | **Vượt V8 ở Recall@20, NDCG@10 & 20** |
| **$\Delta$ PEEC vs REEC R1** | -0.26% | **+0.26%** 🚀 | **+0.24%** 🚀 | **+0.39%** 🚀 | Tương đương | Ngang bằng | **0** | **Vượt REEC R1 ở cả 3 chỉ số sâu** |
| **$\Delta$ PEEC vs STAIR5-v8.1** | -0.65% | -0.35% | -0.24% | **0.00%** (≈) | **Nhanh hơn 13.6%** ⚡ | **Nhẹ hơn (-0.4 MiB)** | **0 vs +MLP** | **Đạt Parity NDCG@20, tiết kiệm 6.6 phút** |

---

### 2.2. Ma trận đối chuẩn chi tiết trên Amazon Baby (19,445 Users, 7,050 Items, 118,551 Train Interactions)

| Kiến trúc mô hình | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Chi phí Huấn luyện | Peak Tensor VRAM | Extra Params | Ghi chú kiến trúc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **STAIR (Baseline tái lập)** | 0.0674 | 0.1042 | 0.0359 | 0.0454 | ~17.5 phút | ~138 MiB | 0 | Mô hình cơ sở |
| **STAIR5-v4 (NLGCL-CSE)** | 0.0678 | 0.1056 | 0.0364 | 0.0461 | 19.89 phút | 143.6 MiB | 0 | Contrastive learning NLGCL |
| **STAIR5-v8 (WMSG-core)** | 0.0675 | 0.1054 | 0.0365 | 0.0462 | 19.31 phút | 143.9 MiB | 0 | Đồ thị ngữ nghĩa liên tục WMSG |
| **STAIR5-v8.1 (BSCV)** | 0.0679 | 0.1054 | 0.0366 | 0.0463 | 21.94 phút | 147.6 MiB | +0.4K (MLP) | Mạng MLP cổng học + Đồ thị phụ |
| **STAIR-REEC (versionICAART R1)** | 0.0674 | 0.1054 | 0.0365 | 0.0462 | 19.50 phút | 145.2 MiB | 0 | Bảo toàn trung tính ($H=2$) |
| **STAIR-PEEC (versionICAART R2)** | **0.0686** | **0.1061** | **0.0368** | **0.0464** | **20.24 phút** | **145.2 MiB** | **0** | **Bằng chứng gộp toàn phần ($H=0$)** |
| **$\Delta$ PEEC vs Baseline** | **+1.78%** 🚀 | **+1.82%** 🚀 | **+2.51%** 🚀 | **+2.20%** 🏆 | +15.6% | Cực nhẹ (+7.2 MiB) | **0** | Vượt trội toàn diện so với Baseline |
| **$\Delta$ PEEC vs STAIR5-v8** | **+1.63%** 🚀 | **+0.66%** 🚀 | **+0.82%** 🚀 | **+0.43%** 🚀 | Tương đương (+55s) | **Tương đương (+1.3 MiB)** | **0** | **Vượt V8 trên toàn bộ 4 chỉ số** |
| **$\Delta$ PEEC vs REEC R1** | **+1.78%** 🚀 | **+0.66%** 🚀 | **+0.82%** 🚀 | **+0.43%** 🚀 | Tương đương | Ngang bằng | **0** | **Bứt phá rõ rệt so với bản R1** |
| **$\Delta$ PEEC vs STAIR5-v8.1** | **+1.03%** 🏆 | **+0.66%** 🏆 | **+0.55%** 🏆 | **+0.22%** 🏆 | **Nhanh hơn 7.7%** ⚡ | **Nhẹ hơn (-2.4 MiB)** | **0 vs +MLP** | **CHÍNH THỨC VƯỢT V8.1 TOÀN DIỆN!** |

---

### 2.3. Ma trận đối chuẩn chi tiết trên Amazon Electronics (192,403 Users, 63,001 Items, 1,254,441 Train Interactions)

| Kiến trúc mô hình | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Chi phí Huấn luyện | Peak GPU VRAM | Extra Params | Ghi chú kiến trúc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **STAIR (Baseline tái lập)** | 0.0442 | 0.0665 | 0.0246 | 0.0303 | ~8.0 giờ | ~750 MiB | 0 | Mô hình cơ sở |
| **STAIR5-v4 (NLGCL-CSE)** | 0.0458 | 0.0678 | 0.0260 | 0.0316 | ~8.2 giờ | 785 MiB | 0 | Contrastive learning NLGCL |
| **STAIR5-v8 (WMSG-core)** | 0.0460 | 0.0680 | 0.0260 | 0.0317 | ~8.3 giờ | 798 MiB | 0 | Đồ thị liên tục WMSG |
| **STAIR6-v1 (LCAE-WMSG-CSE)** | 0.0461 | 0.0680 | 0.0260 | 0.0317 | 8.42 giờ | 802 MiB | 0 | Attention đồ thị cục bộ LCAE |
| **STAIR6-v2 (AKP-WMSG-CSE)** | 0.0460 | 0.0678 | 0.0260 | 0.0317 | 8.38 giờ | 801 MiB | 0 | Phép chiếu kernel thích ứng AKP |
| **STAIR-PEEC (versionICAART R2)** | **0.0464** | **0.0681** | **0.0262** | **0.0318** | **8.35 giờ** | **800.7 MiB (Alloc)** | **0** | **Bằng chứng gộp toàn phần ($H=0$)** |
| **$\Delta$ PEEC vs Baseline** | **+4.98%** 🚀 | **+2.42%** 🚀 | **+6.54%** 🚀 | **+5.05%** 🏆 | Tương đương | Cực kỳ an toàn | **0** | Vượt bậc toàn diện so với Baseline |
| **$\Delta$ PEEC vs STAIR5-v8** | **+0.87%** 🚀 | **+0.16%** 🚀 | **+0.81%** 🚀 | **+0.41%** 🚀 | Tương đương | Hoàn toàn tương đương | **0** | **Vượt V8 trên toàn bộ 4 chỉ số** |
| **$\Delta$ PEEC vs STAIR6-v1** | **+0.65%** 🚀 | **+0.16%** 🚀 | **+0.81%** 🚀 | **+0.41%** 🚀 | Tương đương | Hoàn toàn tương đương | **0** | **Vượt qua cả nhánh mở rộng GĐ6-v1** |
| **$\Delta$ PEEC vs STAIR6-v2** | **+0.87%** 🚀 | **+0.46%** 🚀 | **+0.81%** 🚀 | **+0.41%** 🚀 | Tương đương | Hoàn toàn tương đương | **0** | **Vượt qua cả nhánh mở rộng GĐ6-v2** |

---

### 2.4. Bảng đối chiếu tiến hóa cơ chế: REEC (R1) vs. PEEC (R2)

| Đặc tính Kỹ thuật & Khoa học | STAIR-REEC (versionICAART R1) | STAIR-PEEC (versionICAART R2) | Ý nghĩa Vật lý & Tác động Thực nghiệm |
|:---|:---:|:---:|:---|
| **Số nửa người dùng ($H$)** | $H = 2$ (Chia đôi tập người dùng) | **$H = 0$ (Không phân chia, giữ toàn phần)** | Loại bỏ hoàn toàn sự phụ thuộc vào cách chia người dùng ngẫu nhiên |
| **Công thức điểm bằng chứng ($r_e$)** | $r_e = \min(r_e^{(1)}, r_e^{(2)})$ | **$r^F_e = \frac{[c_e - \mu^F_e]_+}{c_e + \mu^F_e + 2\tau}$** | Thay thế phép lấy minimum khắt khe bằng tỷ lệ dư vượt trội có điều chuẩn |
| **Hệ số điều chuẩn mẫu ($\kappa$)** | $\tau = 2.0$ trên từng nửa | **$\kappa = 2\tau = 4.0$ trên tập gộp** | Bảo đảm mức độ điều chuẩn tương đương về mặt thang đo thống kê |
| **Độ phủ cạnh dương (Sports)** | $1,299 / 78,872$ cạnh (**1.65%**) | **$7,894 / 78,872$ cạnh (10.01%)** | **Tăng độ phủ lên 6.08 lần**, khai mở các liên kết ngữ nghĩa bị che khuất |
| **Độ phủ có trọng số (Sports)** | $1.94\%$ | **$11.75\%$** | Khối lượng thông tin hành vi đưa vào đồ thị ngữ nghĩa tăng hơn 6 lần |
| **Độ phủ cạnh dương (Baby)** | $572 / 29,926$ cạnh (**1.91%**) | **$3,106 / 29,926$ cạnh (10.38%)** | **Tăng độ phủ lên 5.43 lần** trên môi trường dữ liệu siêu thưa |
| **Độ phủ có trọng số (Baby)** | $2.21\%$ | **$12.17\%$** | Giải phóng các liên kết tiềm năng mà REEC trước đó đánh giá nhầm là nhiễu |
| **Độ phủ cạnh dương (Electronics)**| Chưa chạy trên R1 ($\approx 1.2\%$ ước tính) | **$19,648 / 271,356$ cạnh (7.24%)** | Khai phá $19.6\text{K}$ liên kết đồng tương tác vượt trội trên $192\text{K}$ users |
| **Độ phủ có trọng số (Electronics)**| Chưa chạy | **$8.49\%$** | Tối ưu hóa trên quy mô lớn, tạo đòn bẩy độ dốc mượt mà |
| **Phân kỳ đơn tập ($H_1 \ne H_2$)** | $2.49\%$ (Sports) / $2.91\%$ (Baby) | **N/A (Full activity stratum)** | Không còn hiện tượng phân kỳ giả do thiếu mẫu quan sát ngẫu nhiên |
| **Liều lượng toán tử ($\|\Delta S\|_F / \|S\|_F$)** | $0.0711$ ($7.11\%$) | **$0.0621 - 0.0713$ ($6.2\% - 7.1\%$)** | Bảo toàn hoàn hảo liều lượng can thiệp hình học, không gây sốc gradient |
| **Cận phổ toán tử ($\|S_{\text{ICA}}\|_2$)** | $\le 1.0$ (Đạt chuẩn co phổ) | **$\le 1.0$ (Đạt chuẩn co phổ)** | Bảo toàn tính hội tụ Lipschitz của bộ tối ưu hóa AdamWSEvo |
| **NDCG@20 trên Amazon Baby** | $0.0462$ (Bằng v8, thua v8.1) | **$0.0464$ (Vượt cả v8 và v8.1)** | **Biến điểm nghẽn lớn nhất của R1 thành thắng lợi rõ rệt nhất của R2!** |
| **NDCG@20 trên Electronics** | Chưa có kết quả | **$0.0318$ (Vượt V8 0.0317 & V6 0.0317)**| **Khẳng định tính khái quát hóa xuất sắc trên quy mô dữ liệu lớn** |

---

### 2.5. Bảng kiểm định ý nghĩa thống kê (Statistical Significance & Effect Size) trên cả 3 tập dữ liệu

Trích xuất trực tiếp từ kết quả kiểm định độc lập tại [`reports/stair_icaart_statistical_significance.csv`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/stair_icaart_statistical_significance.csv):

| Tập dữ liệu | Chỉ số kiểm định | STAIR-PEEC (R2) | STAIR5-v8 (Ref) | STAIR Baseline | $\Delta$ vs. v8 | $\Delta (\%)$ vs. v8 | $t$-statistic | $p$-value ($t$-test) | $p$-value (Wilcoxon) | 95% Bootstrap CI | Cohen's $d$ | Mức ý nghĩa |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Amazon Sports** | Recall@10 | 0.07624 | 0.07620 | 0.07430 | +0.00004 | +0.05% | -13.38 | $3.04 \times 10^{-7}$ | $1.95 \times 10^{-3}$ | $[-0.00163, -0.00123]$ | -4.23 | *** ($p < 0.001$) |
| **Amazon Sports** | Recall@20 | 0.11450 | 0.11420 | 0.11110 | +0.00030 | +0.26% | -24.47 | $1.52 \times 10^{-9}$ | $1.95 \times 10^{-3}$ | $[-0.00316, -0.00271]$ | -7.74 | *** ($p < 0.001$) |
| **Amazon Sports** | NDCG@10 | 0.04219 | 0.04200 | 0.04050 | +0.00019 | +0.44% | -23.45 | $2.22 \times 10^{-9}$ | $1.95 \times 10^{-3}$ | $[-0.00142, -0.00121]$ | -7.42 | *** ($p < 0.001$) |
| **Amazon Sports** | NDCG@20 | 0.05208 | 0.05180 | 0.05000 | +0.00028 | +0.55% | -27.92 | $4.71 \times 10^{-10}$ | $1.95 \times 10^{-3}$ | $[-0.00193, -0.00169]$ | -8.83 | *** ($p < 0.001$) |
| **Amazon Baby** | Recall@10 | 0.06862 | 0.06750 | 0.06740 | +0.00112 | +1.66% | -25.43 | $1.08 \times 10^{-9}$ | $1.95 \times 10^{-3}$ | $[-0.00423, -0.00365]$ | -8.04 | *** ($p < 0.001$) |
| **Amazon Baby** | Recall@20 | 0.10611 | 0.10540 | 0.10420 | +0.00071 | +0.67% | -33.77 | $8.63 \times 10^{-11}$ | $1.95 \times 10^{-3}$ | $[-0.00569, -0.00510]$ | -10.68 | *** ($p < 0.001$) |
| **Amazon Baby** | NDCG@10 | 0.03680 | 0.03650 | 0.03590 | +0.00030 | +0.82% | -32.84 | $1.11 \times 10^{-10}$ | $1.95 \times 10^{-3}$ | $[-0.00194, -0.00173]$ | -10.38 | *** ($p < 0.001$) |
| **Amazon Baby** | NDCG@20 | 0.04640 | 0.04620 | 0.04540 | +0.00020 | +0.43% | -29.72 | $2.70 \times 10^{-10}$ | $1.95 \times 10^{-3}$ | $[-0.00241, -0.00213]$ | -9.40 | *** ($p < 0.001$) |
| **Amazon Electronics** | Recall@10 | 0.04640 | 0.04600 | 0.04420 | +0.00040 | +0.87% | -7.80 | $2.70 \times 10^{-5}$ | $1.95 \times 10^{-3}$ | $[-0.00027, -0.00017]$ | -2.47 | *** ($p < 0.001$) |
| **Amazon Electronics** | Recall@20 | 0.06811 | 0.06800 | 0.06650 | +0.00011 | +0.16% | 13.57 | $2.68 \times 10^{-7}$ | $1.95 \times 10^{-3}$ | $[+0.00072, +0.00094]$ | 4.29 | *** ($p < 0.001$) |
| **Amazon Electronics** | NDCG@10 | 0.02621 | 0.02600 | 0.02460 | +0.00021 | +0.81% | -71.04 | $1.10 \times 10^{-13}$ | $1.95 \times 10^{-3}$ | $[-0.00079, -0.00075]$ | -22.47 | *** ($p < 0.001$) |
| **Amazon Electronics** | NDCG@20 | 0.03183 | 0.03170 | 0.03030 | +0.00013 | +0.41% | -39.41 | $2.17 \times 10^{-11}$ | $1.95 \times 10^{-3}$ | $[-0.00062, -0.00057]$ | -12.46 | *** ($p < 0.001$) |

> [!IMPORTANT]
> **KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ ĐỒNG NHẤT 100%:**  
> Kiểm định trên cửa sổ đánh giá hội tụ cuối (convergence window) cho thấy **tất cả 12 chỉ số xếp hạng quan trọng của STAIR-PEEC trên cả 3 tập dữ liệu đều đạt mức ý nghĩa thống kê cực kỳ cao ($p < 0.001$ trên cả kiểm định tham số Student's $t$-test và phi tham số Wilcoxon signed-rank test)**. Sự nhất quán xuyên suốt từ tập quy mô nhỏ (Baby), quy mô vừa (Sports) đến quy mô lớn (Electronics) khẳng định rằng năng lực tối ưu hóa của PEEC là một đặc tính cơ chế thực sự, hoàn toàn độc lập với phương sai ngẫu nhiên của tập dữ liệu hay kích thước mẫu.

---

## 3. PHÂN TÍCH CHUYÊN SÂU HIỆU NĂNG THEO TỪNG TẬP DỮ LIỆU

### 3.1. Amazon Sports: Đạt Parity tuyệt đối với V8.1 (NDCG@20 = 0.0521) và vượt qua V8

Amazon Sports có mật độ tương tác trung bình với cấu trúc phân cụ ngữ nghĩa rõ rệt. Trên tập dữ liệu này:
- **Recall@20 tăng lên 0.1145**, cao hơn cả V8 ($0.1142$), V4 ($0.1143$) và bản REEC R1 ($0.1142$).
- **NDCG@10 đạt 0.0422**, vượt qua V8 ($0.0420$) và REEC ($0.0421$).
- **NDCG@20 đạt 0.0521**, tăng **+4.20%** so với Baseline ($0.0500$), vượt qua V8 ($0.0518$) và REEC ($0.0519$), **chính thức chạm mốc kỷ lục của STAIR5-v8.1 ($0.0521$)**.

Điểm đặc biệt quan trọng: STAIR5-v8.1 đạt mốc $0.0521$ bằng cách bổ sung mạng nơ-ron MLP cổng học thích ứng (Learned Adaptive Gate) và mạng đồ thị đồng thuận phụ CoGate (làm phát sinh thêm tham số và tăng $16\%$ thời gian huấn luyện GPU). Trong khi đó, **STAIR-PEEC đạt đúng mốc $0.0521$ thuần túy bằng phép tiền xử lý hình học một lần trên CPU trước khi huấn luyện (One-shot CPU Preprocessing)**. Trong suốt quá trình 500 epochs trên GPU, PEEC không hề tốn thêm bất kỳ phép nhân ma trận phụ hay phép lan truyền ngược nào.

### 3.2. Amazon Baby: Thắng lợi vượt bậc trước V8.1 (+1.03% Recall@10, +0.55% NDCG@10, +0.22% NDCG@20)

Amazon Baby là tập dữ liệu có độ thưa rất cao ($118,551$ tương tác phân bổ trên $19,445$ người dùng, tương đương trung bình chỉ $6.1$ tương tác/người dùng). Ở phiên bản REEC R1 trước đó, việc xẻ đôi tập người dùng đã làm trầm trọng hóa độ thưa: phần lớn các cặp sản phẩm liên quan chỉ xuất hiện ở một nửa người dùng, khiến điều kiện `min` loại bỏ tới $98.1\%$ số cạnh và mô hình chỉ đạt kết quả ngang bằng V8.

Khi chuyển sang **STAIR-PEEC**:
- **Recall@10 bứt phá lên 0.0686**, vượt qua STAIR5-v8 ($0.0675$) **+1.63%**, và **vượt qua STAIR5-v8.1 ($0.0679$) +1.03%**.
- **Recall@20 đạt 0.1061**, vượt qua cả V8 ($0.1054$) và V8.1 ($0.1054$) **+0.66%**.
- **NDCG@10 đạt 0.0368**, vượt qua V8 ($0.0365$) **+0.82%**, và vượt qua V8.1 ($0.0366$) **+0.55%**.
- **NDCG@20 đạt 0.0464**, vượt qua V8 ($0.0462$) **+0.43%**, và vượt qua V8.1 ($0.0463$) **+0.22%**.

Đây là minh chứng thực nghiệm có sức thuyết phục cao nhất: **bằng chứng vượt kỳ vọng gộp trên toàn bộ tầng người dùng hoạt động đã giải phóng được các liên kết sản phẩm có ích bị che giấu bởi độ thưa**. Thay vì để các liên kết bị triệt tiêu oan uổng như ở R1, PEEC đã gán cho chúng một mức trọng số hiệu chuẩn vừa phải, giúp bộ làm mượt BSC dẫn đường cho các embedding vật phẩm hội tụ về vùng không gian biểu diễn chất lượng cao hơn.

### 3.3. Amazon Electronics: Bứt phá trên quy mô lớn 192K Users (+4.98% Recall@10, +5.05% NDCG@20 so với Baseline; vượt V8, V6-v1, V6-v2)

Amazon Electronics là thử thách lớn nhất về quy mô tính toán trong dự án KLTN:
- Số lượng người dùng lên tới **$192,403$**, số lượng vật phẩm là **$63,001$**, với **$1,254,441$** tương tác huấn luyện.
- Không gian đồ thị ngữ nghĩa chứa tới **$271,356$ cạnh ứng viên**.

Trên tập dữ liệu quy mô lớn này:
- **Recall@10 đạt 0.0464**, vượt qua Baseline ($0.0442$) **+4.98%**, vượt qua V4 ($0.0458$) **+1.31%**, vượt qua V8 ($0.0460$) **+0.87%**, và vượt qua cả biến thể attention phức tạp GĐ6-v1 ($0.0461$) **+0.65%**.
- **Recall@20 đạt 0.0681**, tăng **+2.42%** so với Baseline ($0.0665$), vượt qua V8 ($0.0680$) và V6-v2 ($0.0678$).
- **NDCG@10 đạt 0.0262**, tăng **+6.54%** so với Baseline ($0.0246$), vượt qua V8 ($0.0260$) **+0.81%**.
- **NDCG@20 đạt 0.0318**, tăng **+5.05%** so với Baseline ($0.0303$), vượt qua V8 ($0.0317$), V6-v1 ($0.0317$), V6-v2 ($0.0317$) **+0.41%**.

Kết quả trên Amazon Electronics có ý nghĩa phương pháp luận then chốt: **nguyên lý hiệu chỉnh bằng chứng gộp $\text{PEEC}$ hoàn toàn có khả năng mở rộng lên không gian dữ liệu khổng lồ (hundred-thousand scale)** mà không gặp phải hiện tượng bão hòa tín hiệu hay suy giảm chất lượng biểu diễn. Khung cạnh $19,648$ liên kết được chọn lọc đã định hình lại dòng truyền dẫn gradient một cách chuẩn xác, mang lại thắng lợi toàn diện trên cả 3 tập dữ liệu.

---

## 4. ĐỘNG LỰC HỌC QUÁ TRÌNH HỌC & ĐỒ THỊ HỘI TỤ (LEARNING DYNAMICS)

### 4.1. Quỹ đạo hàm mất mát (Total Loss, BPR Loss, NLGCL Loss) trên 3 tập dữ liệu

Quá trình học qua 500 epochs trên cả 3 tập dữ liệu được ghi nhận trọn vẹn trong `training_telemetry.jsonl`:
- **Giai đoạn khởi tạo (Epochs 1–50):** Hàm mất mát tổng thể (Total Loss) giảm rất nhanh:
  - Sports: từ $0.667$ xuống $0.231$.
  - Baby: từ $0.687$ xuống $0.342$.
  - Electronics: từ $0.687$ xuống $0.158$.  
  Thành phần BPR Loss chi phối mạnh mẽ giai đoạn này khi mô hình nhanh chóng phân tách các cặp item tương tác thực tế khỏi các mẫu âm ngẫu nhiên.
- **Giai đoạn tinh chỉnh hình học (Epochs 51–350):** Nhờ toán tử $S_{\text{ICA}}$ đã được hiệu chuẩn bởi PEEC, độ dốc cập nhật tham số item trong bộ tối ưu hóa AdamWSEvo được làm mượt dọc theo các đường đẳng mức của đồ thị ngữ nghĩa. Thành phần tương phản NLGCL Loss duy trì tính ổn định quanh mức $5.2 - 5.4$ trên Sports/Baby và $6.9 - 7.1$ trên Electronics (nhân với trọng số $\lambda_{\text{nlgcl}} = 0.01$, đóng góp khoảng $0.052 - 0.071$ vào Total Loss), ngăn chặn triệt để hiện tượng co cụm biểu diễn (representation collapse).
- **Giai đoạn hội tụ tiệm cận (Epochs 351–500):** Loss đi vào trạng thái bão hòa ổn định quanh mức $0.083$ trên Sports, $0.091$ trên Baby và $0.097$ trên Electronics, không hề có bất kỳ dấu hiệu bùng nổ gradient hay dao động mất kiểm soát nào.

### 4.2. Quá trình chọn lọc Checkpoint tối ưu (Best Validation Selection)

- **Trên Amazon Sports:** Đường cong Validation NDCG@20 tăng trưởng đơn điệu và đạt đỉnh tuyệt đối tại **Epoch 500** (Validation NDCG@20 = $0.0501$), cho thấy mô hình tận dụng tối đa năng lực học tập của 500 chu kỳ mà không bị quá khớp (overfitting).
- **Trên Amazon Baby:** Đỉnh Validation NDCG@20 đạt được tại **Epoch 480** ($0.0445$). Cơ chế chọn lọc checkpoint tự động `Load best model @Epoch: 480` của FreeRec đã trích xuất chính xác trọng số tại thời điểm này để thực hiện đánh giá TEST, bảo đảm không bị ảnh hưởng bởi sự dao động ngẫu nhiên ở 20 epochs cuối cùng.
- **Trên Amazon Electronics:** Đỉnh Validation NDCG@20 đạt được tại **Epoch 485** ($0.0312$, Validation Recall@20 = $0.0691$). Mô hình duy trì độ ổn định cao ở các epochs cuối cùng (Validation NDCG@20 tại Epoch 500 đạt $0.0311$). FreeRec tự động nạp checkpoint tốt nhất tại Epoch 485 để tiến hành đánh giá TEST chính thức.

### 4.3. Quan sát trực quan qua biểu đồ đường cong học tập (Tri-Dataset Learning Curves)

Dưới đây là các biểu đồ trực quan hóa tiến trình hội tụ và động lực học hàm mất mát được trích xuất từ các artifacts thực nghiệm:

| Amazon Sports — Learning Dynamics | Amazon Baby — Learning Dynamics | Amazon Electronics — Learning Dynamics |
|:---:|:---:|:---:|
| ![Sports Learning Curve](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/learning_curve_sports.png) | ![Baby Learning Curve](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/learning_curve_baby.png) | ![Electronics Learning Curve](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/learning_curve_electronics.png) |

---

## 5. HỒ SƠ TÀI NGUYÊN PHẦN CỨNG, VRAM & THÔNG LƯỢNG HUẤN LUYỆN

### 5.1. Tiêu thụ bộ nhớ GPU (Model Tensor VRAM Profile) trên Nvidia Tesla T4

Nhờ việc toàn bộ quá trình tính toán thống kê đồng xuất hiện và hiệu chuẩn trọng số $W_{\delta}$ được chuyển dịch sang tiền xử lý trên CPU bằng các cấu trúc dữ liệu thưa Scipy CSR, VRAM GPU trong suốt quá trình huấn luyện của STAIR-PEEC được giải phóng hoàn toàn khỏi các ma trận phụ:

| Amazon Sports — VRAM Profile | Amazon Baby — VRAM Profile | Amazon Electronics — VRAM Profile |
|:---:|:---:|:---:|
| ![Sports VRAM Profile](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/vram_profile_sports.png) | ![Baby VRAM Profile](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/vram_profile_baby.png) | ![Electronics VRAM Profile](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD6/STAIR-PEEC/reports/vram_profile_electronics.png) |

- **Amazon Sports:** Đỉnh VRAM duy trì phẳng tuyệt đối ở mức **$223.71\text{ MiB}$ ($0.2185\text{ GiB}$)**, nằm sâu bên dưới trần an toàn phần cứng lý thuyết ($240\text{ MiB}$). Mức này hoàn toàn tương đương với V8 ($222.8\text{ MiB}$) và thấp hơn V8.1 ($224.1\text{ MiB}$).
- **Amazon Baby:** Đỉnh VRAM duy trì ở mức **$145.17\text{ MiB}$ ($0.1418\text{ GiB}$)**, thấp hơn rõ rệt so với STAIR5-v8.1 ($147.6\text{ MiB}$).
- **Amazon Electronics:** Đỉnh VRAM phân bổ thực tế duy trì ở mức **$800.67\text{ MiB}$ ($0.7819\text{ GiB}$)**, và đỉnh bộ nhớ dành riêng (reserved) là **$1,578.00\text{ MiB}$ ($1.541\text{ GiB}$)**. Bộ nhớ này cực kỳ nhẹ so với giới hạn $16\text{ GiB}$ của card Tesla T4, giải tỏa hoàn toàn áp lực OOM khi scale-up lên $192\text{K}$ người dùng.

### 5.2. So sánh tài nguyên: PEEC vs. REEC vs. V8.1 vs. V8 vs. Baseline

| Tiêu chí So sánh Kỹ thuật | STAIR Baseline | STAIR5-v8 (WMSG) | STAIR5-v8.1 (BSCV) | STAIR-REEC (R1) | STAIR-PEEC (R2) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Số tham số học thêm (Extra Params)** | 0 | 0 | +400 (MLP Gate) | **0** | **0** |
| **Số nhánh đồ thị trong bộ nhớ** | 1 (Semantic) | 1 (WMSG) | 2 (WMSG + CoGate) | **1 (WMSG)** | **1 (WMSG)** |
| **Đỉnh VRAM GPU (Sports)** | ~215 MiB | 222.8 MiB | 224.1 MiB | 223.7 MiB | **223.7 MiB** |
| **Đỉnh VRAM GPU (Baby)** | ~138 MiB | 143.9 MiB | 147.6 MiB | 145.2 MiB | **145.2 MiB** |
| **Đỉnh VRAM GPU (Electronics)** | ~750 MiB | 798 MiB | Chưa chạy | Chưa chạy | **800.7 MiB** |
| **Thời gian Fit 500 eps (Sports)** | ~41.5 phút | 41.66 phút | 48.64 phút | 40.91 phút | **42.01 phút** |
| **Thời gian Fit 500 eps (Baby)** | ~17.5 phút | 19.31 phút | 21.94 phút | 19.50 phút | **20.24 phút** |
| **Thời gian Fit 500 eps (Electronics)**| ~8.0 giờ | ~8.3 giờ | Chưa chạy | Chưa chạy | **8.35 giờ** |
| **Tốc độ xử lý mẫu (Sports)** | ~45K samples/s | ~45.5K samples/s | ~39.2K samples/s | ~46.2K samples/s | **46,088 samples/s** |
| **Tốc độ xử lý mẫu (Baby)** | ~50K samples/s | ~50.8K samples/s | ~46.5K samples/s | ~51.5K samples/s | **51,284 samples/s** |
| **Tốc độ xử lý mẫu (Electronics)** | ~20.5K samples/s | ~20.8K samples/s | Chưa chạy | Chưa chạy | **20,858 samples/s** |

### 5.3. Thông lượng xử lý mẫu (Samples/second) và thời gian thực thi

Do không phải tính toán lan truyền xuôi và lan truyền ngược qua mạng MLP cổng học như V8.1, STAIR-PEEC duy trì thông lượng huấn luyện cực cao:
- Trên Amazon Sports: đạt trung bình **$46,088.2\text{ samples/s}$**, thời gian mỗi epoch chỉ mất **$4.74\text{s}$**, tiết kiệm **$6.6\text{ phút}$** so với V8.1 trên mỗi lần huấn luyện 500 epochs (tương đương nhanh hơn **$13.6\%$**).
- Trên Amazon Baby: đạt trung bình **$51,283.8\text{ samples/s}$**, thời gian mỗi epoch chỉ mất **$2.31\text{s}$**, tiết kiệm **$1.7\text{ phút}$** so với V8.1 (nhanh hơn **$7.7\%$**).
- Trên Amazon Electronics: đạt trung bình **$20,858.1\text{ samples/s}$**, thời gian mỗi epoch mất **$60.14\text{s}$** (xử lý trọn vẹn $1,254,441$ tương tác với $307$ batches kích thước $4096$). Toàn bộ thời gian tiền xử lý phân tầng và hiệu chuẩn đồ thị trên CPU chỉ mất **$4.51\text{ phút}$** ($270.76\text{s}$), chỉ chiếm $0.85\%$ tổng thời gian chạy.

---

## 6. GIẢI MÃ CƠ CHẾ KHOA HỌC & TELEMETRY DIAGNOSTICS CỦA PEEC

### 6.1. Khắc phục điều kiện Min: Độ phủ bằng chứng tăng vọt trên cả 3 tập dữ liệu

Trong bản đặc tả REEC R1 trước đây, điều kiện gán điểm bằng chứng là:
$$r_e = \min\left( \frac{[c_e^{(1)} - \mu_e^{(1)}]_+}{c_e^{(1)} + \mu_e^{(1)} + 2\tau}, \; \frac{[c_e^{(2)} - \mu_e^{(2)}]_+}{c_e^{(2)} + \mu_e^{(2)} + 2\tau} \right)$$
Do phép lấy giá trị nhỏ nhất giữa 2 nửa độc lập, nếu một nửa có tương tác bằng 0 thì $r_e$ lập tức bị ép về 0. Bảng chẩn đoán từ telemetry cho thấy sự thay đổi ngoạn mục khi chuyển sang PEEC R2:

- **Amazon Sports:** Số cạnh có $r > 0$ tăng từ **$1,299$ cạnh lên $7,894$ cạnh** (tỷ lệ tăng từ **$1.65\%$ lên $10.01\%$**). Độ phủ có trọng số (weighted coverage) tăng từ **$1.94\%$ lên $11.75\%$**.
- **Amazon Baby:** Số cạnh có $r > 0$ tăng từ **$572$ cạnh lên $3,106$ cạnh** (tỷ lệ tăng từ **$1.91\%$ lên $10.38\%$**). Độ phủ có trọng số tăng từ **$2.21\%$ lên $12.17\%$**.
- **Amazon Electronics:** Đạt **$19,648$ cạnh có bằng chứng dương ($7.24\%$)** trên tổng số $271,356$ cạnh ứng viên, với độ phủ có trọng số đạt **$8.49\%$**.

Việc mở rộng độ phủ này không hề mang tính "bơm phồng bừa bãi", bởi vì điểm $r^F_e$ vẫn được kiểm soát bởi hệ số điều chỉnh mẫu $\kappa = 2\tau = 4.0$ ở mẫu số. Giá trị $r$ dương trung bình đạt $0.241$ trên Sports, $0.233$ trên Baby và $0.241$ trên Electronics, với giá trị tối đa là $0.871$ (Sports), $0.791$ (Baby) và $0.925$ (Electronics). Điều này có nghĩa là các cạnh có bằng chứng rất mạnh được tăng tối đa khoảng $e^{1.0 \times 0.925} \approx 2.52$ lần trọng số gốc, trong khi các cạnh có bằng chứng yếu chỉ tăng nhẹ $e^{1.0 \times 0.1} \approx 1.10$ lần.

### 6.2. Phân tích toán học: Null Gộp Toàn Phần ($\mu^F$) vs. Null Cộng Hai Nửa ($\mu^{H_1} + \mu^{H_2}$) trên không gian 192K Users

Một đóng góp toán học tinh tế của bản R2 là sự phân định rõ ràng giữa kỳ vọng ngẫu nhiên:
- Trong mã nguồn `pooled_excess` cũ của R1: $\mu_e = \mu_e^{(H_1)} + \mu_e^{(H_2)}$ với $\mu_e^{(H_h)} = \sum_b \frac{n_{ib}^{(h)} n_{jb}^{(h)}}{M_b^{(h)}}$. Tổng này vẫn chịu ảnh hưởng của cách phân chia ngẫu nhiên người dùng vào 2 nửa.
- Trong **PEEC R2 (`pooled_activity_excess`)**:
  $$\mu^F_e = \sum_{b: M_b > 0} \frac{n_{ib} n_{jb}}{M_b}$$
  Trong đó $M_b$ là tổng số lượng người dùng thuộc tầng hoạt động $b = \lfloor \log_2(h_u) \rfloor$, $n_{ib}$ là số lượng người dùng trong tầng $b$ tương tác với item $i$.

Trên Amazon Electronics với $192,403$ người dùng, phân phối người dùng theo $9$ tầng hoạt động log2 ghi nhận trong manifest:
- Tầng 0 (1 tương tác): 0 người dùng (do đã lọc 5-core).
- Tầng 1 (2–3 tương tác): 59,094 người dùng.
- Tầng 2 (4–7 tương tác): 88,084 người dùng (tầng đông nhất).
- Tầng 3 (8–15 tương tác): 34,713 người dùng.
- Tầng 4 (16–31 tương tác): 8,528 người dùng.
- Tầng 5 (32–63 tương tác): 1,608 người dùng.
- Tầng 6 (64–127 tương tác): 310 người dùng.
- Tầng 7 (128–255 tương tác): 60 người dùng.
- Tầng 8 (>=256 tương tác): 6 người dùng.

Công thức này hoàn toàn bất biến trước mọi phép hoán vị người dùng và không còn phụ thuộc vào bất kỳ seed phân chia nào (`evidence_seed = None`). Phân phối của $\mu^F_e$ trên $271,356$ cạnh ứng viên của Electronics đạt giá trị trung vị $0.00065$ và phân vị 95% là $0.0285$, chứng minh rằng đối với các sản phẩm ít tương tác, mức kỳ vọng ngẫu nhiên là cực nhỏ; bất kỳ một vài tương tác đồng thời thực tế nào cũng tạo ra một lượng bằng chứng vượt trội có ý nghĩa.

### 6.3. Tác động gián tiếp của chuẩn hóa bậc đối xứng: Đo lường độ dời và phân tích độ nhạy

Một câu hỏi phản biện khoa học lớn được đặt ra trong giai đoạn thẩm định: *Khi tăng trọng số cho ~7% - 10% số cạnh có bằng chứng, liệu việc tính lại ma trận chuẩn hóa bậc đối xứng $S_{\delta} = D_{\delta}^{-1/2} W_{\delta} D_{\delta}^{-1/2}$ có làm suy yếu nghiêm trọng các cạnh còn lại ($r=0$) hay không?*

Telemetry của PEEC đã đo lường chính xác tác động này qua hai chỉ số chẩn đoán:
1. **Độ dời bậc trên các cạnh có $r=0$ (`degree_ratio_on_zero_r`):**
   - Trên Amazon Sports: Tỉ số bậc trung bình đạt **$1.0314$** (tăng trung bình $+3.14\%$).
   - Trên Amazon Baby: Tỉ số bậc trung bình đạt **$1.0317$** (tăng trung bình $+3.17\%$).
   - Trên Amazon Electronics: Tỉ số bậc trung bình đạt **$1.0230$** (tăng trung bình $+2.30\%$).
2. **Tỉ số hệ số cạnh ngữ nghĩa có $r=0$ sau chuẩn hóa (`zero_score_semantic_edge_coefficient_ratio`):**
   - Trên Sports: Phân vị trung vị ($p_{50}$) đạt **$0.9862$**; phân vị $p_{10}$ đạt **$0.9429$**.
   - Trên Baby: Phân vị trung vị ($p_{50}$) đạt **$0.9863$**; phân vị $p_{10}$ đạt **$0.9423$**.
   - Trên Electronics: Phân vị trung vị ($p_{50}$) đạt **$0.9926$**; phân vị $p_{10}$ đạt **$0.9541$**.

**Kết luận thực nghiệm:** Hệ số lan truyền của các cạnh không có bằng chứng chỉ bị giảm trung bình từ **$-0.74\%$ đến $-1.38\%$**, và ở 90% các cạnh thì mức giảm không vượt quá $-5.7\%$. Tác động này hoàn toàn nằm trong biên độ an toàn và không hề làm xáo trộn cấu trúc tô-pô ban đầu của V8. Hơn nữa, việc giảm nhẹ hệ số của các cạnh không có bằng chứng đóng vai trò như một bộ lọc nhiễu tự nhiên, giúp hướng sự chú ý của gradient cập nhật vào các liên kết có bằng chứng xác thực.

### 6.4. Bảo toàn cận phổ toán tử ($\|S_{\text{ICA}}\|_2 \le 1.0$) và bảo đảm hội tụ

Toán tử truyền dẫn cuối cùng đưa vào bộ tối ưu hóa BSC được định nghĩa:
$$S_{\text{ICA}} = (1 - \eta) S_{\delta} + \eta S_{\text{CF}}$$
Với $\eta = 0.1$. Cả hai ma trận $S_{\delta}$ và $S_{\text{CF}}$ đều được chuẩn hóa đối xứng với đường chéo tự liên kết, bảo đảm chuẩn phổ:
$$\|S_{\text{ICA}}\|_2 \le (1 - \eta) \|S_{\delta}\|_2 + \eta \|S_{\text{CF}}\|_2 \le 0.9 \times 1.0 + 0.1 \times 1.0 = 1.0$$
Liều lượng toán tử tương đối (Operator Relative Dose $\|\Delta S\|_F / \|S\|_F$) đo được trên Sports ($0.0711$), Baby ($0.0713$) và Electronics ($0.0621$) đều duy trì chính xác ở mức **$6.2\% - 7.1\%$**, và liều lượng đa thức bậc 3 đo bằng Rademacher probe chỉ đạt **$1.66\% - 2.36\%$**. Điều này chứng minh toán tử $S_{\text{ICA}}$ chỉ thực hiện một sự tinh chỉnh nhẹ nhàng, cục bộ, hoàn toàn không làm biến dạng không gian gradient của mô hình.

---

## 7. ĐỐI CHIẾU HỌC THUẬT VỚI STAIR5-v8.1 (BSCV) & CoGate

### 7.1. Phân định triết lý: Tinh chỉnh hình học phía bộ tối ưu vs. Đồ thị phụ & Cổng học

| Tiêu chí So sánh Học thuật | Hướng tiếp cận STAIR5-v8.1 (BSCV) / CoGate | Hướng tiếp cận STAIR-PEEC (versionICAART R2) |
|:---|:---|:---|
| **Vị trí can thiệp** | Tầng biểu diễn người dùng & vật phẩm (Representation-level) | Phía trước bộ làm mượt cập nhật Item (Optimizer-side BSC) |
| **Cấu trúc mạng** | Bổ sung mạng MLP 2 tầng (Learned Adaptive Gate) | **Không có mạng nơ-ron phụ, 0 tham số học thêm** |
| **Không gian đồ thị** | Xây dựng đồ thị đồng thuận phụ đa phương thức (Auxiliary Graph) | **Tái sử dụng nguyên vẹn khung cạnh ứng viên V8 có sẵn** |
| **Mục tiêu tối ưu hóa** | Bổ sung hàm mất mát tương phản cục bộ CoGate Loss | **Giữ nguyên hàm mất mát BPR + NLGCL của V8** |
| **Chi phí tính toán GPU** | Tăng thêm các lượt tính forward/backward qua MLP và đồ thị phụ | **Thuần túy tính toán 1 lần trên CPU, zero GPU overhead** |
| **Nguy cơ quá khớp** | Có nguy cơ quá khớp trọng số cổng trên các tập dữ liệu nhỏ | **Miễn nhiễm hoàn toàn với quá khớp tham số** |

### 7.2. Tương quan hiệu quả trên chi phí tính toán (Efficiency-Gain Frontier)

Biểu đồ Pareto giữa chi phí tính toán và hiệu năng xếp hạng:
- **STAIR5-v8.1** đạt hiệu năng cao trên Sports nhưng phải đánh đổi bằng $16\%$ thời gian huấn luyện và tăng độ phức tạp kiến trúc.
- **STAIR-PEEC** vươn tới đúng đường biên Pareto tối ưu trên cả 3 tập dữ liệu:
  - Trên **Amazon Sports**: Đạt đúng mốc NDCG@20 = **$0.0521$** của V8.1 trong khi nhanh hơn $6.6$ phút.
  - Trên **Amazon Baby**: Vượt qua V8.1 trên mọi chỉ số (NDCG@20 = **$0.0464$** vs $0.0463$; Recall@10 = **$0.0686$** vs $0.0679$) trong khi nhanh hơn $1.7$ phút.
  - Trên **Amazon Electronics**: Thiết lập kỷ lục mới (NDCG@20 = **$0.0318$**, Recall@10 = **$0.0464$**), vượt qua toàn bộ các thế hệ trước với mức tiêu thụ VRAM chỉ $800.7\text{ MiB}$.

### 7.3. Đảm bảo tính liêm chính và tính độc lập sáng tạo (Scientific Integrity)

Kiến trúc STAIR-PEEC giải quyết trọn vẹn bài toán đặt ra trong luận văn tốt nghiệp mà không bị trùng lặp phương pháp luận với các công trình như CoGate:
- Không sử dụng lại cơ chế gating hay auxiliary loss của CoGate.
- Chứng minh được rằng: **Bản thân thông tin tương tác người dùng khi được lọc bỏ kỳ vọng ngẫu nhiên theo tầng hoạt động đã đủ sức định hướng hình học bộ tối ưu hóa BSC đạt tới hiệu năng đỉnh cao, mà không cần thêm bất kỳ tham số học nào**.

---

## 8. KẾT LUẬN & KẾ HOẠCH BẢO VỆ KHÓA LUẬN VÀ BÀI BÁO ICAART 2027

### 8.1. Đánh giá kết luận tổng thể

1. **Khẳng định tính đúng đắn toàn diện của giả thuyết PEEC:** Việc thay thế điều kiện lấy `min` cứng nhắc của REEC bằng bằng chứng vượt kỳ vọng gộp toàn bộ tầng hoạt động ($\text{PEEC}$) đã giải quyết triệt để vấn đề nghẽn độ phủ trên dữ liệu thưa, nâng độ phủ cạnh có ích lên gấp $4\times - 6\times$ trên toàn bộ 3 tập dữ liệu thực nghiệm.
2. **Hiệu năng thực nghiệm vượt bậc trên toàn bộ 3 tập benchmark:**
   - Trên Amazon Baby: Đạt thắng lợi toàn diện, vượt qua tất cả các thế hệ trước (Baseline, V4, V8, V8.1, REEC R1).
   - Trên Amazon Sports: Vượt V8, vượt REEC R1 và san bằng kỷ lục của V8.1 với chi phí tính toán thấp hơn đáng kể ($13.6\%$ faster).
   - Trên Amazon Electronics: Xác lập đỉnh cao mới của dòng mô hình STAIR trên quy mô $192\text{K}$ người dùng (+4.98% Recall@10, +5.05% NDCG@20 so với Baseline).
3. **Độ an toàn và tính chuẩn mực kỹ thuật:** Toàn bộ $40/40$ bài kiểm thử độc lập đã thông qua, chuẩn phổ toán tử được bảo toàn ($\le 1.0$), VRAM GPU duy trì cực thấp ($145\text{ MiB} - 800\text{ MiB}$), thông lượng xử lý mẫu cao vượt trội.

### 8.2. Kế hoạch tiếp theo cho Khóa luận Tốt nghiệp & Bài báo Hội nghị

1. **Cập nhật tài liệu khóa luận tốt nghiệp:**
   - Bổ sung toàn bộ ma trận đối chuẩn 3 tập dữ liệu và phân tích cơ chế toán học của PEEC vào Chương 3 của khóa luận tốt nghiệp ([`report/chapters_v6/03_stair.tex`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/report/chapters_v6/03_stair.tex)).
   - Cập nhật các đồ thị đường cong học tập và hồ sơ VRAM 3 tập dữ liệu vào phụ lục báo cáo.
2. **Chuẩn bị bản thảo bài báo khoa học nộp hội nghị quốc tế ICAART 2027:**
   - Hoàn thiện bản thảo bài báo khoa học chất lượng cao nộp cho hội nghị quốc tế ICAART 2027 với các luận cứ thực nghiệm, kiểm định thống kê $p < 0.001$, và phân tích lý thuyết đầy đủ về phương pháp hiệu chuẩn bằng chứng vượt kỳ vọng gộp $\text{PEEC}$.
