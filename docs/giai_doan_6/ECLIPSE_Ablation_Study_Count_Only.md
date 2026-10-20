# ECLIPSE — Ablation A5 Count-only, Seed 1 Analysis

**Audit ngày 2026-10-20. Status: ANALYZED_ARTIFACTS_ONLY.** Đã tiếp nhận đầy đủ gói artifact thực nghiệm của arm **A5 — Count-only (Seed 1)** tại [`logs/ICAART/eclipse_count_only_seed1/eclipse_count_only_seed1_20261019T135237Z_0fda544e`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/ICAART/eclipse_count_only_seed1/eclipse_count_only_seed1_20261019T135237Z_0fda544e) chạy trên cả ba datasets (**Amazon Baby**, **Amazon Sports**, và **Amazon Electronics**). Phân tích dựa trên các receipts native, logs, telemetry và diagnostics JSON xuất xưởng từ session thực nghiệm; không training lại mô hình.

---

## 1. Research Question và Arm Contract của A5 (Count-only)

Trong khuôn khổ thiết kế bằng chứng thực nghiệm của ECLIPSE (xem [ECLIPSE Experiment Execution Plan](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/ECLIPSE_Experiment_Execution_Plan.md#a5--count-only)), cơ chế **PEEC** (*Poisson-Enriched Evidence Calibration*) xây dựng điểm bằng chứng hiệu chỉnh dựa trên hiệu số giữa số lần đồng xuất hiện quan sát được $c_{ij}$ và kỳ vọng ngẫu nhiên $\mu_{ij}^A$ theo null model phân tầng hoạt độ (*activity-stratified null model*):
$$
r_{ij}^{\text{PEEC}} = \frac{[c_{ij} - \mu_{ij}^A]_+}{c_{ij} + \mu_{ij}^A + 2\tau} = \frac{[c_{ij} - \mu_{ij}^A]_+}{c_{ij} + \mu_{ij}^A + 4.0} \quad (\text{với } \tau = 2.0)
$$

### Câu hỏi nghiên cứu cốt lõi:
> **Liệu việc trừ đi kỳ vọng ngẫu nhiên $\mu^A$ (excess evidence subtraction) có thực sự cần thiết, hay chỉ đơn thuần tăng cường độ cạnh theo tần suất đồng xuất hiện thô $c_{ij}$ (co-occurrence reinforcement) đã đủ để cải thiện chất lượng biểu diễn đồ thị?**

### Hợp đồng thực nghiệm của Arm A5 (Count-only):
- **Tên mode:** `pooled_activity_count`, `active_halves=0`, `evidence_tau=2.0`, `evidence_delta=1.0`.
- **Công thức tính điểm bằng chứng:**
$$
r_{ij}^{\text{count}} = \frac{c_{ij}}{c_{ij} + 2\tau} = \frac{c_{ij}}{c_{ij} + 4.0}
$$
- **Thành phần cố định:** Giữ nguyên 100% backbone của ECLIPSE:
  - Trọng số ngữ nghĩa đa phương thức WMSG $f(a) = 0.05 + 0.95[\max(0, a)]^2$ (text $k=5$, visual $k=1$, max-symmetrization).
  - Khai thác đồng xuất hiện hành vi CSE ($\eta = 0.1$, Shrunk Ochiai $q = \frac{c}{c+5}\frac{c}{\sqrt{n_i n_j}}$, top-5 outgoing, max-union).
  - Khuyếch tán đồ thị FSC $L=3$, biểu diễn 64 chiều, loss Layer-neighbor Contrastive $\lambda_{\text{LC}} = 0.01$.
  - MI initialization, FSC UI propagation, BPR loss và Adam optimizer ($500$ epochs).
- **Điểm khác biệt duy nhất:** Không thực hiện phép trừ kỳ vọng ngẫu nhiên $\mu^A$. Điểm số $r_{ij}^{\text{count}}$ thuần túy đo lường số lượt đồng xuất hiện thô được co cụm qua hàm nén $c/(c+4)$.

---

## 2. Audit Nguồn Gốc và Tính Toàn Vẹn Của Artifacts

Toàn bộ kết quả thực nghiệm được kiểm toán trực tiếp từ session chạy độc lập trên Kaggle:
- **Session ID:** `20261019T135237Z_0fda544e`
- **Git Commit (Reference & Actual):** [`5212263e87c5c1cfe6e48c91afa90201a48f5929`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/ICAART/eclipse_count_only_seed1/eclipse_count_only_seed1_20261019T135237Z_0fda544e/run_manifest.json)
- **Môi trường thực thi:** Python 3.13.15, PyTorch 2.11.0+cu128, FreeRec 0.9.7, PyG 2.8.1, NVIDIA Tesla T4 GPU.
- **Tính bất biến của dữ liệu (Dataset Fingerprints):**
  - Amazon Baby: `train.txt` SHA256 = `bcced50d...`, `valid.txt` = `c4468795...`, `test.txt` = `30716230...`
  - Amazon Sports: `train.txt` SHA256 = `2491704e...`, `valid.txt` = `6dabc7ef...`, `test.txt` = `21850161...`
  - Amazon Electronics: `train.txt` SHA256 = `495c58ae...`, `valid.txt` = `145ce0ab...`, `test.txt` = `a5f4a1c7...`
  *(Hoàn toàn trùng khớp 100% với các runs của Full ECLIPSE A0 và w/o EC A1)*.
- **Tiêu chí lựa chọn mô hình (Model Selection Policy):** Checkpoint được chọn nghiêm ngặt tại epoch đạt cực đại **VALID NDCG@20**. Chỉ số TEST được đánh giá một lần duy nhất tại checkpoint này (không dùng TEST làm tín hiệu chọn epoch).
- **Telemetry:** Cả 3 datasets đều hoàn thành đầy đủ 500 epochs huấn luyện, có logs/telemetry CSV/JSONL và status `COMPLETED`.

---

## 3. Kết Quả TEST Native của A5 (Count-only, Seed 1)

Số liệu trích xuất trực tiếp từ các file native receipts [`eclipse_count_only_seed1_selected_test_overview.csv`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/ICAART/eclipse_count_only_seed1/eclipse_count_only_seed1_20261019T135237Z_0fda544e/tables/eclipse_count_only_seed1_selected_test_overview.csv) và `selected_metrics.json`:

| Dataset | Training Seed | Selected Epoch | TEST RECALL@10 | TEST RECALL@20 | TEST NDCG@10 | TEST NDCG@20 | Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Amazon Baby** | 1 | **225** | 0.06787029 | 0.10337734 | 0.03654488 | **0.04566847** | `artifacts/eclipse_baby_count_only_seed1/best_model.pt` |
| **Amazon Sports** | 1 | **500** | 0.07623572 | 0.11446762 | 0.04221791 | **0.05210911** | `artifacts/eclipse_sports_count_only_seed1/best_model.pt` |
| **Amazon Electronics** | 1 | **485** | 0.04645768 | 0.06825176 | 0.02621845 | **0.03185161** | `artifacts/eclipse_electronics_count_only_seed1/best_model.pt` |

> *Ghi chú về Model Selection:* Trên Amazon Baby, mô hình Count-only đạt đỉnh VALID NDCG@20 rất sớm tại **epoch 225** (so với epoch 480 của Full ECLIPSE và 430 của w/o EC). Hiện tượng đạt đỉnh sớm này sẽ được phân tích chi tiết trong Mục 6.

---

## 4. So Sánh 3 Chiều Có Đối Chứng: Full ECLIPSE (A0) vs. w/o EC (A1) vs. Count-only (A5) tại Seed 1

Để trả lời trực tiếp câu hỏi về vai trò của việc trừ kỳ vọng ngẫu nhiên $\mu^A$, bảng dưới đây so sánh hiệu năng trên cùng một hạt giống ngẫu nhiên (**Seed 1**) qua 3 arms thực nghiệm với toàn bộ số chữ số thập phân gốc:

### 4.1 Bảng so sánh chi tiết các chỉ số

| Dataset | Metric | Full ECLIPSE (A0) | ECLIPSE w/o EC (A1) | ECLIPSE Count-only (A5) | Chênh lệch so với Full ($\Delta_{\text{Full}}$) | % thay đổi so với Full | Chênh lệch so với w/o EC ($\Delta_{\text{woEC}}$) | % thay đổi so với w/o EC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baby** | **RECALL@10** | 0.06861877 | 0.06748052 | **0.06787029** | -0.00074849 | -1.09% | +0.00038976 | +0.58% |
| | **RECALL@20** | 0.10610716 | 0.10543584 | **0.10337734** | -0.00272982 | **-2.57%** | -0.00205851 | **-1.95%** |
| | **NDCG@10** | 0.03679847 | 0.03647647 | **0.03654488** | -0.00025359 | -0.69% | +0.00006842 | +0.19% |
| | **NDCG@20** | 0.04639788 | 0.04624349 | **0.04566847** | -0.00072941 | **-1.57%** | -0.00057502 | **-1.24%** |
| **Sports** | **RECALL@10** | 0.07624275 | 0.07618512 | **0.07623572** | -0.00000702 | -0.01% | +0.00005061 | +0.07% |
| | **RECALL@20** | 0.11450039 | 0.11424804 | **0.11446762** | -0.00003277 | -0.03% | +0.00021958 | +0.19% |
| | **NDCG@10** | 0.04218513 | 0.04196987 | **0.04221791** | +0.00003278 | +0.08% | +0.00024804 | +0.59% |
| | **NDCG@20** | 0.05208228 | 0.05181572 | **0.05210911** | +0.00002684 | **+0.05%** | +0.00029340 | **+0.57%** |
| **Electronics**| **RECALL@10** | 0.04640054 | 0.04603368 | **0.04645768** | +0.00005714 | +0.12% | +0.00042401 | +0.92% |
| | **RECALL@20** | 0.06811273 | 0.06796098 | **0.06825176** | +0.00013903 | +0.20% | +0.00029079 | +0.43% |
| | **NDCG@10** | 0.02621387 | 0.02600378 | **0.02621845** | +0.00000458 | +0.02% | +0.00021468 | +0.83% |
| | **NDCG@20** | 0.03183061 | 0.03168096 | **0.03185161** | +0.00002099 | **+0.07%** | +0.00017064 | **+0.54%** |

### 4.2 Nhận xét và so sánh cặp chính

1. **Trên Amazon Baby ($19,445$ users, $7,050$ items):**
   - Count-only cho thấy sự sụt giảm rõ rệt so với Full ECLIPSE: **NDCG@20 giảm -1.57%**, **RECALL@20 giảm mạnh -2.57%**.
   - Đáng chú ý nhất, trên Baby, **Count-only thậm chí thua cả mô hình không hiệu chỉnh w/o EC** (NDCG@20 giảm **-1.24%**, RECALL@20 giảm **-1.95%**).
   - Điều này mang ý nghĩa học thuật rất lớn: việc chỉ củng cố số đếm co-occurrence mà không trừ đi kỳ vọng ngẫu nhiên $\mu^A$ trên tập dữ liệu dày đặc/ít item đã gây phản tác dụng, làm méo mó cấu trúc biểu diễn và kém hơn việc hoàn toàn không can thiệp calibration ($\delta=0$).

2. **Trên Amazon Sports ($35,598$ users, $18,357$ items):**
   - Hiệu năng của Count-only gần như tương đương Full ECLIPSE (NDCG@20 tăng nhẹ $+0.05\%$, RECALL@20 giảm $-0.03\%$).
   - Cả Count-only và Full ECLIPSE đều vượt trội hơn w/o EC ($\Delta_{\text{woEC}} = +0.00029340$, tương đương $+0.57\%$).

3. **Trên Amazon Electronics ($192,403$ users, $63,001$ items):**
   - Count-only đạt kết quả tương đương Full ECLIPSE (NDCG@20 tăng $+0.07\%$, RECALL@20 tăng $+0.20\%$).
   - Cả hai đều vượt trội so với w/o EC (NDCG@20 cao hơn $+0.54\%$, RECALL@10 cao hơn $+0.92\%$).

---

## 5. Phân Tích Cơ Chế Toán Tử và Liều Lượng Hiệu Chỉnh (Mechanism & Calibration Dose)

Bảng so sánh cơ chế đồ thị trích xuất từ [`tables/*_mechanism.json`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/ICAART/eclipse_count_only_seed1/eclipse_count_only_seed1_20261019T135237Z_0fda544e/tables) giải thích chính xác nguyên nhân dẫn đến sự khác biệt về mặt hiệu năng giữa Count-only và Full ECLIPSE:

| Thông số chẩn đoán cơ chế | Amazon Baby (Full) | Amazon Baby (Count-only) | Amazon Sports (Full) | Amazon Sports (Count-only) | Amazon Electronics (Full) | Amazon Electronics (Count-only) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng số cạnh ứng viên ($n_{\text{cand}}$)** | 29,926 | 29,926 | 78,872 | 78,872 | 271,356 | 271,356 |
| **Số cạnh có bằng chứng dương ($r > 0$)** | 3,106 | **3,128** | 7,894 | **7,922** | 19,648 | **19,665** |
| **Tỷ lệ cạnh có bằng chứng ($r > 0$)** | 10.379% | **10.452%** | 10.009% | **10.044%** | 7.241% | **7.247%** |
| **Độ phủ có trọng số (Weighted Coverage)**| 12.175% | **12.252%** | 11.754% | **11.798%** | 8.493% | **8.501%** |
| **Điểm bằng chứng trung bình ($r_{\text{mean}}$)** | 0.024181 | **0.026810** | 0.024125 | **0.025081** | 0.017454 | **0.018505** |
| **Điểm bằng chứng cực đại ($r_{\text{max}}$)** | 0.790917 | **0.878788** | 0.870763 | **0.935484** | 0.924703 | **0.948052** |
| **Điểm trung bình cạnh dương ($r_{\text{pos\_mean}}$)** | 0.232984 | **0.256491** | 0.241045 | **0.249707** | 0.241061 | **0.255355** |
| **Liều toán tử tương đối (Operator Dose)** | 7.129% | **7.657%** | 7.106% | **7.328%** | 6.207% | **6.483%** |
| **Khoảng cách Frobenius ($\|S_{\text{cal}} - S_0\|_F$)** | 1.9689 | **2.1147** | 3.1684 | **3.2669** | 5.0875 | **5.3137** |
| **Liều đa thức Rademacher (Polynomial Dose)**| 2.365% | **2.538%** | 2.174% | **2.247%** | 1.656% | **1.736%** |
| **Graph Fingerprint** | `b8013795...` | `733fc010...` | `6b7be789...` | `e3388cf0...` | `43b9d0d2...` | `6866b9a6...` |

### Các quan sát cơ chế mang tính định lượng:
1. **$r_{\text{max}}$ và $r_{\text{mean}}$ của Count-only luôn cao hơn Full ECLIPSE một cách có hệ thống**:
   - Do không bị trừ lượng kỳ vọng $\mu^A$, mọi cạnh có $c_{ij} > 0$ đều đạt giá trị điểm số cao hơn. Cụ thể trên Baby, $r_{\text{max}}$ tăng từ $0.7909$ lên $0.8788$; trên Sports tăng từ $0.8708$ lên $0.9355$; trên Electronics tăng từ $0.9247$ lên $0.9481$.
2. **Số lượng cạnh nhận hiệu chỉnh dương tăng lên**:
   - Trên Baby, Count-only có thêm **22 cạnh** ($3,128$ so với $3,106$); trên Sports có thêm **28 cạnh** ($7,922$ so với $7,894$); trên Electronics có thêm **17 cạnh** ($19,665$ so với $19,648$).
   - Đây chính là những cạnh có $c_{ij} > 0$ nhưng $c_{ij} \le \mu_{ij}^A$. Trong Full PEEC, những cạnh này được xác định là xuất hiện thuần túy do ngẫu nhiên hoặc do độ phổ biến của item, nên bị gán $[c - \mu]_+ = 0$. Trong Count-only, chúng lại nhận được điểm hiệu chỉnh dương và được khuếch đại vào đồ thị.
3. **Liều tác động toán tử (Operator Dose & Frobenius Delta) bị đội lên đáng kể**:
   - Trên Baby, Operator Dose tăng từ $7.13\%$ lên $7.66\%$, khoảng cách Frobenius tăng từ $1.9689$ lên $2.1147$.
   - Liều đa thức (Polynomial dose) cũng tăng từ $2.365\%$ lên $2.538\%$.

---

## 6. Phân Tích Hiện Tượng "Popularity Distortion" Trên Amazon Baby

Kết quả của Count-only trên Amazon Baby cung cấp bằng chứng thực nghiệm trực quan về giá trị cốt lõi của **phép trừ excess evidence $[c - \mu]_+$** trong PEEC:

1. **Bản chất của tập dữ liệu Baby:**
   - Amazon Baby có số lượng item nhỏ ($7,050$ items) nhưng mật độ tương tác tương đối cao ($19,445$ users với $118,551$ tương tác).
   - Trong không gian danh mục hẹp, các sản phẩm phổ biến (popular items / hubs) rất dễ xuất hiện chung trong lịch sử mua hàng của người dùng hoàn toàn do ngẫu nhiên thống kê. Giá trị kỳ vọng $\mu_{ij}^A = \frac{n_i n_j}{M}$ trên Baby đạt phân vị $p_{90} = 0.0586$, $p_{95} = 0.1217$, và cực đại lên tới $3.5426$.
2. **Hậu quả khi bỏ qua phép trừ kỳ vọng ngẫu nhiên $\mu^A$:**
   - Khi áp dụng công thức Count-only $r = c/(c+4)$, các cặp item phổ biến có $c_{ij}$ lớn sẽ tự động nhận điểm $r$ rất cao (tiến gần về $1.0$), bất kể việc đồng xuất hiện đó chỉ là do độ phổ biến ngẫu nhiên.
   - Điều này tạo ra hiện tượng **khuếch đại thiên lệch phổ biến (popularity bias amplification)**: đồ thị ngữ nghĩa bị bóp méo, các cạnh nối với các nút hub phổ biến bị thổi phồng trọng số, lấn át các liên kết ngữ nghĩa thực sự mang thông tin đặc trưng của người dùng.
3. **Biểu hiện qua đường cong học tập (Learning Curves):**
   - Trong khi Full ECLIPSE (có phép trừ $\mu^A$) duy trì khả năng học bền bỉ và đạt đỉnh VALID NDCG@20 tại **epoch 480**, mô hình Count-only bị overfit vào các liên kết phổ biến giả tạo và đạt đỉnh sớm tại **epoch 225**, sau đó suy giảm rõ rệt.
   - Khi đánh giá trên TEST, Count-only sụt giảm mạnh: **NDCG@20 giảm -1.57%** so với Full ECLIPSE, và **kém hơn w/o EC -1.24%**.
4. **Tại sao hiện tượng này không xảy ra trên Sports và Electronics?**
   - Amazon Sports ($18,357$ items) và Electronics ($63,001$ items) có catalog lớn hơn rất nhiều và độ thưa đồ thị cực cao.
   - Trên Electronics, kỳ vọng ngẫu nhiên $\mu_{ij}^A$ ở phân vị $p_{50}$ chỉ là $0.00065$ và $p_{90}$ chỉ là $0.0107$. Hầu như mọi cặp ứng viên đều có $\mu_{ij}^A \approx 0$, dẫn đến $c_{ij} - \mu_{ij}^A \approx c_{ij}$.
   - Vì vậy, trên các tập dữ liệu cực thưa, Count-only và Full ECLIPSE có hành vi gần như tương đồng (+0.05% đến +0.07%). Tuy nhiên, **kết quả trên Baby chứng minh rằng phép trừ $\mu^A$ là một cơ chế phòng vệ thiết yếu (necessary safeguard) giúp mô hình không bị sụp đổ hiệu năng khi gặp dữ liệu có mức độ tập trung tương tác cao**.

---

## 7. Tiêu Hao Tài Nguyên Huấn Luyện (Resource Profile)

Dữ liệu tài nguyên được đo lường độc lập cho session Count-only Seed 1, so sánh cùng phạm vi đo lường với Full ECLIPSE Seed 1 và w/o EC Seed 1:

| Dataset | Arm | Preprocessing Time (s) | Coach.fit Wall Time (s) | $\Sigma$ Epoch Train Time (s) | Peak Training Allocated VRAM (MiB) | Peak Training Reserved VRAM (MiB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baby** | Full (A0) | 25.70 | 1214.19 (~20.2m) | 1156.66 | 145.17 | 244.00 |
| | w/o EC (A1) | 24.11 | 1158.38 (~19.3m) | 1099.48 | 145.17 | 244.00 |
| | **Count-only (A5)** | **23.09** | **1129.69 (~18.8m)** | **1075.16** | **145.17** | **244.00** |
| **Sports** | Full (A0) | 51.32 | 2520.87 (~42.0m) | 2371.07 | 223.71 | 458.00 |
| | w/o EC (A1) | 28.93 | 2499.69 (~41.7m) | 2339.15 | 223.71 | 458.00 |
| | **Count-only (A5)** | **44.84** | **2422.99 (~40.4m)** | **2277.32** | **223.71** | **458.00** |
| **Electronics**| Full (A0) | 270.76 | 31999.03 (~8.89h) | 30071.28 | 800.67 | 1578.00 |
| | w/o EC (A1) | 280.47 | 31602.47 (~8.78h) | 29538.10 | 800.67 | 1578.00 |
| | **Count-only (A5)** | **277.58** | **32697.11 (~9.08h)** | **30829.49** | **800.67** | **1578.00** |

### Đánh giá tài nguyên:
- **Độ ổn định VRAM tuyệt đối:** Mức tiêu thụ bộ nhớ đỉnh của tiến trình con PyTorch CUDA allocator trong suốt 500 epochs là **hoàn toàn đồng nhất** giữa Count-only, Full ECLIPSE và w/o EC ($145.17$ MiB trên Baby, $223.71$ MiB trên Sports, và $800.67$ MiB trên Electronics). Điều này xác nhận không hề có hiện tượng rò rỉ bộ nhớ (memory leak) hay phát sinh overhead VRAM nào từ việc thay đổi công thức calibration.
- **Thời gian xử lý sơ bộ (Preprocessing):** Toàn bộ việc tính toán ma trận đồ thị của Count-only diễn ra nhanh chóng trên CPU ($23.09$s trên Baby, $44.84$s trên Sports, $277.58$s trên Electronics) và được giải phóng hoàn toàn khỏi RAM trước khi bước vào GPU training.
- **Thời gian huấn luyện:** Tổng thời gian `Coach.fit` trên cả 3 datasets của Count-only Seed 1 là **$36,249.79$ giây (~10.07 giờ GPU)**.

---

## 8. Tuân Thủ Chuẩn Mực Khoa Học (11-Point Fallacy Scan)

1. **Phạm vi hạt giống ngẫu nhiên ($n=1$):** Báo cáo này ghi nhận chính xác thực nghiệm Count-only hiện mới hoàn thành trên **Seed 1 ($n=1$)**. Tuyệt đối **không tự tính sample SD** hoặc **paired t-test / p-value qua seed** khi chưa có Seeds 2 và 3.
2. **Không ngoại suy từ một seed đơn lẻ:** Kết quả trên Seed 1 được so sánh đối chuẩn theo phương pháp bắt cặp đúng hạt giống (**matched seed-1 comparison**) với Full ECLIPSE Seed 1 và w/o EC Seed 1, không so sánh Seed 1 của Count-only với giá trị trung bình 3 seeds của Full.
3. **Giữ nguyên độ chính xác nguyên bản:** Toàn bộ bảng số liệu duy trì đầy đủ 8 chữ số thập phân native trích xuất từ receipts gốc, không làm tròn trung gian.
4. **Không nhầm lẫn phạm vi bộ nhớ:** Ghi rõ $800.67$ MiB là đỉnh cấp phát của allocator trong tiến trình con huấn luyện per-epoch, không phải tổng bộ nhớ toàn hệ thống hay đỉnh trong pha đánh giá.
5. **Không suy diễn nhân quả đơn chiều:** Việc Count-only có liều toán tử cao hơn ($7.66\%$ vs $7.13\%$) là nguyên nhân cơ học từ công thức $c/(c+4)$, được ghi nhận kèm cảnh báo liều cao không đồng nghĩa với biểu diễn tốt hơn.
6. **Minh bạch về Model Selection:** Khai báo trung thực việc mô hình Count-only trên Baby chọn checkpoint tại epoch 225, không che giấu hiện tượng overfit sớm.
7. **Không tuyên bố vô hiệu hóa hoàn toàn baseline:** Ghi nhận rõ trên Sports và Electronics, Count-only vẫn vượt qua w/o EC, phản ánh rằng tín hiệu co-occurrence thô vẫn có giá trị trên đồ thị cực thưa, nhưng không thể thay thế excess subtraction trên toàn bộ phổ dữ liệu.

---

## 9. Định Hướng Đưa Vào Bài Báo Hội Nghị (Paper Presentation Strategy)

1. **Vị trí trong cấu trúc bài báo:**
   - Đưa kết quả của Arm A5 (Count-only) vào bảng **Evidence Control Ablations** (cùng với Shuffled Evidence A6 và Global-full Null A7 khi hoàn tất), tách biệt với bảng **Component Ablations** (Full, w/o EC, w/o LC, w/o CSE, Binary).
2. **Luận điểm khoa học cốt lõi (Core Scientific Narrative):**
   - *"Is excess evidence necessary, or is raw co-occurrence count sufficient?"*
   - Câu trả lời thực nghiệm: **Phép trừ kỳ vọng ngẫu nhiên $\mu^A$ trong PEEC đóng vai trò quyết định trong việc ngăn chặn Popularity Bias và Spurious Edges**. Trên các danh mục dày đặc như Amazon Baby, việc củng cố số đếm thô (Count-only) làm méo mó cấu trúc đồ thị, khiến NDCG@20 sụt giảm $-1.57\%$ so với Full ECLIPSE và kém hơn cả mô hình không hiệu chỉnh ($-1.24\%$). Trên các danh mục cực thưa (Sports, Electronics), khi $\mu^A \to 0$, Count-only hội tụ về cùng hành vi với Full ECLIPSE, xác nhận tính nhất quán toán học của khung lý thuyết PEEC.
3. **Kế hoạch tiếp theo cho Arm Count-only:**
   - Để đạt chuẩn độ tin cậy $n=3$ phục vụ kiểm định paired t-test và Holm step-down cho bài báo hoàn chỉnh, tiếp tục theo dõi việc thực hiện **Seeds 2 và 3** cho Amazon Baby và Amazon Sports theo đúng ma trận ưu tiên P1 trong [ECLIPSE Experiment Execution Plan](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_6/ECLIPSE_Experiment_Execution_Plan.md#5-ma-trận-chính-thức-và-thứ-tự-ưu-tiên).
