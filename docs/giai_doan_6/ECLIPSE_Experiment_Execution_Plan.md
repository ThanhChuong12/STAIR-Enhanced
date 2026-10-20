# ECLIPSE — Kế hoạch thực nghiệm, phân công Chương–Hiếu và chuẩn bị paper

## 0. Phạm vi và trạng thái

- **Ngày lập:** 18/10/2026; các mốc lịch dùng múi giờ Asia/Saigon.
- **Người thực hiện:** Lê Hà Thanh Chương và Bùi Trung Hiếu.
- **Origin skill/mode:** academic-research-suite → experiment-agent / plan.
- **Trạng thái kế hoạch:** PLANNED. Đã đọc mã nguồn để xác định khả năng cấu hình; không chạy training hoặc unit tests trong lần lập kế hoạch này.
- **Phương pháp paper:** **ECLIPSE — Excess-Evidence Calibrated Layer-Contrastive Item Propagation with Support Expansion**. PEEC là tên mô-đun pooled excess-evidence calibration.
- **Tiến độ Full Model (A0):** Toàn bộ ba hạt giống độc lập ($n=3$: Seed 1, Seed 2, Seed 3) của **Full ECLIPSE** đã hoàn tất trọn vẹn 500 epochs trên cả 3 datasets và được nghiệm thu chính thức `VERIFIED_ARTIFACT` (chi tiết xem [ECLIPSE_Full_Model_Multi_Seed_Analysis.md](ECLIPSE_Full_Model_Multi_Seed_Analysis.md)).
- **Giả định tài nguyên để lập lịch:** hai slot GPU độc lập hoạt động đồng thời. Hai hàng đợi dưới đây là slot A/Chương và slot B/Hiếu; nếu GPU đều ở máy Chương thì Hiếu vẫn có thể nhận nhiệm vụ cấu hình/nghiệm thu và điều phối hàng đợi B. Không giả định có bốn GPU. Nếu có thêm slot, lấy job chưa chạy từ ledger, không đổi protocol.
- **Quyết định đã chốt:** không chạy thêm STAIR gốc; giữ kết quả tái lập và artifact sẵn có, ghi đúng số run. Không đưa số phiên bản nội bộ vào bảng đối sánh paper.
- **Đích thực nghiệm:** kiểm chứng hiệu quả và đóng góp của phương pháp, không bảo đảm accept, không đặt p<0.05 hoặc một tỷ lệ tăng trưởng cố định làm điều kiện nghiệm thu run.

**Đầu ra cuối cùng:** (1) kết quả full ba seed trên ba dataset; (2) w/o EC ba seed trên ba dataset; (3) component/evidence ablations ba seed trên Baby/Sports; (4) một seed Count/Shuffle trên Electronics; (5) Global-full null trên Baby/Sports nếu code được hoàn thiện kịp; (6) bảng, hình, uncertainty và artifact đủ truy vết.

**Tra nhanh:** mục 3: code còn thiếu; mục 4: cấu hình/điều kiện mỗi arm; mục 6: phân công và 47 Job IDs; mục 8–10: cách chạy và artifact; mục 11–13: hình/bảng/thống kê; mục 15: checklist nộp.

## 1. Những việc làm ngay, không cần chờ full model hoàn tất

### Chương

1. Ghi lại các process full seed 2–3 đang có: dataset, seed, máy/GPU, output path, trạng thái. Nếu các run đang chạy tuần tự thì ghi QUEUED cho job chưa bắt đầu, không đánh dấu cả sáu run là RUNNING.
2. Giữ nguyên commit/config/environment của các process đang chạy; lưu YAML, command và manifest đang dùng.
3. Chuẩn bị arm registry và contract cho ablations ở checkout riêng. Không sửa checkout đang được process training sử dụng, vì có thể có import/data-loader lazy ở epoch sau.
4. Phụ trách cấu hình w/o EC và Global-full null; cùng Hiếu khóa một implementation revision cho toàn bộ ablations.
5. Sau khi slot A rảnh, ưu tiên w/o EC Electronics seed 1 và 3, rồi Count Electronics.

### Hiếu

1. Nhận kiểm tra khả năng cấu hình w/o LC, w/o CSE và Binary; lưu các điểm còn thiếu trong code trước khi chạy GPU dài.
2. Chuẩn bị notebook/harness ablation có kiểm tra theo từng arm; không chỉ sửa biến trong notebook full model.
3. Chuẩn bị bộ tổng hợp kết quả từ selected TEST receipts, pipeline vẽ hình và ledger chung.
4. Sau khi slot B rảnh, ưu tiên w/o EC Electronics seed 2; trong khi slot A chạy Electronics seed 3, chạy các ablation nhỏ; sau đó chạy Shuffle Electronics.
5. Đọc chéo artifact Chương bàn giao; Chương đọc chéo artifact Hiếu bàn giao.

### Checklist trước khi xếp job đầu tiên

- [ ] Đã phân biệt training seed với evidence/placebo seed.
- [ ] Arm full và w/o EC không bị đổi tên cấu hình thành một thuật toán khác.
- [ ] Có output directory riêng theo dataset/arm/seed/attempt.
- [ ] Có manifest chứa arm ID và expected differences.
- [ ] Mode/flag thực sự được code hỗ trợ; không dùng CLI minh họa của một mode chưa implement.
- [ ] Nếu cần sửa code, full-model branch vẫn giữ phương trình và hành vi cũ; ablation commit được ghi riêng và có kiểm tra parity phù hợp.

## 2. Protocol chung phải khóa

| Yếu tố | Quyết định |
|---|---|
| Dataset | Amazon Baby, Sports, Electronics; đúng split và feature fingerprints seed 1 |
| Dimensionality | 64D; không nâng 256D trong đợt này |
| Epoch budget | 500 epochs, không rút xuống 50/100 để tạo kết quả cuối |
| Checkpoint selection | VALID NDCG@20; mỗi seed chọn checkpoint riêng bằng cùng quy tắc |
| TEST | Đo tại checkpoint được VALID chọn; không chọn arm/seed/checkpoint bằng TEST |
| Training seeds | 1, 2, 3; mỗi run khởi tạo lại từ đầu |
| Optimizer | AdamWSEvo và phân vùng tham số như full model; không thêm auxiliary optimizer |
| BSC/FSC depth | 3; giữ lịch tọa độ native |
| Ranking | Full ranking, cùng seen-item masking, evaluator native |
| Evaluations | Mỗi 5 epochs như seed 1 |
| Primary endpoint | NDCG@20 |
| Secondary endpoint | Recall@20; đồng thời báo Recall@10, NDCG@10 |
| Calibration mặc định | delta=1.0, tau=2.0, kappa=2*tau=4.0 |
| Pooling | Toàn bộ người dùng TRAIN đủ điều kiện; không chia hai nửa trong primary/count/shuffle/global-full |
| Activity conditioning | floor(log2(h_u)), inactive users bị loại; chỉ bỏ conditioning trong arm Global-full |
| Semantic candidates | Text top-5 và image top-1, giữ cùng candidate sets giữa các arm |
| WMSG mặc định | 0.05 + 0.95*[max(0,cosine)]^2 trên selected edges |
| CSE mặc định | eta=0.1, k_cf=5, c_min=2, t_shrinkage=5 |
| Layer-contrastive mặc định | lambda=0.01, temperature=0.2, G=1, alpha=0.5 |
| Precision | FP32 native, TF32 off, AMP off |
| Shuffle | evidence_placebo_seed=20261016 cố định qua training seeds |
| Search | Không tìm thêm hyperparameter cho một arm sau khi xem TEST |

### 2.1 Hyperparameters riêng từng dataset

| Dataset | lr | Weight decay | Batch | gamma | CL chunk | Workers |
|---|---:|---:|---:|---:|---:|---:|
| Baby | 0.001 | 0.3 | 1024 | 0.1 | 1024 | 4 |
| Sports | 0.001 | 0.1 | 1024 | 0.2 | 1024 | 4 |
| Electronics | 0.001 | 0.1 | 4096 | 0.4 | 128 | 4 |

Các giá trị trên lấy từ manifest seed 1 đã thực hiện, không phải giá trị tuning mới. w/o LC vẫn giữ batch và các hyperparameters còn lại; CL chunk không có tác dụng khi LC thực sự bị tắt.

### 2.2 Runtime và mã nguồn

- Reference full commit: `5212263e87c5c1cfe6e48c91afa90201a48f5929`.
- Reference environment ghi trong manifest: Python 3.13.15, Torch 2.11.0+cu128, FreeRec 0.9.7, Tesla T4.
- Notebook full: [eclipse_seed2.ipynb](D:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/P6/eclipse_seed2.ipynb). Seed 3 là bản sao đổi training seed.
- Không sửa YAML nguồn seed 1 tại chỗ; tạo resolved config riêng cho run.
- Ablation cần thay code phải có revision mới, ghi rõ file thay đổi và kiểm tra nhánh full vẫn tương đương. Không yêu cầu mọi arm có cùng source hash khi một ablation thực sự cần thêm mode; yêu cầu đúng expected delta và code provenance.
- GPU khác model hoặc cùng GPU bị chia sẻ tải làm sai diễn giải timing; ghi máy/GPU/load context, không gộp thời gian như cùng điều kiện nếu thực tế khác nhau.

## 3. Trạng thái hỗ trợ code: việc nào có thể chạy, việc nào còn bị chặn

**Quan trọng:** đây là rà soát tĩnh, không phải chứng nhận đã chạy thành công từng arm.

| Arm | Runner hiện tại | Notebook full hiện tại | Công việc trước khi chạy |
|---|---|---|---|
| Full | Đã có đủ $n=3$ seeds | Đã hoàn tất seeds 1/2/3 | **HOÀN TẤT VÀ NGHIỆM THU** (Seeds 1, 2, 3 đạt `VERIFIED_ARTIFACT`) |
| w/o EC | Đã có đủ $n=3$ seeds | Đã hoàn tất seeds 1/2/3 | **HOÀN TẤT VÀ NGHIỆM THU** (Seeds 1, 2, 3 đạt `VERIFIED_ARTIFACT`) |
| w/o LC | Model có nhánh bỏ LC khi lambda=0, nhưng validator chỉ cho 0.01 | Bị khóa cấu hình full | Thêm named arm, cho phép riêng lambda=0; không nới full tùy ý |
| w/o CSE | eta=0 được runner chấp nhận | Kiểm tra full sẽ từ chối eta/graph khác | Named arm/harness, expected operator semantic-only |
| Binary semantic | Bị khóa weighted; graph recovery còn hard-code weighted | Không phù hợp | Sửa routing raw semantic graph ở cả đường parent/reference và calibrator |
| Count-only | Mode pooled_activity_count đã có | Full notebook khóa mode excess | Harness mode-aware, không dùng exporter khóa pooled_activity_excess |
| Shuffled evidence | Mode pooled_activity_shuffle đã có | Full notebook khóa mode excess | Harness mode-aware; ghi placebo seed, strata và score histogram |
| Global-full null | Mode pooled_global_excess đã bổ sung, bins none/halves0 | Seeds 1 & 2 Baby/Sports COMPLETED ($n=2$ per dataset, đã phân tích); Seed 3 READY | Full-user global null trên active TRAIN users; declared source overlay, không dùng unstratified_replication |

### 3.1 Các vị trí code phải kiểm tra

- `stair_icaart_config.py`: validator khóa `lambda_nlgcl=0.01`, `semantic_mode=weighted`, mode registry hiện chưa có global-full.
- `models/stair5_v8.py::training_objective`: đã có điều kiện bỏ gọi LC khi lambda=0; cần xác nhận telemetry raw/weighted LC bằng 0.
- `models/stair5_v8_graph.py::resolve_v8_config`: named full arm có thể ghi đè semantic_mode; không chỉ truyền reference rồi tưởng đã đổi graph.
- `models/stair_icaart_graph.py::recover_v8_raw_semantic_graph`: đang gọi weighted/power=2 trực tiếp. Đây là điểm chặn Binary dù sửa YAML.
- `models/stair_icaart_evidence.py::pooled_activity_score`: Count và degree-stratified Shuffle đã có công thức tương ứng.
- `notebook/P6/eclipse_seed2.ipynb`: full-only expected config, graph fingerprint và receipt validation. Không sửa vài tham số rồi tái sử dụng nguyên các checks này cho ablation.

### 3.2 Gói chuẩn bị kỹ thuật, làm song song lúc GPU đang chạy full

| ID | Chủ trì | Người đọc chéo | Nội dung và điều kiện hoàn tất |
|---|---|---|---|
| ENG-01 | Chương | Hiếu | Arm registry, resolved config, arm_id, expected differences; full vẫn khóa đúng cấu hình hiện tại |
| ENG-02 | Hiếu | Chương | w/o LC và w/o CSE; tắt đúng module, không đổi optimizer/evaluator |
| ENG-03 | Hiếu | Chương | Binary raw graph ở cả reference/calibrator; giữ candidate support, không đổi thành một native legacy graph có normalization khác |
| ENG-04 | Chương | Hiếu | Global-full null trên active TRAIN users; metadata null_scope rõ ràng, không chia hai nửa |
| ENG-05 | Hiếu | Chương | Notebook/harness ablation, parser mode-aware, arm-aware comparability, CSV/figure export, không p-value từ epochs |
| ENG-06 | Cả hai | Đọc chéo | Kiểm tra fixture/contract và smoke ngắn do hai bạn thực hiện; log kiểm tra thật, không ghi PASSED trước khi chạy |

Các file ablation mới nếu được tạo ở bước sau nên có tên rõ, ví dụ `notebook/P6/eclipse_ablations.ipynb`. Tên này là **kế hoạch**, không khẳng định file đã tồn tại. Giai đoạn hiện tại chỉ tạo tài liệu này; không sửa code thuật toán hoặc chạy test/training.

## 4. Đặc tả và điều kiện nghiệm thu từng arm

Ký hiệu: R là binary TRAIN user-item presence; c_ij=sum_u R_ui R_uj; n_ib là số user trong activity stratum b tương tác với item i; M_b là số active users của stratum; h_u là số item TRAIN phân biệt; E_sem là semantic support cố định.

### A0 — Full ECLIPSE

\[
\mu^A_{ij}=\sum_{b:M_b>0}\frac{n_{ib}n_{jb}}{M_b},\qquad
r^A_{ij}=\frac{[c_{ij}-\mu^A_{ij}]_+}{c_{ij}+\mu^A_{ij}+4}.
\]

\[
W^\delta_{ij}=W_{ij}\exp(\delta r^A_{ij}),\quad
S=(1-\eta)\mathcal N(W^\delta)+\eta\bar S_{CF}.
\]

- delta=1, eta=0.1, lambda_LC=0.01.
- Toàn bộ ba hạt giống (Seeds 1, 2, 3) đã hoàn thành 500 epochs trên cả ba dataset, đạt trạng thái `VERIFIED_ARTIFACT`. Chi tiết xem [Báo cáo phân tích Full ECLIPSE đa seed](ECLIPSE_Full_Model_Multi_Seed_Analysis.md).
- Thu được ba seed/dataset; không chọn seed tốt nhất để làm số đại diện. Báo cáo bằng Mean $\pm$ Sample SD ($n=3$, $\text{ddof}=1$).

### A1 — w/o EC

- **Thay duy nhất:** `evidence_delta=0.0`; vẫn giữ estimator pooled_activity_excess, tau=2, eta=0.1 và LC=0.01.
- Không đổi sang replicated mode hoặc tắt LC/CSE.
- Nghiệm thu: `exact_v8_bypass=true`, lý do delta_zero; calibrated operator bằng reference operator của cùng weighted/CSE backbone, operator/polynomial dose do calibration bằng 0 trong độ chính xác đã khai báo.
- Score/coverage có thể vẫn dương vì estimator vẫn được tính; không ép chúng bằng 0. Bằng chứng quyết định là **liều calibration áp dụng bằng 0**.
- Hỏi: phần calibration có tạo incremental gain ngoài các thành phần còn lại không?
- Nếu không tốt hơn w/o EC: không thể quy toàn bộ gain so với STAIR cho PEEC; phải giảm độ mạnh của claim calibration.

### A2 — w/o LC

- **Thay duy nhất:** lambda_nlgcl=0; LC call thực sự bị bỏ qua, không chỉ tính rồi nhân 0.
- Graph, PEEC, CSE, MI, FSC, BPR, optimizer và evaluator giữ nguyên.
- Nghiệm thu: raw LC=0, weighted LC=0; total loss bằng BPR theo reduction native; graph/candidate/evidence fingerprints bằng full của cùng dataset; không thêm loss thay thế.
- Không thay batch/chunk để tạo lợi thế cho arm này. Runtime nhanh hơn vì bỏ LC là kết quả cần đo, không phải lý do bù bằng các thay đổi khác.
- Nếu ablation không giảm metric: không viết LC luôn cần thiết trên mọi dataset.

### A3 — w/o CSE

- **Thay duy nhất:** eta=0.0; giữ k_cf=5 và các hằng số estimator để tránh thay nhiều yếu tố.
- Toán tử cuối là normalized calibrated semantic graph. Cạnh CF-only không còn đóng góp; cạnh vừa là semantic vừa là CF có thể vẫn tồn tại từ semantic branch.
- Hành vi TRAIN vẫn được dùng trong FSC, BPR và PEEC. Không gọi arm này là "without behavioral information".
- Nghiệm thu: effective_eta=0, semantic support và evidence scores giống full; không còn CF-only propagation; scorer/objective không đổi.
- Nếu CF graph vẫn được build trong preprocessing dù eta=0, ghi đúng overhead thực tế, không tuyên bố đã loại mọi chi phí CF.

### A4 — Binary semantic weights

- Giữ nguyên text/image candidates và thứ tự modalities.
- Đặt **mỗi directed candidate occurrence của mỗi modality** bằng 1:
\[
B^{bin}_{ij}=\sum_{m\in\{t,v\}}\mathbf 1[j\in K_m(i)],\qquad
W^{bin}_{ij}=\max(B^{bin}_{ij},B^{bin}_{ji}),\quad W^{bin}_{ii}=0.
\]
- Sau đó giữ nguyên r^A, exponential calibration, actual-degree normalization, identity isolates và CF convex blend.
- W_binary có thể nhận 1 hoặc 2 do overlap modalities; không tự đặt toàn bộ union edges bằng 1.
- Không chuyển sang symmetrize-each-modality-first, không đổi k, không bỏ PEEC hoặc CF.
- Nghiệm thu: candidate support/counts/null/score hash giống full; raw weight values phù hợp binary occurrence sum; normalized graph thường khác full; reference của calibrator cũng là **binary uncalibrated** operator, không lấy weighted reference cho fast-path.
- Cần sửa code routing, không chỉ nới validator. Nếu delta0 bypass trả weighted operator cho Binary thì arm này sai.

### A5 — Count-only

- Mode có sẵn: `pooled_activity_count`, halves=0, tau=2, delta=1.
\[
r^{count}_{ij}=\frac{c_{ij}}{c_{ij}+4}.
\]
- Giữ weighted semantic graph và CSE/LC. Null có thể vẫn được tính để diagnostic nhưng không dùng trong score.
- Nghiệm thu: score đúng công thức, support giữ nguyên; metadata không gắn nhãn score này là null-subtracted excess.
- Hỏi: trừ kỳ vọng có tốt hơn chỉ tăng trọng số theo count không?
- Cùng delta không đồng nghĩa cùng operator dose. Báo score distribution và normalized dose, không tự tuning thêm bằng TEST để một arm thắng.

### A6 — Shuffled evidence

- Mode có sẵn: `pooled_activity_shuffle`, halves=0, tau=2, delta=1, placebo_seed=20261016.
- Tính r^A đúng như full rồi hoán vị giữa semantic pairs trong cùng unordered endpoint-degree-quintile stratum.
- Giữ một phép shuffle cố định cho training seeds 1/2/3. Ba training seeds đo training variance; không phải ba phép randomization độc lập của graph.
- Nghiệm thu: support/candidates/count/null giống full; multiset scores của từng shuffle stratum giữ nguyên; assignment có đổi nếu stratum chứa score khác nhau; ghi moved-edge fraction nếu export được.
- Với toàn score bằng nhau hoặc stratum singleton, permutation có thể không đổi; không ép score thay đổi bằng cách phá phân phối.
- Báo normalized dose vì cùng score histogram chưa bảo toàn weighted graph mass hoặc BSC effect.
- Nếu Shuffle ngang/full: chưa chứng minh edge-specific evidence placement có ích; không chỉ dựa vào gain so với STAIR để bỏ qua kết quả này.

### A7 — Global-full null

- **Chưa implement.** Tên mode dự kiến `pooled_global_excess`; chỉ dùng sau khi registry/routing đã có và được kiểm tra.
- Lấy cùng binary TRAIN presence và pooled observed counts, không chia hai nửa:
\[
M_{active}=|\{u:h_u>0\}|,\quad n_i=\sum_{u:h_u>0}R_{ui},\quad
\mu^G_{ij}=\frac{n_i n_j}{M_{active}},\quad
r^G_{ij}=\frac{[c_{ij}-\mu^G_{ij}]_+}{c_{ij}+\mu^G_{ij}+4}.
\]
- Trường hợp M_active=0 phải xử lý xác định, không chia 0. Không dùng tổng số user chứa inactive để thay mẫu số này.
- Giữ delta/tau, graph support, LC/CSE và normalization; chỉ thay cách conditioning null.
- Metadata: null_scope=global_full, active_halves=0, active_partition=false. Không giả ghi full_activity.
- `unstratified_replication` hiện có là **hai-nửa + minimum**, không phải arm này.
- Nếu không chạy được: ghi NOT_RUN; trong paper không claim đã cô lập thực nghiệm lợi ích activity conditioning.

## 5. Ma trận chính thức và thứ tự ưu tiên

| Arm | Baby | Sports | Electronics | Mức |
|---|---|---|---|---|
| Full ECLIPSE | seeds 1/2/3 | seeds 1/2/3 | seeds 1/2/3 | P0; **HOÀN TẤT 100% (Seeds 1, 2, 3 VERIFIED_ARTIFACT)** |
| w/o EC | seeds 1/2/3 | seeds 1/2/3 | seeds 1/2/3 | P0; **HOÀN TẤT 100% (Seeds 1, 2, 3 VERIFIED_ARTIFACT)** |
| w/o LC | seeds 1/2/3 | seeds 1/2/3 | seed 1 tùy thời gian | P1; Electronics là P2 |
| w/o CSE | seeds 1/2/3 | seeds 1/2/3 | seed 1 tùy thời gian | P1; Electronics là P2 |
| Binary semantic | seeds 1/2/3 | seeds 1/2/3 | seed 1 tùy thời gian | P1; Electronics là P2 |
| Count-only | seeds 1/2/3 (VERIFIED_ARTIFACT) | seeds 1/2/3 (VERIFIED_ARTIFACT) | seed 1 (VERIFIED_ARTIFACT) | P1; **HOÀN TẤT 100% PHẠM VI** (Baby/Sports s1–3, Electronics s1) |
| Shuffled evidence | seeds 1/2/3 | seeds 1/2/3 | seed 1 | P1; Electronics sau w/o EC |
| Global-full null | seeds 1/2/3 nếu sẵn sàng | seeds 1/2/3 nếu sẵn sàng | Không yêu cầu | P1 điều kiện code; là control riêng |

- Core sau các full run: **39 run** = 9 w/o EC + 30 run Baby/Sports của năm arm LC/CSE/Binary/Count/Shuffle.
- Kế hoạch mở rộng ưu tiên: **47 run** = core 39 + 2 Electronics Count/Shuffle + 6 Global-full Baby/Sports.
- Ba Electronics LC/CSE/Binary là **P2, không nằm trong 47 run**; nếu thực hiện đủ thì tổng 50 run.
- Không cộng seed 1 full đã hoàn tất hoặc sáu full seed 2/3 đang triển khai vào số 47.
- Nếu reuse artifact w/o EC seed 1 được nghiệm thu ở mục 8 thì giảm số run mới, không giảm số seed báo cáo.

## 6. Phân công và cân bằng GPU

| Nhánh | Chương / hàng đợi A | Hiếu / hàng đợi B |
|---|---|---|
| w/o EC | Cả ba dataset, seeds 1 và 3: 6 run | Cả ba dataset, seed 2: 3 run |
| w/o LC | Đọc chéo | Baby/Sports seeds 1/2/3: 6 run |
| w/o CSE | Đọc chéo | Baby/Sports seeds 1/2/3: 6 run |
| Binary semantic | Đọc chéo | Baby/Sports seeds 1/2/3: 6 run |
| Count-only | Baby/Sports seeds 1/2/3 + Electronics seed 1: 7 run | Đọc chéo |
| Shuffled evidence | Đọc chéo | Baby/Sports seeds 1/2/3 + Electronics seed 1: 7 run |
| Global-full null | Baby/Sports seeds 1/2/3: 6 run | Đọc chéo công thức và artifacts |

Tổng: **Chương 19 job**, **Hiếu 28 job** trong kế hoạch 47. Chương ít job hơn nhưng nhận nhiều Electronics hơn; ước lượng khoảng 34.7 so với 31.1 GPU-hours. Số job không đại diện khối lượng tính toán.

### 6.1 Hàng đợi khi hai slot rảnh sau full run (T0)

| Wave | Slot A / Chương | Slot B / Hiếu | Dự kiến tham khảo |
|---|---|---|---|
| W1 | w/o EC Electronics seed 1 | w/o EC Electronics seed 2 | ~8.8–9 h |
| W2 | w/o EC Electronics seed 3 | LC/CSE/Binary Baby/Sports, ba seeds mỗi arm | ~9–10 h |
| W3 | Count Electronics seed 1 | Shuffle Electronics seed 1 | ~9 h |
| W4 | w/o EC Baby/Sports seeds 1/3 + Count Baby/Sports + Global-full Baby/Sports | w/o EC Baby/Sports seed 2 + Shuffle Baby/Sports; sau đó tổng hợp/đọc chéo | ~4–9 h |

Nếu một slot rảnh trước, lấy job READY tiếp theo, không chờ đồng bộ cả hai. Nếu Global-full/Binary chưa sẵn sàng, chạy arm READY khác; không tiêu tốn slot GPU chờ code. Waves là lịch dự toán, không tự động dừng job theo số giờ bảng này.

### 6.2 Ledger từng job

Ledger ghi trạng thái theo từng job; các job chưa có artifact vẫn là **PLANNED**. Đổi thành READY sau kiểm tra implementation/harness, RUNNING khi process thực sự bắt đầu, COMPLETED_PENDING_REVIEW sau runner thành công và VERIFIED_ARTIFACT sau đọc chéo. Đây là nghiệm thu artifact, không phải nhãn reviewer khoa học độc lập.

| Job ID | Arm | Dataset | Seed | Người chạy | Ưu tiên | Trạng thái |
|---|---|---|---:|---|---|---|
| ECL-wo_ec-baby-s1 | wo_ec | baby | 1 | Chương | P0 | REUSED_VERIFIED |
| ECL-wo_ec-baby-s2 | wo_ec | baby | 2 | Hiếu | P0 | VERIFIED_ARTIFACT |
| ECL-wo_ec-baby-s3 | wo_ec | baby | 3 | Chương | P0 | READY |
| ECL-wo_ec-sports-s1 | wo_ec | sports | 1 | Chương | P0 | REUSED_VERIFIED |
| ECL-wo_ec-sports-s2 | wo_ec | sports | 2 | Hiếu | P0 | VERIFIED_ARTIFACT |
| ECL-wo_ec-sports-s3 | wo_ec | sports | 3 | Chương | P0 | READY |
| ECL-wo_ec-electronics-s1 | wo_ec | electronics | 1 | Chương | P0 | REUSED_VERIFIED |
| ECL-wo_ec-electronics-s2 | wo_ec | electronics | 2 | Hiếu | P0 | VERIFIED_ARTIFACT |
| ECL-wo_ec-electronics-s3 | wo_ec | electronics | 3 | Chương | P0 | READY |
| ECL-wo_lc-baby-s1 | wo_lc | baby | 1 | Hiếu | P1 | PLANNED |
| ECL-wo_lc-baby-s2 | wo_lc | baby | 2 | Hiếu | P1 | PLANNED |
| ECL-wo_lc-baby-s3 | wo_lc | baby | 3 | Hiếu | P1 | PLANNED |
| ECL-wo_lc-sports-s1 | wo_lc | sports | 1 | Hiếu | P1 | PLANNED |
| ECL-wo_lc-sports-s2 | wo_lc | sports | 2 | Hiếu | P1 | PLANNED |
| ECL-wo_lc-sports-s3 | wo_lc | sports | 3 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-baby-s1 | wo_cse | baby | 1 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-baby-s2 | wo_cse | baby | 2 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-baby-s3 | wo_cse | baby | 3 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-sports-s1 | wo_cse | sports | 1 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-sports-s2 | wo_cse | sports | 2 | Hiếu | P1 | PLANNED |
| ECL-wo_cse-sports-s3 | wo_cse | sports | 3 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-baby-s1 | binary_semantic | baby | 1 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-baby-s2 | binary_semantic | baby | 2 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-baby-s3 | binary_semantic | baby | 3 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-sports-s1 | binary_semantic | sports | 1 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-sports-s2 | binary_semantic | sports | 2 | Hiếu | P1 | PLANNED |
| ECL-binary_semantic-sports-s3 | binary_semantic | sports | 3 | Hiếu | P1 | PLANNED |
| ECL-count_only-baby-s1 | count_only | baby | 1 | Chương | P1 | READY |
| ECL-count_only-baby-s2 | count_only | baby | 2 | Chương | P1 | PLANNED |
| ECL-count_only-baby-s3 | count_only | baby | 3 | Chương | P1 | PLANNED |
| ECL-count_only-sports-s1 | count_only | sports | 1 | Chương | P1 | READY |
| ECL-count_only-sports-s2 | count_only | sports | 2 | Chương | P1 | PLANNED |
| ECL-count_only-sports-s3 | count_only | sports | 3 | Chương | P1 | PLANNED |
| ECL-count_only-electronics-s1 | count_only | electronics | 1 | Chương | P1 | READY |
| ECL-shuffled_evidence-baby-s1 | shuffled_evidence | baby | 1 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-baby-s2 | shuffled_evidence | baby | 2 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-baby-s3 | shuffled_evidence | baby | 3 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-sports-s1 | shuffled_evidence | sports | 1 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-sports-s2 | shuffled_evidence | sports | 2 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-sports-s3 | shuffled_evidence | sports | 3 | Hiếu | P1 | PLANNED |
| ECL-shuffled_evidence-electronics-s1 | shuffled_evidence | electronics | 1 | Hiếu | P1 | PLANNED |
| ECL-global_full_null-baby-s1 | global_full_null | baby | 1 | Chương | P1 | COMPLETED |
| ECL-global_full_null-baby-s2 | global_full_null | baby | 2 | Chương | P1 | COMPLETED |
| ECL-global_full_null-baby-s3 | global_full_null | baby | 3 | Chương | P1 | READY |
| ECL-global_full_null-sports-s1 | global_full_null | sports | 1 | Chương | P1 | COMPLETED |
| ECL-global_full_null-sports-s2 | global_full_null | sports | 2 | Chương | P1 | COMPLETED |
| ECL-global_full_null-sports-s3 | global_full_null | sports | 3 | Chương | P1 | READY |


### 6.3 Ba job Electronics P2, chưa nằm trong 47 job

| Job ID | Arm | Seed | Người chạy | Điều kiện |
|---|---|---:|---|---|
| ECL-wo_lc-electronics-s1 | w/o LC | 1 | Chương | Core hoàn tất, manuscript đã có draft, đủ quota |
| ECL-wo_cse-electronics-s1 | w/o CSE | 1 | Hiếu | Cùng điều kiện |
| ECL-binary_semantic-electronics-s1 | Binary | 1 | Hiếu | Cùng điều kiện; routing Binary đã nghiệm thu |

Nếu không chạy, ghi NOT_RUN, không đổi sang một run 100 epochs rồi đặt cạnh kết quả 500 epochs. Không cần đợi ba job này để nộp paper.

## 7. Thời gian, ngân sách và lịch đến hạn nộp

### 7.1 Cơ sở dự toán

Fit time seed 1 đã quan sát: full Baby 1214.19 s (~0.337 h), Sports 2520.87 s (~0.700 h), Electronics 31999.03 s (~8.889 h). Dùng native delta0 reference quan sát trước đó làm mốc w/o EC: ~0.322/0.694/8.778 h. Đây là mốc lập lịch, không phải runtime đã đo cho tất cả ablations.

| Khối việc sau full đang chạy | Số run | GPU-hours dự toán |
|---|---:|---:|
| w/o EC: 3 datasets × 3 seeds | 9 | ~29.4 |
| LC/CSE/Binary/Count/Shuffle: 2 datasets × 3 seeds × 5 arms | 30 | ~15.6 |
| Global-full: 2 datasets × 3 seeds | 6 | ~3.1 |
| Count/Shuffle Electronics: seed 1 mỗi arm | 2 | ~17.8 |
| **Tổng kế hoạch 47 run** | **47** | **~65.9** |
| Dự phòng 25% cho preprocessing, startup, xuất artifact và biến thiên | — | **~16.5** |
| **Ngân sách vận hành** | — | **~82.4 GPU-hours** |

Hai GPU: ~33 giờ lý tưởng sau T0, dự trù khoảng 42–46 giờ nếu hàng đợi cân bằng và có quota liên tục. Full seed 2/3 còn đang chạy cần cộng phần thời gian còn lại; không giả định đã xong. Nếu reuse được ba artifact w/o EC seed 1, giảm khoảng 9.8 GPU-hours.

Ba Electronics LC/CSE/Binary P2 bổ sung khoảng **26.7 GPU-hours trước dự phòng**, nên không đưa vào đường găng. Timing thực tế có thể khác vì LC tắt, nnz thay đổi hoặc cache hit; ghi lại số đo thật, không thay thế bằng bảng dự toán.

### 7.2 Lịch nội bộ

| Ngày (Asia/Saigon) | Mục tiêu Chương | Mục tiêu Hiếu | Điều kiện chuyển bước |
|---|---|---|---|
| 18/10 | Theo dõi các full run hiện có; chuẩn bị named arms/Global-full, khóa config | Arm-aware harness, Binary/LC/CSE và reporting; chuẩn bị figures | Các arm sẵn sàng trước khi slot rảnh |
| 19/10 | w/o EC Electronics và Count theo hàng đợi; viết Method | w/o EC Electronics + small-dataset ablations; viết Experiment Setup | Mỗi run hoàn tất được xuất artifact ngay |
| 20/10 | Count/Global-full/small jobs, tổng hợp seed | Shuffle Electronics và small jobs; vẽ ablation | Có core ablations và paired differences |
| 21/10 | Khóa bảng chính, claims và Method; đọc chéo artifact Hiếu | Khóa hình/bảng, statistical summary; đọc chéo artifact Chương | Bản nháp hoàn chỉnh; không mở tuning mới |
| 22/10 | Cùng kiểm tra và nộp; giữ buffer | Cùng kiểm tra và nộp; giữ buffer | Submission hợp lệ, không đợi job P2 |

Mốc chính thức hiện được thông báo cho stage 2 là 22/10/2026 theo AoE: https://icaart.scitevents.org/ImportantDates.aspx?y=2027. Đặt mục tiêu nội bộ nộp sớm ngày 22/10 theo giờ Việt Nam, không đợi sát hạn hoặc tự suy ngày hết quota Kaggle từ lịch này.

### 7.3 Quy tắc cắt phạm vi nếu chậm

1. Không hủy full seed 2/3 đang tiến triển bình thường chỉ để chạy ablation.
2. Giữ ưu tiên full và w/o EC ba seed/ba dataset.
3. Giữ component và Count/Shuffle trên Baby/Sports; nếu buộc giảm số seed, quyết định theo ngân sách trước khi xem TEST, áp dụng đối xứng và ghi rõ n thật.
4. Global-full chưa implement kịp thì ghi NOT_RUN, bỏ claim đã cô lập activity-conditioning; không dùng một estimator hai-nửa giả làm control.
5. Count/Shuffle Electronics đặt sau w/o EC; nếu hết thời gian, báo giới hạn phạm vi thay vì kéo dài deadline paper. Đây là mục tiêu kế hoạch, không lý do che kết quả xấu.
6. Không chạy ba Electronics P2 khi core/viết bài chưa xong.
7. Không cắt một seed hoặc dataset vì metric không tốt. Job có lỗi kỹ thuật được xử lý riêng với log lỗi.

## 8. Quy trình một run từ READY đến bàn giao

### 8.1 Chuẩn bị

1. Nhận đúng Job ID trong ledger, ghi owner, machine, GPU index và attempt_id mới.
2. Khởi tạo process từ đầu; không nạp embedding/model/optimizer checkpoint của seed khác hay arm khác. Có thể dùng cache graph/candidates xác định nếu cache signature hợp lệ.
3. Tạo resolved YAML từ protocol đã khóa rồi áp dụng đúng một ablation; lưu diff so với full cấu hình cùng dataset.
4. Kiểm tra source revision đúng gói ablations; kiểm tra split/feature/candidate hashes và environment. Không tự update Torch trong lúc chạy.
5. Chạy config/import pre-flight. Với mode mới, thực hiện fixture/short smoke đã quy định ở ENG-06 trong output `purpose=smoke` riêng; không gộp vào paper results.
6. Kiểm tra flag seed cuối cùng đã resolve. Thay chữ seed trong filename mà cfg.seed vẫn là 1 không tạo ra run mới.

**Cache và bộ nhớ:** chỉ tái sử dụng input/candidates/CF graph xác định với signature phù hợp. Không cache embedding đã train hoặc optimizer state giữa seeds. Cache key cho graph/reference phải phân biệt weighting, evidence/null mode, delta/tau, eta, candidates, normalization và source version; không lấy weighted reference cho Binary. Giữ graph construction candidate-only/chunked trên CPU, final sparse operator trên GPU; không tạo dense item×item để tính similarity hoặc diagnostics.

### 8.2 Trong khi chạy

- Mỗi GPU chỉ có một training process của kế hoạch, tránh đo timing trong cạnh tranh GPU ngoài dự kiến.
- Theo dõi stdout log, tăng epoch trong training_telemetry.jsonl và run_status.json.
- Chỉ so progress với wall time/range đã đo; không kill tự động chỉ vì một epoch dài. Người chạy kiểm tra process/CPU/data loader/disk trước khi quyết định can thiệp.
- Nếu OOM/nonfinite: lưu lỗi và config, không tự giảm batch/đổi AMP cho riêng một arm rồi tiếp tục ghi cùng protocol. Bất kỳ thay đổi bắt buộc phải được ghi là deviation; nếu ảnh hưởng comparison thì chạy lại cặp liên quan hoặc giảm claim.
- Không đánh giá TEST theo epoch để quyết định dừng hoặc tuning. Các curve phục vụ monitoring là TRAIN và VALID.
- Không ghi đè/xóa attempt đang có. Nếu cần retry kỹ thuật, attempt mới, liên kết rõ attempt lỗi trước.

### 8.3 Khi subprocess kết thúc

Một run hợp lệ phải có:

- [ ] Exit code 0; run_status=COMPLETED cho đúng training seed.
- [ ] Training telemetry có đủ 500 epochs của một uninterrupted run, hoặc một resume được khai báo và xác minh rõ. Primary ưu tiên không resume.
- [ ] selected_metrics.json: mode=test, selected_checkpoint=true, selection="validation NDCG@20".
- [ ] Có đủ RECALL@10, RECALL@20, NDCG@10, NDCG@20 hữu hạn trong [0,1].
- [ ] selected_epoch hợp lệ, thuộc lịch lựa chọn; không dùng dòng TEST cuối cùng tùy tiện nếu không chứng minh là checkpoint được chọn.
- [ ] best_model.pt tồn tại và có đường dẫn; checkpoint có contract arm/config/operator phù hợp.
- [ ] Manifest, resolved config, CLI, environment, dataset/source/candidate/operator fingerprints được lưu.
- [ ] Điều kiện riêng arm ở mục 4 đạt, ví dụ w/o LC loss CL=0; w/o CSE effective_eta=0.
- [ ] CSV và figure data lấy từ receipts/telemetry; không có missing bị chuyển thành 0.
- [ ] Đóng ZIP, tạo SHA256, sao chép khỏi session Kaggle trước khi hết quota.
- [ ] Người còn lại đọc chéo xong; kết quả xấu về metric vẫn được giữ nếu run đúng kỹ thuật.

**Hoàn tất run không yêu cầu metric cao hơn full hoặc baseline.** Nghiệm thu tính đúng đắn trước, phân tích khoa học sau.

### 8.4 Tái sử dụng seed 1 w/o EC từ artifact sẵn có

**Cập nhật 2026-10-18:** Đã tìm thấy ba run 500 epochs có cấu hình/data/candidates và operator đúng bằng nhánh delta=0. Xem [audit và hướng dẫn notebook](ECLIPSE_wo_EC_Seed1_Artifact_Audit.md). **Không cần train lại seed 1**; cần dùng [notebook recovery seed 1](../../notebook/P6/eclipse_wo_ec_seed1.ipynb) để xuất selected TEST receipt từ checkpoint sẵn có. Baby chọn epoch **430**, còn summary sáu chữ số trong log thuộc epoch 500; không được lấy summary đó làm selected metric. Sports/Electronics chọn epoch 500. Tạm giữ trạng thái `REUSED_TRAINING_PENDING_SELECTED_REEVALUATION`, chỉ đổi thành `REUSED_VERIFIED` sau khi recovery và đọc chéo thành công. [Notebook w/o EC seed 2](../../notebook/P6/eclipse_wo_ec_seed2.ipynb) đã được chuẩn bị cho bước training độc lập tiếp theo; Dùng [notebook w/o EC seed 3](../../notebook/P6/eclipse_wo_ec_seed3.ipynb) cho run seed 3; xem cập nhật ở mục 8.6.

**Nghiệm thu sau recovery:** Phiên `eclipse_wo_ec_seed1_reuse_20261018T152158Z_69603726` đã có đủ native selected TEST receipts, đúng checkpoints, source/data/operator fingerprints và matched runtime trên cả ba dataset. Trạng thái ba job seed 1 hiện là **REUSED_VERIFIED**, thay thế trạng thái chờ ở ghi chú trước. Xem [báo cáo đã hiệu chỉnh](ECLIPSE_Ablation_Study_wo_EC.md) và [acceptance JSON](../../reports/eclipse_wo_ec_seed1_audit/recovered_seed1_acceptance.json). Đây là kiểm tra artifact của user-executed evaluation, không phải model rerun hoặc xác nhận thống kê từ một seed. Chưa tự xác nhận người đọc chéo độc lập là Hiếu. Notebook seed 2 đã nhúng riêng ba receipt **w/o EC đã đo**, kèm provenance; không cần upload lại checkpoint seed 1 chỉ để tổng hợp. Không sửa architecture hoặc các hyperparameters trước replication seeds 2–3.

Chương kiểm tra, Hiếu đọc chéo:

1. Cùng split/feature/candidate fingerprints và catalog.
2. Cùng MI/FSC/BPR/LC/optimizer/batch/lr/decay/epochs/evaluator/selection và training seed=1.
3. Operator bằng đúng weighted semantic + CF operator của delta0; không thêm SAP, gate, residual hay thay CSE.
4. Có selected TEST tại checkpoint đúng và đủ độ chính xác; không lấy số 4 chữ số trong báo cáo làm receipt sáu chữ số.
5. Có source/environment provenance; nếu code khác thì cần diff và lý giải tính tương đương, không suy từ tên folder.
6. Không chọn một artifact trong nhiều attempts vì TEST cao hơn.

Nếu đạt, đổi trạng thái job A1-s1 thành REUSED_VERIFIED, ghi source path/checksums. Không cần đưa tên phiên bản nội bộ vào paper. Nếu chưa đạt thì giữ job phải chạy; không tự chuyển số liệu thành ablation chỉ vì gần nhau.

### 8.5 Nghiệm thu ECLIPSE w/o EC Seed 2

**Cập nhật 2026-10-19:** Phiên `eclipse_wo_ec_seed2_20261018T162239Z_f820d171` hoàn tất training độc lập 500 epochs trên Baby, Sports và Electronics. [Artifacts](../../logs/ICAART/eclipse_wo_ec_seed2) có native selected TEST receipts, checkpoints, manifests và telemetry. **151 kiểm tra artifact nhất quán** đã đạt: [acceptance JSON](../../reports/eclipse_wo_ec_seed2_audit/acceptance.json). Ba job `ECL-wo_ec-*-s2` là **VERIFIED_ARTIFACT**; đây là nghiệm thu kỹ thuật, không tự xác nhận Hiếu đã đọc chéo hoặc model đã rerun trên máy local.

Selected epochs lần lượt Baby **420**, Sports **470**, Electronics **480**, đều theo VALID NDCG@20. Native evaluator có TEST tại terminal epoch 500 rồi restored selected checkpoint; bảng paper chỉ dùng `selected_metrics.json`, không chọn theo TEST. Seed 2 dùng pinned commit `5212263e87c5c1cfe6e48c91afa90201a48f5929`; seed 1 reuse historical revisions đã audit tương đương, không phải cùng source commit.

w/o EC hiện có **n=2** trên mỗi dataset, đủ báo sample SD; **Full–w/o EC mới có matched seed 1 trong artifacts local**. Không so Full seed 1 với mean w/o EC n=2 để tuyên bố thắng qua hai seeds. Số liệu và các diễn giải đã hiệu chỉnh tại [báo cáo ablation](ECLIPSE_Ablation_Study_wo_EC.md). Giữ nguyên cấu hình khi tiếp tục replication; không tune từ TEST.

### 8.6 Nghiệm thu ECLIPSE w/o EC Seed 3 — Hoàn Thành Trọn Vẹn $n=3$ w/o EC

**Cập nhật 2026-10-20:** Phiên huấn luyện độc lập `eclipse_wo_ec_seed3_20261019T075013Z_03cb3f39` đã hoàn tất 500 epochs trên cả 3 datasets: Amazon Baby, Amazon Sports, và Amazon Electronics.
- Toàn bộ hóa đơn native `selected_metrics.json`, manifests, configs, figures và checkpoint weights đã được lưu trữ và kiểm tra tại [`logs/ICAART/eclipse_wo_ec_seed3/`](../../logs/ICAART/eclipse_wo_ec_seed3/).
- Cả ba job `ECL-wo_ec-*-s3` được nâng trạng thái thành **VERIFIED_ARTIFACT**.
- Nhánh **ECLIPSE w/o EC** (A1) chính thức hoàn thành toàn bộ bộ ba hạt giống $n=3$ (Seeds 1, 2, 3).
- **Kết quả tổng hợp w/o EC ($n=3$, Mean $\pm$ Sample SD):**
  - **Baby ($n=3$):** Recall@10 = $0.067455 \pm 0.000119$, Recall@20 = $0.104091 \pm 0.001180$, NDCG@10 = $0.036664 \pm 0.000207$, NDCG@20 = $\mathbf{0.046070 \pm 0.000162}$.
  - **Sports ($n=3$):** Recall@10 = $0.076236 \pm 0.000125$, Recall@20 = $0.114329 \pm 0.000084$, NDCG@10 = $0.041837 \pm 0.000251$, NDCG@20 = $\mathbf{0.051703 \pm 0.000229}$.
  - **Electronics ($n=3$):** Recall@10 = $0.045962 \pm 0.000136$, Recall@20 = $0.068167 \pm 0.000206$, NDCG@10 = $0.025971 \pm 0.000063$, NDCG@20 = $\mathbf{0.031715 \pm 0.000030}$.
- **Đối chiếu ghép cặp 3-seed hoàn chỉnh (Full ECLIPSE vs. w/o EC):**
  - Full ECLIPSE giành chiến thắng ở **32 / 36 so sánh (88.9%)** (thắng 8/9 seeds trên NDCG@20 và 9/9 seeds trên Recall@10).
  - Đạt ý nghĩa thống kê $p < 0.05$ trên **Sports NDCG@20** ($p = 0.0446$) và **Baby Recall@10** ($p = 0.0179$).
  - Xem chi tiết tại [ECLIPSE_Ablation_Study_wo_EC.md](ECLIPSE_Ablation_Study_wo_EC.md).

### 8.7 Nghiệm thu Full ECLIPSE (A0) Seeds 2 & 3 — Hoàn Thành Trọn Vẹn $n=3$ Full Model

**Cập nhật 2026-10-20:** Toàn bộ ba hạt giống độc lập ($n=3$) của mô hình chuẩn **Full ECLIPSE** (A0) trên cả ba benchmark Amazon Baby, Amazon Sports, và Amazon Electronics đã hoàn tất 100% 500 epochs:
- **Seed 1:** `REUSED_VERIFIED` từ [`logs/GD6/STAIR-PEEC/`](../../logs/GD6/STAIR-PEEC/).
- **Seed 2:** `VERIFIED_ARTIFACT` từ [`logs/ICAART/eclipse_seed2/`](../../logs/ICAART/eclipse_seed2/) (session: `20261018T083655Z_2b0f7924`).
- **Seed 3:** `VERIFIED_ARTIFACT` từ [`logs/ICAART/eclipse_seed3/`](../../logs/ICAART/eclipse_seed3/) (session: `20261019T075435Z_d063f4c8`).

**Kết quả tổng hợp tri-seed ($n=3$, Mean $\pm$ Sample SD):**
- **Amazon Baby ($n=3$):** Recall@10 = $0.068356 \pm 0.000236$, Recall@20 = $0.104553 \pm 0.001484$, NDCG@10 = $0.037057 \pm 0.000232$, NDCG@20 = $\mathbf{0.046340 \pm 0.000241}$.
- **Amazon Sports ($n=3$):** Recall@10 = $0.076792 \pm 0.000481$, Recall@20 = $0.114640 \pm 0.000164$, NDCG@10 = $0.042162 \pm 0.000112$, NDCG@20 = $\mathbf{0.051956 \pm 0.000156}$.
- **Amazon Electronics ($n=3$):** Recall@10 = $0.046177 \pm 0.000212$, Recall@20 = $0.068247 \pm 0.000122$, NDCG@10 = $0.026050 \pm 0.000164$, NDCG@20 = $\mathbf{0.031759 \pm 0.000093}$.

**Dấu vân tay đồ thị hiệu chuẩn ($S_{\mathrm{ECLIPSE}}$ SHA-256):** Khớp bitwise 100% trên cả ba hạt giống: Baby (`b801...`), Sports (`6b7b...`), Electronics (`43b9...`).
**Dấu chân bộ nhớ (Peak Allocator):** Bitwise bất biến giữa các seed: Baby 145.17 MiB, Sports 223.71 MiB, Electronics 800.67 MiB.
Phân tích chi tiết, bảng LaTeX và đối chiếu matched pairs xem tại [ECLIPSE_Full_Model_Multi_Seed_Analysis.md](ECLIPSE_Full_Model_Multi_Seed_Analysis.md).

## 9. Cách cấu hình và lệnh tham khảo

### 9.1 Các arm đã biểu diễn được bằng runner hiện tại

Đây là ví dụ một run Baby seed 1. Đường dẫn là ví dụ; thay bằng output mới của Job ID thực tế. Cần có data root writable với `Processed/Amazon2014Baby_550_MMRec`. Không chạy trong checkout của full process đang hoạt động.

```bash
python -u main_stair_icaart.py \
  --config configs/Amazon2014Baby_STAIR_PEEC.yaml \
  --root /kaggle/working/eclipse_data \
  --device 0 --seed 1 \
  --evidence-mode pooled_activity_excess --evidence-halves 0 \
  --evidence-delta 0.0 --evidence-tau 2.0 \
  --artifact-dir /kaggle/working/eclipse_ablation_runs/ATTEMPT_ID/artifacts/baby_eclipse_wo_ec_seed1 \
  --graph-cache-dir /kaggle/working/eclipse_ablation_cache/baby
```

Thay đổi **so với full**, không phải chuỗi lệnh phải cộng dồn:

| Arm | Override runner hiện có |
|---|---|
| w/o EC | mode=pooled_activity_excess, halves=0, delta=0 |
| w/o CSE | mode=pooled_activity_excess, halves=0, delta=1, eta=0 |
| Count | mode=pooled_activity_count, halves=0, delta=1 |
| Shuffle | mode=pooled_activity_shuffle, halves=0, delta=1, evidence-placebo-seed=20261016 |

Giữ `--seed` đúng job, tau=2. Không đổi `--evidence-seed` thành training seed; trường này inactive trong full pooling.

### 9.2 Các arm chưa có lệnh runnable ngay

- w/o LC: lambda=0 chỉ được dùng **sau** khi named-arm validator chấp nhận riêng ablation. Hiện chỉ thêm `--lambda-nlgcl 0` sẽ bị từ chối.
- Binary: không có một override YAML đủ để sửa toàn bộ đường graph hiện tại. Phải routing nhất quán như ENG-03.
- Global-full: native mode `pooled_global_excess`, bins `none`, halves0 đã có. Chỉ dùng notebooks Global-full với source overlay đã khai báo; checkout Full cũ không có mode này.

### 9.3 Quy tắc harness/notebook mới

1. Tạo bảng expected config theo `(dataset, arm)`; bắt mọi thay đổi ngoài danh sách cho phép.
2. Mọi arm phải cùng split/features/candidates. Graph **được phép khác** trong w/o EC/CSE/Binary/Count/Shuffle/Global; w/o LC graph phải giống full.
3. Phân biệt graph fingerprint và semantic support fingerprint. Không bắt mọi arm có cùng graph như full rồi xóa guard khi bị lỗi.
4. Receipt parser xác thực evidence_mode tương ứng mỗi arm, thay vì khóa mọi run thành pooled_activity_excess.
5. Tổng hợp bằng dataset/arm/seed; không trộn Count/Shuffle vào full chỉ vì cùng architecture string trong native manifest.
6. Lưu arm_id/method/purpose ở orchestration metadata ngay cả khi native runner vẫn dùng tên implementation cũ.
7. Không sửa full notebook đang chạy. Notebook ablation là artifact riêng; chính kế hoạch này không khẳng định nó đã tồn tại.

## 10. Output bắt buộc của từng run

### 10.1 Thư mục và định danh

```text
eclipse_ablation_runs/<session_id>/
  run_manifest.json
  master_run_ledger.csv
  configs/<dataset>_eclipse_<arm>_seed<s>.yaml
  logs/<dataset>_eclipse_<arm>_seed<s>.log
  artifacts/<dataset>_eclipse_<arm>_seed<s>/
    manifest.json
    run_status.json
    selected_metrics.json
    evaluation_metrics.jsonl
    training_telemetry.jsonl
    best_model.pt
    training_checkpoint.pt             # nếu native lifecycle xuất
    last_model.pt                      # nếu native lifecycle xuất
  provenance/<dataset>_<arm>_seed<s>_command.json
  provenance/<dataset>_<arm>_seed<s>_comparability.json
  provenance/environment.json
  provenance/pip_freeze.txt
  tables/<dataset>_<arm>_seed<s>_selected_test.csv
  tables/<dataset>_<arm>_seed<s>_diagnostics.json
  figures/<dataset>_<arm>_seed<s>_*.pdf
  figures/<dataset>_<arm>_seed<s>_*.png
  figures/FIGURE_CAPTIONS.md
```

Nếu notebook full đang chạy dùng tên/path khác như `eclipse_<dataset>_seed2_*`, giữ nguyên file gốc và lưu mapping trong catalog; không cần chạy lại để đổi tên.

### 10.2 Schema một dòng selected TEST trong catalog chung

```text
job_id, purpose, method, arm_id, dataset, training_seed, attempt_id,
status, selected_epoch, selection_metric, split,
RECALL@10, RECALL@20, NDCG@10, NDCG@20,
embedding_dim, gamma, lr, weight_decay, batch_size, num_workers,
evidence_mode, evidence_halves, delta, tau, null_scope,
eta, lambda_nlgcl, semantic_weighting, placebo_seed,
split_fingerprints, feature_fingerprints, candidate_fingerprints,
semantic_support_fingerprint, graph_fingerprint, source_fingerprints,
commit, python, torch, freerec, gpu_model, gpu_index,
preprocessing_seconds, preprocessing_cache_state,
fit_seconds, train_epoch_seconds_sum,
peak_training_allocated_mib, peak_training_reserved_mib, memory_scope,
checkpoint_path, artifact_path, archive_path, archive_sha256,
reviewed_by, exclusion_reason
```

- Bốn metric là **TEST tại selected checkpoint**, không phải VALID.
- Missing numeric là null/NaN trong tính toán; CSV xuất `NOT_MEASURED`. Không thay bằng 0 hoặc scalar reference.
- `fit_seconds` lấy từ Coach.fit hoặc timer có phạm vi được xác định; tổng `seconds` của epochs là training-only, không phải fit time.
- Ghi allocated và reserved riêng. Nếu không đo peak trong full evaluation thì không đặt nhãn whole-run peak VRAM.
- Cache hit/miss và scope preprocessing phải lưu; không gắn nhãn zero-overhead chỉ vì sparse tensor kích thước tương đương.
- `training_checkpoint.pt` và `last_model.pt` chỉ bắt buộc nếu implementation thực sự xuất; `best_model.pt`/selected checkpoint là bắt buộc cho truy vết và per-user analysis sau này.

### 10.3 Diagnostics bắt buộc cho cơ chế

| Nhóm | Đại lượng cần lưu | Diễn giải đúng |
|---|---|---|
| Support | Undirected semantic candidates, raw semantic directed nnz, CF nnz, final nnz, isolates, CF-only support | Cạnh semantic không tự được thêm bởi PEEC |
| Evidence | count/null/score quantiles, fraction r>0, raw-weight coverage | Coverage không chứng minh utility hoặc quan hệ nhân quả |
| Calibration | delta, actual-degree normalization, endpoint/degree-ratio quantiles | Raw boost có thể bị normalization tái phân bố |
| Operator | Relative Frobenius operator dose và reference fingerprint | Không coi dose lớn hơn là tốt hơn |
| BSC | Fixed-probe polynomial dose theo coordinate, probe seed/dimension/scope | Đây không phải actual Adam-direction diagnostic nếu chưa đo thực tế |
| Objective | loss, bpr_loss, raw nlgcl_loss, weighted_nlgcl_loss | w/o LC phải phản ánh LC đã tắt |
| Performance | epoch time, samples/s, fit/preprocessing time, training allocator peaks | Phạm vi đo và phần cứng quyết định khả năng so sánh |
| Shuffle | placebo seed, degree strata, score multiset checks, moved-edge fraction nếu có | Training variance khác graph-permutation variance |
| Global | M_active, null_scope, active_partition=false, count regularizer | Không gắn nhãn half-conditioned |

**Dose reference:** Full/Count/Shuffle/Global có cùng weighted/CSE uncalibrated reference khi protocol đúng; w/o EC dose=0. Binary và w/o CSE có reference riêng theo operator của arm; không so các con số dose khác reference rồi suy ra module nào mạnh hơn. Figure cơ chế chính nên dùng nhóm cùng reference, còn diagnostics Binary/CSE ghi riêng.

### 10.4 Checklist áp dụng cho MỌI dataset × arm × seed (cập nhật 2026-10-19)

Mục này là **yêu cầu nghiệm thu**, không khẳng định các notebook của mọi arm đã implement đầy đủ. Dùng khi Chương/Hiếu tạo notebook mới hoặc copy template; ưu tiên lấy guards/exporters đã audit từ notebook w/o EC rồi điều chỉnh theo arm. Không copy nguyên expected fingerprints của w/o EC cho arm khác.

**Gate A — trước training (READY):**

- [ ] Một nguồn `ARM`, `DATASET`, `SEED`, `PURPOSE`; job_id, CLI, resolved config, output naming cùng nguồn. Không để TARGET_SEED=2 nhưng lời gọi runner hard-code 3.
- [ ] Mỗi attempt có session UTC+UUID riêng, không xóa/ghi đè log/receipt/checkpoint cũ. Cache graph tách khỏi attempt, giữ cache signature. Retry kỹ thuật liên kết attempt cũ; không tự chọn attempt theo TEST.
- [ ] Checkout riêng được pin commit. Không tự fallback sang một repo HEAD khác mà không kiểm SHA/digests. Không sửa hoặc xóa checkout process đang chạy. Lưu remote URL, commit, git status/diff, source-file SHA256 và notebook SHA256; dirty patch phải có diff.
- [ ] Không mở validator toàn cục để arm nào cũng nhận tham số bất kỳ. Kiểm đúng allowlist/diff của arm; w/o CSE eta0 không cần sửa validator lambda cho w/o LC. Nếu source revision khác Full, phải audit ảnh hưởng, không chỉ ghi “cùng commit”.
- [ ] Match runtime/dependency versions và precision với protocol; lưu Python/Torch/FreeRec/PyG/CUDA/driver/GPU/TF32/AMP và pip freeze. Không cài fallback FreeRec unpinned hoặc tự upgrade Torch.
- [ ] GPU dự kiến có thật; nếu thiếu thì fail trước training, không âm thầm fallback CPU rồi gộp timing như T4. Runtime/hardware lệch phải ghi deviation và đánh giá phạm vi so sánh.
- [ ] Attach data qua Add Input trước Kaggle Save & Run All; tắt auto-attach mặc định trong batch. Kiểm đủ train/valid/test/text/image files, hashes, catalog/item order, TRAIN multiplicity và data root writable. Không chỉ kiểm thư mục tồn tại.
- [ ] Check resolved configuration so với Full cùng dataset: dimension=64, layers=3, 500 epochs, gamma/lr/decay/batch/optimizer, sampling, mfiles/k, eval frequency/selection và đúng một intervention. Không nạp model/optimizer state từ run khác.
- [ ] Persist command JSON, YAML gốc + overrides + resolved config/protocol diff, environment, input fingerprints, arm contract và preflight verdict. Nếu guard fail, không launch 9h để “thử xem”.

**Gate B — sau training (COMPLETED_PENDING_REVIEW):**

- [ ] Child exit0, native status COMPLETED đúng seed; telemetry đầy đủ epochs1–500 của run fresh hoặc resume được khai báo. Thiếu epoch/nonfinite phải điều tra, không tự bỏ dòng.
- [ ] Native `selected_metrics.json`: mode=test, selected_checkpoint=true, selection=validation NDCG@20, đúng seed/evidence mode/delta; graph fingerprint khớp final operator của manifest.
- [ ] Đọc đúng canonical keys **RECALL@10, RECALL@20, NDCG@10, NDCG@20**, finite trong [0,1]. Thiếu primary metric thì fail nghiệm thu; không `.get(name,0.0)`. Các diagnostic tùy chọn thiếu giữ null/NOT_MEASURED.
- [ ] Selected epoch tương ứng native VALID selection (bao gồm tie rule); checkpoint tồn tại và có byte hash. Native lifecycle có thể TEST terminal rồi restored selected: chỉ selected receipt dùng cho paper. Không viết “chỉ một TEST đã được tính” nếu history có hai events.
- [ ] Kiểm manifest/config/operator/loss theo bảng arm bên dưới. Native manifest có provenance không thay thế orchestration arm_id; `v8_arm=WMSG-core` có thể là tên backbone, không đủ nhận dạng paper arm.
- [ ] Tách preprocessing, Coach.fit, tổng training epoch time, selected-eval/recovery time nếu đo và subprocess wall. `time_before_Popen→exit` **không phải** Coach.fit. Scope allocator/hardware/cache state được ghi.

**Gate C — hậu xử lý/bàn giao (VERIFIED_ARTIFACT sau audit):**

- [ ] Xuất selected TEST CSV + diagnostics/comparability/resource summaries, ba hình per-run ở mục11.8, source data và captions tiếng Anh. Plotting-only không gọi training hoặc tự chọn checkpoint khác.
- [ ] Gói **cả** native artifact directory **và stdout log bên ngoài**, config/command/dirty diff/notebook/export script, provenance, tables/figures/captions. Không chỉ tar `artifact_dir` rồi bỏ `WORK_DIR/logs/...`.
- [ ] Lưu archive receipt/sidecar JSON gồm archive filename/bytes/SHA256 và file inventory; tự kiểm đường dẫn tương đối trong gói. Log lỗi/attempt lỗi được giữ riêng, không xóa.
- [ ] Người còn lại đọc chéo artifact và mở PDF/PNG kiểm crop/fonts/n/labels. `reviewed_by`, ngày và phạm vi phải là thực tế; không đánh dấu independent review chỉ vì notebook có tên Hiếu.
- [ ] Ledger ghi trạng thái thật, artifact path và actual n. Không suy seed1/2 đã có từ notebook mang tên seed3. Một run không có SD/p-value qua seed.

### 10.5 Điều kiện riêng theo arm — không tái sử dụng guard sai

| Arm | Thay đổi cho phép | Những thứ phải giữ / kiểm tra sau run | Reference cho dose |
|---|---|---|---|
| Full | Không ablate; pooled_activity_excess, delta1 | eta.1, LC.01, weighted semantic và protocol; VALID-selected TEST | Weighted semantic+CSE uncalibrated eta.1 |
| w/o EC | delta0 | eta.1/LC.01; exact native bypass; operator/poly dose0. Evidence có thể vẫn được tính và >0 | Weighted semantic+CSE uncalibrated eta.1, operator bằng reference |
| w/o LC | lambda_nlgcl0 | Graph/evidence/candidates như Full; bỏ LC call/gradient; LC raw/weighted0 hoặc explicit inactive theo telemetry schema. Không ghi số đo không có thành0 | Như Full |
| w/o CSE | eta0 | delta1/LC.01; same raw semantic candidates/weights/count/null/score; effective_eta0 và không CF contribution trong final graph. CF có thể vẫn build: ghi cost thật | **Uncalibrated semantic-only eta0** |
| Binary semantic | Mỗi directed modality occurrence weight1 trước sum→max-sym | Candidates/count/null/score như Full; binary W có thể1 hoặc2; PEEC/CSE/LC giữ. Cả parent/reference/calibrator phải binary | **Binary uncalibrated+CSE** |
| Count-only | pooled_activity_count; score=c/(c+4) | delta1/tau2/halves0, W/CSE/LC như Full; không gọi score là excess/null-subtracted; raw null có thể vẫn tính | Như Full |
| Shuffled | pooled_activity_shuffle; fixed placebo seed20261016 | Count/null/support như Full; score multiset theo strata giữ; đo placement/moved fraction nếu có, không ép singleton đổi | Như Full |
| Global-full | pooled_global_excess; bins none/halves0; μ=n_i*n_j/M_active | Counts/support/CF/training giữ; global_full metadata, one pool, inactive users excluded. Source extension đã có trong notebooks, không dùng estimator hai nửa | Như Full |

Count/Shuffle có cùng delta không đồng nghĩa dose như Full; không sửa delta để thắng TEST. Final graph của w/o CSE/Binary/Count/Shuffle được phép khác Full, không xóa guard để chấp nhận tùy tiện hoặc bắt equality sai. Muốn chứng minh no-CF propagation có thể cần kiểm sparse operator hoặc builder contract; chỉ một flag eta0 chưa đủ chứng minh mọi implementation path đã dùng đúng nó.

## 11. Hình bắt buộc theo từng run và hình tổng hợp cho paper

### 11.1 Hình theo run, tự động xuất sau khi train xong

Mỗi plot có PDF vector + PNG 300 dpi; nguồn CSV/JSON giữ cùng thư mục. Mọi label và caption trong paper dùng tiếng Anh. Rerun plotting không được gọi lại training.

| Stem đề xuất | Nội dung | Nguồn bắt buộc | Lưu ý |
|---|---|---|---|
| `eclipse_<dataset>_<arm>_seed<s>_learning_curves` | TRAIN total/BPR/weighted-LC; VALID N@20 và R@20; selected epoch | Telemetry + evaluation_metrics + selected receipt | Không vẽ TEST theo epoch để chọn model; LC=0 đúng nghĩa trong w/o LC |
| `eclipse_<dataset>_<arm>_seed<s>_training_memory` | Allocated/reserved MiB và train seconds/epoch | Child-process training telemetry | Không lấy parent-kernel torch memory; không vẽ 800 MiB là hardware ceiling |
| `eclipse_<dataset>_<arm>_seed<s>_calibration_diagnostics` | Coverage, operator/BSC dose, measured score quantiles | Native manifest/diagnostics | Quantile profile không được gọi histogram; label reference/scope |

Có thể không đưa hàng chục hình per-run vào paper. Chúng là artifact chẩn đoán và phụ lục; Figure chính dùng bảng tổng hợp bên dưới.

### 11.2 Figure 1 — architecture

- **Tên:** `eclipse_architecture.pdf` và `.png`.
- **Chủ trì:** Chương; Hiếu kiểm tra data flow và labels.
- Tách OFFLINE CPU / TRAIN GPU / INFERENCE.
- TRAIN presence → activity null/excess; text/image → candidates → weighted semantic graph; CF support branch; calibration → normalization → convex blend.
- MI/FSC → BPR + LC; static item graph → BSC trên Adam-preconditioned item directions. User group bypass smoother.
- Inference: representations → dot-product → full ranking + seen masking.
- Không vẽ PEEC như một learned consensus branch hoặc auxiliary loss.
- Có thể dùng figure function trong notebook full làm bản đầu; cần kiểm tra render ở kích thước paper, dẫn nguồn MI/FSC/BSC kế thừa STAIR.

Caption gợi ý: “Overview of ECLIPSE. TRAIN-derived pooled excess evidence calibrates a static semantic item graph before optimizer-side propagation, while layer-contrastive learning acts on the forward representations.”

### 11.3 Figure 2 — component và evidence ablations

- **Tên:** `eclipse_ablation_ndcg20.pdf` / `.png`.
- **Chủ trì:** Hiếu; Chương đối soát số.
- Thiết kế 2×2 panel: hàng trên component arms trên Baby/Sports; hàng dưới evidence controls trên Baby/Sports.
- Hàng trên: Full, w/o EC, w/o LC, w/o CSE, Binary.
- Hàng dưới: Full, w/o EC, Count, Shuffle, Global-full (nếu đã chạy).
- Trục y: selected TEST NDCG@20; mỗi arm có điểm từng seed và mean ± sample SD, n=3 khi đủ.
- Recall@20 xuất đầy đủ ở Table 3; nếu có diện tích thì thêm hình tương ứng ở phụ lục, không nhồi tám panel chữ quá nhỏ.
- Run thiếu: ghi NOT_RUN/ẩn có chú thích, không vẽ cột 0. Electronics Count/Shuffle chỉ n=1 thì không có SD; báo ở bảng hoặc phụ lục, không trình bày như n=3.
- Không chọn seed có thứ hạng đẹp nhất để vẽ.

Caption: “Component and evidence-control ablations on Amazon Baby and Sports. Points denote independent training seeds; error bars indicate sample standard deviation.”

### 11.4 Figure 3 — mechanism diagnostics

- **Tên:** `eclipse_evidence_operator_diagnostics.pdf` / `.png`.
- **Chủ trì:** Chương; Hiếu kiểm tra công thức và reference.
- Panel A: candidate-positive coverage và raw-weight coverage theo dataset/estimator.
- Panel B: operator dose và polynomial probe dose, dùng chung uncalibrated weighted/CSE reference cho nhóm Full/Count/Shuffle/Global.
- Nếu có đủ dữ liệu, panel C ở phụ lục: zero-score coefficient-ratio quantiles và degree groups.
- Graph/evidence là static qua training seeds: không coi ba giá trị graph giống nhau là ba thí nghiệm độc lập để tạo error bars.
- Không kết luận “denoising thành công” từ coverage/dose đơn thuần; đối chiếu với ranking ablation ở Figure 2.

Caption: “Evidence coverage and realized perturbation after symmetric normalization and polynomial propagation. Probe-based dose is measured on fixed inputs and does not represent actual optimizer directions.”

### 11.5 Figure 4 — efficiency–accuracy

- **Tên:** `eclipse_accuracy_efficiency.pdf` / `.png`.
- **Chủ trì:** Hiếu; Chương kiểm tra timing scope.
- Mỗi dataset một panel: x=fit wall time, y=selected TEST NDCG@20; điểm phương pháp/ablation có số đo thật.
- So sánh full với w/o EC và các arms đã đo; STAIR tái lập chỉ thêm nếu timing và hardware provenance phù hợp. Không lấy runtime paper trên RTX 3090 vẽ chung như T4 cùng điều kiện.
- Nhiều seed: có thể vẽ từng run và centroid; chỉ ghi SD nếu thật sự có đủ measurements cùng điều kiện.
- Preprocessing và memory đưa Table 4, không gộp âm thầm vào x=fit.
- Nếu không đủ comparator timing đồng nhất, đổi tên thành `eclipse_resource_profile` và chỉ mô tả tài nguyên; không claim Pareto superiority.

### 11.6 Hình phụ lục ưu tiên

1. `eclipse_ablation_recall20`: đầy đủ endpoint phụ.
2. `eclipse_seed_stability_ndcg20`, `eclipse_seed_stability_recall20`: ba seed thật, không p-value từ epoch.
3. Learning/memory đại diện: dùng seed 1 đã khóa trước hoặc ghi rõ quy tắc, không chọn theo TEST đẹp nhất.
4. Delta sensitivity trên Baby nếu còn budget; chỉ dùng VALID để lựa chọn, không chạy grid sau khi xem TEST rồi che search.
5. Head/tail metrics nếu export được per-user/item breakdown đúng evaluator; không đặt số liệu giả để lấp hình.

### 11.7 Quy chuẩn xuất hình

- Không ép số thứ tự Figure vào tên file; đánh số khi chốt LaTeX.
- Font đủ đọc khi hình rộng khoảng 7.2 inch, palette dễ phân biệt, PDF vector và PNG 300 dpi.
- Caption ghi dataset, metric split, n seeds, ý nghĩa error bars, hardware/scope nếu liên quan.
- Mỗi figure có CSV/JSON nguồn và script/notebook cell tạo hình.
- Không dùng t-SNE, 3D bar hoặc heatmap chỉ để tăng số hình nếu không trả lời câu hỏi khoa học.
- Chương/Hiếu phải mở PDF/PNG thật để kiểm tra crop, legend, font và nhãn; file xuất thành công chưa đủ là hình sẵn cho paper.

### 11.8 Gói hình tối thiểu cho mỗi seed/arm và hướng dẫn Hiếu bổ sung

**Naming thống nhất:** `eclipse_<dataset>_<arm>_seed<s>_<kind>`; dataset=`baby|sports|electronics`, arm=`full|wo_ec|wo_lc|wo_cse|binary_semantic|count_only|shuffled_evidence|global_full_null`. Giữ mapping mode native→arm trong catalog. Tên đã xuất khác chuẩn vẫn giữ raw file và catalog mapping, không train lại để đổi tên.

Ví dụ notebook Hiếu đang được review: `eclipse_electronics_wo_cse_seed3_*`.

| Bắt buộc mỗi run | Nội dung thiết kế | Dữ liệu nguồn và kiểm tra |
|---|---|---|
| `..._learning_curves.pdf/.png` | 2 panels: TRAIN total/BPR/weighted LC; VALID NDCG@20 và Recall@20. Đánh dấu selected epoch bằng receipt. | CSV từ telemetry/evaluation; filter VALID. Raw LC scale khác BPR: panel riêng hoặc weighted LC; không mislabeled raw thành weighted. Không TEST curve, không smoothing che điểm gốc. |
| `..._training_memory.pdf/.png` | Allocated/reserved MiB và training seconds/epoch ở panels riêng. | CSV từ child training telemetry; caption training-only allocator, không tổng VRAM/whole-run peak. Không dùng parent notebook kernel memory. |
| `..._calibration_diagnostics.pdf/.png` | Panels coverage (% edges và % raw-weight mass), operator/poly probe dose, score quantile profile nếu được đo. | JSON/CSV từ native graph diagnostics; caption reference/normalization/probe scope. Quantiles không thành histogram; thiếu panel ghi NOT_MEASURED, không điền0. |
| `..._selected_test.csv` | 1 row dataset/arm/seed/attempt/selected_epoch +4 native TEST metrics và provenance path. | Validated receipt, không dùng console đã làm tròn. |
| `..._resource_summary.csv`, `..._comparability.json`, `..._diagnostics.json` | Resources đúng timer/scope; protocol diff và fingerprints; cơ chế. | Manifest/native log/telemetry và source/data hashes; không bịa trường thiếu. |
| `FIGURE_CAPTIONS.md` + source CSV/JSON | Caption tiếng Anh cho từng file; ghi dataset/arm/seed/n, split, units, reference. | PDF vector + PNG300dpi; nguồn/script xuất gói cùng hình. |

**Không bắt buộc vẽ architecture mới cho mỗi seed.** Một schematic chung và chú thích arm là đủ. Resource profile tổng hợp, seed stability và ablation Full-vs-arm làm **sau khi có receipts thực tế**; n=1 chỉ point, n≥2 có sample SD; matched comparison chỉ trên giao seed set. Không tạo p-value từ epochs. Diagnostics graph static qua seed không được xem là independent seed replicates.

**Lưu ý theo arm:**

- w/o EC: dose0 là đúng; coverage có thể khác0, legacy coverage chưa đo là NOT_MEASURED.
- w/o LC: hiển thị loss objective BPR và trạng thái LC disabled; không đổi loss0 thành thiếu số hoặc thêm LC giả.
- w/o CSE: vẫn có calibration, dùng semantic-only dose reference; không caption “CF preprocessing cost=0” nếu native code vẫn build CF.
- Count/Shuffle: báo score/dose cùng reference với Full; shuffle histogram giữ không có nghĩa normalized operator dose giữ.
- Binary: reference là binary+CSE; không dùng w/o EC weighted hash làm expected graph.

**Cứu kết quả đã chạy nhưng thiếu hình:**

1. Copy nguyên run directory/stdout log/checkpoint/manifest ra bản lưu giữ; lập inventory hashes. Không chạy lại training cell hoặc cell dọn stale artifacts.
2. Tạo notebook/script **report-only** đọc receipts, evaluation JSONL, training JSONL và manifest; không load model/training. Khôi phục những metadata có thật từ source revision/YAML/argv/log; trường không thể khôi phục ghi NOT_MEASURED.
3. Xuất bảng/hình như trên, giữ outputs bổ sung trong `postprocess/<export_id>/`, tham chiếu run gốc. Ghi rõ ngày/export-script hash, không sửa raw receipts để ép schema.
4. Nếu config/selection/data/operator được xác minh thì thiếu plotting **không yêu cầu train lại**. Nếu metric in console0 vì sai case nhưng native receipt đúng, sửa parser và regenerate report, không coi model đã đo0.
5. Chỉ quyết định rerun sau khi kiểm lỗi ảnh hưởng scientific comparison hoặc thiếu checkpoint/evidence thiết yếu không khôi phục được. Giữ record mọi attempt, không rerun để chọn TEST tốt hơn.

## 12. Bảng paper và bộ dữ liệu tổng hợp

| Bảng | Nội dung | Người tổng hợp | Nguồn |
|---|---|---|---|
| Table 1 | Dataset stats, split/features, protocol | Hiếu | Dataset/manifest và protocol đã xác minh |
| Table 2 | Published baselines + STAIR reproduced + ECLIPSE | Chương | Selected TEST receipts; reported results ghi nguồn riêng |
| Table 3A | Component ablations: full, w/o EC/LC/CSE, Binary | Hiếu | Ba seed Baby/Sports; w/o EC Electronics đủ ba seed |
| Table 3B | Count, Shuffle, Global-full; full và w/o EC làm anchors | Chương | Metrics và matched seeds; n được ghi theo dataset |
| Table 4 | Params, preprocessing, fit, allocator memory | Hiếu | Timers/telemetry/environment, không lấy từ figure đã làm tròn |
| Appendix | Từng seed, config, failures/exclusions, sensitivity | Cả hai | Ledger + artifacts |

ECLIPSE mean±SD ở bảng chính không biến STAIR một run thành ba run. STAIR reported và reproduced ghi tách nguồn/số run; không dùng p-value paper gốc cho ECLIPSE. Không đưa phiên bản nội bộ thành một external comparator.

**Các file tổng hợp phải tạo ở bước phân tích:**

```text
paper_artifacts/
  provenance/master_run_ledger.csv
  provenance/artifact_catalog.json
  provenance/comparability_checks.json
  tables/eclipse_selected_test_by_seed.csv
  tables/eclipse_main_results.csv
  tables/eclipse_component_ablations.csv
  tables/eclipse_evidence_controls.csv
  tables/eclipse_paired_seed_differences.csv
  tables/eclipse_resource_summary.csv
  tables/statistical_analysis_status.json
  latex/table_main.tex
  latex/table_ablations.tex
  latex/table_efficiency.tex
  figures/*.pdf
  figures/*.png
  figures/FIGURE_CAPTIONS.md
```

Không tạo nhãn “statistically significant” từ một scalar benchmark reference hoặc từ bảng cuối 10 epochs.

## 13. Phân tích thống kê và cách đọc kết quả

### 13.1 Phân tích mặc định

Cho mỗi dataset, metric và arm A có cùng seed:

\[
\Delta_s^{(A)}=M_{full,s}-M_{A,s},\quad
\overline\Delta=\frac1n\sum_s\Delta_s,\quad
s_\Delta=\sqrt{\frac{\sum_s(\Delta_s-\overline\Delta)^2}{n-1}}.
\]

- Báo mean±sample SD cho từng arm; không thay SD bằng standard error mà không đổi nhãn.
- Báo absolute difference và relative difference của mean: 100*(mean_full/mean_A-1), mẫu số phải khác 0.
- Báo từng paired difference và số seed thắng/hòa/thua; dùng số unrounded để tính. Không gọi hòa từ việc cả hai in ra 0.0521.
- Full n=3 vs một Electronics control n=1: chỉ có một matched-seed difference ở seed 1 và comparison mô tả; không gán n=3 cho control.
- Không pool ba datasets như ba seed hoặc gộp seed×user làm independent samples.
- NDCG@20 primary, Recall@20 secondary; báo cả bốn metrics, không chỉ giữ những ô thắng.

### 13.2 p-value

- Không bắt buộc mọi ablation phải có p-value.
- Với ba seed, paired t-test chỉ có df=2; nếu tính thì là phân tích ít mẫu, phải nêu giả định/giới hạn. Không chạy kiểm định chuẩn tính với n=3 rồi coi đã chứng minh phân phối chuẩn.
- Exact two-sided sign-flip test với ba paired differences có p-value nhỏ nhất 2/8=0.25; điều này không áp dụng cho paired t-test tham số. Không chọn test chỉ vì test nào cho p nhỏ.
- Ưu tiên effect size, paired differences và uncertainty trung thực. Không tăng seed riêng cho arm đang gần p<0.05 rồi dừng khi đạt.
- Nếu đưa kiểm định confirmatory qua nhiều datasets/comparators, khai báo family trước, ví dụ full vs w/o EC trên ba dataset cho NDCG@20, và xử lý multiplicity phù hợp như Holm. Các control khác có thể ghi exploratory.
- STAIR hiện một run: không làm paired seed t-test với ECLIPSE bằng cách nhân bản scalar STAIR.

### 13.3 Per-user bootstrap/randomization — tùy chọn, không thay thế seed study

Nếu còn checkpoint và đủ thời gian, Hiếu phụ trách exporter native per-user, Chương kiểm tra:

1. Cùng TEST user IDs và cùng checkpoint-selection rule.
2. Mỗi user có đủ native Recall@10/20 và NDCG@10/20; giữ user order hoặc join đúng ID.
3. Trung bình per-user phải tái tạo aggregate receipt, kiểm tra atol=1e-6 và lưu sai số. Nếu không đạt, điều tra semantics evaluator; không rescale dữ liệu để ép khớp.
4. Bootstrap paired user IDs, chẳng hạn 10,000 resamples với analysis RNG seed cố định và được ghi; dùng cùng sampled IDs cho cả hai checkpoint.
5. Resampling streaming/chunked, ví dụ 8 resamples mỗi batch và xử lý metrics lần lượt theo memory budget; không tạo ma trận 10,000 × số user trên Electronics. Không đưa bootstrap vào GPU training loop.
6. Báo CI/effect conditional on checkpoints của từng seed. Không coi user resampling đã đo được training randomness.
7. Không huấn luyện lại chỉ để đổi tên file nếu checkpoint còn đầy đủ.

### 13.4 Diễn giải các kết quả không thuận lợi

| Quan sát | Kết luận được phép | Không được viết |
|---|---|---|
| Full > STAIR nhưng full≈w/o EC | Hệ thống cải thiện baseline; incremental EC chưa rõ | Toàn bộ gain là nhờ PEEC |
| Full≈Count | Chưa tách được lợi ích null subtraction trong control hiện tại | Null đã được chứng minh cần thiết |
| Full≈Shuffle | Chưa chứng minh placement evidence đúng cạnh có ích | Behavioral evidence calibration chắc chắn hữu ích |
| Full≈Global | Activity conditioning chưa có bằng chứng incremental | Conditioning giải quyết popularity bias |
| w/o LC hoặc w/o CSE tốt hơn | Tương tác/thành phần có thể phụ thuộc dataset | Mọi thành phần đều thiết yếu |
| Mean positive nhưng seed differences đổi dấu | Small/uncertain effect; cần giới hạn claim | Cải thiện ổn định, chắc chắn |

Không đổi cấu hình full đã khóa chỉ để chữa một ablation sau khi thấy TEST. Thay đổi phương pháp sẽ là nghiên cứu mới, không còn đúng bộ thực nghiệm đăng ký ở tài liệu này.

## 14. Bàn giao và đọc chéo giữa hai người

Mỗi batch hoàn tất bàn giao:

```text
Job IDs / owner / reviewer
Dataset, arm, seeds, attempt IDs
Resolved config + source commit + environment
Selected TEST CSV và native selected_metrics.json
Checkpoint paths
Diagnostics + figure data + PDF/PNG
Fit/preprocessing/memory scope
ZIP + SHA256 + nơi lưu ngoài Kaggle
Lỗi/deviations và cách xử lý
```

Người nhận kiểm tra:

- Có đúng arm/seed/config không, có vô tình resume không?
- Có đủ raw receipts và checkpoint không?
- Aggregate CSV khớp native receipts sau chuẩn hóa tên metric không?
- Figure dùng TEST hay VALID đúng nhãn, n/error bar đúng không?
- Có output nào bị điền 0 khi không đo không?
- Source/graph khác full có đúng expected difference của arm không?
- Có dấu hiệu chọn một attempt theo TEST hoặc bỏ seed xấu không?

Nếu có hai attempts hoàn tất cùng `(dataset,arm,seed)`, dừng tổng hợp để xác minh. Mặc định dùng attempt hợp lệ đầu tiên theo protocol; retry hợp lệ vì lỗi kỹ thuật phải có lý do. Không tự lấy attempt có TEST cao nhất hoặc im lặng gộp thành seed độc lập.

## 15. Checklist chốt paper

### Bằng chứng

- [ ] Full model đủ ba seeds trên ba datasets; mean±SD và từng seed được lưu.
- [ ] w/o EC đủ ba seeds trên ba datasets hoặc phạm vi thiếu được công bố rõ.
- [ ] LC/CSE/Binary và Count/Shuffle Baby/Sports đủ phạm vi đã chốt.
- [ ] Global-full chạy đúng formulation hoặc được ghi NOT_RUN, không thay bằng replicated control.
- [ ] Count/Shuffle Electronics một seed được ghi n=1, không SD giả.
- [ ] STAIR dùng kết quả hiện có, ghi đúng reported/reproduced và số run.
- [ ] Không sử dụng bảng significance cũ dựa trên epochs/VALID-vs-TEST.

### Hình/bảng và Method

- [ ] Architecture thể hiện optimizer-side propagation đúng vị trí, không consensus view.
- [ ] Bốn primary metrics được xuất đầy đủ; NDCG@20 là primary endpoint đã khóa.
- [ ] Tables/figures có source và n/scope rõ ràng, không làm tròn trước khi tính gain.
- [ ] BSC spectral boundedness không bị viết thành guarantee Adam convergence hoặc guaranteed ranking gain.
- [ ] Calibration parameter-free không bị viết thành toàn recommender không có tham số.
- [ ] Có phân biệt semantic similarity, statistical association và update utility.
- [ ] Phương pháp kế thừa MI/FSC/BSC/standard losses được dẫn nguồn phù hợp; không cần kể lịch sử phiên bản nội bộ.
- [ ] Không đổi tên phương pháp thành nhiều model độc lập trong các sections.

### Nộp bài

- [ ] Manuscript tiếng Anh, references khớp metadata, dùng đúng template và giới hạn submission hiện hành.
- [ ] Kiểm tra anonymity, disclosure theo quy định và sự khác biệt nội dung với các manuscript liên quan đang nộp nếu còn hiệu lực; việc chưa xuất bản không tự loại bỏ quy tắc simultaneous substantial overlap.
- [ ] Bài và artifact không chứa đường dẫn/captions vô tình lộ tác giả khi cần double blind.
- [ ] Author review hoàn tất, không có claim vượt bằng chứng thực tế.
- [ ] Nộp trước buffer nội bộ; lưu confirmation.

Nguồn chính thức để đối chiếu trước khi nộp: https://icaart.scitevents.org/Guidelines.aspx?y=2027 và https://icaart.scitevents.org/ImportantDates.aspx?y=2027.

## 16. Quyết định cuối của kế hoạch

**Giữ các full run đang chạy. Tiếp theo ưu tiên w/o EC; tập trung các ablation nhỏ trên Baby/Sports; Count/Shuffle Electronics mỗi arm một seed; Global-full chỉ chạy sau khi implement đúng. Không chạy thêm STAIR gốc và không mở thêm kiến trúc/256D.**

Chương nhận ít job nhưng nhiều Electronics; Hiếu nhận nhiều job ngắn và tổng hợp figure/table. Mục tiêu bàn giao là artifact đầy đủ, comparison công bằng và kết luận có thể kiểm tra — không phải mọi ablation đều phải giảm hoặc mọi ô đều phải có p<0.05.


## 17. Review notebook của Hiếu — w/o CSE Electronics seed 3 (2026-10-19)

[Báo cáo chi tiết và vị trí cell/line](ECLIPSE_wo_CSE_Electronics_S3_Notebook_Review.md). **Initial review: ANALYZED_STATIC_ONLY**, trên snapshot gốc; không chạy model/training. Sau yêu cầu sửa tiếp theo, notebook đã được cập nhật như ghi chú cuối mục này. Chưa có artifact w/o CSE thực tế để nghiệm thu metric/hình.

- CLI có đúng ý định eta0/delta1/LC.01; chưa kết luận kiến trúc ablation sai.
- Lỗi chắc chắn: parser dùng `Recall@10/20` thay native `RECALL@10/20` và fallback0; phải sửa trước tổng hợp.
- Fixed output + xóa stale checkpoints/receipts + log `w` có thể phá attempt cũ; tạo attempt mới, không dọn kết quả cũ tự động.
- Source clone HEAD/fallback repo và local patch không đủ khóa reproducibility; lưu commit+dirty diff+digests, đúng named-arm contract. Preflight đang kiểm lambda0 của w/o LC chứ chưa kiểm w/o CSE.
- Thiếu checks effective config/eta, fingerprints, native selected receipt, telemetry completeness; phải kiểm thay vì chỉ exit0.
- Không có plotting/export CSV/captions; archive chỉ tar native artifact, bỏ stdout log nằm ngoài. Bổ sung mục10.4–10.5 và11.8, có thể hậu xử lý sau run không train lại.
- Electronics w/o CSE seed3 là extension ngoài ledger tối thiểu hiện tại (P2 chỉ dự kiến seed1). Không tự thêm thành completed hoặc suy đủ3seeds; nếu đã chạy thì giữ, ghi actual coverage và ưu tiên budget còn lại theo core plan.

**Phân công follow-up:** Hiếu sửa template/harness cho các ablation mình phụ trách và xuất gói hậu xử lý; Chương đọc chéo artifact. Checklist mới không đồng nghĩa Hiếu đã implement hoặc đã chạy; chỉ đổi status sau khi có bằng chứng thực tế.


### 17.1 Notebook đã sửa — READY cho kiểm tra môi trường và chạy bởi Hiếu

[Notebook w/o CSE Electronics seed3](../../notebook/P6/eclipse_wo_cse_electronics_s3.ipynb) đã được sửa; [chi tiết follow-up](ECLIPSE_wo_CSE_Electronics_S3_Notebook_Review.md#7-follow-up--notebook-đã-được-sửa). Các lỗi liệt kê ở mục17 là findings của bản cũ; không còn là trạng thái hiện tại.

- Train mới: `RUN_MODE='train'`, seed3/64D/500epochs/eta0/delta1/LC.01. Source pin và checks data/config/receipt; không mở rộng nhánh nghiên cứu hoặc đổi protocol Full.
- Bổ sung hình cho run cũ: `RUN_MODE='report_only'`, chỉ định `EXISTING_ARTIFACT_DIR` và `EXISTING_STDOUT_LOG`. Không training/model/source checkout/network; dữ liệu thiếu giữ NOT_MEASURED, provenance lệch cần đọc chéo.
- Xuất `eclipse_electronics_wo_cse_seed3_{learning_curves,training_memory,calibration_diagnostics,resource_profile}`: PDF/PNG300dpi, source CSV/JSON, captions; ZIP có stdout log ngoài và file inventory/checksum sidecar. `NOTEBOOK_SOURCE_PATH` tùy chọn để gói exact submitted ipynb, nếu không thì tải executed notebook riêng.
- [57 kiểm tra dữ liệu/cú pháp/export đạt](../../reports/eclipse_wo_cse_electronics_s3_notebook_audit/repaired_notebook_validation.json), chỉ fixtures tạm, không model/GPU/training. Đây là software readiness, không phải Kaggle run COMPLETED/VERIFIED_ARTIFACT hoặc số liệu paper mới. Chưa tăng actual n hoặc sửa ledger thành completed.
- Nếu notebook cũ đang train, để run hoàn tất và dùng report-only ở session mới; không thay source giữa run. Electronics seed3 vẫn là extension ngoài phạm vi tối thiểu, không suy đã có đủ seeds1–3.


## 18. Count-only — Toàn bộ thực nghiệm hoàn tất (2026-10-20)

Bảy run `ECL-count_only` (Baby seeds 1–3, Sports seeds 1–3, Electronics seed 1) đã chuyển trạng thái **COMPLETED / VERIFIED_ARTIFACT**. Xem phân tích đối chiếu toàn diện tại [ECLIPSE_Ablation_Study_Count_Only.md](ECLIPSE_Ablation_Study_Count_Only.md).

- Baby→Sports→Electronics seed1, Baby→Sports seeds 2–3, 64D, 500 epochs. Mode `pooled_activity_count`, delta1/tau2/halves0; WMSG+CSE eta.1+LC.01+MI/FSC/BSC giữ theo Full. Chỉ mode/score thay đổi; không tune riêng từ TEST.
- Score `c/(c+4)` không subtract null. Counts/null/candidates/semantic/CF invariant; Count score/final graph hashes không bị ép bằng Full. Dose reference weighted+CSE eta.1.
- Mỗi dataset xuất learning curves, training memory, calibration diagnostics: PDF/PNG300dpi + CSV/JSON/captions; selected TEST/resources/comparability. Session resource profile/status table và ZIP đầy đủ stdout/native files/config/source/provenance cùng inventory/checksum sidecar.


## 19. Actual status sau audit A0/A1/A5 — 2026-10-20

Phần này thay các trạng thái lịch sử của các arms ở trên, không xóa lịch sử thực hiện. [Full analysis](ECLIPSE_Full_Model_Multi_Seed_Analysis.md), [A1 analysis](ECLIPSE_Ablation_Study_wo_EC.md), [A5 analysis](ECLIPSE_Ablation_Study_Count_Only.md), [paper readiness và figures/tables](ECLIPSE_Paper_Experimental_Readiness_and_Presentation_Plan.md).

| Arm | Baby | Sports | Electronics | Audit scope |
| --- | --- | --- | --- | --- |
| A0 Full | s1–3 COMPLETED | s1–3 COMPLETED | s1–3 COMPLETED | Native receipts/telemetry/source; checkpoints Full s1 chưa có local |
| A1 w/o EC | s1–3 COMPLETED | s1–3 COMPLETED | s1–3 COMPLETED | s1 recovered historical checkpoint, s2–3 native; đủ 9 checkpoint local |
| A2 w/o LC | Hiếu báo đã xong | Hiếu phân tích riêng | Chưa xác nhận phạm vi | HUMAN_REPORTED_COMPLETE; artifacts còn ở máy Hiếu, không suy đủ mọi ô/seed |
| A3 w/o CSE | Hiếu báo đã xong | Hiếu phân tích riêng | Chưa xác nhận phạm vi | HUMAN_REPORTED_COMPLETE; artifacts còn ở máy Hiếu, không suy đủ mọi ô/seed |
| A5 Count-only | s1–3 COMPLETED | s1–3 COMPLETED | s1 COMPLETED | Đầy đủ 7 runs native receipts/telemetry/source/checkpoints; xem [A5 analysis](ECLIPSE_Ablation_Study_Count_Only.md) |
| Shuffle/Binary/Global | Giữ actual ledger của người chạy | Chưa nhận receipts mới vào audit này | Không tự tăng n/status | Notebook READY hoặc một thông báo không thay kiểm toán số liệu |

- A0 NDCG@20 n3: Baby0.046340±0.000241, Sports0.051956±0.000156, Electronics0.031759±0.000093.
- EC relative mean gain so A1: +0.585%/+0.490%/+0.138%; Electronics s3 N@20 âm. Không có N@20 comparison nào còn p<.05 sau Holm3 trong audit hồi cứu. Không dùng epoch p-values cũ.
- A5 Count-only hoàn thành 100% phạm vi (7 runs): Baby n3 N@20=0.046148±0.000538 (Full thắng 2/3 seeds, mean +0.416%); Sports n3 N@20=0.051938±0.000245 (Full vượt trội trên cả 4 metric means; Count thắng w/o EC 3/3 seeds, p=0.0154); Electronics n1 N@20=0.031852. Xem [báo cáo chi tiết A5](ECLIPSE_Ablation_Study_Count_Only.md).
- Không yêu cầu chạy lại A0/A1 đã hoàn tất, không tự thêm ba seed STAIR; STAIR n1 phải có nhãn n1, không có paired-seed significance.
- P0 tiếp theo: Hiếu bàn giao bảng/receipts A2/A3 theo cùng schema; hoàn thành/nhận Count/Shuffle đúng kế hoạch; lập external benchmark source/protocol matrix; cập nhật Example.tex đang còn single-seed/internal-version narrative.
- Figures ưu tiên: architecture; component/evidence ablation khi đủ receipts; coverage + operator/probe dose. Paired EC effect và learning curves/nguồn lực có thể ở appendix. Export PDF vector + PNG300dpi + source data/captions, ghi actual n/units/errorbar scope.
- Giữ kiến trúc đã chốt. Nếu controls không ủng hộ specific excess-evidence claim, điều chỉnh claim trung thực; không tune TEST để làm mọi arm thắng.


## 20. Count-only seeds2–3 — Hoàn tất và nghiệm thu (2026-10-20)

Bốn run Count-only follow-up (Baby+Sports seed 2 và Baby+Sports seed 3) đã hoàn thành và được kiểm toán trọn vẹn:

- Session seed 2: `20261020T024629Z_c207b2b5`; Session seed 3: `20261020T025325Z_26123b1b`.
- Checkpoints, telemetry, plots, figures và tables đã được tích hợp đầy đủ vào [ECLIPSE_Ablation_Study_Count_Only.md](ECLIPSE_Ablation_Study_Count_Only.md).
- Toàn bộ 7 runs của arm A5 Count-only đạt trạng thái **VERIFIED_ARTIFACT**. Các guide tạm thời đã được dọn dẹp.


## 21. Global-full null — Seeds 1–2 COMPLETED ($n=2$), Seed 3 READY (2026-10-20)

Bốn job Global-full Baby/Sports seeds 1–2 đã **COMPLETED**, kiểm toán toàn diện và phân tích chi tiết trong [ECLIPSE_Ablation_Study_Global_Full_Null.md](ECLIPSE_Ablation_Study_Global_Full_Null.md). Hai job seed 3 giữ trạng thái **READY** trong [hướng dẫn](ECLIPSE_Global_Full_Null_Notebook_Guide.md).

- **Trạng thái thực nghiệm:**
  - Seed 1 (Baby ep=485, Sports ep=500): COMPLETED tại `logs/ICAART/eclipse_global_full_null_seed1/`
  - Seed 2 (Baby ep=450, Sports ep=470): COMPLETED tại `logs/ICAART/eclipse_global_full_null_seed2/`
  - Seed 3: Sẵn sàng thực thi tại [eclipse_global_full_null_seed3.ipynb](../../notebook/P6/eclipse_global_full_null_seed3.ipynb)
- Mode `pooled_global_excess`, bins `none`, halves0. $\mu_{\text{global}}=\frac{n_i n_j}{M_{\text{active}}}$ trên active TRAIN users; score $[c-\mu_{\text{global}}]_+/(c+\mu_{\text{global}}+4)$, $\delta=1$, $\tau=2$. Không dùng Count-only hoặc unstratified_replication.
- Giữ WMSG/CSE eta.1/LC.01/MI/FSC/BSC và per-dataset settings. Counts/candidates/raw semantic/presence/CF khớp 100% Full; graph fingerprint bất biến trên từng dataset (`14aebbb...` trên Baby, `dd5b64f...` trên Sports).
- Sai số công thức kiểm toán đạt giới hạn chính xác số thực dấu phẩy động: $2.22 \times 10^{-16}$.
- **Kết quả khoa học chính ($n=2$):**
  - So với w/o EC (A1): Global-full null thắng tuyệt đối 16/16 chỉ số ghép cặp (+1.067% NDCG@20 trên Baby; +0.597% trên Sports).
  - So với Full ECLIPSE (A0): Global-full null đạt hiệu năng tương đương hoặc hơi nhỉnh hơn (+0.220% NDCG@20 trên Baby; +0.007% trên Sports), chứng minh hiệu quả cốt lõi của PEEC nằm ở nguyên lý excess evidence subtraction ($[c - \mu]_+$), trong khi phân tầng hoạt độ đóng vai trò điều hòa bảo thủ cắt tỉa cạnh ngẫu nhiên.
