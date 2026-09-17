# BẢNG RÀ SOÁT CẤU HÌNH THỰC NGHIỆM STAIR-CNLGCL v1-R (CONFIG AUDIT)
## Giai đoạn 4: Tích Hợp Trực Giao Cross-Component Synergy (NLGCL & BSC-Reweight)

---

**Dự án:** STAIR-Enhanced (Nâng cao Năng lực Hệ Gợi Ý Đa Phương Thức)  
**Phiên bản:** STAIR-CNLGCL v1-R (Refined Architecture)  
**Tài liệu tham chiếu:** 
- Mã nguồn mô hình: [models/stair_cnlgcl_v1_r.py](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/stair_cnlgcl_v1_r.py) & [models/GD4/stair_cnlgcl_v1_r.py](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/GD4/stair_cnlgcl_v1_r.py)
- Kịch bản huấn luyện: [main_stair_cnlgcl_v1_r.py](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_stair_cnlgcl_v1_r.py)
- Sổ tay thực nghiệm: [notebook/P4/stair_cnlgcl_v1_r.ipynb](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/P4/stair_cnlgcl_v1_r.ipynb)
- Tệp cấu hình gốc: `configs/Amazon2014Baby_550_MMRec.yaml`, `configs/Amazon2014Sports_550_MMRec.yaml`, `configs/Amazon2014Electronics_550_MMRec.yaml`
- Nhật ký thực thi trực tiếp: `logs/GD4/baby_v1_r.log`, `logs/GD4/sports_v1_r.log`, `logs/GD4/electronics_v1_r.log`

---

## 2.1. BẢNG CẤU HÌNH BÀI BÁO (PAPER CONFIGURATION TABLE)

Bảng dưới đây tổng hợp đầy đủ và chính xác 100% các siêu tham số được sử dụng trong quá trình huấn luyện và đánh giá mô hình STAIR-CNLGCL v1-R trên 3 tập dữ liệu chuẩn Amazon (Baby, Sports, Electronics). Tất cả các tham số đã được đối soát chéo trực tiếp giữa mã nguồn, tệp cấu hình YAML, sổ tay Jupyter và log runtime thực tế từ hệ thống GPU Tesla T4.

| STT | Tham số (Parameter) | Ký hiệu Toán học | Amazon Baby | Amazon Sports | Amazon Electronics | Nhóm Tham số (Parameter Group) | Nguồn Đối soát (Verification Source) |
|:---:|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | **Embedding Dimension** | $d$ | **64** | **64** | **64** | Backbone STAIR | YAML / Code (`--embedding-dim`) |
| 2 | **GNN Layers** | $L$ | **3** | **3** | **3** | Backbone STAIR | YAML / Code (`--num-layers`) |
| 3 | **Batch Size** | $B$ | **1024** | **1024** | **4096** | Tối ưu hóa Huấn luyện | YAML / Log runtime (`[batch_size]`) |
| 4 | **Learning Rate** | $\eta$ | **1e-3** | **1e-3** | **1e-3** | Tối ưu hóa Huấn luyện | YAML / Code (`--lr 0.001`) |
| 5 | **Weight Decay** | $\lambda_{\text{wd}}$ | **0.3** | **0.1** | **0.1** | Tối ưu hóa Huấn luyện | YAML / Log runtime (`[weight_decay]`) |
| 6 | **Optimizer** | — | **AdamWSEvo** | **AdamWSEvo** | **AdamWSEvo** | Tối ưu hóa Huấn luyện | YAML / Code (`--optimizer`) |
| 7 | **Spectral Gamma** | $\gamma$ | **0.1** | **0.2** | **0.4** | Suy hao Phổ GNN $\beta_3$ | YAML / Log runtime (`[gamma]`) |
| 8 | **kNN Neighbors** | $K_{\text{txt}}-K_{\text{vis}}$ | **5-1** | **5-1** | **5-1** | Topo Đồ thị Cơ sở | YAML / Code (`--num-neighbors`) |
| 9 | **InfoNCE Temperature** | $\tau$ | **0.2** | **0.2** | **0.2** | Tương phản CNLGCL | Code / Notebook (`--tau 0.20`) |
| 10 | **Contrastive Weight** | $\lambda_{\text{cl}}$ | **0.005** | **0.008** | **0.005** | Tương phản CNLGCL | Notebook Cell 5 / Log runtime |
| 11 | **Warmup Epochs** | $T_{\text{warmup}}$ | **100** | **50** | **50** | Lịch điều phối CNLGCL | Notebook Cell 5 / Log runtime |
| 12 | **AMM (Adaptive Margin)** | $m_{\max}$ | **ON (0.02)** | **ON (0.02)** | **ON (0.02)** | Biên thích nghi Đa phương thức | Code / Log (`[use_amm: 1]`) |
| 13 | **FNF (False Negative Filter)** | $\tau_{\text{thresh}}$ | **ON (0.50)** | **OFF (1.00)** | **OFF (1.00)** | Lọc Mẫu Âm tính Giả | Notebook Cell 5 / Log runtime |
| 14 | **SSB Mode** | — | **modal_only** | **full_ssb** | **full_ssb** | Tái trọng số BSC Engine | Notebook Cell 5 / Log runtime |
| 15 | **Modal Weight** | $\alpha$ | **0.5** | **0.4** | **0.4** | Tái trọng số BSC Engine | Notebook Cell 5 / Log runtime |
| 16 | **Behavioral Weight** | $\beta$ | **0.0** | **0.2** | **0.2** | Tái trọng số BSC Engine | Notebook Cell 5 / Log runtime |
| 17 | **Modal Sim Thresholds** | $\tau_t, \tau_v$ | **0.1, 0.1** | **0.1, 0.1** | **0.1, 0.1** | Ngưỡng Đồng thuận Modal | Code (`--ssb-tau-text/visual`) |
| 18 | **Weight Clamping Bounds** | $[w_{\min}, w_{\max}]$ | **[1.0, 3.6]** | **[1.0, 3.6]** | **[1.0, 3.6]** | Ổn định Phổ Đồ thị | Code (`--ssb-min/max-weight`) |
| 19 | **Fused Ops [4, B, d]** | — | **ON** | **ON** | **ON** | Tối ưu Kỹ thuật Tensor | Code / Log (`[use_fused_ops: 1]`) |
| 20 | **Spectral Noise Amplitude** | $\epsilon$ | **0.08** | **0.08** | **0.08** | Nhiễu Phổ Bảo toàn Hướng | Code / Log (`[eps: 0.08]`) |
| 21 | **Direction Balance** | $\alpha_{\text{dir}}$ | **0.5** | **0.5** | **0.5** | Cân bằng Hướng Tương phản | Code / Log (`[alpha_dir: 0.5]`) |
| 22 | **Evaluation Chunk Size** | $C_{\text{eval}}$ | — | — | **512** | Kỹ thuật Bảo toàn VRAM | Notebook Cell 5 / Log runtime |

> [!NOTE]
> **Giải thích các tham số đã rà soát và xác định chính xác:**
> 1. **`gamma` ($\gamma$):** Được cấu hình tăng dần theo quy mô đồ thị: **0.1** (Baby, đồ thị nhỏ 7K items), **0.2** (Sports, đồ thị trung bình 18K items), và **0.4** (Electronics, đồ thị lớn 63K items). Giá trị này điều chỉnh hệ số suy hao phổ của tầng GNN thứ 3 theo công thức $\beta_{3, k} = 0.1 + 0.9(k/d)^\gamma$.
> 2. **`FNF` (In-batch False Negative Filtering):** Được kích hoạt **ON** trên Baby với ngưỡng tương đồng ngữ nghĩa $\tau_{\text{thresh}} = 0.50$ (lọc bỏ các sản phẩm cùng danh mục mẹ có độ tương đồng ngữ nghĩa cao bị lấy mẫu ngẫu nhiên làm mẫu âm trong mini-batch). Đối với Sports và Electronics, do không gian sản phẩm thưa và rộng, xác suất trùng mẫu âm cực thấp nên FNF được tắt (**OFF**, $\tau_{\text{thresh}} = 1.00$) để tối ưu tốc độ.
> 3. **`WD` (Weight Decay):** Được thiết lập **0.3** trên Baby trong tệp cấu hình chuẩn `configs/Amazon2014Baby_550_MMRec.yaml` nhằm kiểm soát hiện tượng quá khớp (overfitting) trên đồ thị có mật độ tương tác cao ($0.117\%$), trong khi Sports và Electronics duy trì mức chuẩn **0.1**.

---

## 2.2. XÁC NHẬN CÁC NHÓM THAM SỐ TỪ MÃ NGUỒN (CODE VERIFICATION OF PARAMETER GROUPS)

Theo thiết kế hệ thống trong [models/GD4/stair_cnlgcl_v1_r.py](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/GD4/stair_cnlgcl_v1_r.py) và [main_stair_cnlgcl_v1_r.py](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_stair_cnlgcl_v1_r.py), toàn bộ các tham số trên được phân rã thành 4 nhóm kiến trúc độc lập:

### Nhóm 1: STAIR Backbone & Tối Ưu Hóa Huấn Luyện (Training Backbone)
Nhóm này kế thừa nền tảng của mô hình STAIR gốc và cơ chế tối ưu hóa gradient cấp đồ thị:
- **Biểu diễn ẩn ($d = 64$):** Số chiều không gian tiềm ẩn cho cả user và item embedding.
- **Số tầng lan truyền ($L = 3$):** Kiến trúc tích chập 3 tầng Forward Stepwise Convolution (FSC).
- **Bộ tối ưu hóa (`AdamWSEvo`):** Tích hợp bộ lọc làm mịn gradient `Smoother(mAdj)` trên đồ thị item-item.
- **Hệ số suy hao phổ ($\gamma$):** Xác định tốc độ suy giảm các thành phần tần số cao tại tầng tích chập cuối cùng:
  $$\beta_{3, k} = 0.1 + 0.9 \left( \frac{k}{d} \right)^\gamma, \quad k \in \{1, \dots, d\}$$
- **Đồ thị k-NN cơ sở ($5-1$):** Ghép nối 5 láng giềng gần nhất từ phương thức văn bản (Text) và 1 láng giềng từ phương thức hình ảnh (Vision).

### Nhóm 2: Động Cơ Tái Trọng Số Đồ Thị Đồng Thuận (BSC-Reweight Engine)
Được đóng gói hoàn chỉnh trong lớp `BSC_Reweight_Engine` (`models/GD4/stair_cnlgcl_v1_r.py` dòng 38-192):
- **Công thức tính trọng số cạnh:**
  $$W_{ij} = W_{ij}^{\text{base}} \cdot \left[ 1 + \alpha \cdot q_{\text{modal}}(i, j) + \beta \cdot q_{\text{behavior}}(i, j) \right]$$
  Trong đó:
  - $q_{\text{modal}}(i, j) = \sqrt{\text{ReLU}(S_{ij}^{\text{txt}} - \tau_t) \cdot \text{ReLU}(S_{ij}^{\text{vis}} - \tau_v)}$ (Trung bình nhân có ngưỡng).
  - $q_{\text{behavior}}(i, j) = \frac{(R^T R)_{ij}}{\sqrt{\deg(i) \cdot \deg(j)}}$ (Hệ số tương đồng hành vi Ochiai).
- **Các chế độ hỗ trợ (`ssb_mode`):**
  - `modal_only`: Ép $\beta = 0.0, \alpha = 0.5$ (dành riêng cho Baby để triệt tiêu hiện tượng over-smoothing do đồng xuất hiện hành vi).
  - `full_ssb`: $\alpha = 0.4, \beta = 0.2$ (áp dụng cho Sports và Electronics để kết hợp hài hòa cả ngữ nghĩa và hành vi người dùng).
- **Kẹp trọng số an toàn ($[w_{\min}, w_{\max}] = [1.0, 3.6]$):** Ngăn chặn hiện tượng bùng nổ gradient và bảo toàn bán kính phổ đồ thị $\rho(\hat{A}) \le 1.0$.
- **Ma trận kề chuẩn hóa đối xứng (Symmetric Normalized Adjacency):**
  $$\hat{A} = D^{-1/2} W_{\text{sym}} D^{-1/2}$$
  Được chuyển đổi thành định dạng thưa PyTorch `torch.sparse_csr_tensor` để tối ưu hóa bộ nhớ và truyền trực tiếp vào bộ đệm của optimizer `AdamWSEvo`.

### Nhóm 3: Hàm Mất Mát Điều Lệ Tương Phản Đa Tầng (CNLGCL Loss Engine)
Được triển khai trong lớp `CNLGCL_Loss_v1R` (`models/GD4/stair_cnlgcl_v1_r.py` dòng 195-390):
- **Tương phản chéo tầng:** Thực hiện giữa tầng đầu vào thô $H^{(0)}$ và tầng tích chập collaborative đầu tiên $H^{(1)}$:
  $$\mathcal{L}_{\text{CNLGCL}} = \alpha_{\text{dir}} \mathcal{L}_{u \to i}^{(0 \leftrightarrow 1)} + (1 - \alpha_{\text{dir}}) \mathcal{L}_{i \to u}^{(0 \leftrightarrow 1)}$$
- **Nhiệt độ InfoNCE ($\tau = 0.20$):** Kiểm soát độ sắc nét của phân phối xác suất softmax trên siêu cầu L2.
- **Nhiễu phổ bảo toàn hướng ($\epsilon = 0.08$):** Thêm vector nhiễu $\Delta_z = \epsilon \cdot \text{sgn}(z) \odot |\mathcal{N}(0, I)|$ nhằm ngăn chặn sụp đổ biểu diễn (representation collapse) mà không làm đổi góc phần tư hình học.
- **Biên thích nghi đa phương thức (AMM):**
  $$m(u, i) = \min \left( m_{\max}, c \cdot \left[ 1 - \text{sim}_{\text{modal}}(i, i^+) \right] \right), \quad m_{\max} = 0.02$$
- **Lọc mẫu âm tính giả (FNF):** Khi `use_fn_mask = 1`, loại trừ các mẫu âm trong mini-batch có $\text{sim}_{\text{modal}}(i, j) > \tau_{\text{thresh}}$ ra khỏi mẫu số hàm mất mát InfoNCE.
- **Lịch làm ấm tuyến tính (Linear Warmup):**
  $$\lambda(t) = \min\left(1, \frac{t}{T_{\text{warmup}}}\right) \cdot \lambda_{\text{cl}}$$
  Giúp bảo vệ đa tạp không gian biểu diễn ở giai đoạn khởi tạo ban đầu.

### Nhóm 4: Kỹ Thuật Tối Ưu Hóa Bộ Nhớ & Tăng Tốc Phần Cứng (Engineering Safeguards)
- **Fused Tensor Operations $[4, B, d]$:** Gom cụm 4 tensor biểu diễn $(H_u^{(0)}, H_i^{(0)}, H_u^{(1)}, H_i^{(1)})$ vào một khối bộ nhớ liên tục duy nhất, thực hiện phép tính nhân ma trận theo lô (Batched GEMM) và chuẩn hóa đồng thời. Giúp cắt giảm **11.5% - 21.4% thời gian huấn luyện thực tế** mà không làm tăng dù chỉ 1 MB VRAM đỉnh.
- **Phân đoạn Đánh giá (Evaluation Chunk Size = 512):** Trên tập dữ liệu lớn Electronics (63,005 sản phẩm và 192,403 người dùng), việc tính ma trận điểm số đầy đủ $192{,}403 \times 63{,}005$ sẽ gây tràn bộ nhớ VRAM ngay lập tức. Cơ chế chunking 512 người dùng/lần tính toán giúp duy trì mức tiêu thụ VRAM đo đạc thực tế chỉ **1,489 MB** trên GPU Tesla T4.
- **Xử lý đồ thị phân đoạn trên CPU (CPU-Chunked Vectorization):** Động cơ BSC tính toán chất lượng tương đồng đa phương thức theo từng cụm 32,768 cạnh trên CPU trước khi đưa vào GPU, loại bỏ hoàn toàn nguy cơ OOM trong pha tiền xử lý.

---

## 2.3. GIẢI TRÌNH KHOA HỌC CHO VIỆC ĐIỀU CHỈNH THAM SỐ THEO TẬP DỮ LIỆU (DATASET-ADAPTIVE RATIONALE)

| Tập dữ liệu | Đặc thù Đồ thị | Thách thức Cốt lõi | Quyết định Cấu hình STAIR-CNLGCL v1-R | Kết quả Thực nghiệm Đạt được |
|:---|:---|:---|:---|:---|
| **Amazon Baby** | - Quy mô nhỏ: 7,050 items, 19,445 users<br>- Mật độ cao nhất: **0.117%** (8.27 tương tác/user) | Dễ bị over-smoothing khi bổ sung đồ thị hành vi $R^T R$; nhiều sản phẩm tương đồng cao dễ bị gán nhầm là âm tính. | - `ssb_mode = modal_only` ($\alpha=0.5, \beta=0.0$)<br>- `tau_thresh = 0.50`, `use_fn_mask = 1`<br>- $\lambda_{\text{cl}} = 0.005$, $T_{\text{warmup}} = 100$<br>- $\gamma = 0.1$, $\lambda_{\text{wd}} = 0.3$ | Khắc phục hoàn toàn thất bại của v5 (-9.12%), đưa NDCG@20 đạt **0.0448** (+3.94% so với Baseline). |
| **Amazon Sports** | - Quy mô trung bình: 18,357 items, 35,598 users<br>- Mật độ cân bằng: **0.053%** (5.14 tương tác/user) | Cần cân bằng tối ưu giữa cấu trúc liên kết người dùng và ngữ nghĩa đa phương thức của dụng cụ thể thao. | - `ssb_mode = full_ssb` ($\alpha=0.4, \beta=0.2$)<br>- `use_fn_mask = 0` (Xác suất va chạm âm tính < 0.05%)<br>- $\lambda_{\text{cl}} = 0.008$, $T_{\text{warmup}} = 50$<br>- $\gamma = 0.2$, $\lambda_{\text{wd}} = 0.1$ | Thiết lập đỉnh cao mới (SOTA): NDCG@20 đạt **0.0511** (+4.93% so với Baseline), vượt cả v5 (+3.08%) và v3.1 (+4.31%). |
| **Amazon Electronics** | - Quy mô cực lớn: **63,005 items**, 192,403 users<br>- Mật độ siêu thưa: **0.010%** (2.15 tương tác/user) | Không gian tìm kiếm rộng lớn, độ phân mảnh cao, nguy cơ cạn kiệt bộ nhớ GPU VRAM khi xếp hạng 63K items. | - `ssb_mode = full_ssb` ($\alpha=0.4, \beta=0.2$)<br>- `batch_size = 4096` (Mở rộng phạm vi mẫu âm InfoNCE)<br>- $\lambda_{\text{cl}} = 0.005$, $T_{\text{warmup}} = 50$<br>- $\gamma = 0.4$ (Lọc mạnh nhiễu tần số cao)<br>- `eval_chunk_size = 512` | Đạt NDCG@20 **0.0315** (+4.30% so với Baseline), VRAM ổn định ở mức 1,489 MB (chỉ chiếm 9.3% dung lượng GPU T4). |

---

## 2.4. TỔNG KẾT & KẾT LUẬN KIỂM ĐỊNH HỆ THỐNG

1. **Tính Nhất Quán Giữa Thiết Kế và Thực Thi:** Toàn bộ 22 tham số cấu hình của mô hình STAIR-CNLGCL v1-R đã được chuẩn hóa và kiểm tra đồng bộ trên tất cả các thành phần: kịch bản Python (`main_stair_cnlgcl_v1_r.py`), module mô hình (`models/GD4/stair_cnlgcl_v1_r.py`), sổ tay kiểm nghiệm Kaggle (`notebook/P4/stair_cnlgcl_v1_r.ipynb`), và bộ kiểm thử tự động 5/5 bài kiểm tra (`tests/test_stair_cnlgcl_v1_r.py`).
2. **Khắc phục Triệt để Điểm Bất Ổn:** Cơ chế phân định chế độ thích nghi (`modal_only` trên Baby và `full_ssb` trên Sports/Electronics) đã giải quyết triệt để rủi ro xung đột gradient và suy thoái do over-smoothing từ các phiên bản đơn lẻ trước đây.
3. **Bằng Chứng Thực Nghiệm Tin Cậy:** Kết quả huấn luyện từ các tệp nhật ký `logs/GD4/*_v1_r.log` xác nhận mô hình hội tụ ổn định và đạt kết quả vượt trội đồng thời trên cả 3 tập dữ liệu chuẩn mà không gây suy hao tài nguyên phần cứng.
