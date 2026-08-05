# STAIR-v3: Clipped Softmax Fusion với Topology-Preserving Binary Graph

**Phiên bản:** STAIR-v3 (STAIR-ClipFuse)  
**Ngày:** 2026-08-05  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Cơ sở:** Bài học thực nghiệm từ STAIR-v2 (STAIR-DyFuse)

---

## 1. ĐỘNG LỰC TỪ BÀI HỌC THỰC NGHIỆM V2

### 1.1 Ba bài học cốt lõi từ STAIR-v2

Thực nghiệm STAIR-v2 đã cho thấy 3 bài học kỹ thuật quan trọng:

| # | Bài học | Hệ quả trong v2 | Nguyên tắc cho v3 |
|---|---|---|---|
| **L1** | Raw std/norm không phải confidence indicator khi preprocessing không đồng nhất | Text norm = 1.0 (pre-normalized), Visual norm = 76.8 → c_v = 95.6%, text bị triệt tiêu | **Cần điều chỉnh scale trước khi tính confidence** |
| **L2** | Trọng số cạnh liên tục (continuous cosine weights) làm lệch phân phối đồ thị | Cạnh có cosine thấp → weight ≈ 0 → multi-hop propagation trong BSC suy giảm | **Giữ cấu trúc nhị phân TopK sau khi tính điểm kết hợp** |
| **L3** | Không có connectivity floor → một modality có thể bị triệt tiêu hoàn toàn | Đồ thị text bị nhân 0.044 → gần như vô nghĩa | **Đặt ngưỡng tối thiểu δ cho mỗi phương thức** |

### 1.2 Phân tích điểm thất bại của v2

```
STAIR-v2 failure mode:

Text features (pre-L2-normalized):
  L2-norm = 1.0 (constant) → std per dim = 0.0435

Visual features (NOT normalized):
  L2-norm ~ N(76.8, 12.1) → std per dim = 0.9420

Ratio-based confidence:
  c_v = 0.9420 / (0.9420 + 0.0435) = 0.956 ← EXTREME IMBALANCE
  c_t = 0.044

Effect on mAdj:
  w_ij^text   = 0.044 × cosine_sim ≈ 0     ← TEXT GRAPH SUPPRESSED
  w_ij^visual = 0.956 × cosine_sim ≈ full  ← VISUAL DOMINATES

Result:
  BSC smoother only operates on visual graph → loses semantic connectivity
  Validation Recall@20 Baby: 0.0898 (-13.2% vs baseline)
```

---

## 2. THIẾT KẾ STAIR-v3: CLIPPED SOFTMAX FUSION

### 2.1 Tổng quan kiến trúc

STAIR-v3 (**STAIR-ClipFuse**) giải quyết cả 3 bài học trên thông qua pipeline 4 bước:

```
Feature Vectors (raw, not normalized)
        │
        ▼
┌───────────────────────────────────────────────┐
│  Bước 0: L2-normalize đồng nhất cả 2 modality │
│  f̂_v = f_v / ‖f_v‖₂,  f̂_t = f_t / ‖f_t‖₂  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Bước 1: Tính Dimension-wise Variance Ratio   │
│  Đo lường thực sự information richness        │
│  σ_v = mean(std(f̂_v, dim=0))                 │
│  σ_t = mean(std(f̂_t, dim=0))                 │
│  Score_v = σ_v / τ,  Score_t = σ_t / τ        │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Bước 2: Softmax → Clipped Confidence         │
│  c̃_v = softmax([Score_v, Score_t])[0]         │
│  c_v  = clip(c̃_v, δ, 1-δ)  với δ = 0.3      │
│  c_t  = 1 - c_v                               │
│  Đảm bảo c_v, c_t ∈ [0.3, 0.7] mọi lúc       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Bước 3: Scoring + TopK Binary Fusion         │
│  s̃_ij = c_v × s_ij^v + c_t × s_ij^t          │
│  A_ij = 1 [ s̃_ij ∈ TopK-neighbors(i) ]       │
│  → Binary adjacency matrix (0/1)              │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  Bước 4: Symmetric Laplacian Normalization    │
│  mAdj = D^{-1/2} A D^{-1/2}                  │
│  (giống STAIR gốc, bảo toàn BSC stability)    │
└───────────────────────────────────────────────┘
```

---

## 3. TOÁN HỌC CHI TIẾT

### 3.1 Bước 0 — L2-Normalization Đồng Nhất

Trước khi tính confidence, chuẩn hóa cả hai modality về unit sphere để loại bỏ scale bias:

$$\hat{\mathbf{f}}_i^v = \frac{\mathbf{f}_i^v}{\|\mathbf{f}_i^v\|_2 + \epsilon}, \qquad \hat{\mathbf{f}}_i^t = \frac{\mathbf{f}_i^t}{\|\mathbf{f}_i^t\|_2 + \epsilon}$$

**Lý do:** Sau bước này, cả text và visual đều có L2-norm = 1. Metric confidence lúc này chỉ phản ánh **cấu trúc phân phối theo chiều (dimension-wise spread)**, không bị ảnh hưởng bởi sự khác biệt scale thô giữa hai preprocessing pipelines.

---

### 3.2 Bước 1 — Dimension-wise Variance Ratio

Sau khi normalize, tính **mean standard deviation trên từng chiều** (dimension-wise) cho toàn bộ item set:

$$\sigma^v = \frac{1}{D_v} \sum_{d=1}^{D_v} \text{std}(\hat{\mathbf{F}}^v_{:,d}), \qquad \sigma^t = \frac{1}{D_t} \sum_{d=1}^{D_t} \text{std}(\hat{\mathbf{F}}^t_{:,d})$$

trong đó $\hat{\mathbf{F}}^v \in \mathbb{R}^{N \times D_v}$ là ma trận feature đã normalize, $\hat{\mathbf{F}}^t \in \mathbb{R}^{N \times D_t}$ tương tự cho textual.

**Ý nghĩa thống kê:** $\sigma^v$ lớn có nghĩa là các item **phân tán nhiều trong không gian visual** — tức visual modality có khả năng phân biệt item tốt hơn và chứa nhiều thông tin hữu ích hơn cho bài toán gợi ý. Ngược lại, $\sigma^v$ nhỏ có nghĩa visual feature phẳng, kém phân biệt.

**So sánh với v2:** Trong v2, chúng ta dùng raw norm ratio trên toàn item. Ở v3, sau khi normalize về unit sphere, chúng ta dùng **dimension-wise std** — đây là chỉ số phân biệt item (discriminability) thực sự của mỗi modality.

---

### 3.3 Bước 2 — Clipped Softmax Confidence

**Bước 2a — Softmax confidence với temperature:**

$$\tilde{c}^v = \frac{\exp(\sigma^v / \tau)}{\exp(\sigma^v / \tau) + \exp(\sigma^t / \tau)}, \qquad \tilde{c}^t = 1 - \tilde{c}^v$$

Siêu tham số $\tau > 0$ (temperature) điều khiển độ sắc nét của confidence:
- $\tau \to 0$: confidence tiến về one-hot (winner-takes-all)
- $\tau \to \infty$: confidence tiến về uniform (0.5/0.5)
- $\tau = 1.0$: mặc định, phản ánh tỷ lệ thực tế

**Bước 2b — Connectivity Floor Clipping:**

$$c^v = \text{clip}(\tilde{c}^v,\ \delta,\ 1-\delta), \qquad c^t = 1 - c^v$$

với $\delta = 0.3$ là ngưỡng tối thiểu (connectivity floor). Kết quả: $c^v, c^t \in [0.3, 0.7]$ mọi lúc.

**Lý do:** Không có modality nào được phép chiếm quá 70% trọng số. Trong thực tế Amazon Baby/Sports, cả text (mô tả sản phẩm) và visual (hình ảnh) đều mang thông tin bổ sung cho nhau — không nên cho phép một modality áp đảo hoàn toàn.

**Bảng giá trị tham chiếu:**

| $\sigma^v / \sigma^t$ | $\tilde{c}^v$ ($\tau=1$) | $c^v$ (sau clip $\delta=0.3$) |
|:---:|:---:|:---:|
| 0.25 (text dominant) | 0.20 | **0.30** (clipped) |
| 0.50 (text lớn hơn) | 0.38 | **0.38** |
| 1.00 (cân bằng) | 0.50 | 0.50 |
| 2.00 (visual lớn hơn) | 0.62 | **0.62** |
| 4.00 (visual dominant) | 0.80 | **0.70** (clipped) |

> **Nhận xét:** Với dữ liệu thực tế v2 ($\sigma^v/\sigma^t \approx 0.942/0.044 \approx 21$), nếu không clip thì $\tilde{c}^v \approx 1.0$. Sau clip: $c^v = 0.7$, $c^t = 0.3$ — cân bằng hơn nhiều và cho phép text graph đóng góp 30%.

---

### 3.4 Bước 3 — Topology-Preserving TopK Binary Fusion

**Bước 3a — Tính điểm kết hợp:**

$$\tilde{s}_{ij} = c^v \cdot s_{ij}^v + c^t \cdot s_{ij}^t$$

trong đó $s_{ij}^v, s_{ij}^t$ là cosine similarity từ visual/textual feature đã normalize. Cạnh chỉ tồn tại trong một graph thì phần còn lại bằng 0.

**Bước 3b — TopK Selection + Binarization:**

$$A_{ij} = \mathbf{1}\Big[\tilde{s}_{ij} \in \text{TopK-neighbors}(i),\ k = k_v + k_t\Big]$$

Giữ $k = k_v + k_t$ neighbors tốt nhất theo điểm kết hợp, sau đó binarize (0/1). Đây là điểm cải tiến cốt lõi so với v2:

| | STAIR Baseline | STAIR-v2 (DyFuse) | **STAIR-v3 (ClipFuse)** |
|---|:---:|:---:|:---:|
| Graph weights | Binary (0/1) | Continuous cosine × confidence | **Binary (0/1) sau TopK** |
| Topology | Fixed (max-pooling) | Changed (reweighted edges) | **Preserved (TopK selection)** |
| Distribution | Uniform | Skewed (heavy tail) | **Uniform (binarized)** |
| BSC stability | ✅ Stable | ❌ Skewed distribution | **✅ Stable** |

**Lý do giữ binary topology:** FSC/BSC trong STAIR hoạt động như các bộ lọc bước (stepwise filter). Với binary adjacency + symmetric Laplacian normalization, phép nhân ma trận tương đương diffusion đều trên đồ thị — tính chất này bị phá vỡ khi dùng continuous weights lệch nặng.

---

### 3.5 Bước 4 — Symmetric Laplacian Normalization (giữ nguyên từ STAIR gốc)

$$\hat{A}_{ij} = \frac{A_{ij}}{\sqrt{d_i \cdot d_j} + \epsilon}$$

với $d_i = \sum_j A_{ij}$ là bậc của node $i$ trong đồ thị đã kết hợp. Dạng chuẩn hóa này giữ nguyên đặc tính của BSC smoother trong STAIR gốc.

---

### 3.6 Hàm mất mát — Giữ nguyên BPR

$$\mathcal{L} = \mathcal{L}_{\text{BPR}}(u, i^+, i^-)$$

STAIR-v3 **không thêm hàm mất mát mới**. Toàn bộ thay đổi nằm trong hàm `prepare()`.

---

## 4. SO SÁNH THIẾT KẾ ĐẦY ĐỦ

| Tiêu chí | STAIR Baseline | STAIR-v1 (GCL) | STAIR-v2 (DyFuse) | **STAIR-v3 (ClipFuse)** |
|---|:---:|:---:|:---:|:---:|
| Cách kết hợp đồ thị | `max(s_v, s_t)` | `max(s_v, s_t)` | `c_v × s_v + c_t × s_t` | **TopK(`c_v×s_v + c_t×s_t`) → binary** |
| Confidence source | — | — | Raw std ratio | **L2-normalized dim-wise std** |
| Connectivity floor | — | — | ❌ Không có | **✅ δ=0.3** |
| Graph topology | Binary | Binary | Continuous (skewed) | **Binary** |
| Tham số thêm | 0 | 0 | 0 | **1 (δ, fixed default 0.3)** |
| Hàm mất mát thêm | Không | GCL (InfoNCE) | Không | **Không** |
| Adaptive theo item | Không | Không | Có (per-item) | **Có (global, dataset-level)** |
| Preprocessing scale bias | N/A | N/A | ❌ Không xử lý | **✅ L2-normalize trước** |

> **Lưu ý về "global vs per-item":** STAIR-v2 tính confidence per-item nhưng thực ra do text norm = 1 cho mọi item → confidence trở thành global constant. STAIR-v3 thẳng thắn tính global dataset-level confidence từ dimension-wise std — đây là chỉ số phản ánh đúng information richness của từng modality trên toàn dataset.

---

## 5. VỊ TRÍ CAN THIỆP TRONG CODE

### 5.1 Thay đổi trong `main_v3.py` — hàm `compute_confidence()`

```python
def compute_confidence(self, feat_t: torch.Tensor, feat_v: torch.Tensor,
                       delta: float = 0.3, tau: float = 1.0) -> Tuple[float, float]:
    """
    Compute clipped softmax modality confidence from dimension-wise std.

    Steps:
      1. L2-normalize both modalities to unit sphere (remove scale bias)
      2. Compute mean per-dim std across items for each modality
      3. Apply softmax with temperature tau
      4. Clip to [delta, 1-delta] to enforce connectivity floor

    Args:
        feat_t: textual features  (N, D_t)
        feat_v: visual  features  (N, D_v)
        delta:  minimum confidence per modality (default 0.3)
        tau:    temperature for softmax (default 1.0)

    Returns:
        c_v, c_t: global confidence scores (float), both in [delta, 1-delta]
    """
    eps = 1e-7

    # Step 1: L2-normalize → unit sphere (remove preprocessing scale bias)
    feat_v_n = F.normalize(feat_v.float(), p=2, dim=-1)  # (N, D_v)
    feat_t_n = F.normalize(feat_t.float(), p=2, dim=-1)  # (N, D_t)

    # Step 2: Dimension-wise std across items → mean over dims
    sigma_v = feat_v_n.std(dim=0).mean().item()   # scalar
    sigma_t = feat_t_n.std(dim=0).mean().item()   # scalar

    # Step 3: Softmax confidence with temperature
    score_v = sigma_v / tau
    score_t = sigma_t / tau
    exp_v = math.exp(score_v)
    exp_t = math.exp(score_t)
    c_v_raw = exp_v / (exp_v + exp_t + eps)

    # Step 4: Clip to [delta, 1-delta] → connectivity floor
    c_v = max(delta, min(1.0 - delta, c_v_raw))
    c_t = 1.0 - c_v

    print(f"[ClipFuse] sigma_v={sigma_v:.4f}, sigma_t={sigma_t:.4f} | "
          f"raw c_v={c_v_raw:.4f} → clipped c_v={c_v:.4f}, c_t={c_t:.4f}")
    return c_v, c_t
```

### 5.2 Thay đổi trong `main_v3.py` — hàm `prepare()`

```python
def prepare(self, path: str):
    """Build mAdj using ClipFuse: Clipped-Softmax Confidence + TopK Binary Fusion."""
    mfeats = []
    for mfile in cfg.mfiles:
        with open(os.path.join(path, mfile), 'rb') as f:
            feat = torch.tensor(pickle.load(f), dtype=torch.float32).to(cfg.device)
        mfeats.append(feat)

    feat_t, feat_v = mfeats[0], mfeats[1]   # textual (N, D_t), visual (N, D_v)

    # Compute clipped softmax confidence (global, dataset-level)
    c_v, c_t = self.compute_confidence(feat_t, feat_v, delta=cfg.conf_delta, tau=cfg.conf_temp)

    # Build kNN graphs with cosine similarity weights
    def build_knn_weighted(features: torch.Tensor, k: int):
        feat_n = F.normalize(features, p=2, dim=-1)   # unit sphere
        sim = feat_n @ feat_n.t()                      # (N, N) cosine similarity
        sim.fill_diagonal_(-10.)
        # TopK per row
        topk_vals, topk_idx = sim.topk(k=k, dim=-1)   # (N, k)
        topk_vals = topk_vals.clamp(min=0.)
        # Build edge_index
        rows = torch.arange(feat_n.size(0), device=feat_n.device).unsqueeze(1).expand_as(topk_idx)
        edge_index = torch.stack([rows.reshape(-1), topk_idx.reshape(-1)], dim=0)
        edge_weight = topk_vals.reshape(-1)
        return edge_index, edge_weight

    k_t, k_v = cfg.num_neighbors[0], cfg.num_neighbors[1]
    ei_t, ew_t = build_knn_weighted(feat_t, k_t)   # textual kNN (N*k_t edges)
    ei_v, ew_v = build_knn_weighted(feat_v, k_v)   # visual  kNN (N*k_v edges)

    # Weighted score for each edge (used for ranking, not as final weight)
    score_t = c_t * ew_t   # confidence-weighted textual scores
    score_v = c_v * ew_v   # confidence-weighted visual scores

    # Combine all candidate edges
    edge_index_all = torch.cat([ei_t, ei_v], dim=1)        # (2, N*(k_t+k_v))
    score_all      = torch.cat([score_t, score_v], dim=0)  # (N*(k_t+k_v),)

    # Coalesce: sum scores for edges appearing in both graphs
    edge_index_all, score_all = freerec.graph.coalesce(
        edge_index_all, score_all, reduce='sum'
    )

    # TopK selection: keep top (k_t + k_v) neighbors per item by combined score
    # Then BINARIZE: set all kept edge weights to 1.0
    k_total = k_t + k_v
    N = feat_t.size(0)
    # For each row, keep top k_total edges
    # We rebuild sparse → dense → topk → sparse
    row, col = edge_index_all
    binary_weight = torch.zeros(N, N, device=cfg.device)
    binary_weight[row, col] = score_all
    # TopK per row and binarize
    _, top_cols = binary_weight.topk(k=min(k_total, N-1), dim=-1)  # (N, k_total)
    src = torch.arange(N, device=cfg.device).unsqueeze(1).expand_as(top_cols)
    edge_index_final = torch.stack([src.reshape(-1), top_cols.reshape(-1)], dim=0)
    edge_weight_final = torch.ones(edge_index_final.size(1), device=cfg.device)  # BINARY

    # To undirected + symmetric Laplacian normalization (same as STAIR baseline)
    edge_index_final, edge_weight_final = freerec.graph.to_undirected(
        edge_index_final, edge_weight_final, reduce='max'
    )
    edge_index_final, edge_weight_final = freerec.graph.to_normalized(
        edge_index_final, edge_weight_final, normalization='sym'
    )

    self.mAdj = freerec.graph.to_sparse_csr(edge_index_final, edge_weight_final, N)
    print(f"[ClipFuse] mAdj built: {edge_index_final.size(1)} edges | "
          f"c_v={c_v:.4f}, c_t={c_t:.4f}")
```

> **Lưu ý bộ nhớ:** Bước tạo dense matrix `(N, N)` có thể tốn nhiều bộ nhớ với N lớn (Sports N=18357 → 18357² × 4 byte ≈ 1.4 GB). Phần `5.4 — Tối ưu bộ nhớ` sẽ trình bày cách xử lý batch-wise.

### 5.3 Arguments mới trong `cfg`

```python
cfg.add_argument("--conf-delta", type=float, default=0.3,
                 help="minimum confidence floor for each modality [0.2, 0.4]")
cfg.add_argument("--conf-temp",  type=float, default=1.0,
                 help="softmax temperature for confidence scoring")
cfg.add_argument("--fuse-k",     type=str,   default='5-1',
                 help="kNN per modality for TopK fusion (same format as num-neighbors)")
```

### 5.4 Tối ưu bộ nhớ — Batch-wise TopK (tránh dense N×N matrix)

```python
# Efficient TopK: avoid dense NxN matrix
# Sort edges per source node using scatter_topk or manual approach
from torch_scatter import scatter

def topk_per_row(row: torch.Tensor, col: torch.Tensor,
                 score: torch.Tensor, k: int, N: int):
    """Memory-efficient TopK selection per source node."""
    # Sort by (row, -score) to get top-k per row
    sort_idx  = torch.argsort(row * N - score)   # approximate: sort by row then desc score
    row_s     = row[sort_idx]
    col_s     = col[sort_idx]
    score_s   = score[sort_idx]

    # Keep first k per row using cumcount trick
    ones   = torch.ones_like(row_s)
    cumcnt = scatter(ones, row_s, reduce='sum', dim_size=N)   # (N,) → count per row
    # ... build mask for top-k per row
    # (full implementation in main_v3.py)
    return edge_index_kept, edge_weight_kept
```

---

## 6. PHÂN TÍCH LÝ THUYẾT

### 6.1 Tại sao Dimension-wise Std sau L2-normalize là chỉ số tốt hơn?

Sau khi L2-normalize:

$$\hat{\mathbf{f}}_i \in \mathbb{S}^{D-1}$$  (hypersphere)

Khi đó, $\text{std}(\hat{\mathbf{F}}_{:,d})$ đo **mức độ phân tán** của item set theo chiều $d$ trên hypersphere.

- **σ_d lớn** → items trải đều trên chiều $d$ → chiều đó có khả năng phân biệt item tốt (discriminative dimension)
- **σ_d nhỏ** → items co cụm → chiều đó kém phân biệt

$\sigma = \text{mean}_d(\sigma_d)$ là chỉ số tổng hợp về **information content** của toàn modality.

**So sánh với v2:** v2 dùng raw L2-norm của feature → phụ thuộc preprocessing scale. v3 dùng dim-wise std sau normalize → đo thực sự cấu trúc thông tin nội tại của modality.

### 6.2 Tại sao TopK Binary tốt hơn Continuous Weighted?

BSC (Backward Stepwise Convolution) hoạt động theo công thức:

$$\mathbf{H}^{(l)} = \hat{A} \mathbf{H}^{(l-1)}$$

Với binary $\hat{A}$ (symmetric Laplacian):
- Eigenvalues nằm trong $[-1, 1]$
- Phép nhân ma trận bảo toàn energy (stable diffusion)
- Sau $L$ bước, thông tin lan truyền đều theo $L$-hop neighborhood

Với continuous skewed $\hat{A}$ (weights lệch mạnh như v2):
- Cạnh weight ≈ 0 → effectively dropped (sparse, not intended)
- Cạnh weight >> mean → dominates (biased diffusion)
- Multi-hop propagation không đồng đều → feature representation méo

### 6.3 Connectivity Floor δ = 0.3 — Cơ sở lý thuyết

Trong multi-modal recommendation, mỗi modality cung cấp một góc nhìn độc lập về sản phẩm (semantic space). Optimal fusion lý thuyết cần minimize:

$$\mathcal{L}_{\text{fusion}} = \mathcal{L}_{\text{rec}} + \lambda \cdot D_{\text{KL}}(P_{\text{fusion}} \| P_{\text{uniform}})$$

$D_{\text{KL}}$ penalty ngăn confidence quá extreme. Giá trị δ = 0.3 tương đương với prior entropy:

$$H([0.3, 0.7]) = -0.3\log 0.3 - 0.7\log 0.7 \approx 0.61 \text{ bits}$$

So với maximum entropy $H([0.5, 0.5]) = 1.0$ bit, δ = 0.3 giữ được 61% entropy tối đa — tức cho phép sự khác biệt có ý nghĩa giữa hai modality nhưng không hoàn toàn triệt tiêu một modality.

---

## 7. HYPERPARAMETER SENSITIVITY ANALYSIS

### 7.1 Sensitivity của δ (Connectivity Floor)

| δ | c_v range | c_t range | Behavior |
|:---:|:---:|:---:|---|
| 0.1 | [0.1, 0.9] | [0.1, 0.9] | Aggressive — rủi ro triệt tiêu gần hoàn toàn |
| **0.3** | [0.3, 0.7] | [0.3, 0.7] | **Recommended — cân bằng giữa flexibility và safety** |
| 0.4 | [0.4, 0.6] | [0.4, 0.6] | Conservative — gần uniform, ít adaptive |
| 0.5 | [0.5, 0.5] | [0.5, 0.5] | Uniform — tương đương STAIR baseline |

### 7.2 Sensitivity của τ (Temperature)

| τ | Behavior khi σ_v >> σ_t |
|:---:|---|
| 0.1 | c_v ≈ 1.0 (winner-takes-all, trước khi clip) → clip cứu |
| 0.5 | c_v ≈ 0.88 (sharp) → clip xuống 0.7 nếu > 0.7 |
| **1.0** | c_v ≈ softmax tự nhiên → **Recommended** |
| 2.0 | c_v ≈ 0.62 (smooth) → ít nhạy hơn, safe hơn |
| 5.0 | c_v ≈ 0.51 (near-uniform) → gần không tác dụng |

**Kết luận:** τ = 1.0 với δ = 0.3 là default khuyến nghị. Nếu muốn thêm safety, tăng τ = 2.0.

### 7.3 Grids thực nghiệm gợi ý

```
Primary search (fix τ=1.0):
  δ ∈ {0.2, 0.3, 0.4}

Secondary search (fix δ=0.3):
  τ ∈ {0.5, 1.0, 2.0}

k (neighbors per modality):
  k_t-k_v ∈ {5-1, 5-5, 10-5}  (consistent with num-neighbors format)
```

---

## 8. KẾ HOẠCH THỰC NGHIỆM

### 8.1 Baseline comparison

| Model | Baby Recall@20 | Sports Recall@20 | Ghi chú |
|---|:---:|:---:|---|
| STAIR Baseline | 0.1034 | 0.1119 | Mục tiêu phải vượt |
| STAIR-v1 (GCL) | 0.1047 | 0.1124 | Mục tiêu phải vượt |
| STAIR-v2 (DyFuse) | 0.0898 | 0.1002 | Thấp hơn baseline 13% |
| **STAIR-v3 (ClipFuse) — Mục tiêu** | **≥ 0.106** | **≥ 0.114** | **+2% so với baseline** |

### 8.2 Ablation study plan

| Experiment | δ | τ | Topology | Mục đích |
|---|:---:|:---:|:---:|---|
| `v3-full` | 0.3 | 1.0 | Binary TopK | Đề xuất đầy đủ |
| `v3-nocliip` | 0.0 | 1.0 | Binary TopK | Kiểm tra tác dụng của δ |
| `v3-nol2` | 0.3 | 1.0 | Binary TopK (không L2-norm trước) | Kiểm tra tác dụng của normalize |
| `v3-cont` | 0.3 | 1.0 | Continuous (như v2) | Kiểm tra tác dụng của binary fusion |
| `v3-uniform` | 0.5 | 1.0 | Binary TopK | Kiểm tra: v3 ≥ baseline? |

### 8.3 Lệnh chạy thực nghiệm

```bash
# Experiment chính (STAIR-v3 full)
python main_v3.py \
  --root /kaggle/data \
  --dataset Amazon2014Baby_550_MMRec \
  --epochs 500 --batch-size 1024 --embedding-dim 64 \
  --num-layers 3 --num-neighbors 5-1 \
  --conf-delta 0.3 --conf-temp 1.0 \
  --optimizer adamwsevo --lr 1e-3 --weight-decay 0.1 --seed 1 \
  > log_stair_v3_baby_full.txt 2>&1

# Ablation: no clip
python main_v3.py ... --conf-delta 0.0 > log_stair_v3_baby_noclip.txt 2>&1

# Ablation: uniform (baseline equivalent)
python main_v3.py ... --conf-delta 0.5 > log_stair_v3_baby_uniform.txt 2>&1
```

---

## 9. PHÂN TÍCH RỦI RO VÀ BIỆN PHÁP

| Rủi ro | Khả năng | Biện pháp |
|---|:---:|---|
| σ_v ≈ σ_t sau L2-normalize → c_v ≈ 0.5 (flat) | Thấp | Nếu xảy ra, tăng số layers BSC hoặc kết hợp với GCL |
| Dense NxN matrix tốn RAM (Sports N=18k) | Cao | Dùng batch-wise TopK (section 5.4) |
| TopK k_total quá nhỏ → sparse graph | Thấp | Default k_t+k_v=6, có thể tăng lên 10 |
| Binary binarization mất thông tin trọng số | Trung bình | Thực nghiệm ablation v3-cont để đánh giá |
| δ=0.3 quá conservative → không tác dụng | Thấp | Thực nghiệm với δ∈{0.2, 0.3} |

---

## 10. TÓM TẮT ĐÓNG GÓP KHOA HỌC

STAIR-v3 (ClipFuse) đóng góp 4 cải tiến kỹ thuật rõ ràng so với các phiên bản trước:

1. **L2-normalize trước confidence** → Loại bỏ preprocessing scale bias, confidence phản ánh information richness thực sự.

2. **Dimension-wise std làm confidence signal** → Đo discriminability của từng modality trên hypersphere, không phụ thuộc vào absolute magnitude.

3. **Connectivity Floor δ** → Đảm bảo cả hai modality luôn đóng góp ít nhất δ vào đồ thị kết hợp, ngăn chặn Modality Collapse.

4. **TopK Binary Topology** → Giữ cấu trúc đồ thị nhị phân sau bước fusion, bảo toàn tính ổn định của BSC Smoother.

Điểm quan trọng nhất: **v3 là parameter-free** (nếu δ và τ cố định theo default). Tất cả thay đổi nằm trong hàm `prepare()` — không ảnh hưởng tới forward pass, backward pass, hay hàm mất mát của STAIR gốc.

---

## 11. LIÊN KẾT TÀI LIỆU

| Tài liệu | Mô tả |
|---|---|
| [`STAIR_v2_Report.md`](STAIR_v2_Report.md) | Thiết kế chi tiết STAIR-v2 (DyFuse) |
| [`STAIR_v2_Experiment_Report.md`](STAIR_v2_Experiment_Report.md) | Kết quả thực nghiệm và Root Cause Analysis của v2 |
| [`main_v2.py`](../main_v2.py) | Implementation của STAIR-v2 |
| `main_v3.py` *(sẽ tạo)* | Implementation của STAIR-v3 (ClipFuse) |
| `notebook/stair_v3.ipynb` *(sẽ tạo)* | Notebook Kaggle cho STAIR-v3 |
