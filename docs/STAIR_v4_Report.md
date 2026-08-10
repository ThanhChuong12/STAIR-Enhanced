# STAIR-v4: SG-URInit Integration — Semantically Guaranteed User Representation Initialization

**Phiên bản:** STAIR-v4 (STAIR-SGInit)  
**Ngày:** 2026-08-10  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Cơ sở lý thuyết:** SG-URInit (SIGIR'26) + STAIR-v3.1 (ClipFuse-Consensus)  
**Trạng thái:** 📋 Planning / Design Phase

---

## 1. BỐI CẢNH VÀ ĐỘNG LỰC

### 1.1 Nhìn lại hành trình cải tiến STAIR

| Phiên bản | Tên | Đóng góp cốt lõi | Baby R@20 (Test) | Sports R@20 (Test) |
|---|---|---|:---:|:---:|
| Baseline | STAIR | STAIR gốc: kNN + BSC smoother | 0.1034 | 0.1119 |
| v1 | STAIR-GCL | Graph Contrastive Learning loss | 0.1047 | 0.1124 |
| v2 | STAIR-DyFuse | Dynamic Modality Fusion (L2-norm) | 0.0898 ↓ | 0.1002 ↓ |
| v3.1 | STAIR-ClipFuse | kNN-Discriminability + Prior-Preserving + Consensus Boost | 0.0938 | **0.1124** ✓ |
| **v4** | **STAIR-SGInit** | **SG-URInit: Semantically Guaranteed User Init** | **?** | **?** |

> **Quan sát chính:**  
> v3.1 đã giải quyết hoàn toàn vấn đề **Modality Collapse** của v2 và lần đầu tiên vượt Baseline trên **Sports**. Tuy nhiên, tập **Baby** vẫn kém Baseline khoảng 9.3% (R@20: 0.0938 vs 0.1034). Root cause chưa được giải quyết nằm ở **phía User Embedding**, không phải phía Item Graph.

### 1.2 Phân tích Root Cause còn lại: "User Initialization Noise"

Sau khi v3.1 tối ưu hoàn toàn phía item-item graph, điểm nghẽn còn lại là **khởi tạo user embedding**:

```
current STAIR prepare() [dòng 266]:
  user_emb = R @ mfeats
  
  trong đó R = left-normalized user-item interaction matrix (sparse)
  và mfeats = SVD-whitened multimodal features (prior-weighted)
```

**Vấn đề cụ thể** (theo phân tích SG-URInit paper):

1. **User tương tác ít (Cold-start / Sparse users):** Mỗi user trên Baby trung bình chỉ tương tác với ~5-7 items. Khi $R$ có hàng thưa, phép nhân `R @ mfeats` chỉ trung bình trên 5-7 vector item — rất dễ bị nhiễu ngẫu nhiên, không đại diện được sở thích toàn cục.

2. **Local Semantics chỉ — không có Global Context:** `R @ mfeats` thu thập thông tin **item-level local** (chỉ những items user đã click), bỏ qua **cluster-level global** (nhóm sở thích tổng quát của user).

3. **Asymmetric Initialization Gap:** Item embeddings được neo bởi FSC với đặc trưng whitened chất lượng cao ngay từ đầu, trong khi User embeddings chỉ được neo qua sparse `R` — hai bên không đối xứng về chất lượng thông tin ban đầu.

4. **Tác động lớn hơn ở Baby:** Sports (18K items, nhiều tương tác) có neighborhood phong phú giúp `R @ mfeats` có thể ước lượng tốt preference vector. Baby (7K items, ít tương tác) gặp trường hợp ngược lại — neighborhood quá nhỏ, trung bình `R @ mfeats` không đủ chất lượng.

---

## 2. GIẢI PHÁP: SG-URInit (STAIR-v4)

### 2.1 Ý tưởng cốt lõi của SG-URInit (SIGIR'26)

**Paper:** *"Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation"* (Xu et al., arXiv:2604.14839)

Thay vì dùng khởi tạo random hoặc sparse matrix aggregation thuần túy, SG-URInit xây dựng user embedding từ **hai nguồn thông tin bổ sung nhau**:

$$\mathbf{u}_u^{\text{init},m} = \lambda \cdot \underbrace{\mathbf{u}_u^{\text{item},m}}_{\text{Local: Item-level}} + (1-\lambda) \cdot \underbrace{\mathbf{u}_u^{\text{cluster},m}}_{\text{Global: Cluster-level}}$$

- **$\mathbf{u}_u^{\text{item},m}$ (Local Aggregation):** Trung bình có trọng số degree của item multimodal features trong neighborhood của user $u$, giúp capture sở thích đặc thù.
- **$\mathbf{u}_u^{\text{cluster},m}$ (Global Aggregation):** Trung bình của centroid K-Means tương ứng với mỗi item đã tương tác, giúp capture sở thích tổng quát và kháng nhiễu.

---

## 3. TOÁN HỌC CHI TIẾT

### 3.1 Định nghĩa ký hiệu

| Ký hiệu | Định nghĩa |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | Tập users và items |
| $\mathcal{N}(u)$ | Tập items user $u$ đã tương tác (training set) |
| $\mathbf{f}_i^{m,\text{white}}$ | SVD-whitened feature của item $i$, modality $m \in \{t, v\}$ (đã có trong STAIR) |
| $K$ | Số clusters K-Means (hyperparameter) |
| $\mathcal{C}^m = \{c_1^m, \ldots, c_K^m\}$ | Centroids sau K-Means trên tất cả item features, modality $m$ |
| $\lambda$ | Mixing weight ($\lambda = 0.1$, ưu tiên Global Cluster) |

### 3.2 Bước 1: Local Item-Level Aggregation

Dùng degree-normalized aggregation kiểu LightGCN:

$$\mathbf{u}_u^{\text{item},m} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)|} \cdot \sqrt{|\mathcal{N}(i)|}} \, \mathbf{f}_i^{m,\text{white}}$$

trong đó:
- $|\mathcal{N}(u)|$ là degree của user $u$
- $|\mathcal{N}(i)|$ là degree của item $i$ (số users đã tương tác với item $i$)
- Chuẩn hóa hai bên $D_u^{-1/2} R D_i^{-1/2}$ — giống symmetric normalization của LightGCN, tránh user phổ biến chiếm ưu thế

**Lưu ý quan trọng về STAIR:** Phép nhân này chính xác là: $\mathbf{U}^{\text{item},m} = D_U^{-1/2} R D_I^{-1/2} \mathbf{F}^m$ với $R$ là sparse interaction matrix, hoàn toàn khác với `R_left @ mfeats` hiện tại. Sự khác biệt là chuẩn hóa **hai phía** (sym) thay vì một phía (left).

### 3.3 Bước 2: Global Cluster-Level Aggregation

**Sub-bước 2a: K-Means clustering trên item features (per modality)**

$$\{c_1^m, \ldots, c_K^m\} = \text{KMeans}(\{\mathbf{f}_i^{m,\text{white}}\}_{i \in \mathcal{I}}, K)$$

**Sub-bước 2b: Gán mỗi item về centroid gần nhất**

$$\text{clust}(i, m) = \arg\min_{k \in [K]} \| \mathbf{f}_i^{m,\text{white}} - c_k^m \|_2$$

$$\mathbf{f}_i^{m,\text{clust}} = c_{\text{clust}(i,m)}^m$$

**Sub-bước 2c: Aggregation cho user**

$$\mathbf{u}_u^{\text{cluster},m} = \frac{1}{|\mathcal{N}(u)|} \sum_{i \in \mathcal{N}(u)} \mathbf{f}_i^{m,\text{clust}}$$

> **Ý nghĩa:** Thay vì lấy chính xác vector đặc trưng của item $i$ (nhiễu cục bộ), ta lấy centroid cluster của $i$ — một vector "mượt" hơn, đại diện cho nhóm sở thích mà item $i$ thuộc về. Đặc biệt hiệu quả khi users có ít interaction.

**Ma trận hóa hiệu quả:**

Đặt $\mathbf{F}^{m,\text{clust}} \in \mathbb{R}^{N_I \times D}$ là ma trận cluster-centroid-embedding của tất cả items. Khi đó:

$$\mathbf{U}^{\text{cluster},m} = D_U^{-1} R \cdot \mathbf{F}^{m,\text{clust}}$$

Toàn bộ có thể tính bằng sparse matrix multiplication — **O(nnz)** complexity, không tốn tài nguyên.

### 3.4 Bước 3: Multimodal Fusion & Final Initialization

**Per-modality fusion:**

$$\mathbf{u}_u^{\text{init},m} = \lambda \cdot \mathbf{u}_u^{\text{item},m} + (1-\lambda) \cdot \mathbf{u}_u^{\text{cluster},m}, \quad \lambda = 0.1$$

**Cross-modality fusion (Prior-Preserving, consistent với v3.1):**

$$\mathbf{u}_u^{\text{init}} = \frac{k_t \cdot \mathbf{U}^{\text{init},t} + k_v \cdot \mathbf{U}^{\text{init},v}}{k_t + k_v}$$

với $k_t = 5, k_v = 1$ — **bảo toàn đúng cấu trúc prior 5:1 đã thành công ở v3.1**.

**Thay thế user embedding initialization:**

$$\texttt{User.embeddings.weight.data} \leftarrow \mathbf{U}^{\text{init}}$$

và **giữ `requires_grad=True`** — SG-URInit chỉ thay điểm khởi đầu, không đóng băng.

---

## 4. TẠI SAO $\lambda = 0.1$ (CLUSTER DOMINANT)?

Theo paper SG-URInit, $\lambda = 0.1$ cho mô hình có item-item graph sẵn có (như STAIR có `mAdj`):

- $\lambda$ nhỏ → **Cluster-level chiếm 90%** → người dùng ít tương tác vẫn có embedding ổn định.
- Với users nhiều tương tác (Sports), item-level aggregation đã chất lượng → cluster và item tương đồng → $\lambda$ không ảnh hưởng nhiều.
- Với users ít tương tác (Baby), cluster-level aggregation cung cấp "semantic fallback" giúp embedding không bị nhiễu bởi 5 items ngẫu nhiên.

---

## 5. VỊ TRÍ TÍCH HỢP VÀO STAIR (Chi tiết kỹ thuật)

### 5.1 Cấu trúc `prepare()` hiện tại (v3.1)

```python
def prepare(self, path: str):
    # 1. Load raw features
    raw_mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
    feat_t_raw, feat_v_raw = raw_mfeats[0], raw_mfeats[1]
    
    # 2. Build kNN graphs
    ei_t, ew_t = self.build_knn_weighted(feat_t_raw, k_t)
    ei_v, ew_v = self.build_knn_weighted(feat_v_raw, k_v)
    
    # 3. Compute confidence (ClipFuse-v3.1)
    c_v, c_t = self.compute_confidence(ew_t, ew_v)
    
    # 4-6. Build mAdj (ClipFuse + Consensus Boosting)
    ...
    self.register_buffer('mAdj', mAdj.to_sparse_csr())
    
    # 7. Initialize embeddings [CẦN SỬA ĐỔI PHẦN NÀY]
    mfeats = [self.whitening(feat) * k for feat, k in zip(raw_mfeats, cfg.num_neighbors)]
    mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
    self.Item.embeddings.weight.data.copy_(mfeats)        # OK - giữ nguyên
    
    edge_index_u2i, edge_weight_u2i = ...
    R = torch.sparse_coo_tensor(...)
    self.User.embeddings.weight.data.copy_(R @ mfeats)    # ← SỬA THÀNH SG-URInit
```

### 5.2 Patch cụ thể cho v4: Thêm `sg_urinit()`

```python
def sg_urinit(
    self,
    mfeats_whitened: torch.Tensor,   # (N_I, D) — prior-weighted whitened item features
    n_clusters: int = 8,
    lambda_mix: float = 0.1,
) -> torch.Tensor:
    """
    SG-URInit: Compute semantically guaranteed user initialization.
    
    Args:
        mfeats_whitened: (N_I, D) prior-weighted multimodal whitened item embeddings.
        n_clusters: K for K-Means clustering.
        lambda_mix: mixing weight for local (item-level) vs global (cluster-level).
    
    Returns:
        user_init: (N_U, D) user initialization tensor.
    """
    from sklearn.cluster import KMeans
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)  # suppress sklearn warnings
    
    N_I = mfeats_whitened.shape[0]
    N_U = self.User.count
    
    # --- Build sparse interaction matrix (symmetric degree normalization) ---
    train_graph = self.dataset.train().to_bigraph(edge_type='u2i')['u2i']
    edge_index = train_graph.edge_index  # (2, E): row=user, col=item
    
    users_idx = edge_index[0]   # (E,)
    items_idx  = edge_index[1]  # (E,)
    
    # Compute degrees
    deg_u = torch.zeros(N_U, dtype=torch.float32)
    deg_i = torch.zeros(N_I, dtype=torch.float32)
    deg_u.scatter_add_(0, users_idx, torch.ones_like(users_idx, dtype=torch.float32))
    deg_i.scatter_add_(0, items_idx, torch.ones_like(items_idx, dtype=torch.float32))
    
    deg_u = deg_u.clamp(min=1.0)
    deg_i = deg_i.clamp(min=1.0)
    
    # Symmetric normalized edge weights: 1 / (sqrt(deg_u) * sqrt(deg_i))
    ew_sym = 1.0 / (deg_u[users_idx].sqrt() * deg_i[items_idx].sqrt())
    
    # Left normalized edge weights for cluster aggregation: 1 / deg_u
    ew_left = 1.0 / deg_u[users_idx]
    
    F_white = mfeats_whitened.cpu().float()  # (N_I, D) on CPU for KMeans
    
    # --- Bước 1: Local Item-Level Aggregation (sym-normalized) ---
    # U_item = D_u^{-1/2} R D_i^{-1/2} @ F_white
    weighted_F = F_white[items_idx] * ew_sym.unsqueeze(-1)  # (E, D)
    U_item = torch.zeros(N_U, F_white.shape[1], dtype=torch.float32)
    U_item.scatter_add_(0, users_idx.unsqueeze(-1).expand_as(weighted_F), weighted_F)
    # U_item: (N_U, D)
    
    # --- Bước 2: K-Means Clustering ---
    print(f"  [SG-URInit] K-Means clustering (K={n_clusters})...")
    feats_np = F_white.numpy()
    km = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=10)
    km.fit(feats_np)
    cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # (K, D)
    
    # Assign each item to nearest centroid
    dists = torch.cdist(F_white, cluster_centers)           # (N_I, K)
    _, clust_idx = torch.min(dists, dim=1)                  # (N_I,)
    F_cluster = cluster_centers[clust_idx]                  # (N_I, D) — centroid embeddings
    
    # Cluster-level aggregation (left-normalized): U_cluster = D_u^{-1} R @ F_cluster
    weighted_Fc = F_cluster[items_idx] * ew_left.unsqueeze(-1)  # (E, D)
    U_cluster = torch.zeros(N_U, F_white.shape[1], dtype=torch.float32)
    U_cluster.scatter_add_(0, users_idx.unsqueeze(-1).expand_as(weighted_Fc), weighted_Fc)
    # U_cluster: (N_U, D)
    
    # --- Bước 3: Mix ---
    U_init = lambda_mix * U_item + (1.0 - lambda_mix) * U_cluster  # (N_U, D)
    
    print(
        f"  [SG-URInit] Done: "
        f"item_agg_norm={U_item.norm(dim=-1).mean():.4f}, "
        f"cluster_agg_norm={U_cluster.norm(dim=-1).mean():.4f}, "
        f"init_norm={U_init.norm(dim=-1).mean():.4f}"
    )
    return U_init
```

### 5.3 Gọi trong `prepare()` (thay dòng hiện tại)

```python
# Phần cuối của prepare() — thay vì:
# self.User.embeddings.weight.data.copy_(R @ mfeats)

# === SG-URInit: Semantically Guaranteed User Initialization ===
print(f"[SG-URInit] Computing semantically guaranteed user embeddings (K={cfg.sg_num_clusters}, λ={cfg.sg_lambda})...")
user_init = self.sg_urinit(
    mfeats_whitened=mfeats,
    n_clusters=cfg.sg_num_clusters,
    lambda_mix=cfg.sg_lambda,
)
self.User.embeddings.weight.data.copy_(user_init.to(cfg.device))
```

### 5.4 Hyperparameters mới cần thêm vào `cfg`

```python
# STAIR-v4 SG-URInit Hyperparameters
cfg.add_argument("--sg-num-clusters", type=int,   default=8,    help="K for K-Means in SG-URInit [4, 8]")
cfg.add_argument("--sg-lambda",       type=float, default=0.1,  help="mixing weight: lambda=local, (1-lambda)=cluster")
```

---

## 6. SO SÁNH KIẾN TRÚC CÁC PHIÊN BẢN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAIR Initialization Pipeline                        │
├─────────────┬───────────────────────────────────────────────────────────────┤
│  Version    │  Item Init                │  User Init                        │
├─────────────┼───────────────────────────┼───────────────────────────────────┤
│  Baseline   │  Multimodal whitened      │  R_left @ mfeats                  │
│  v1 (GCL)  │  Multimodal whitened      │  R_left @ mfeats                  │
│  v2 (Dyn.) │  Multimodal whitened      │  R_left @ mfeats                  │
│  v3.1      │  Multimodal whitened      │  R_left @ mfeats (giống baseline)  │
│  ▶ v4      │  Multimodal whitened      │  SG-URInit (KMeans + sym-norm)    │
└─────────────┴───────────────────────────┴───────────────────────────────────┘

SG-URInit = λ × (D_u^{-1/2} R D_i^{-1/2} @ F) + (1-λ) × (D_u^{-1} R @ F_cluster)
                        ↑ Local (exact)                    ↑ Global (semantic fallback)
```

---

## 7. CHI PHÍ TÍNH TOÁN

| Bước | Complexity | Thời gian ước tính |
|---|---|---|
| Build interaction matrix | O(E) | < 1 giây |
| Sym-normalized item aggregation | O(E × D) | < 2 giây |
| K-Means (K=8, N=7K-18K items) | O(N × K × D × iter) | 4–20 giây |
| Cluster assignment + aggregation | O(E × D) | < 2 giây |
| Tổng cộng | — | **< 30 giây** |

> Chi phí chạy **một lần duy nhất** trong `prepare()` trước khi training bắt đầu. Không ảnh hưởng tốc độ training.

---

## 8. KẾ HOẠCH THỰC THI

### Bước 1: Viết `main_v4.py`
- Copy `main_v3.py` → `main_v4.py`
- Thêm hyperparameters `--sg-num-clusters`, `--sg-lambda`
- Thêm phương thức `sg_urinit()` vào class `STAIR`
- Sửa phần cuối `prepare()` để gọi `sg_urinit()` thay vì `R_left @ mfeats`
- Cập nhật description string

### Bước 2: Test nhanh trên local
```bash
# Test build với Baby (1 epoch để kiểm tra khởi tạo)
python main_v4.py \
  --root ../../data \
  --dataset Amazon2014Baby_550_MMRec \
  --epochs 1 \
  --sg-num-clusters 8 --sg-lambda 0.1 \
  --seed 1
```
Kiểm tra output:
- `[SG-URInit] K-Means clustering (K=8)...` xuất hiện
- Norm của `U_init` ≈ 0.01–0.1 (hợp lý với embedding-dim=64)
- Không có NaN/Inf

### Bước 3: Chạy full trên Kaggle
```bash
# Baby
python main_v4.py \
  --root /kaggle/data \
  --dataset Amazon2014Baby_550_MMRec \
  --epochs 500 --batch-size 1024 --embedding-dim 64 \
  --num-layers 3 --num-neighbors 5-1 \
  --conf-delta 0.3 --conf-temp 1.0 --alpha-consensus 0.5 \
  --sg-num-clusters 8 --sg-lambda 0.1 \
  --optimizer adamwsevo --lr 1e-3 --weight-decay 0.1 --seed 1 \
  > log_stair_v4_baby.txt 2>&1

# Sports
python main_v4.py \
  --root /kaggle/data \
  --dataset Amazon2014Sports_550_MMRec \
  --epochs 500 --batch-size 1024 --embedding-dim 64 \
  --num-layers 3 --num-neighbors 5-1 \
  --conf-delta 0.3 --conf-temp 1.0 --alpha-consensus 0.5 \
  --sg-num-clusters 8 --sg-lambda 0.1 \
  --optimizer adamwsevo --lr 1e-3 --weight-decay 0.1 --seed 1 \
  > log_stair_v4_sports.txt 2>&1
```

### Bước 4: Phân tích kết quả
- So sánh Test metrics tại `Load best model` checkpoint
- Phân tích learning curves: Xem v4 có converge nhanh hơn v3.1 không (SG-URInit nên giúp training bắt đầu từ điểm tốt hơn → converge sớm hơn)
- Nếu Baby cải thiện rõ rệt (> +2%), confirm SG-URInit là đúng hướng

---

## 9. ABLATION STUDY ĐỀ XUẤT

Để làm rõ đóng góp của từng thành phần:

| Experiment | K | λ | Mô tả |
|---|---|---|---|
| v3.1 (baseline) | — | — | ClipFuse-Consensus, R_left init |
| v4-Local-Only | — | 1.0 | Chỉ dùng sym-normalized local aggregation |
| v4-Cluster-Only | 8 | 0.0 | Chỉ dùng cluster aggregation |
| **v4-Full (K=4)** | **4** | **0.1** | SG-URInit với K=4 |
| **v4-Full (K=8)** | **8** | **0.1** | SG-URInit với K=8 (khuyến nghị) |
| v4-Full (λ=0.5) | 8 | 0.5 | SG-URInit với trọng số balanced |

---

## 10. KỲ VỌNG KẾT QUẢ

Dựa trên kết quả của SG-URInit paper (tăng +1-3% trên FREEDOM, DRAGON, MMGCN), và phân tích root cause trên STAIR:

| Dataset | Metric | v3.1 (hiện tại) | v4 kỳ vọng | Mức cải thiện |
|---|---|:---:|:---:|:---:|
| **Baby** | Recall@20 | 0.0938 | ~0.096–0.103 | **+2% đến +10%** |
| **Baby** | NDCG@20 | 0.0412 | ~0.042–0.047 | **+2% đến +14%** |
| **Sports** | Recall@20 | 0.1124 | ~0.113–0.115 | **+0% đến +2%** |
| **Sports** | NDCG@20 | 0.0506 | ~0.051–0.052 | **+0% đến +3%** |

> **Lưu ý quan trọng:** Kỳ vọng cải thiện lớn hơn ở **Baby** vì user initialization noise ảnh hưởng mạnh hơn ở tập thưa. Sports đã vượt Baseline nên cải thiện sẽ nhỏ hơn.

---

## 11. RỦI RO VÀ BIỆN PHÁP GIẢM THIỂU

| Rủi ro | Xác suất | Biện pháp |
|---|---|---|
| K-Means không ổn định (random init) | Thấp | Dùng `random_state=seed`, `n_init=10` |
| K-Means chọn sai K → cluster quá thô | Trung bình | Ablation K=4 và K=8, report cả hai |
| `sklearn` chưa cài trên Kaggle | Thấp | `pip install scikit-learn` (có sẵn Kaggle) |
| User init norm quá lớn/nhỏ → gradient explosion | Thấp | `whitening()` đảm bảo scale hợp lý; kiểm tra norm trước khi copy |
| Sports có thể bị ảnh hưởng xấu | Thấp | Sports tốt ở v3.1 nhờ graph → user init là thứ yếu |

---

## 12. TÓM TẮT ĐÓNG GÓP KHOA HỌC CỦA v4

STAIR-v4 tích hợp SG-URInit vào pipeline STAIR theo hướng **model-agnostic, training-free**:

1. **Không thêm tham số học:** Toàn bộ tính toán SG-URInit là tiền xử lý, không có weight mới.
2. **Không phá vỡ các cải tiến v3.1:** Phần item graph (ClipFuse + Consensus Boosting) giữ nguyên 100%.
3. **Giải quyết đúng root cause còn lại:** Sparse user initialization trên Baby.
4. **Nhất quán về triết lý:** Zero-learnable-parameter principle, chỉ thay đổi khởi điểm embedding.
5. **Hiệu quả tính toán cao:** < 30 giây overhead, không ảnh hưởng training speed.

---

## 13. PHỤ LỤC: SG-URInit vs. `R_left @ mfeats` (Phân tích so sánh)

| Tiêu chí | `R_left @ mfeats` (v3.1) | SG-URInit (v4) |
|---|---|---|
| Normalization | Left (chia theo degree_u) | Sym (LightGCN-style) + Left (cluster) |
| Semantic scope | Local only (clicked items) | Local + Global (cluster context) |
| Sparse user handling | Kém (5 items → trung bình tệ) | Tốt (cluster fallback cung cấp global prior) |
| Dependency | Matrix multiply only | K-Means + 2 matrix multiplications |
| Complexity | O(E × D) | O(N_I × K + E × D) |
| Time | < 1s | 4–30s |

> **Kết luận:** SG-URInit cung cấp initialization chất lượng cao hơn đáng kể với chi phí overhead nhỏ — hoàn toàn phù hợp triết lý zero-extra-learnable-params của STAIR.
