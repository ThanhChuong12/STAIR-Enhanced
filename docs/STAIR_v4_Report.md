# STAIR-v4: SG-URInit Integration — Semantically Guaranteed User Representation Initialization

**Phiên bản:** STAIR-v4 (STAIR-SGInit)  
**Ngày:** 2026-08-11 (Cập nhật sau review)  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Cơ sở lý thuyết:** SG-URInit (SIGIR'26) + STAIR-v3.1 (ClipFuse-Consensus)  
**Trạng thái:** ✅ Implementation Ready — `main_v4.py` hoàn chỉnh

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
> v3.1 đã giải quyết hoàn toàn vấn đề **Modality Collapse** của v2 và lần đầu tiên vượt Baseline trên **Sports**. Tuy nhiên, tập **Baby** vẫn kém Baseline khoảng 9.3% (R@20: 0.0938 vs 0.1034). Root cause còn lại nằm ở **phía User Embedding**, không phải phía Item Graph.

### 1.2 Phân tích Root Cause còn lại: "User Initialization Noise"

Sau khi v3.1 tối ưu hoàn toàn phía item-item graph, điểm nghẽn còn lại là **khởi tạo user embedding**:

```
current STAIR prepare() user init:
  user_emb = R_left @ mfeats
  
  trong đó R_left = left-normalized user-item interaction matrix
  và mfeats = SVD-whitened multimodal features (prior-weighted)
```

**Bốn vấn đề cụ thể:**

1. **User tương tác ít (Sparse users):** Mỗi user trên Baby trung bình chỉ tương tác với ~5-7 items. Khi $R$ có hàng thưa, `R_left @ mfeats` chỉ trung bình trên 5-7 vector item — dễ bị nhiễu, không đại diện được sở thích toàn cục.

2. **Local Semantics Only — thiếu Global Context:** `R_left @ mfeats` thu thập thông tin **item-level local** (chỉ những items user đã click), bỏ qua **cluster-level global** (nhóm sở thích tổng quát).

3. **Asymmetric Initialization Gap:** Item embeddings được neo bởi FSC với đặc trưng whitened chất lượng cao, trong khi User embeddings chỉ được neo qua sparse `R_left` — hai bên không đối xứng.

4. **Tác động lớn hơn ở Baby vs. Sports:** Sports (18K items, nhiều tương tác) có neighborhood phong phú. Baby (7K items, ít tương tác) neighborhood quá nhỏ → init kém chất lượng.

---

## 2. GIẢI PHÁP: SG-URInit (STAIR-v4)

### 2.1 Ý tưởng cốt lõi của SG-URInit (SIGIR'26)

**Paper:** *"Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation"* (Xu et al., arXiv:2604.14839)

Thay vì sparse aggregation thuần túy, SG-URInit xây dựng user embedding từ **hai nguồn thông tin bổ sung nhau**:

$$\mathbf{u}_u^{\text{init}} = \lambda \cdot \underbrace{\mathbf{u}_u^{\text{local}}}_{\text{Item-level (sym-norm)}} + (1-\lambda) \cdot \underbrace{\mathbf{u}_u^{\text{global}}}_{\text{Cluster-level (K-Means)}}$$

---

## 3. TOÁN HỌC CHI TIẾT (Đã đồng bộ với code thực tế)

> **Lưu ý thiết kế:** SG-URInit trong STAIR-v4 hoạt động trên **một lần duy nhất** trên tensor `mfeats_whitened` — đây là tensor item feature đã được prior-weighted 5:1 và SVD-whitened. Không có bước tính riêng từng modality rồi fuse vì User và Item đều sống trong cùng không gian embedding đã fuse từ trước. Điều này đảm bảo nhất quán 100% giữa User và Item initialization.

### 3.1 Định nghĩa ký hiệu

| Ký hiệu | Định nghĩa |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | Tập users và items |
| $\mathcal{N}(u)$ | Tập items user $u$ đã tương tác (training set) |
| $\mathbf{F} \in \mathbb{R}^{N_I \times D}$ | Prior-weighted whitened item features (đầu vào cho User init, trùng với Item embedding) |
| $K$ | Số clusters K-Means (mặc định 8) |
| $\mathcal{C} = \{c_1, \ldots, c_K\}$ | Centroids sau K-Means trên `mfeats_whitened` |
| $\lambda$ | Local mixing weight ($\lambda = 0.1$, khuyến nghị từ paper cho model có item-item graph) |

### 3.2 Bước 1: Local — Sym-normalized Item Aggregation

Dùng symmetric degree normalization kiểu LightGCN (khác `R_left` của baseline):

$$\mathbf{U}^{\text{local}} = D_U^{-1/2} R D_I^{-1/2} \mathbf{F}$$

trong đó:
- $D_U[u,u] = |\mathcal{N}(u)|$ (user degree)
- $D_I[i,i] = |\mathcal{N}(i)|$ (item degree — số users đã tương tác với item $i$)
- Symmetric normalization tránh user có nhiều interaction chiếm ưu thế quá mức

### 3.3 Bước 2: Global — K-Means Cluster Aggregation

**Sub-bước 2a: K-Means clustering**
$$\{c_1, \ldots, c_K\} = \text{KMeans}(\mathbf{F}, K=8, \text{seed}=1)$$

**Sub-bước 2b: Gán centroid cho từng item**
$$\mathbf{F}^{\text{clust}}[i] = c_{\arg\min_{k} \|\mathbf{F}[i] - c_k\|_2}$$

**Sub-bước 2c: Left-normalized aggregation cho user**
$$\mathbf{U}^{\text{global}} = D_U^{-1} R \mathbf{F}^{\text{clust}}$$

> **Ý nghĩa:** Thay vì lấy đặc trưng cụ thể của item $i$ (nhiễu cục bộ), lấy centroid cluster của $i$ — vector "mượt hơn", đại diện cho nhóm sở thích mà item $i$ thuộc về. Đặc biệt hiệu quả khi users có ít interaction.

### 3.4 Bước 3: Mix

$$\mathbf{U}^{\text{mix}} = \lambda \cdot \mathbf{U}^{\text{local}} + (1-\lambda) \cdot \mathbf{U}^{\text{global}}, \quad \lambda = 0.1$$

### 3.5 Bước 4: Scale Alignment *(bổ sung quan trọng)*

K-Means centroids là trung bình các vector → norm nhỏ hơn item gốc. Với $\lambda=0.1$, `U_mix` bị chi phối 90% bởi `U_global` có norm nhỏ → lệch scale với `Item.embeddings`.

**Giải pháp:** rescale để user và item ở cùng norm space tại epoch 0:

$$\mathbf{U}^{\text{init}} = \mathbf{U}^{\text{mix}} \cdot \frac{\overline{\|\mathbf{F}\|_2}}{\overline{\|\mathbf{U}^{\text{mix}}\|_2}}$$

trong đó $\overline{\|\cdot\|_2}$ là mean per-vector norm. Điều này đảm bảo BPR scores $\mathbf{u} \cdot \mathbf{i}$ không bị méo ngay tại epoch 0.

---

## 4. VỊ TRÍ TÍCH HỢP VÀO STAIR (Chi tiết kỹ thuật)

### 4.1 Hai thay đổi duy nhất trong code

```
main_v3.py → main_v4.py:
  (1) Thêm phương thức sg_urinit() vào class STAIR
  (2) Thay 1 dòng ở cuối prepare():
        CŨ: self.User.embeddings.weight.data.copy_(R @ mfeats)
        MỚI: user_init = self.sg_urinit(mfeats, edge_index_u2i, ...)
             self.User.embeddings.weight.data.copy_(user_init.to(cfg.device))
```

### 4.2 Thiết kế `sg_urinit()` hoàn chỉnh (với tất cả các fix)

```python
def sg_urinit(
    self,
    mfeats_whitened: torch.Tensor,  # (N_I, D) — SAME tensor as Item.embeddings
    edge_index_u2i: torch.Tensor,   # (2, E)   — REUSE từ prepare(), không gọi API mới
    n_clusters: int = 8,
    lambda_mix: float = 0.1,
) -> torch.Tensor:
    # Fix 1: Explicit int64
    users_idx = edge_index_u2i[0].long()
    items_idx  = edge_index_u2i[1].long()
    
    # Compute degrees
    deg_u = torch.zeros(N_U).scatter_add_(0, users_idx, torch.ones(E))
    deg_i = torch.zeros(N_I).scatter_add_(0, items_idx, torch.ones(E))
    
    # Fix 2: Cold user warning
    cold_users = (deg_u == 0).sum().item()
    if cold_users > 0:
        print(f"WARNING: {cold_users} users with 0 interactions → global mean fallback")
    
    # Bước 1: Local (sym-norm)
    ew_sym = 1.0 / (deg_u[users_idx].sqrt() * deg_i[items_idx].sqrt())
    U_local = scatter_add(F_white[items_idx] * ew_sym, users_idx)
    
    # Bước 2: K-Means
    km = KMeans(K, random_state=seed, n_init=10).fit(F_white.numpy())
    
    # Fix 3: Cluster diagnostics
    counts = torch.bincount(clust_idx)
    print(f"Cluster sizes: {counts.tolist()}")
    if counts.max() / N_I > 0.7:
        print("WARNING: degenerate clustering detected")
    
    # Left-norm aggregation
    U_global = scatter_add(F_cluster[items_idx] * ew_left, users_idx)
    
    # Bước 3: Mix
    U_mix = lambda_mix * U_local + (1 - lambda_mix) * U_global
    
    # Bước 4: Fix 4 — Scale alignment
    scale = F_white.norm(dim=-1).mean() / U_mix.norm(dim=-1).mean().clamp(1e-8)
    U_init = U_mix * scale
    
    return U_init
```

### 4.3 Tại sao dùng lại `edge_index_u2i` từ `prepare()`?

Trong `prepare()` đã có sẵn:
```python
edge_index_u2i, edge_weight_u2i = freerec.graph.to_normalized(
    self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index,
    normalization='left'
)
```

`sg_urinit()` nhận trực tiếp `edge_index_u2i` (không dùng `edge_weight_u2i` vì tự tính sym/left norm riêng). Điều này đảm bảo **cùng một topology đồ thị** — không rủi ro freerec xử lý khác nhau ở hai lần gọi API.

---

## 5. CÁC KỸ THUẬT FIX ĐÃ TÍCH HỢP (Engineering Hardening)

| # | Vấn đề | Fix trong `main_v4.py` |
|---|---|---|
| 1 | Scale mismatch User vs Item init | Rescale `U_init` theo mean item norm |
| 2 | API inconsistency | Tái sử dụng `edge_index_u2i` có sẵn trong `prepare()` |
| 3 | Math/code inconsistency (modality-by-modality) | Dùng single-pass trên `mfeats` đã fused |
| 4 | `scatter_add_` cần int64 | Ép kiểu tường minh `users_idx.long()` |
| 5 | Cold users (0 interactions) | Global mean item embedding fallback + warning |
| 6 | Degenerate K-Means (cluster quá lớn) | Print cluster size distribution, warning nếu >70% |
| 7 | Sports biên độ mỏng | Ablation λ=1.0 có sẵn như "safety brake" |

---

## 6. SO SÁNH KIẾN TRÚC CÁC PHIÊN BẢN

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STAIR Pipeline Comparison                                 │
├──────────┬────────────────────────┬────────────────────────────────────────────────┤
│ Version  │ Item Graph             │ User Initialization                             │
├──────────┼────────────────────────┼────────────────────────────────────────────────┤
│ Baseline │ Static (5:1 kNN)       │ R_left @ mfeats                                │
│ v1 GCL   │ Static + GCL loss      │ R_left @ mfeats                                │
│ v2 Dyn.  │ DyFuse (collapsed)     │ R_left @ mfeats                                │
│ v3.1     │ ClipFuse-Consensus ✓   │ R_left @ mfeats (unchanged)                    │
│ ▶ v4     │ ClipFuse-Consensus ✓   │ SG-URInit: λ·Sym_norm + (1-λ)·KMeans_cluster  │
│          │ (identical to v3.1)    │ + scale alignment to item norm                 │
└──────────┴────────────────────────┴────────────────────────────────────────────────┘
```

---

## 7. CHI PHÍ TÍNH TOÁN

| Bước | Complexity | Thời gian ước tính |
|---|---|---|
| Sym-normalized item aggregation | O(E × D) | < 2 giây |
| K-Means (K=8, N_I=7K-18K, D=64) | O(N_I × K × D × iter) | 5–20 giây |
| Cluster assignment + left aggregation | O(N_I × K + E × D) | < 3 giây |
| Scale alignment | O(N_U × D) | < 0.1 giây |
| **Tổng** | — | **< 30 giây** |

> Chi phí chạy **một lần duy nhất** trước training. Không ảnh hưởng tốc độ training.

---

## 8. KẾ HOẠCH THỰC NGHIỆM

### 8.1 Lệnh chạy trên Kaggle (main experiment)

```bash
# Baby — K=8, λ=0.1 (cluster-dominant)
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
  [same args] > log_stair_v4_sports.txt 2>&1
```

### 8.2 Ablation Study

| Config | K | λ | Mục đích |
|---|---|---|---|
| v4-LocalOnly | — | 1.0 | Sym-norm only (safety brake cho Sports) |
| **v4-Full (K=8)** | **8** | **0.1** | **Main experiment (cluster-dominant)** |
| v4-Full (K=4) | 4 | 0.1 | Baby có 7K items — K nhỏ hơn có thể tốt hơn |
| v4-Balanced | 8 | 0.5 | Balanced local+global |

---

## 9. RỦI RO VÀ BIỆN PHÁP GIẢM THIỂU

| Rủi ro | Xác suất | Biện pháp |
|---|---|---|
| K-Means không ổn định | Thấp | `random_state=seed`, `n_init=10` |
| K-Means chọn sai K → cluster thô | Trung bình | Ablation K=4 và K=8; print cluster sizes |
| Sports tụt về dưới Baseline | **Trung bình** | Chạy ablation λ=1.0 (local-only) làm safety brake |
| Scale mismatch | Thấp | Đã fix: rescale theo item norm |
| Cold users | Thấp | Đã fix: global mean fallback |
| Degenerate clustering (>70%) | Thấp-Trung | Đã fix: print warning + suggest giảm K |

> **Ghi chú Sports:** Biên độ vượt Baseline của v3.1 chỉ là +0.45% đến +1.1% — rất mỏng. SG-URInit tác động chủ yếu vào phía User, trong khi ưu thế của Sports ở v3.1 đến từ Item Graph (ClipFuse). Do đó rủi ro Sports tụt xuống là Trung bình, không Thấp.

---

## 10. KỲ VỌNG KẾT QUẢ

> **Lưu ý:** $\lambda=0.1$ là khởi điểm từ khuyến nghị của paper gốc (calibrate trên FREEDOM/DRAGON/MMGCN). Đây **chưa phải** giá trị tối ưu đã xác nhận cho kiến trúc FSC/BSC của STAIR. Ablation là cần thiết để xác nhận.

| Dataset | Metric | v3.1 | v4 kỳ vọng | Mức cải thiện |
|---|---|:---:|:---:|:---:|
| **Baby** | Recall@20 | 0.0938 | ~0.096–0.104 | **+2% đến +11%** |
| **Baby** | NDCG@20 | 0.0412 | ~0.042–0.047 | **+2% đến +14%** |
| **Sports** | Recall@20 | 0.1124 | ~0.112–0.115 | **−0.5% đến +2%** |
| **Sports** | NDCG@20 | 0.0506 | ~0.050–0.052 | **−1% đến +3%** |

---

## 11. TÓM TẮT ĐÓNG GÓP KHOA HỌC CỦA v4

STAIR-v4 tích hợp SG-URInit theo hướng **model-agnostic, training-free, zero-learnable-parameter**:

1. **Không thêm tham số học:** Toàn bộ tính toán SG-URInit là tiền xử lý (< 30 giây).
2. **Không phá vỡ các cải tiến v3.1:** Item graph (ClipFuse + Consensus Boosting) giữ nguyên 100%.
3. **Giải quyết đúng root cause còn lại:** Sparse user initialization noise trên Baby.
4. **Nhất quán về triết lý:** Tiếp nối zero-parameter principle đã thành công ở v1, v3.1.
5. **Bốn engineering hardening:** Scale alignment, graph consistency, cluster diagnostics, cold-user fallback.

---

## 12. PHỤ LỤC: So sánh `R_left @ mfeats` vs. SG-URInit

| Tiêu chí | `R_left @ mfeats` (v3.1) | SG-URInit (v4) |
|---|---|---|
| Normalization | Left (÷ deg_u) | Sym-norm (local) + Left (cluster) |
| Semantic scope | Local (clicked items only) | Local + Global (cluster context) |
| Sparse user quality | Kém (5 items → noisy mean) | Tốt (centroid fallback = semantic prior) |
| Scale alignment | Không | Có (rescaled to item norm) |
| Complexity | O(E × D) | O(N_I × K + E × D) + KMeans |
| Time overhead | < 1s | 5–30s (one-time) |
