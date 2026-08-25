# BÁO CÁO PHÂN TÍCH THẤT BẠI V1 VÀ KẾ HOẠCH CẢI TIẾN GIAI ĐOẠN 2 — ĐỢT 2 (STAIR-Enhanced v2)
## Module 1 Revised: Residual-Whitening Projector với Warm-start SVD và Micro-Ablation

**Phiên bản:** STAIR-Enhanced v2 (STAIR-ResProjector)  
**Ngày:** 2026-08-25  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Cơ sở lý thuyết:** Negative Results Analysis từ v1 + Residual Design Pattern + Warm-start SVD  
**Trạng thái:** 🔬 Phân tích hoàn thành — Đang triển khai implementation

---

## 1. KẾT QUẢ THỰC NGHIỆM V1 — SỐ LIỆU CỤ THỂ

### 1.1 Bảng so sánh định lượng: STAIR-Baseline vs. STAIR-Enhanced v1

| Tập dữ liệu | Chỉ số | Baseline (Table 2) | Enhanced v1 | Delta (%) |
|---|---|:---:|:---:|:---:|
| **Baby** | Recall@10 | 0.0674 | 0.0611 | **−9.4%** |
| | Recall@20 | 0.1042 | 0.0948 | **−9.0%** |
| | NDCG@10 | 0.0359 | 0.0325 | **−9.5%** |
| | NDCG@20 | 0.0454 | 0.0412 | **−9.3%** |
| **Sports** | Recall@10 | 0.0743 | 0.0695 | **−6.5%** |
| | Recall@20 | 0.1111 | 0.1040 | **−6.4%** |
| | NDCG@10 | 0.0405 | 0.0376 | **−7.2%** |
| | NDCG@20 | 0.0500 | 0.0466 | **−6.8%** |
| **Electronics** | Recall@10 | 0.0442 | 0.0401 | **−9.3%** |
| | Recall@20 | 0.0665 | 0.0601 | **−9.6%** |
| | NDCG@10 | 0.0246 | 0.0223 | **−9.3%** |
| | NDCG@20 | 0.0303 | 0.0274 | **−9.6%** |

**Nhận xét:**
> Sụt giảm đồng đều trên **cả 3 dataset** (không có ngoại lệ) và đồng đều trên **cả 4 chỉ số**, với biên độ từ $-6.4\%$ đến $-9.6\%$. Đây **không phải nhiễu ngẫu nhiên** — đây là một *thất bại hệ thống sạch* (systematic failure), tức là một thành phần thiết kế cốt lõi đang gây hại nhất quán.

---

## 2. CHẨN ĐOÁN NGUYÊN NHÂN GỐC RỄ (Root Cause Analysis)

### 2.1 Nguyên nhân 1 (Nghiêm trọng nhất): Phá vỡ "Structural Prior" của SVD Whitening

**SVD Whitening trong STAIR gốc không chỉ đơn thuần là khởi tạo — nó định hình không gian embedding.**

Hàm `whitening()` trong `main.py`:
```python
def whitening(self, feats: torch.Tensor):
    feats = feats - feats.mean(0, keepdim=True)   # 1. Khử kỳ vọng
    feats, _, _ = torch.linalg.svd(feats, full_matrices=False)  # 2. SVD
    return feats[:, :64] * math.sqrt(N_items / 64)  # 3. Cắt top-64 & scale
```

Phép biến đổi này tạo ra:
- **Decorrelated features:** Các cột của ma trận U trong SVD là trực giao hoàn toàn → tất cả 64 chiều đều độc lập tuyến tính với nhau.
- **Equal-variance spectrum:** Mỗi chiều có phương sai gần bằng 1 (do tính chất của singular vectors).
- **Magnitude chuẩn:** Scale $\sqrt{N_{items}/D}$ đặt embedding ở đúng vùng tham chiếu mà Smoother (BSC) dự kiến.

**Hệ quả khi xóa bỏ whitening:**

Công thức $\beta_j$ trong paper (Eq. 5) — tham số BSC:
$$\beta_j = 0.1 + 0.9 \cdot \left(\frac{j}{D}\right)^{\gamma}, \quad j = 0, \ldots, D-1$$

Và $\gamma$ được tune thực nghiệm cho từng dataset:
- Baby: $\gamma = 0.1$ → spectrum gần bằng phẳng (nhạy cảm với nhiễu)  
- Sports: $\gamma = 0.2$ → spectrum trung bình  
- Electronics: $\gamma = 0.4$ → spectrum dốc mạnh (giảm smoothing ở chiều cao)

Tất cả các giá trị $\gamma$ này **được tune riêng trên phân phối whitened embeddings** (decorrelated, equal-variance). Khi `DeRedundantGatedProjector` thay thế, item embeddings đầu ra từ projector là **heteroscedastic** (phương sai không đồng nhất theo chiều), khiến Smoother nhận tín hiệu có phân phối spectrum hoàn toàn khác → mismatch hyperparameter nghiêm trọng.

### 2.2 Nguyên nhân 2: Mismatch Learning Rate — Projector Học Quá Chậm

**Code `marked_params()` trong Enhanced v1:**
```python
def marked_params(self):
    return [
        {'params': self.User.parameters(),      'smoother': None},
        {'params': self.Item.parameters(),      'smoother': Smoother(...)},
        {'params': self.projector.parameters(), 'smoother': None},  # Dùng chung lr=1e-3
    ]
```

`AdamWSEvo` với `lr=1e-3` và `weight_decay=0.3` (Baby) được **tune để fine-tune embeddings đã có sẵn điểm khởi đầu tốt** (từ whitened features). Đây là learning rate rất nhỏ.

Projector là tham số **mới hoàn toàn** với random init (Kaiming) — cần learning rate cao hơn ít nhất 5-10x ở giai đoạn đầu để thoát khỏi vùng không gian nhiễu và học cấu trúc của dữ liệu. Với `lr=1e-3`, trong 500 epoch (mỗi epoch ~5-7 giây), projector chưa hội tụ đến điểm tốt thì training đã kết thúc.

### 2.3 Nguyên nhân 3: Gate Collapse — Cổng Gated Hội tụ Thiên Lệch

Không có logging riêng cho gate weight $\mathbf{z} \in (0,1)^{64}$ qua các epoch, nên không thể xác nhận chắc chắn — nhưng đây là hiện tượng phổ biến trong Gated Fusion:

Nếu gradient từ BPR loss truyền ngược qua gate thấy rằng **chỉ giảm loss khi ưu tiên text** (do text thường stable hơn visual sau projector), thì $\mathbf{z} \to \mathbf{1}$ (luôn chọn text, bỏ qua visual). Kết quả: $\mathbf{h}_{fused} \approx \mathbf{h}_t^{proj}$ — đây là một mô hình text-only kém whitened STAIR gốc cả về chất lượng lẫn đa phương thức.

### 2.4 Nguyên nhân 4: Null-Space Projection Quá Tích cực (Over-aggressive)

EMA Null-space Projection với `null_rank=16` loại bỏ 16/64 chiều (25%) của embedding space. Các chiều bị loại bỏ là những PC có phương sai chung cao nhất — nhưng trong bối cảnh STAIR, những PC này có thể chứa **tín hiệu semantic quan trọng** (chủng loại sản phẩm, màu sắc nổi bật, giá cả) chứ không phải nhiễu thuần túy.

---

## 3. TẠI SAO KHOẢNG CÁCH CỐ ĐỊNH ~9% TRÊN MỌI DATASET

**Lý thuyết:** Nếu mô hình không học được gì có ích từ projector (do mismatch lr + gate collapse), thì:
- `Item.embeddings` xuất phát từ random projector output (không có structural prior)
- FSC/BSC smooth embeddings ban đầu không tốt → propagate noise
- Kết quả hội tụ về một "capacity ceiling" nhất định, thấp hơn baseline một khoảng cố định

Khoảng $-9\%$ phản ánh **chi phí mất đi** khi xóa bỏ structural prior của SVD mà không có cơ chế bù đắp hiệu quả. Đây là giải thích hợp lý nhất cho hiện tượng sụt giảm nhất quán.

---

## 4. KẾ HOẠCH STAIR-ENHANCED V2: RESIDUAL-WHITENING PROJECTOR

### 4.1 Nguyên tắc Thiết kế (Design Philosophy)

**"Không đập bỏ — hãy bồi thêm" (Don't Replace — Augment)**

Thay vì thay thế hoàn toàn SVD Whitening, v2 giữ nguyên "món quà miễn phí" từ SVD và thêm projector như một **nhánh residual** học phần bổ sung:

$$\mathbf{e}_i^{v2} = \underbrace{\text{whitening}(x_{t,i}, x_{v,i})}_{\text{Structural Prior (frozen)}} + \lambda_{res} \cdot \underbrace{\text{ResProjector}(x_{t,i}, x_{v,i})}_{\text{Nonlinear Correction (learned)}}$$

Trong đó $\lambda_{res} \in (0, 1)$ là scalar điều chỉnh mức độ đóng góp của nhánh correction (mặc định $\lambda_{res} = 0.1$ để bắt đầu nhỏ).

### 4.2 Warm-start SVD cho Projector

Thay vì khởi tạo ngẫu nhiên (Kaiming), khởi tạo trọng số projector gần với phép chiếu SVD:
- **`text_proj`:** Khởi tạo $W_t \approx V_t[:64, :]^\top$ từ SVD của text feature matrix, sao cho $h_t \approx \text{whitening}(x_t)$ tại epoch 0.
- **`vis_proj`:** Tương tự với $W_v \approx V_v[:64, :]^\top$.

Điều này đảm bảo **điểm khởi đầu của projector tương đương whitening baseline** — projector chỉ cần learn phần deviation/correction nhỏ thay vì học lại từ đầu.

### 4.3 Learning Rate Tách biệt cho Projector

```python
def marked_params(self):
    return [
        {'params': self.User.parameters(),           'smoother': None,        'lr': cfg.lr},
        {'params': self.Item.parameters(),           'smoother': Smoother(...), 'lr': cfg.lr},
        {'params': self.res_projector.parameters(),  'smoother': None,        'lr': cfg.lr * 5},
    ]
```

Projector dùng `lr * 5` để học nhanh hơn trong giai đoạn đầu.

### 4.4 Cô lập Thành phần qua Micro-Ablation

Thay vì gộp Projector + Gate + Null-space + EMA cùng lúc (như v1), v2 tách thành **3 biến thể riêng biệt** để cô lập:

| Biến thể | Cơ chế | Mục đích Ablation |
|---|---|---|
| **v2a (ResOnly)** | `e = whitening(x) + λ·MLP(x)`, không có Gate/EMA | Kiểm tra: Residual connection có cải thiện không? |
| **v2b (GateOnly)** | `e = whitening(x)`, thêm Gate để chọn text/visual weight | Kiểm tra: Gate riêng lẻ có tác dụng không? |
| **v2c (Full)** | `e = whitening(x) + λ·GatedMLP(x)`, Gate + Residual | Tổng hợp nếu cả 2 biến thể trên đều có hiệu quả |

---

## 5. TOÁN HỌC CHI TIẾT: RESIDUAL-WHITENING PROJECTOR (V2A)

### 5.1 Bước 1: Whitening (giữ nguyên từ STAIR gốc)

$$\mathbf{F}_{wt} = \mathbf{U}_t[:, :D] \cdot \sqrt{\frac{N_I}{D}} \in \mathbb{R}^{N_I \times D}$$
$$\mathbf{F}_{wv} = \mathbf{U}_v[:, :D] \cdot \sqrt{\frac{N_I}{D}} \in \mathbb{R}^{N_I \times D}$$

Gộp theo tỉ lệ `num_neighbors` ($k_t=5$, $k_v=1$):
$$\mathbf{E}_{svd} = \frac{5 \cdot \mathbf{F}_{wt} + 1 \cdot \mathbf{F}_{wv}}{6} \in \mathbb{R}^{N_I \times D}$$

### 5.2 Bước 2: Residual Projector (học được)

$$\mathbf{h}_{t,i} = \text{GELU}(W_t x_{t,i} + b_t) \in \mathbb{R}^D$$
$$\mathbf{h}_{v,i} = \text{GELU}(W_v x_{v,i} + b_v) \in \mathbb{R}^D$$
$$\Delta_i = \text{LayerNorm}\left(\frac{\mathbf{h}_{t,i} + \mathbf{h}_{v,i}}{2}\right) \in \mathbb{R}^D$$

LayerNorm đảm bảo nhánh correction có phân phối chuẩn, không phá vỡ scale của nhánh SVD.

### 5.3 Bước 3: Kết hợp Residual

$$\mathbf{e}_i^{final} = \mathbf{E}_{svd, i} + \lambda_{res} \cdot \Delta_i$$

Scale cuối:
$$\mathbf{e}_i^{final} \leftarrow \frac{\mathbf{e}_i^{final}}{\|\mathbf{e}_i^{final}\|_2} \cdot \sqrt{\frac{N_I}{D}}$$

### 5.4 Warm-start Projector (Khởi tạo xấp xỉ SVD)

Để warm-start $W_t$:
```python
# Tính SVD của text_feat (N x 384) → U_t (N x 64)
U_t, _, V_t = torch.linalg.svd(text_feat, full_matrices=False)
# V_t: (384 x 384), cột đầu tiên là PC1
# W_t ~ V_t[:, :64].T  sao cho W_t @ x_t ≈ U_t (trước scale)
W_t_init = V_t[:64, :]  # (64 x 384)
self.res_projector.text_proj[0].weight.data.copy_(W_t_init)
```

Với warm-start, tại epoch 0: $\mathbf{h}_{t,i} = \text{GELU}(W_t x_{t,i}) \approx$ phép chiếu SVD, tức $\Delta_i \approx 0$ (correction nhỏ). Mô hình bắt đầu từ điểm tương đương baseline và chỉ cần **learn deviation nhỏ** để cải thiện.

---

## 6. SO SÁNH THIẾT KẾ V1 VỚI V2

| Khía cạnh | Enhanced v1 (Thất bại) | Enhanced v2 (Kế hoạch sửa) |
|---|---|---|
| **Cách dùng SVD** | Xóa hoàn toàn → mất structural prior | Giữ làm nhánh chính (frozen) |
| **Vai trò Projector** | Thay thế whitening | Bổ sung correction (residual) |
| **Khởi tạo Projector** | Kaiming random → bắt đầu từ noise | Warm-start từ SVD → bắt đầu từ baseline |
| **Learning rate Projector** | Chung lr=1e-3 với embedding | Tách riêng lr_proj = 5e-3 |
| **Null-space Projection** | Loại bỏ 16/64 chiều → quá tích cực | Bỏ, thay bằng LayerNorm nhẹ hơn |
| **Gate Collapse** | Không có logging → không phát hiện | Thêm log gate entropy mỗi 50 epoch |
| **Ablation Strategy** | Gộp tất cả (black-box) | Micro-ablation 3 biến thể độc lập |
| **Tính tương thích YAML** | Dùng chung γ của baseline | Re-tune γ riêng cho từng dataset |

---

## 7. CHIẾN LƯỢC THỰC NGHIỆM V2

### 7.1 Thứ tự ưu tiên chạy Kaggle

```
Đợt 1 (Session 1, ~3h):
  v2a-baby:   ResOnly  + warm-start, Baby,  lr_proj=5e-3, λ_res=0.1
  v2b-baby:   GateOnly + whitening,  Baby,  lr_proj=5e-3
  → So sánh ngay: v2a vs v2b vs Baseline trên Baby để chọn winner

Đợt 2 (Session 2, ~5h):
  Winner-sports:    Chạy biến thể tốt nhất trên Sports
  Winner-electronics: Chạy biến thể tốt nhất trên Electronics

Đợt 3 (nếu cần, Session 3):
  v2c-full: ResOnly + Gate + warm-start trên tất cả 3 datasets
```

### 7.2 Tiêu chí Thành công

- **Ngưỡng tối thiểu:** Recall@20 không thấp hơn Baseline trên mọi dataset.
- **Ngưỡng kỳ vọng:** $+2\%$ trở lên trên ít nhất 1 dataset.
- **Ngưỡng lý tưởng:** $+3\%$ đến $+5\%$ trên Baby (dataset dễ cải thiện nhất).

### 7.3 Giá trị Khoa học của Kết quả Âm V1

Kể cả khi v1 không cải thiện, đây là **Negative Result có giá trị** cho Khóa luận:
- Chứng minh SVD Whitening trong STAIR không thể được thay thế đơn giản mà không gây regression.
- Phân tích định lượng lý do thất bại (structural prior mismatch, lr mismatch, gate collapse).
- Thiết lập rõ ràng cận dưới của chi phí khi mất đi decorrelated spectrum (~9%).
- Mở ra hướng nghiên cứu về **Hybrid Architecture** (whitening + learned correction) thay vì end-to-end replacement.

---

## 8. PHỤ LỤC: PHÂN TÍCH CODE THỰC TẾ V1

### 8.1 Đoạn Code Gây Ra Structural Mismatch

```python
# main_enhanced_v1.py — prepare() — NGUYÊN NHÂN CHÍNH
text_feat, visual_feat = mfeats_raw[0], mfeats_raw[1]
mfeats = prepare_item_embeddings(
    self.projector, text_feat, visual_feat,
    n_items=self.Item.count, embedding_dim=cfg.embedding_dim
)
self.Item.embeddings.weight.data.copy_(mfeats)
# ↑ mfeats từ projector (random init) KHÔNG DECORRELATED
# BSC Smoother với γ=0.1/0.2/0.4 đang giả định mfeats là whitened!
```

```python
# main.py (STAIR gốc) — SO SÁNH
mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, num_neighbors)]
mfeats = sum(mfeats).div(sum(num_neighbors))
# ↑ mfeats từ whitening: decorrelated, equal-variance spectrum
# cfg.beta3 được tune trực tiếp cho phân phối này
```

### 8.2 Kiến trúc V1 vs V2 (Flow Diagram)

```
V1 (Thất bại):
Text(384) ──►[MLP+GELU]──►h_t──►[P_null]──►h_t_proj──►[Gate z]──►fused──►[L2+scale]──► e_item
Vis(4096) ──►[MLP+GELU]──►h_v──►[P_null]──►h_v_proj──►[1-z   ]──►      ↑
(whitening() bị xóa hoàn toàn)

V2a (Kế hoạch - ResOnly):
Text(384) ──►[SVD+scale]────────────────────────────────────────────────────► E_svd (FROZEN)
Vis(4096) ──►[SVD+scale] (weighted 5:1)                                                  │ +
                                                                                          ↓
Text(384) ──►[WarmMLP+GELU]──►h_t──►                                                     │
                                    ├──►[sum/2]──►[LayerNorm]──►Δ──►[λ_res scale]────────┘
Vis(4096) ──►[WarmMLP+GELU]──►h_v──►                           (residual correction)
```

---

*Báo cáo này phục vụ cho mục "Phân tích Kết quả Âm" trong Chương 5 của Khóa luận tốt nghiệp.*  
*Kết quả thực nghiệm v2 sẽ được cập nhật sau khi hoàn thành các session Kaggle.*
