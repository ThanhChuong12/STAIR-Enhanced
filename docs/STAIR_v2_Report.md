# STAIR-v2: Modality-Adaptive kNN Graph Reweighting (STAIR-DyFuse)

## 1. Tổng quan

STAIR-DyFuse là hướng cải tiến thứ hai của mô hình STAIR, tập trung vào việc xây dựng đồ thị kNN (kNN graph) cho backward stepwise convolution (BSC) chính xác và phù hợp hơn với từng sản phẩm. Điểm khác biệt so với cải tiến v1 là phương pháp này can thiệp trực tiếp vào đồ thị `mAdj` thay vì can thiệp vào hàm mất mát.

---

## 2. Phân tích mã nguồn STAIR gốc

Trong `main.py` của STAIR (xem hàm `prepare`, dòng 122–167), quy trình xây dựng `mAdj` hiện tại là:

```python
# Tách riêng từng kNN graph cho từng modality
edge_index = torch.cat(
    [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
    dim=1
)
# Gộp tất cả cạnh bằng reduce='sum', sau đó to_undirected với reduce='max'
edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
```

**Vấn đề cốt lõi:** Mặc định `num-neighbors='5-1'` tức textual kNN dùng k=5, visual kNN dùng k=1. Sau khi coalesce và `reduce='max'`, toàn bộ cạnh được gộp lại với trọng số bằng nhau (all weights = 1.0 sau normalize). Cách này không phân biệt được sản phẩm nào thiên về visual, sản phẩm nào thiên về textual — mọi sản phẩm đều nhận được gradient BSC theo cùng một tỷ lệ.

---

## 3. Học hỏi từ TAMER

Từ mã nguồn `TAMER/models/tamer.py`, cụ thể đoạn (dòng 134–136):

```python
# TAMER gộp nhiều đồ thị kNN bằng weighted sum với alpha tĩnh
self.mm_adj = (self.alpha_text_p * self.p_adj
             + self.alpha_image * self.image_adj
             + self.alpha_interest * self.co_adj
             + self.alpha_text_z * self.z_adj)
```

TAMER đã đi đúng hướng khi dùng weighted sum thay vì max-pooling. Tuy nhiên, trọng số `alpha_text_p`, `alpha_image` là các **siêu tham số tĩnh** được cấu hình thủ công — tức cùng một trọng số áp dụng cho mọi sản phẩm, không phân biệt được những sản phẩm có visual feature mạnh hay textual feature mạnh.

**Đột phá STAIR-DyFuse so với TAMER:** Thay vì dùng trọng số tĩnh toàn cục, tính trọng số ở mức **từng sản phẩm** dựa trên norm của feature vector — hoàn toàn parameter-free.

---

## 4. Học hỏi từ NLGCL+

Từ mã nguồn `NLGCL-Plus/src/models/freedom_plus.py`, cụ thể hàm `calculate_loss` (dòng 226–253):

```python
# NLGCL+ dùng weighted sum tĩnh cho mm_adj
self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
```

NLGCL+ cũng dùng weighted sum tĩnh với một siêu tham số `mm_image_weight` duy nhất. Điều thú vị là NLGCL+ giới thiệu **adaptive sample weighting** cho InfoNCE loss — tức dùng norm của multimodal feature để điều chỉnh trọng số contrastive loss, không phải điều chỉnh graph. Ý tưởng dùng norm vector feature như một **confidence signal** đã được NLGCL+ chứng minh là hiệu quả. STAIR-DyFuse vận dụng nguyên lý đó nhưng áp dụng vào bài toán graph reweighting.

---

## 5. Thiết kế STAIR-DyFuse: Modality-Adaptive Graph Reweighting

### 5.1. Động lực

Mỗi sản phẩm có đặc tính khác nhau:
- Sản phẩm thời trang (quần áo, giày dép): visual feature nắm bắt hình dạng, màu sắc rõ ràng hơn textual feature.
- Sản phẩm đồ chơi trẻ em (Baby dataset): textual feature mô tả chức năng, độ tuổi, tính năng chi tiết hơn.

Khi áp dụng cùng một tỷ lệ kNN cho mọi sản phẩm, đồ thị bị nhiễu ở những sản phẩm có một modality yếu. Sản phẩm có visual feature nhiễu (ảnh mờ, không đặc trưng) nhưng textual feature mạnh vẫn đang bị kéo gradient theo cả visual kNN graph — gây nhiễu cho quá trình BSC.

### 5.2. Bước 1: Tính Modality Confidence Score (parameter-free)

Với mỗi sản phẩm $i$, tính confidence score dựa trên norm của feature sau whitening:

$$c_i^v = \frac{\|\mathbf{f}_i^v\|_2}{\|\mathbf{f}_i^v\|_2 + \|\mathbf{f}_i^t\|_2 + \epsilon}$$

$$c_i^t = 1 - c_i^v = \frac{\|\mathbf{f}_i^t\|_2}{\|\mathbf{f}_i^v\|_2 + \|\mathbf{f}_i^t\|_2 + \epsilon}$$

**Lý do dùng norm:** Sau SVD whitening trong STAIR gốc (`whitening()` trong `main.py`), các feature vector đã được chuẩn hóa về cùng magnitude. Feature có norm lớn hơn trong không gian đã whitened có nghĩa là modality đó giải thích được nhiều phương sai hơn trong không gian ẩn — tức mang nhiều thông tin phân biệt hơn cho sản phẩm đó.

### 5.3. Bước 2: Item-level Edge Reweighting

Thay vì dùng `reduce='max'` khi gộp đồ thị, STAIR-DyFuse tính trọng số từng cạnh theo confidence của item nguồn:

$$\tilde{s}_{ij} = c_i^v \cdot s_{ij}^v + c_i^t \cdot s_{ij}^t$$

Ở đây:
- $s_{ij}^v$ là trọng số cạnh $(i,j)$ trong visual kNN graph (cosine similarity từ visual features)
- $s_{ij}^t$ là trọng số cạnh $(i,j)$ trong textual kNN graph (cosine similarity từ textual features)
- Cạnh chỉ có mặt trong một graph thì phần còn lại bằng 0

**So sánh với STAIR gốc:** STAIR gốc sau `reduce='max'` chỉ giữ lại cạnh có trọng số lớn hơn — tương đương `max(s_{ij}^v, s_{ij}^t)`. Cách này bảo toàn topo đồ thị nhưng mất thông tin về mức độ đóng góp của từng modality. STAIR-DyFuse tổng hợp cả hai với trọng số thích nghi.

### 5.4. Bước 3: Row Normalization để ổn định gradient

Sau khi tính $\tilde{s}_{ij}$, thực hiện symmetric normalization:

$$\hat{S}_{ij} = \frac{\tilde{s}_{ij}}{\sqrt{d_i \cdot d_j} + \epsilon}$$

với $d_i = \sum_k \tilde{s}_{ik}$ là degree của node $i$ trong đồ thị đã reweight. Đây là dạng chuẩn hóa sym (symmetric Laplacian) giống với `to_normalized(normalization='sym')` trong STAIR gốc, đảm bảo không thay đổi bản chất của BSC smoother.

### 5.5. Hàm mất mát tổng thể

STAIR-DyFuse **không thêm hàm mất mát mới**. Toàn bộ thay đổi nằm ở bước xây dựng `mAdj` trong `prepare()`. Hàm mất mát giữ nguyên BPR:

$$\mathcal{L} = \mathcal{L}_{BPR}(u, i^+, i^-)$$

---

## 6. Vị trí can thiệp chính xác trong code STAIR

**File cần sửa:** `main.py`, trong hàm `prepare()` (dòng 122–167).

Hiện tại STAIR xây dựng `mAdj` bằng:

```python
# STAIR gốc: gộp 2 kNN graph với max
edge_index = torch.cat(
    [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
    dim=1
)
edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
```

**Sửa thành (STAIR-DyFuse):**

```python
# Bước 1: Tính confidence score per item
f_text, f_visual = mfeats[0], mfeats[1]  # textual, visual sau whitening
norm_v = f_visual.norm(dim=-1)  # (N,)
norm_t = f_text.norm(dim=-1)    # (N,)
eps = 1e-7
c_v = norm_v / (norm_v + norm_t + eps)   # (N,) confidence visual
c_t = 1.0 - c_v                           # (N,) confidence textual

# Bước 2: Xây dựng từng kNN graph với trọng số cosine similarity thực tế
def get_knn_graph_weighted(features, k):
    features_norm = F.normalize(features, dim=-1)
    sim = features_norm @ features_norm.t()  # (N, N)
    sim.fill_diagonal_(-10.)
    edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
    row, col = edge_index[0], edge_index[1]
    # Lấy cosine similarity thực tế làm edge weight
    edge_weight = sim[row, col].clamp(min=0.)
    return edge_index, edge_weight

ei_t, ew_t = get_knn_graph_weighted(mfeats[0], cfg.num_neighbors[0])  # textual graph
ei_v, ew_v = get_knn_graph_weighted(mfeats[1], cfg.num_neighbors[1])  # visual graph

# Bước 3: Áp dụng confidence reweighting
row_t, col_t = ei_t[0], ei_t[1]
row_v, col_v = ei_v[0], ei_v[1]
ew_t = c_t[row_t] * ew_t   # confidence của item nguồn cho textual
ew_v = c_v[row_v] * ew_v   # confidence của item nguồn cho visual

# Bước 4: Gộp và normalize
edge_index = torch.cat([ei_t, ei_v], dim=1)
edge_weight = torch.cat([ew_t, ew_v], dim=0)
edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='sum')
edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
```

---

## 7. So sánh với các thiết kế liên quan

| Tiêu chí | STAIR gốc | TAMER | NLGCL+ | STAIR-DyFuse |
|---|---|---|---|---|
| Cách gộp kNN graph | `max(s_v, s_t)` | `alpha * s1 + beta * s2` (tĩnh) | `w * s_v + (1-w) * s_t` (tĩnh) | `c_i^v * s_v + c_i^t * s_t` (per-item) |
| Tham số thêm | 0 | 4 alpha siêu tham số | 1 siêu tham số | 0 tham số |
| Adaptive theo item | Không | Không | Không | Có |
| Rủi ro sparse graph | Thấp (giữ tất cả cạnh) | Thấp | Thấp | Thấp (không cắt cạnh) |
| Cơ sở toán học | Đơn giản | Alpha grid search | Đơn giản | Norm-based confidence |
| Hàm mất mát thêm | Không | Không | Có (NLGCL) | Không |

---

## 8. Phân tích rủi ro

**Ưu điểm:**
- Hoàn toàn parameter-free, không gia tăng số tham số và không cần grid search.
- Can thiệp vào giai đoạn preprocessing (hàm `prepare`), không ảnh hưởng tới tốc độ training.
- Bảo toàn toàn bộ cạnh đồ thị, không gây sparse graph như attention threshold.
- Logic toán học đơn giản và minh bạch.

**Rủi ro cần lưu ý:**
- Confidence score dựa trên norm của feature đã whitened. Sau SVD whitening, norm phụ thuộc vào singular values của feature matrix — cần kiểm tra xem sau whitening các norm có phân phối rõ ràng giữa 2 modality hay gần bằng nhau. Nếu norm quá cân bằng, confidence score sẽ luôn gần 0.5 và không tạo ra sự khác biệt.
- Trên tập Baby dataset (độ thưa 99.88%), ảnh sản phẩm có thể nhiễu hơn textual, nên kỳ vọng `c_t > c_v` trung bình cho hầu hết sản phẩm Baby. Hiệu quả sẽ cao hơn trên tập dày hơn như Sports.

**Kiểm tra trước khi chạy đầy đủ:**

```python
# Chạy cell kiểm tra phân phối confidence score
import matplotlib.pyplot as plt
print(f"Mean c_v: {c_v.mean():.4f}, Std: {c_v.std():.4f}")
print(f"Mean c_t: {c_t.mean():.4f}, Std: {c_t.std():.4f}")
plt.hist(c_v.cpu().numpy(), bins=50)
plt.title("Distribution of visual confidence score per item")
plt.show()
```

Nếu phân phối của `c_v` có độ lệch chuẩn > 0.05, phương pháp sẽ tạo ra sự phân biệt rõ ràng và hiệu quả. Nếu std < 0.02, nên xem xét dùng softmax-scaled norm thay vì normalized norm.

---

## 9. Hướng mở rộng (nếu DyFuse chưa đủ mạnh)

Nếu norm-based confidence score quá đồng đều sau whitening, có thể nâng cấp sang **Entropy-based Confidence**:

$$c_i^v = \frac{\exp(\|\mathbf{f}_i^v\|_2 / \tau_c)}{\exp(\|\mathbf{f}_i^v\|_2 / \tau_c) + \exp(\|\mathbf{f}_i^t\|_2 / \tau_c)}$$

với $\tau_c = 1.0$ là temperature cho confidence. Cách này khuếch đại sự phân biệt giữa hai modality hơn so với linear normalization.

---

## 10. Tóm tắt đóng góp khoa học

STAIR-DyFuse đề xuất cơ chế reweight đồ thị kNN ở mức từng sản phẩm dựa trên độ mạnh tương đối của từng modality, được đo bằng norm vector feature sau SVD whitening. So với STAIR gốc dùng max-pooling tĩnh, so với TAMER và NLGCL+ dùng trọng số toàn cục không phân biệt sản phẩm, STAIR-DyFuse tạo ra một `mAdj` phản ánh chính xác hơn cấu trúc ngữ nghĩa của từng sản phẩm mà không thêm bất kỳ tham số học nào, giữ nguyên tính hiệu quả tính toán của mô hình gốc.
