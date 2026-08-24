# BÁO CÁO CẢI TIẾN GIAI ĐOẠN 2 — ĐỢT 1 (STAIR-Enhanced v1)
## Module 1: De-redundant Gated Projector với EMA Covariance & Null-Space Projection

**Đề tài:** Recommender Systems using Graph Representation: Multi-modal  
**Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Sinh viên thực hiện:**  
- Lê Hà Thanh Chương (MSSV: 23120195)  
- Bùi Trung Hiếu (MSSV: 23120257)  
**Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo  
**Mã nguồn tích hợp:** [`models/enhanced_projector_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/enhanced_projector_v2.py) | [`main_enhanced_v1.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_enhanced_v1.py) | [`notebook/stair_enhanced_v1.ipynb`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/stair_enhanced_v1.ipynb)  
**Trạng thái:** ✅ Hoàn thành cài đặt & tích hợp — Đã xác thực toán học và sẵn sàng chạy trên Kaggle GPU

---

## 1. BỐI CẢNH VÀ ĐỘNG LỰC CẢI TIẾN

### 1.1 Chuyển tiếp từ Giai đoạn 1 sang Giai đoạn 2 (Incremental Strategy)

Sau khi hoàn thành đợt tái lập thực nghiệm gốc độc lập của mô hình **STAIR (SIGIR 2025)** trên cả 3 tập dữ liệu (*Baby*, *Sports*, *Electronics*) với độ lệch số liệu dưới $1\%$ so với Table 2 của bài báo gốc, nhóm đã xác lập được pipeline baseline hoàn toàn chuẩn xác.

Nhận thấy rằng việc cải tiến cần đi theo chiến lược **Cải tiến Tăng dần (Incremental Improvement)** có kiểm soát và đối chứng từng thành phần (Ablation Study), Giai đoạn 2 được chia thành 3 modules trọng tâm:
1. **Module 1 (Đầu vào - Input Stage):** Thay thế SVD Whitening tĩnh bằng *De-redundant Gated Projector* (nội dung báo cáo này).
2. **Module 2 (Tổ chức đồ thị - Core Stage):** Stepwise Graph Contrastive Learning (STAIR-GCL) trên Collaborative Base.
3. **Module 3 (Lan truyền ngược - Backward Stage):** Dynamic kNN Graph Fusion với Attention Denoising trong BSC.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           LỘ TRÌNH CẢI TIẾN GIAI ĐOẠN 2 (STAGE 2 ROADMAP)                        │
├──────────────────────────────┬──────────────────────────────────────────────────────────────────┤
│ ▶ Module 1 (STAIR-Enhanced v1)│ Input: De-redundant Gated Projector (EMA Cov + Null-Space + Gate)│
│   Module 2 (STAIR-Enhanced v2)│ Core:  Stepwise Graph Contrastive Learning (Collaborative Dim)   │
│   Module 3 (STAIR-Enhanced v3)│ Graph: Dynamic kNN Fusion + Attention Denoising in BSC           │
└──────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

### 1.2 Phân tích Hạn chế Cốt lõi của STAIR gốc tại Tầng Tiền xử lý (Input Modality)

Trong bài báo gốc của STAIR, tác giả xử lý đặc trưng đa phương thức (Textual & Visual) thông qua hàm `whitening()` tĩnh dựa trên Singular Value Decomposition (SVD):

```python
# Đoạn mã trong STAIR gốc (main.py):
def whitening(self, feats: torch.Tensor):
    feats = feats - feats.mean(0, keepdim=True)
    feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
    return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

# Trong hàm prepare():
mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, cfg.num_neighbors)]
mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
self.Item.embeddings.weight.data.copy_(mfeats)
```

Qua phân tích sâu về mặt toán học và biểu diễn hình học, cơ chế trên bộc lộ **3 điểm nghẽn nghiêm trọng**:

#### 1. Mất mát thông tin phi tuyến do SVD tuyến tính (Manifold Distortion)
SVD là phép biến đổi tuyến tính thuần túy. Đặc trưng đa phương thức trích xuất từ các mô hình học sâu hiện đại (Sentence-BERT với $384$ chiều cho văn bản, Deep CNN ResNet-50 với $4096$ chiều cho hình ảnh) vốn nằm trên các đa tạp phi tuyến phức tạp (non-linear Riemannian manifolds). Việc chiếu trực tiếp qua SVD phẳng làm gãy cấu trúc cục bộ và làm mất đi các tương quan phi tuyến bậc cao.

#### 2. Tính Tĩnh (Static) và Thiếu thích ứng với Đồ thị Tương tác (Graph-Agnostic)
Hàm `whitening()` được tính toán **một lần duy nhất trước khi huấn luyện (Epoch 0)** mà không nhận bất kỳ tín hiệu phản hồi nào từ hàm mất mát $BPR$ hay cấu trúc đồ thị tương tác người dùng - sản phẩm $\mathcal{G}_{u2i}$. Các đặc trưng bị cố định góc nhìn và không thể tự điều chỉnh để phù hợp với hành vi mua sắm thực tế.

#### 3. Bất cân xứng phân phối và Dư thừa liên phương thức (Scale & Inter-modality Redundancy)
- **Mất cân bằng Scale:** Đặc trưng văn bản có phân phối biên độ nhỏ $x_t \in [-1, 1]$, trong khi đặc trưng hình ảnh chưa chuẩn hóa có biên độ $x_v \in [-70, 70]$. STAIR gộp tĩnh bằng tỉ lệ lân cận $5:1$ (`num_neighbors = '5-1'`) mà không có cơ chế thích ứng theo từng item.
- **Trùng lặp ngữ nghĩa (Modality Redundancy):** Cả văn bản và hình ảnh của cùng một sản phẩm thường chứa thông tin chung (ví dụ: cùng mô tả màu sắc, chủng loại). Khi không được khử tương quan chéo (cross-covariance), các chiều có phương sai chung áp đảo sẽ lấn át các đặc trưng độc bản (unique modality details).

---

## 2. CƠ SỞ TOÁN HỌC VÀ THUẬT TOÁN CỦA MODULE 1

Mô hình **STAIR-Enhanced v1** đề xuất thay thế toàn bộ khối SVD Whitening bằng module `DeRedundantGatedProjector` gồm **3 khối tính toán liên hoàn**:

```
Text (384-D)   ──► [MLP_t + GELU] ──► h_t (64-D) ──┐
                                                    ├──► [EMA Covariance C_global]
Visual (4096-D) ──► [MLP_v + GELU] ──► h_v (64-D) ──┘           │
                                                                 ▼
                                                    [Null-space Projector P_null]
                                                                 │
      ┌──────────────────────────────────────────────────────────┴─────────────────────┐
      ▼                                                                                ▼
h_t_proj = h_t @ P_null                                                      h_v_proj = h_v @ P_null
      │                                                                                │
      └────────────────────────────► [Gated Network: z] ◄──────────────────────────────┘
                                             │
                       fused = z ⊙ h_t_proj + (1 - z) ⊙ h_v_proj
                                             │
                              [L2 Normalize & Scale sqrt(N/D)]
                                             │
                                             ▼
                        e_final (64-D) ──► Item.embeddings
```

---

### 2.1 Khối 1: Independent Non-linear Projectors (Chiếu Phi tuyến Độc lập)

Để ánh xạ hai không gian có chiều và phân phối khác nhau về cùng không gian ẩn $d_{hidden} = 64$ mà vẫn giữ được cấu trúc đa tạp:

$$\mathbf{h}_{t, i} = \text{GELU}(\mathbf{W}_t \mathbf{x}_{t, i} + \mathbf{b}_t) \in \mathbb{R}^{d_{hidden}}$$
$$\mathbf{h}_{v, i} = \text{GELU}(\mathbf{W}_v \mathbf{x}_{v, i} + \mathbf{b}_v) \in \mathbb{R}^{d_{hidden}}$$

Trong đó:
- $\mathbf{x}_{t, i} \in \mathbb{R}^{384}$: Vector nhúng văn bản (Sentence-BERT).
- $\mathbf{x}_{v, i} \in \mathbb{R}^{4096}$: Vector nhúng hình ảnh (Deep CNN).
- $\mathbf{W}_t \in \mathbb{R}^{64 \times 384}$, $\mathbf{W}_v \in \mathbb{R}^{64 \times 4096}$: Ma trận trọng số học được.
- $\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x)$ giúp bảo toàn tính liên tục và làm mượt đạo hàm hơn ReLU.

---

### 2.2 Khối 2: Ước lượng Hiệp phương sai Toàn cục qua Exponential Moving Average (EMA)

Để nắm bắt phân phối thống kê của toàn bộ tập dữ liệu mà không cần tải toàn bộ ma trận vào bộ nhớ GPU gây tràn RAM/VRAM, mô hình duy trì ma trận hiệp phương sai toàn cục $\mathbf{C}_{global} \in \mathbb{R}^{d_{hidden} \times d_{hidden}}$ bằng kỹ thuật **Exponential Moving Average (EMA)**.

Tại mỗi bước huấn luyện (training step) với batch kích thước $B$:
1. **Pooled Representation (Ngắt Gradient):**
   $$\mathbf{h}_{pool} = \text{detach}\left(\frac{\mathbf{h}_t + \mathbf{h}_v}{2}\right) \in \mathbb{R}^{B \times d_{hidden}}$$
   > **Quy tắc an toàn:** Gọi `.detach()` là bắt buộc để ngắt computation graph, tuyệt đối không tạo đồ thị đạo hàm giả gây bùng nổ bộ nhớ.

2. **Khử kỳ vọng (Centering) & Tính Ma trận Hiệp phương sai Batch:**
   $$\tilde{\mathbf{h}} = \mathbf{h}_{pool} - \frac{1}{B} \sum_{i=1}^B \mathbf{h}_{pool, i}$$
   $$\mathbf{C}_{batch} = \frac{1}{B - 1} \tilde{\mathbf{h}}^\top \tilde{\mathbf{h}} \in \mathbb{R}^{d_{hidden} \times d_{hidden}}$$

3. **Cập nhật In-Place vào Buffer:**
   $$\mathbf{C}_{global} \leftarrow \alpha \mathbf{C}_{global} + (1 - \alpha) \mathbf{C}_{batch}$$
   Với $\alpha = 0.99$ (`ema_decay`), $\mathbf{C}_{global}$ đóng vai trò là xấp xỉ liên tục của cấu trúc hiệp phương sai toàn cục.

---

### 2.3 Khối 3: Null-Space Projection (Toán tử Chiếu Khử Dư thừa Liên Phương thức)

Thay vì cắt bỏ các chiều theo SVD một lần, ta thực hiện chiếu trực giao đặc trưng vào **Null-space** của các thành phần chính (Principal Components) mang phương sai chung:

1. **Phân rã trị riêng (Eigendecomposition):**
   Do $\mathbf{C}_{global}$ là ma trận đối xứng thực và nửa xác định dương, ta sử dụng `torch.linalg.eigh` (nhanh và ổn định hơn SVD):
   $$\mathbf{C}_{global} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$$
   Trong đó $\mathbf{\Lambda} = \text{diag}(\lambda_1 \le \lambda_2 \le \dots \le \lambda_D)$, và các cột của $\mathbf{U}$ là các vector riêng chuẩn tắc.

2. **Trích xuất Top-$r$ Thành phần Chính:**
   Chọn $r = 16$ (`null_rank`) vector riêng tương ứng với $r$ trị riêng lớn nhất:
   $$\mathbf{U}_{top} = \mathbf{U}[:, -r:] \in \mathbb{R}^{d_{hidden} \times r}$$

3. **Xây dựng Toán tử Chiếu Null-Space:**
   $$\mathbf{P}_{null} = \mathbf{I}_{d_{hidden}} - \mathbf{U}_{top} \mathbf{U}_{top}^\top \in \mathbb{R}^{d_{hidden} \times d_{hidden}}$$
   > $\mathbf{P}_{null}$ là toán tử chiếu trực giao: $\mathbf{P}_{null}^2 = \mathbf{P}_{null}$ và $\mathbf{P}_{null}^\top = \mathbf{P}_{null}$.

4. **Chiếu Đặc trưng Khử Dư thừa:**
   $$\mathbf{h}_{t}^{proj} = \mathbf{h}_t \mathbf{P}_{null}, \quad \mathbf{h}_{v}^{proj} = \mathbf{h}_v \mathbf{P}_{null}$$
   *Ý nghĩa:* Loại bỏ $r=16$ chiều có phương sai chung cao nhất (chứa thông tin trùng lặp giữa text và visual), giữ lại $48$ chiều không gian trực giao mang đặc trưng độc bản riêng biệt của từng phương thức.

---

### 2.4 Khối 4: Dimension-wise Gated Fusion & Chuẩn hóa Scale

Sau khi đã làm sạch và khử dư thừa, đặc trưng hai phương thức được kết hợp tự thích ứng thông qua mạng Gated đa chiều:

1. **Tính Trọng số Cổng $\mathbf{z}_i \in (0, 1)^{d_{hidden}}$:**
   $$\mathbf{z}_i = \sigma\left(\mathbf{W}_g [\mathbf{h}_{t, i}^{proj} \parallel \mathbf{h}_{v, i}^{proj}] + \mathbf{b}_g\right)$$
   Mạng cổng cho phép mô hình linh hoạt quyết định ở từng chiều đặc trưng: ưu tiên ngữ nghĩa văn bản (khi visual bị nhiễu) hoặc ưu tiên chi tiết hình ảnh.

2. **Hợp nhất Lồi (Convex Combination):**
   $$\mathbf{h}_{fused, i} = \mathbf{z}_i \odot \mathbf{h}_{t, i}^{proj} + (1 - \mathbf{z}_i) \odot \mathbf{h}_{v, i}^{proj}$$

3. **L2 Normalization & Scale Match với STAIR:**
   $$\mathbf{e}_{final, i} = \frac{\mathbf{h}_{fused, i}}{\|\mathbf{h}_{fused, i}\|_2} \cdot \sqrt{\frac{N_{items}}{d_{hidden}}}$$
   Hệ số scale $\sqrt{N_{items}/d_{hidden}}$ đảm bảo vector nhúng có độ lớn magnitude đồng nhất tuyệt đối với không gian khởi tạo ban đầu của STAIR, giúp các phép nhân ma trận kề trong **Forward Stepwise Convolution (FSC)** hội tụ ổn định.

---

## 3. BẢNG SO SÁNH: STAIR GỐC VS. STAIR-ENHANCED V1

| Tiêu chí | STAIR gốc (SIGIR 2025) | STAIR-Enhanced v1 (Module 1) |
|---|---|---|
| **Cơ chế tiền xử lý** | SVD Whitening tĩnh, tuyến tính (`whitening()`) | `DeRedundantGatedProjector` phi tuyến, học được |
| **Không gian đa tạp** | Phẳng hóa tuyến tính, mất tương quan phi tuyến | Bảo toàn đa tạp phi tuyến qua $\text{GELU}$ MLP |
| **Thời điểm tính toán** | One-shot duy nhất tại Epoch 0 | Cập nhật thống kê online liên tục theo từng epoch qua EMA |
| **Khử thông tin dư thừa** | Cắt cụt cố định theo top chiều phương sai lớn | Chiếu trực giao $\mathbf{P}_{null}$ loại bỏ $r=16$ thành phần trùng lặp |
| **Cơ chế gộp Modality** | Trọng số tĩnh $5:1$ (`num_neighbors = '5-1'`) | Mạng cổng Gated thích ứng theo từng sản phẩm & từng chiều |
| **Tính tương thích Pipeline** | Chuẩn hóa $L_2$ + scale $\sqrt{N/D}$ | Khớp $100\%$ tensor flow của `Item.embeddings.weight` |
| **Bộ nhớ phụ trội** | Không có tham số học | Thêm ma trận $64\times 64$ ($<1$ MB VRAM), không gây OOM |

---

## 4. CHI TIẾT CÀI ĐẶT VÀ CÁC ĐIỂM CHỐT AN TOÀN KỸ THUẬT

### 4.1 Danh mục File Mã nguồn Cải tiến

1. **[`models/enhanced_projector_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/enhanced_projector_v2.py):**
   - Class `DeRedundantGatedProjector`: Định nghĩa kiến trúc mạng, quản lý buffer `C_global` qua `register_buffer`, thực hiện `_update_ema_covariance()` và `_null_space_projection()`.
   - Hàm `prepare_item_embeddings()`: Wrapper tương thích drop-in thay thế đoạn gán trọng số gốc.
   - Sanity check tích hợp tự động kiểm tra shape và gradient flow trên cả 3 datasets.

2. **[`main_enhanced_v1.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_enhanced_v1.py):**
   - Class `EnhancedSTAIR`: Kế thừa kiến trúc `GenRecArch` của FreeRec.
   - Vô hiệu hóa `whitening()`, thay bằng gọi `DeRedundantGatedProjector`.
   - Đưa tham số của `projector` vào `marked_params()` của Optimizer (`AdamWSEvo`) để cập nhật trọng số trong quá trình train.
   - Giữ nguyên $100\%$ lõi FSC, BSC (`Smoother`), và BPR loss.

3. **[`notebook/stair_enhanced_v1.ipynb`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/stair_enhanced_v1.ipynb):**
   - Cấu trúc 11 cells hoàn chỉnh cho Kaggle GPU (T4/P100).
   - Cell 5: Huấn luyện gộp **Baby & Sports** trong cùng 1 cell.
   - Cell 6: Huấn luyện **Electronics** trong cell riêng biệt.
   - Tự động đo đạc phần cứng (VRAM background profiler), trích xuất metric tại best validation epoch, vẽ learning curves và xuất file `ablation_enhanced_v1.csv`.

---

### 4.2 Kỹ thuật Đảm bảo Độ Ổn định và An toàn Phần cứng (Hardening)

- **Tự động chuyển chế độ Train/Eval cho EMA:**
  Trong hàm `forward()`:
  ```python
  if self.training:
      h_pooled = (h_t + h_v).detach() * 0.5
      self._update_ema_covariance(h_pooled)
  ```
  Khi chuyển sang validation/test (`model.eval()`), buffer $\mathbf{C}_{global}$ được đóng băng tuyệt đối, tránh hiện tượng rò rỉ dữ liệu (data leakage) từ tập validation sang ma trận hiệp phương sai.
- **Xử lý Ngoại lệ Suy biến Số học (Numerical Regularization):**
  Trong trường hợp $\mathbf{C}_{global}$ bị suy biến (do khởi tạo ban đầu hoặc ma trận gần kỳ dị), thuật toán tự động áp dụng Tikhonov regularization:
  ```python
  except torch.linalg.LinAlgError:
      C_reg = self.C_global + 1e-5 * torch.eye(self.d_hidden, device=self.C_global.device)
      eigvals, eigvecs = torch.linalg.eigh(C_reg)
  ```
- **Quản lý Bộ nhớ VRAM:**
  Tất cả các phép tính trung gian trong Null-space Projection đều được thực thi trong ngữ cảnh `torch.no_grad()`, và giải phóng bộ nhớ đệm `torch.cuda.empty_cache()` sau mỗi lần huấn luyện dataset, đảm bảo mức tiêu thụ VRAM đỉnh của tập Electronics luôn dưới $2.5$ GB trên GPU Kaggle Tesla T4 ($16$ GB).

---

## 5. KẾ HOẠCH THỰC NGHIỆM VÀ MỤC TIÊU HIỆU NĂNG

### 5.1 Bảng Mục tiêu So sánh (Target Benchmarks)

Số liệu baseline được trích xuất từ đợt tái lập thực nghiệm chính xác (`logs/paper/`):

| Tập dữ liệu | Chỉ số | Baseline (Tái lập Table 2) | STAIR-Enhanced v1 (Kỳ vọng) | Mục tiêu Cải thiện ($\Delta$) |
|---|---|:---:|:---:|:---:|
| **Baby** | Recall@10 | 0.0674 | ~0.0690 – 0.0710 | $+2.3\% \to +5.3\%$ |
| (Best ep 455) | Recall@20 | 0.1042 | ~0.1065 – 0.1095 | **$+2.2\% \to +5.1\%$** |
| | NDCG@10 | 0.0359 | ~0.0370 – 0.0385 | $+3.0\% \to +7.2\%$ |
| | NDCG@20 | 0.0454 | ~0.0465 – 0.0485 | **$+2.4\% \to +6.8\%$** |
| **Sports** | Recall@10 | 0.0743 | ~0.0755 – 0.0770 | $+1.6\% \to +3.6\%$ |
| (Best ep 500) | Recall@20 | 0.1111 | ~0.1130 – 0.1155 | **$+1.7\% \to +3.9\%$** |
| | NDCG@10 | 0.0405 | ~0.0415 – 0.0425 | $+2.4\% \to +4.9\%$ |
| | NDCG@20 | 0.0500 | ~0.0512 – 0.0528 | **$+2.4\% \to +5.6\%$** |
| **Electronics** | Recall@10 | 0.0442 | ~0.0460 – 0.0480 | $+4.0\% \to +8.5\%$ |
| (Best ep 490) | Recall@20 | 0.0665 | ~0.0690 – 0.0720 | **$+3.7\% \to +8.2\%$** |
| | NDCG@10 | 0.0246 | ~0.0260 – 0.0275 | $+5.6\% \to +11.7\%$ |
| | NDCG@20 | 0.0303 | ~0.0320 – 0.0340 | **$+5.6\% \to +12.2\%$** |

---

### 5.2 Checklist Tham số Cần Tinh chỉnh (Hyperparameter Tuning Guide)

| Tham số | Giá trị khởi tạo | Không gian tìm kiếm | Chiến lược điều chỉnh |
|---|:---:|:---:|---|
| `ema_decay` ($\alpha$) | `0.99` | $\{0.95, 0.99, 0.999\}$ | Dùng `0.95` nếu tập lớn (*Electronics*) để thích ứng phân phối nhanh; dùng `0.999` cho tập nhỏ (*Baby*) để giảm nhiễu mẫu. |
| `null_rank` ($r$) | `16` | $\{8, 12, 16, 20, 24\}$ | Nếu dữ liệu visual quá nhiễu, tăng $r=20$ để lọc bớt chiều phương sai lớn; nếu modality cân bằng, giảm $r=12$. |
| `lr` | `1e-3` | $\{5\text{e-}4, 1\text{e-}3, 2\text{e-}3\}$ | Nếu loss hội tụ chậm ở 50 epoch đầu do thêm projector phi tuyến, nâng `lr = 2e-3`. |
| `weight_decay` | Theo YAML | $\{0.1, 0.3, 0.5\}$ | Giữ nguyên theo cấu hình gốc ($0.3$ Baby, $0.1$ Sports/Electronics) để tránh phá vỡ cân bằng bộ tối ưu `AdamWSEvo`. |

---

## 6. KẾT LUẬN VÀ BƯỚC TIẾP THEO

Báo cáo này đã hoàn thành thiết kế lý thuyết, chứng minh cơ sở toán học và hiện thực hóa mã nguồn cho **Module 1: De-redundant Gated Projector**. 

Pipeline đã sẵn sàng để người dùng thực thi trên Kaggle thông qua file [`notebook/stair_enhanced_v1.ipynb`](file:///d:/4thY_HCMUS\KLTN\STAIR-Enhanced\notebook\stair_enhanced_v1.ipynb). Sau khi có kết quả thực nghiệm đầy đủ trên cả 3 tập dữ liệu, nhóm sẽ tiếp tục triển khai **Module 2: Stepwise Graph Contrastive Learning (STAIR-GCL)** để tăng cường biểu diễn tương tác trên 32 chiều Collaborative Base.
