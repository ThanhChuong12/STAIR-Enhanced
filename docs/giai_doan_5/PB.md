## Phản biện chi tiết Kiến trúc STAIR5-v3 (STAIR-PC-LHC)

Trước khi đi vào chi tiết, có **một vấn đề tiên quyết cực kỳ nghiêm trọng**: bạn vẫn đang trích dẫn số liệu **"+6.6% Recall@20 Sports, +11.2% NDCG@20 Baby"** từ báo cáo cũ. Nhưng chính bạn đã đối chiếu lại trong `STAIR5_v1_Experiment_Report.md` bản cập nhật và xác nhận các con số đúng phải là:

| Dataset | Con số đang trích dẫn (SAI) | Con số đúng sau đối soát |
|:---|:---:|:---:|
| Sports R@20 | +6.6% | **+1.98%** |
| Sports NDCG@20 | +5.4% | **+1.20%** |
| Baby NDCG@20 | +11.2% | **−1.54%** (thua baseline) |
| Baby R@20 | +5.2% | **−1.15%** (thua baseline) |

**Bạn đang thiết kế v3 dựa trên một tiền đề sai.** Đây là vấn đề nghiêm trọng nhất — nếu v3 được build trên giả định "v1 đã tăng +11% trên Baby", thì toàn bộ logic cải tiến phía sau sẽ lệch hướng.

Với tiền đề đã sửa, bức tranh thực tế là: **v1 thắng nhẹ trên Sports, thua trên Baby, và trung tính trên Electronics.** Đây là điểm xuất phát hoàn toàn khác — và nó thay đổi mục tiêu của v3: **không phải tăng thêm trên Sports, mà là cứu Baby và Electronics.**

---

### 1. Vấn đề Chiến lược: Tái phạm đúng sai lầm của v2

v3 gộp **5 thành phần tín hiệu** vào một công thức confidence duy nhất:

$$c_{ij} = \sigma(w_1 s^{\text{Lorentz}} + w_2 s^{\text{behavior}} + w_3 s^{\text{mom}} - w_4 u^{\text{unc}})$$

Cộng với hai thay đổi lớn:
- **Thay đổi BSC graph** (Component 1)
- **Thay đổi InfoNCE temperature** (Component 2)

**Đây chính xác là pattern đã thất bại 3 lần liên tiếp** trong đề tài (DLIA v2, REARM v6, MHD v3). Nếu v3 thắng, bạn không biết đóng góp đến từ đâu. Nếu v3 thua, bạn không biết thành phần nào gây hại. Không có cách nào viết kết luận khoa học cho luận văn.

**Khuyến nghị tối thiểu:** Tách thành 3 arm độc lập — v3a (chỉ BSC reweight, dùng **một** signal), v3b (chỉ PC-LHC temperature), v3c (kết hợp). Chạy v3a trước, chỉ mở v3b khi v3a có tín hiệu.

---

### 2. Phản biện Component 1: Preference-Compatible BSC

#### 2.1. Thành phần `s_mom` (Momentum Alignment) có lỗi logic nghiêm trọng

Đây là **vấn đề toán học nặng nhất** của v3. Công thức:

$$\mathbf{M}_i^{(t)} = \mu \mathbf{M}_i^{(t-1)} + (1-\mu) \Delta E_i^{(t)}$$

**Lỗi #1 — Momentum không phải "preference direction".** Trong SGD/AdamW, vector momentum $\mathbf{M}_i$ là **trung bình có trọng số của các gradient BPR** mà item $i$ đã nhận. Gradient BPR cho một cặp $(u, i^+, i^-)$ là:

$$\nabla_{E_i} \mathcal{L}_{\text{BPR}} = -\sigma(-\Delta) \cdot (E_u - E_{i^-})$$

Vector này phụ thuộc vào **user và negative item** trong batch, không phải vào "preference" thuần túy của item $i$. Hai item có cùng category sẽ có momentum hướng về trung bình của các user đã tương tác với chúng — mà trung bình này là **bias theo tần suất**, không phải compatibility.

**Lỗi #2 — Vòng lặp phản hồi (runaway feedback).** Nếu $s^{\text{mom}}_{ij}$ cao → $c_{ij}$ cao → BSC boost mạnh → $E_i$ và $E_j$ bị làm mịn về gần nhau → gradient của chúng càng giống nhau → $s^{\text{mom}}_{ij}$ càng cao → $c_{ij}$ càng cao...

Đây là **positive feedback loop không có damping**, dễ dẫn đến **collapse cục bộ** — một cụm item sẽ hút nhau vào một điểm, làm mất tính đa dạng biểu diễn. Đặc biệt nguy hiểm trên Baby (đồ thị dày, nhiều feedback cycles).

**Lỗi #3 — Không có cơ chế reset/decay.** Momentum được cập nhật EMA qua 500 epochs, nhưng không có cơ chế "quên" khi representation thay đổi mạnh. Nếu một item đổi cluster (do training dynamics), momentum cũ vẫn ảnh hưởng hàng trăm epochs sau.

#### 2.2. Thành phần `u_ij^unc` (Uncertainty) bị diễn giải sai

$$u_{ij}^{\text{unc}} = \frac{1}{2}(\sigma_i^2 + \sigma_j^2)$$

**Lỗi #1 — Đây là pairwise sum, không phải pairwise divergence.** Uncertainty of an edge nên là **mức độ không chắc chắn về cạnh đó**, không phải trung bình uncertainty của hai node. Ví dụ: hai item đều uncertain cao nhưng vì cùng lý do (cùng category nhiễu), thì cạnh giữa chúng có thể rất reliable.

**Lỗi #2 — Trừ `u_ij` làm global down-weight, không phải per-edge.** Nếu $\sigma_i^2$ lớn, mọi edge chứa $i$ đều bị penalty. Điều này chỉ push các uncertain items về confidence ≈ 0, không phân biệt được "uncertain nhưng vẫn compatible" vs "uncertain và không compatible".

**Lỗi #3 — MURAL 2026 không dùng uncertainty cho edge weighting.** Trong MURAL, aleatoric uncertainty được dùng để **down-weight modality features** trong fusion (feature-level), không phải để gate item-item edges (graph-level). Bạn đang diễn giải sai paper nguồn.

#### 2.3. Thành phần `s_Lorentz` tạo double-counting với LHC

`s_Lorentz` được tính từ $H^{(0)}$ — chính là embedding mà LHC loss đang tối ưu. Nếu LHC làm hai item gần nhau trong Lorentz, thì $s^{\text{Lorentz}}_{ij}$ tăng → $c_{ij}$ tăng → BSC boost mạnh → hai item càng gần nhau → LHC lại thấy chúng gần hơn...

**Đây là double-counting:** cùng một tín hiệu (Lorentz proximity) được dùng hai lần — một lần trong loss, một lần trong graph. Không có justification lý thuyết cho việc này.

**Cách sửa:** Nếu muốn dùng thông tin hình học cho graph, phải dùng **Lorentz similarity của raw features đã whitening**, không phải của trainable $H^{(0)}$.

#### 2.4. `w = [1.0, 1.0, 0.5, 0.5]` — không có cơ sở

- Nếu learnable: gradient chảy qua đâu? Offline graph construction không có training signal.
- Nếu fixed: tại sao chọn các giá trị này? Không có ablation hay lý thuyết justify.
- Trong code: `self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.5, 0.5]))` — nhưng `forward()` không dùng `self.w`. Đây là **dead code** hoặc bug.

#### 2.5. "0% Edge Pruning" là tuyên bố misleading

$$\tilde{A}_{ij}^{\text{BSC}} = A_{ij}^{\text{kNN}} \cdot (1 + \alpha c_{ij})$$

Về mặt kỹ thuật, đúng là không có edge nào bị xóa. Nhưng **cấu trúc graph thay đổi hoàn toàn** vì:
- BSC Smoother chuẩn hóa theo degree
- Nếu $\alpha = 0.5$ và nhiều edge có $c_{ij} \approx 1$, degree tăng 50%
- Phân phối lại toàn bộ trọng số spectral của graph

**Đây không phải "no structural change"** — đây là **rescaling toàn cục** có thể gây ra vấn đề tương tự như pruning.

#### 2.6. Recompute graph mỗi epoch vi phạm thiết kế v1

v1 đã cẩn thận **freeze BSC graph** để bảo toàn gradient isolation (§3.5 trong v1 report). v3 đề xuất **recompute mỗi epoch** với embedding thay đổi. Điều này:

- Phá vỡ tính ổn định của optimizer (Smoother assumes fixed adjacency)
- Tạo **meta-learning problem** (learning the graph while learning on the graph)
- Không rõ gradient có chảy qua graph construction không

Nếu có: backprop through graph → computational cost lớn và instability.
Nếu không: graph là một **hàm của epoch** — và bạn phải chứng minh nó hội tụ.

---

### 3. Phản biện Component 2: PC-LHC Temperature

#### 3.1. Kế thừa nguyên xi lỗi của v2

$$\tau_i = \tau_0 \cdot (1 + \eta \sigma_i^2) \cdot \exp\left(-\gamma \frac{\log(1+\deg_i)}{\max_k \log(1+\deg_k)}\right)$$

**Phản biện toán học từ v2 vẫn còn nguyên giá trị:**
- Temperature asymmetric giữa positive và negative → vi phạm tính hợp lệ của InfoNCE
- Quan hệ "τ nhỏ → kéo về gốc" là **sai về mặt toán học** — radial position phụ thuộc vào $\|H_i\|$, không phụ thuộc vào $\tau$
- Không có bằng chứng thực nghiệm về correlation giữa degree và radial position

#### 3.2. Thêm $\sigma_i^2$ vào temperature là **đổ thêm dầu vào lửa**

- Nếu $\sigma_i^2$ lớn: $\tau_i$ tăng → phân phối softmax mềm hơn
- Softmax mềm hơn nghĩa là **gradient nhỏ hơn** cho mọi cặp
- Item uncertain sẽ nhận ít gradient hơn → **càng uncertain hơn** → feedback loop

Đây là **vòng lặp tiêu cực**: item noisy được down-weight → không học được → vẫn noisy → lại bị down-weight.

#### 3.3. Interaction giữa $\sigma_i^2$ và $\deg_i$ không được phân tích

Công thức có **hai** yếu tố điều chỉnh temperature: uncertainty và degree. Nếu hai yếu tố này correlate (item long-tail thường uncertain hơn), thì tác động của chúng sẽ **khuếch đại lẫn nhau**, không phải độc lập. Cần phân tích tương quan $\sigma_i^2$ vs $\log \deg_i$ trước.

---

### 4. Các vấn đề Engineering thực tế

#### 4.1. Chi phí tính toán chưa được đánh giá

- Recompute Lorentz embeddings cho 63K items mỗi epoch
- Tính `s_Lorentz` cho tất cả kNN edges (~300K-500K edges trên Electronics)
- Update momentum vector [63K, 64] mỗi training step
- Uncertainty network forward cho 63K items mỗi epoch

**Ước tính sơ bộ:** thêm ~5-10 giây/epoch trên Electronics, tức ~1-2 giờ cho 500 epochs. Nhưng nếu cộng thêm v1 (79s/epoch), có thể vượt ngưỡng 12h Kaggle.

#### 4.2. Code có bug và thiếu sót

```python
self.momentum_vec.mul_(decay).add_((1 - decay) * item_grads)
```

- `(1 - decay) * item_grads` tạo tensor mới → tốn memory
- Correct: `self.momentum_vec.mul_(decay).add_(item_grads, alpha=1-decay)`

```python
self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.5, 0.5]))
```

- Không dùng trong `forward()`
- Nếu dùng, gradient flow path không rõ ràng

#### 4.3. Data leakage risk trong momentum update

`update_momentum(item_grads)` không chỉ định **khi nào** và **từ batch nào** để update:
- Nếu update sau validation step → leakage
- Nếu update trong training loop, phải chỉ dùng gradient từ train batches
- Code không có comment hay safeguard

---

### 5. So sánh Literature: Table IV có vấn đề về tính chính xác

Bảng đối chiếu của bạn viết:
- **"EVEN (AAAI'25)"**: OK có paper
- **"SIGER (NTU'25)"**: không rõ paper này có thật không — cần xác minh
- **"MURAL (2026)"**: cần xác minh
- **"IGDMRec (TMM'26)"**: OK

**Nếu SIGER và MURAL là do AI tự sinh, đây là vấn đề học thuật nghiêm trọng.** Phải verify URL/DOI trước khi trích dẫn.

Thêm nữa, bảng so sánh gợi ý rằng STAIR5-v3 "vượt trội" về tốc độ, nhưng thực tế:
- v3 phải recompute graph mỗi epoch → có thể chậm hơn IGDMRec ở steady state
- Tuyên bố "Bảo tồn 100% SVD Whitening" là đúng nhưng không có nghĩa v3 "tốt hơn" — EVEN và SIGER không áp dụng SVD Whitening vì kiến trúc khác

---

### 6. Mục tiêu thực nghiệm không có cơ sở

Bạn viết:
> *"Kỳ vọng: Recall@20 Sports 0.1150+, NDCG@20 Baby 0.0460+"*

| Metric | v1 thực tế | Mục tiêu v3 | Gap cần |
|:---|:---:|:---:|:---:|
| Sports R@20 | 0.1133 | 0.1150 | +1.5% |
| Baby NDCG@20 | 0.0447 | 0.0460 | **+2.9%** (và phải vượt baseline 0.0454) |
| Electronics NDCG@20 | 0.0301 | không đề cập | cần bù -0.66% |

**Với 5 component chưa được kiểm chứng độc lập, mục tiêu +2.9% trên Baby là phi thực tế.** Chưa kể Baby hiện đang **thua** baseline — để đạt 0.0460 bạn cần **cải thiện +1.5% so với baseline**, không chỉ "cứu v1".

---

### 7. Khuyến nghị Cụ thể

Nếu bạn vẫn muốn theo đuổi hướng Preference-Compatible BSC, đây là phiên bản tối giản hóa đề xuất:

#### Bước 1 — Diagnostic trước khi code

Chạy trên v1 checkpoint (không cần train lại):
```python
# 1. Correlation giữa radial position và degree
corr_rd = corr(radial_positions, np.log1p(degrees))
# Nếu |corr_rd| < 0.15: bỏ Radial Temperature

# 2. Correlation giữa s_Lorentz và s_Euclidean (raw cosine)
corr_le = corr(lorentz_sim, euclidean_cos)
# Nếu |corr_le| > 0.85: s_Lorentz dư thừa, bỏ

# 3. Correlation giữa uncertainty và degree
corr_ud = corr(sigma_squared, np.log1p(degrees))
# Nếu |corr_ud| > 0.5: 2 signals trùng lặp, chỉ giữ 1
```

#### Bước 2 — Nếu phải thử, chỉ dùng 1 signal

Thay vì 4 signals, chỉ dùng **signal mạnh nhất** (theo diagnostic). Đây là ví dụ nếu `s_behavior` là mạnh nhất:

$$c_{ij} = \sigma(w_2 \cdot s_{ij}^{\text{behavior}})$$
$$\tilde{A}_{ij}^{\text{BSC}} = A_{ij}^{\text{kNN}} \cdot (1 + \alpha c_{ij})$$

**Đây thực chất là Giai đoạn 3 BSC-Reweight** với Sigmoid thay vì linear boost. Nếu không cải thiện, bỏ hoàn toàn.

#### Bước 3 — Không làm PC-LHC

Temperature modulation có lỗi toán học chưa giải quyết. **Đừng triển khai.**

#### Bước 4 — Tập trung cứu Baby

Baby đang thua baseline (−1.5%). Đây là vấn đề nghiêm trọng hơn Sports (đang thắng +1.2%).

**Thử hướng khác cho Baby:**
- Chạy κ-grid trên Baby: $\kappa \in \{0.05, 0.1, 0.2, 0.5\}$
- Nếu $\kappa = 0.1$ phục hồi baseline → xác nhận hypothesis "Baby cần curvature thấp"
- Đây là **đóng góp khoa học thực sự** — không phải giả thuyết "thêm component để tăng số"

---

### 8. Tóm tắt Đánh giá

| Khía cạnh | Đánh giá | Hành động |
|:---|:---:|:---|
| **Tiền đề (số liệu v1)** | ❌ **SAI** | Sửa lại theo báo cáo đối soát |
| **Chiến lược Isolation** | ❌ Vi phạm | Tách arms |
| **s_mom component** | ❌ **Có feedback loop nguy hiểm** | Bỏ hoàn toàn |
| **u_ij component** | ❌ Sai interpretation của MURAL | Bỏ hoàn toàn |
| **s_Lorentz trong graph** | ⚠️ Double-counting với LHC | Dùng raw features |
| **PC-LHC temperature** | ❌ Kế thừa lỗi v2 | Không triển khai |
| **Recompute graph mỗi epoch** | ❌ Vi phạm v1 design | Freeze như v1 |
| **Mục tiêu thực nghiệm** | ❌ Không cơ sở | Đặt lại theo gap thực |
| **Code** | ⚠️ Có bug, dead code | Rewrite sau khi design chốt |

---

### 9. Kết luận

Kiến trúc v3 có **hai ý tưởng tốt về mặt trực giác** (edge trust weighting, uncertainty-aware learning), nhưng **cách triển khai có 4 lỗi toán học** và **1 lỗi chiến lược nghiêm trọng** (gộp 5 components).

**Đóng góp thực sự cần thiết lúc này không phải là thêm v3**, mà là:

1. **Fix hiểu lầm số liệu**: Xây dựng narrative đúng cho v1 — Sports thắng nhẹ, Baby thua, Electronics trung tính. Đây là **conditional contribution** với câu hỏi khoa học rõ ràng: *"When does hyperbolic geometry help e-commerce recommendation?"*

2. **Chạy κ-grid trên Baby**: Nếu curvature thấp (0.1) cứu được Baby → đây là **first-principles discovery** đáng giá hơn bất kỳ cải tiến số nào.

3. **Chạy ablation E0, HC trên Sports**: Xác định đóng góp đến từ Lorentz geometry hay kernel effect. Đây là yêu cầu bắt buộc để bảo vệ luận văn.

4. **Đo correlation diagnostics** trước khi thử bất kỳ v3 nào — có thể tiết kiệm 20+ giờ GPU.

**Nếu v3 được triển khai, xác suất nó thất bại (kết quả ≤ v1) tôi ước tính là 70-80%** dựa trên 3 lần thất bại tương tự trước đó (DLIA, REARM, MHD). Xác suất nó vượt baseline trên Baby là <20%.

Đừng để v3 làm loãng đóng góp của v1. Bảo vệ v1 bằng phân tích sâu, không phải bằng cách thêm component mới.