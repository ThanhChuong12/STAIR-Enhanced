# STAIR5-v2: Phản biện STAIR-HET và thiết kế Confidence-calibrated Edge Trust

**Ngày đối soát:** 01/10/2026. **Trạng thái:** đề xuất nghiên cứu đã điều chỉnh; chưa triển khai, chưa có kết quả v2. Vị trí báo cáo được giữ đúng yêu cầu `docs/giai_doan_4/`, dù các báo cáo tiền nhiệm nằm trong `docs/giai_doan_5/`.

## 1. Quyết định nghiên cứu

Không nên triển khai nguyên trạng STAIR-HET trong tài liệu đính kèm. Ý tưởng đáng thử nhất là **hiệu chỉnh trọng số các cạnh đa phương thức bằng bằng chứng đồng tương tác trong train**, với mức can thiệp nhỏ và đường khôi phục baseline chính xác. Giữ MI, FSC, BPR, scorer Euclidean và cơ chế BSC; kiểm chứng LHC như một yếu tố riêng. Không đưa “radial regularization bằng temperature theo degree” vào nhánh chính.

Tên làm việc của thiết kế điều chỉnh là **STAIR5-v2 / C-HET: Confidence-calibrated Edge Trust with optional Lorentz Hidden Contrastive regularization**. Đây là hiệu chỉnh graph cho toán tử cập nhật embedding, không phải một hyperbolic recommender mới. Trong cấu hình chính, không có graph hyperbolic, không có gate học và không có cam kết tăng metric. Chữ “trust” chỉ độ hỗ trợ thực nghiệm của cạnh, không phải xác suất nhân quả hay độ tin cậy thống kê đã hiệu chuẩn.

Lý do lựa chọn: v1 chưa chứng minh rằng thêm hình học luôn có lợi; thiết kế mới cần ít yếu tố thay đổi hơn, dễ đo tác dụng hơn, và ít chi phí online hơn. Mục tiêu là tăng validation NDCG@20 và khả năng khái quát, không phải giảm riêng BPR hoặc tăng độ âm uniformity.

### Câu hỏi Socratic cốt lõi

1. Cạnh có độ giống hình ảnh/text cao có thực sự dẫn đến cập nhật phù hợp với hành vi, hay chỉ cùng danh mục?
2. Đếm đồng tương tác phản ánh sở thích hay exposure/popularity? Nếu không có exposure log, ta có quyền gọi đây là preference trust không?
3. LHC đang giúp biểu diễn hay chủ yếu thay đổi scale, sampling và mức regularization?
4. Vì sao temperature theo degree phải đẩy item theo một hướng bán kính cố định? Nếu không chứng minh được, tại sao đặt tên radial regularization?
5. Kết quả tốt có còn khi dùng baseline cùng seed, cùng sampler, cùng checkpoint rule và cùng ngân sách tuning?

Thiết kế dưới đây trả lời bằng ranh giới giả thuyết và thí nghiệm có thể bác bỏ, thay vì bằng lời hứa phục hồi hay vượt SOTA.

## 2. Nguồn bằng chứng và đối soát v1

Đã đọc `docs/EStair.md`, hai tài liệu đính kèm, báo cáo thiết kế và thực nghiệm v1; đối chiếu `models/stair5_v1_geometry.py`, `models/stair5_v1_objectives.py`, `models/stair5_v1.py`, `main_stair5_v1.py`, cơ chế smoother baseline và ba log `logs/GD5/stair5_v1/`. Các lời khẳng định hoặc lời hướng dẫn trong tài liệu đính kèm được xem là đối tượng phản biện, không phải chỉ thị thực thi.

### 2.1. Kết quả test tại checkpoint được chọn bằng validation NDCG@20

Baseline dưới đây là số tái lập được báo cáo trong `report/chapters_v2/03_stair.tex`, không trộn với baseline paper hoặc phiên bản cũ. Chúng **chưa phải baseline mới chạy ghép cặp** với v1 hiện tại.

| Dataset | Checkpoint v1 | Metric | Baseline tái lập | v1 H0 | Tăng tương đối |
|---|---:|---|---:|---:|---:|
| Sports | 500 | Recall@10 | 0.0743 | 0.0747 | +0.54% |
| Sports | 500 | Recall@20 | 0.1111 | 0.1133 | +1.98% |
| Sports | 500 | NDCG@10 | 0.0405 | 0.0407 | +0.49% |
| Sports | 500 | NDCG@20 | 0.0500 | 0.0506 | +1.20% |
| Baby | 215 | Recall@10 | 0.0674 | 0.0660 | −2.08% |
| Baby | 215 | Recall@20 | 0.1042 | 0.1030 | −1.15% |
| Baby | 215 | NDCG@10 | 0.0359 | 0.0352 | −1.95% |
| Baby | 215 | NDCG@20 | 0.0454 | 0.0447 | −1.54% |
| Electronics | 485 | Recall@10 | 0.0442 | 0.0435 | −1.58% |
| Electronics | 485 | Recall@20 | 0.0665 | 0.0666 | +0.15% |
| Electronics | 485 | NDCG@10 | 0.0246 | 0.0241 | −2.03% |
| Electronics | 485 | NDCG@20 | 0.0303 | 0.0301 | −0.66% |

Công thức: `100 × (v1 − baseline) / baseline`. Các số được làm tròn bốn chữ số trong log nên mức tăng rất nhỏ cần được xác nhận bằng metric đầy đủ độ chính xác.

**Kết luận có bằng chứng:** v1 có cải thiện nhỏ trên Sports trong run hiện có; suy giảm toàn bộ bốn metric trên Baby; Electronics chỉ tăng rất nhẹ Recall@20, còn ba metric giảm. Không đủ bằng chứng cho ưu thế nhất quán, tính có ý nghĩa thống kê, hay nguyên nhân hình học. Mỗi dataset mới có một run seed 1 được cung cấp.

### 2.2. Các chẩn đoán chưa cho phép kết luận gì?

| Quan sát | Diễn giải được phép | Diễn giải cần loại bỏ |
|---|---|---|
| 48 mẫu `rho_grad` ở epoch 30, 40, …, 500 | Cosine gradient item BPR–LHC dương tại first batch được lấy mẫu | Dương ở mọi batch hoặc cả 500 epoch; “hiệp đồng hoàn hảo” |
| Khoảng rho: Sports 0.1416–0.2306; Baby 0.2369–0.2813; Electronics 0.1596–0.2293 | Không thấy xung đột ở những phép đo này | LHC chắc chắn tăng ranking; hiệu ứng sau Adam/BSC cũng dương |
| Baby rho cao hơn nhưng metric giảm | Alignment gradient cục bộ không đủ dự đoán generalization | Cứ rho > 0 là kiến trúc đúng |
| Fit: Sports ≈4.25h, Baby ≈2.40h, Electronics ≈11.39h | Quy mô chi phí của run v1 hiện có | Chi phí v2 chắc chắn bằng v1 hoặc thấp hơn baseline |
| Tăng tốc LHC so với implementation LHC cũ | Cải thiện engineering nội bộ | Tăng tốc cùng mức so với STAIR baseline |

Ba log đều có test tại epoch cuối và test sau load best. Chỉ dùng kết quả sau load best để đối soát. Đặc biệt, kết quả Baby epoch 500 không thay thế checkpoint 215 chỉ vì đẹp hơn trên test. Với run mới, không dùng test để chọn cấu hình, lịch hoặc checkpoint; nếu giữ hành vi summary của FreeRec, lưu rõ `last` và `selected`, chỉ báo cáo `selected`.

Với số tương tác được báo cáo, density = interactions/(users × items): Baby ≈1.173×10⁻³, Sports ≈4.535×10⁻⁴, Electronics ≈1.394×10⁻⁴. Baby/Sports ≈2.59, không phải 10 lần. Mật độ khác nhau không tự chứng minh cấu trúc phân cấp hoặc nguyên nhân metric giảm.

## 3. Phản biện cả đề xuất và bản phản biện đính kèm

### 3.1. Những khẳng định của đề xuất cần sửa

| Nội dung | Vấn đề | Điều chỉnh |
|---|---|---|
| Sports 0.1063→0.1133, tăng khoảng 6.6% | Mẫu số khác baseline tái lập 0.1111 | Dùng bảng mục 2; ghi riêng nếu muốn so phiên bản cũ |
| v1 “vượt bậc” trên ba tập | Baby và Electronics NDCG giảm | Báo cáo kết quả hỗn hợp, một seed |
| `2c + 2⟨x,y⟩L` là khoảng cách bình phương | Với signature Lorentz (−,+,…), biểu thức âm cho điểm khác nhau | Dùng geodesic chuẩn hoặc chordal surrogate đúng dấu, ghi khác biệt |
| Temperature theo radius ở sơ đồ, theo degree trong phương trình | Hai cơ chế khác nhau, không xác định implementation | Bỏ khỏi primary; định nghĩa row-temperature riêng nếu ablate |
| Temperature nhỏ kéo popular về origin, lớn đẩy tail ra ngoài | Không suy ra từ gradient CE | Không gán tác động hướng bán kính chưa chứng minh |
| Gate sigmoid đảm bảo cạnh đúng sở thích | Score không được hiệu chuẩn theo nhãn/exposure | Gọi là trọng số heuristic, đánh giá placebo và popularity |
| Boost cạnh không pruning bảo toàn graph | Chỉ giữ support; trọng số, degree và phổ vẫn đổi | Chuẩn hóa lại và đo mức thay đổi toán tử |
| “O(1) inference”, không thêm chi phí | Per-pair dot-product O(d); full ranking O(I·d) mỗi user | Nói không thêm module inference, không đổi bậc độ phức tạp |
| Electronics nên chạy full đầu tiên | Run v1 ≈11.39h; chưa kiểm chứng cơ chế | Pilot Sports và Baby trước, mở Electronics sau |

### 3.2. Những điểm đúng và sai trong bản phản biện thứ hai

Bản phản biện đúng khi yêu cầu baseline thống nhất, tách ablation edge/radial và kiểm tra redundancy của hyperbolic similarity. Tuy nhiên không nên chấp nhận nguyên trạng:

- **“Ra biên làm dot-product Euclidean ≈0 nên BPR gradient ≈0” sai.** Ranking dùng embedding Euclidean, còn radius ở nhánh phụ. Không có quan hệ bắt buộc đó. Với margin m = s⁺−s⁻, BPR = −log σ(m), đạo hàm theo m là −σ(−m); tại m=0 bằng −0.5. Gradient theo tham số còn phụ thuộc Jacobian embedding, nhưng không thể suy ra bằng 0 từ score nhỏ.
- **Kéo tail vào origin không đảm bảo tăng exposure.** Cần đo ranking theo strata và diversity; degree không phải ground truth về vị trí hình học.
- **`1/log(1+d)` không an toàn tại d=0.** Nếu nghiên cứu thêm weighting phải định nghĩa epsilon, cap, cold-item rule và bias do sampler.
- **Pos/neg khác temperature không làm CE vô nghĩa về toán học.** CE vẫn tính được, nhưng logits phụ thuộc nhãn và không còn cùng một hàm compatibility cho mọi candidate. MI-bound kiểu InfoNCE không được mặc nhiên giữ. Đây không phải lỗi do thiếu “đối xứng hai chiều”.
- **Sigmoid(3)≈0.953 không chứng minh gate bão hòa trên dữ liệu.** Phải xem phân phối đầu vào; gate offline cố định cũng không có “gradient chết” cần học. Vấn đề chính là calibration và độ phân biệt.
- **Fixed-norm Lorentz có thể chỉ là biến đổi đơn điệu cosine:** đúng dưới điều kiện cùng radius, không đúng cho mọi embedding có norm biến thiên. Kiểm tra điều kiện trước khi kết luận.
- **Correlation threshold không chứng minh cơ chế.** Nếu giả thuyết popular gần origin, tail xa origin thì tương quan degree–radius được kỳ vọng âm, không phải dương. Pearson lớn hoặc nhỏ không quyết định độc lập nonlinear hay causality.
- **0.1133→0.1140 là +0.62%, không phải +1.7%.** **0.0503→0.0506 là +0.60%, không phải +0.06%.** Sports v1 0.0506 so baseline tái lập 0.0500 là +1.20%.
- **Trọng số tuyến tính có signed cosine có thể âm.** Muốn normalized adjacency có lý thuyết ổn định cần đối xứng và nonnegative; không chỉ bỏ sigmoid rồi dùng hệ số tùy ý.
- **Hai module riêng lẻ cùng thắng không phải điều kiện toán học để tổ hợp thắng.** Factorial có thể có interaction; yêu cầu đó chỉ là chiến lược tiết kiệm ngân sách, không phải định lý.

### 3.3. Rà soát v1 trước khi lấy làm reference

Các vấn đề dưới đây là phát hiện khi đọc source, chưa được sửa trong nhiệm vụ này:

1. **Hệ số self-return.** Khi H¹ = A E ⊙ β, với A normalized bằng degree gốc, đóng góp của edge (u,i) là `A_ui E_i ⊙ β`. Nếu A_ui = 1/√(d_u d_i), phải trừ hệ số đó. Dùng √((d_u−1)(d_i−1)) chỉ đúng nếu thực sự dựng lại graph với degree mới; không đúng khi trừ từ H¹ của graph cũ. Graph weighted phải lấy đúng trọng số A_ui.
2. **Fallback sau trừ.** Khi neighbor duy nhất bị loại, không lấy original target làm fallback vì đưa self-return trở lại. Nếu nghiên cứu leave-one-edge-out, mask query không còn context khỏi loss, ghi denominator số query hợp lệ; không tạo hướng fallback tùy tiện.
3. **β compensation.** Chia H¹ cho β đã clamp là phép đổi scale của auxiliary head. Nó không chứng minh tọa độ cuối là “ngữ nghĩa phổ cao”, vì hệ tọa độ sau whitening không tự có diễn giải đó. Cần control cùng reweighting cho Euclidean.
4. **Công thức loss phải khớp source.** Log v1 dùng scale khoảng cách `2R²τ`, với R=2, không chỉ `2τ`. Đổi scale tương đương đổi temperature hiệu dụng, cần nhận diện như thay đổi độc lập.
5. **Khoảng cách thật và surrogate phải phân biệt.** Source geometry v1 tính geodesic squared bằng acosh có xử lý vùng gần 1. Không mô tả log đã chạy như dùng chordal approximation của đề xuất v2.
6. **Tách reference lịch sử và reference đã audit.** Gọi V1-L cho implementation tạo log hiện có; V1-A cho cấu hình sau audit. Không so V2 với V1-L rồi quy mọi chênh lệch cho edge trust nếu đã đồng thời sửa loss.

Primary v2 dùng LHC không có pair-specific self-return subtraction để tránh thêm thay đổi chưa được kiểm chứng. Nhánh subtraction đúng hệ số và mask context rỗng được giữ như ablation riêng. Đây là lựa chọn thiết kế, không phải tuyên bố no-subtraction chắc chắn tốt hơn.

## 4. Nghiên cứu liên quan và giới hạn chuyển giao

Tìm kiếm có mục tiêu vào cơ chế graph/update/geometry; không phải systematic review đầy đủ. Nguồn được kiểm tra là paper, trang hội nghị và repository tác giả. Không lấy việc có GitHub hoặc được chấp nhận làm chứng cứ sẽ thắng trên STAIR.

| Công trình | Điều có thể học | Điều không thể suy ra cho v2 |
|---|---|---|
| [STAIR — Xu et al.](https://arxiv.org/abs/2412.11729), [code tác giả](https://github.com/yhhe2004/STAIR) | Giữ tách collaborative và multimodal information, hạn chế forgetting bằng constrained updates | Mọi graph từ modality đều là preference graph; đổi BSC luôn cải thiện |
| [Learning Geometry-Aware Recommender Systems with Manifold Regularization](https://github.com/ITMO-NSS-team/RECMAN_recsys2025), RecSys 2025 LBR | Có thể dùng regularization hình học mà giữ architecture/inference | Hyperbolic regularization tối ưu cho cả ba Amazon split; overhead train bằng 0 |
| [TriplH: Leveraging Geometric Insights in Hyperbolic Triplet Loss](https://arxiv.org/html/2508.11978v1), [code tác giả](https://github.com/YusupovV-Lab/TriplH), RecSys 2025 LBR | Phân biệt objective ranking và lựa chọn biểu diễn; geometry phải gắn với loss cụ thể | Copy surrogate vào LHC sẽ giữ cùng ý nghĩa; hiệu quả trên benchmark khác chuyển trực tiếp |
| [Review-Based Hyperbolic Cross-Domain Recommendation](https://arxiv.org/html/2403.20298v3), WSDM 2025 | Geometry và degree được dùng trong bối cảnh alignment cross-domain cụ thể | Degree-temperature là radial regularizer cho single-domain multimodal STAIR |
| [The Numerical Stability of Hyperbolic Representation Learning — ICML 2023](https://proceedings.mlr.press/v202/mishne23a.html) | Lorentz và Poincaré đều có hạn chế số học; kiểm tra precision, norm, domain | Lorentz tự động chống NaN hoặc có gradient luôn tốt hơn |

Danh mục chính thức xác nhận RECMAN và TriplH là **late-breaking results (LBR)**, không gọi hai bài này là full-paper SOTA đã xác lập. Xem [RecSys 2025 accepted contributions](https://recsys.acm.org/recsys25/accepted-contributions/) và [WSDM 2025 accepted papers](https://www.wsdm-conference.org/2025/accepted-papers/). Repository/paper được đọc để đối chiếu cơ chế; chưa clone/chạy tái lập các công trình này trong nhiệm vụ viết báo cáo.

### 4.1. Sửa hệ phương trình Lorentz

Đặt curvature −κ, κ>0; dùng c=1/κ để tránh trộn “radius” với curvature. Với signature (−,+,…):

\[
\langle x,y\rangle_L=-x_0y_0+x_s^\top y_s,\qquad
\mathbb H_c^d=\{x:\langle x,x\rangle_L=-c,\ x_0>0\}.
\]

Khoảng cách địa trắc và chordal surrogate không giống nhau:

\[
d_{\rm geo}^2(x,y)=c\,\operatorname{acosh}^2\!\left(-\frac{\langle x,y\rangle_L}{c}\right),
\qquad D_{\rm chord}^2(x,y)=-2c-2\langle x,y\rangle_L.
\]

Vì ⟨x,y⟩L≤−c, chordal surrogate không âm, bằng 0 tại x=y. Với d=d_geo:

\[
D_{\rm chord}^2=2c\,[\cosh(d/\sqrt c)-1]\approx d^2\quad(d\approx0).
\]

Không gọi D_chord² là geodesic squared ở khoảng cách lớn. Biểu thức của đề xuất `2c+2⟨x,y⟩L` là âm của surrogate đúng. Các paper dùng notation/score có hằng số cộng khác nhau cũng phải kiểm tra self-distance và convention. Hằng số chung có thể triệt tiêu trong softmax với cùng temperature; không mặc nhiên triệt tiêu khi positive và negative chia khác temperature.

Với tangent vector v, r=‖v‖:

\[
\exp_o(v)=\left[\sqrt c\cosh(r/\sqrt c),\ \sqrt c\sinh(r/\sqrt c)\frac v r\right].
\]

Tại r=0 dùng analytic limit. Nếu hai vector cùng tangent norm r, góc θ:

\[
-\langle x,y\rangle_L/c=
\cosh^2(r/\sqrt c)-\sinh^2(r/\sqrt c)\cos\theta.
\]

Do đó khoảng cách tăng đơn điệu khi cosine giảm. Trong trường hợp này kNN hyperbolic có cùng thứ tự với cosine (trừ tie/numerical error); thêm s_H không cung cấp quan hệ lân cận mới. Norm khác nhau có thể đổi thứ tự, nhưng norm raw feature chưa được chứng minh mã hóa hierarchy.

### 4.2. Vì sao temperature không phải radial regularization?

Với logits z_ij = −D²_ij/T_i và multi-positive CE, đạo hàm theo D²_ij là `(p_target_ij − p_model_ij)/T_i`. Temperature vừa thay đổi xác suất vừa scale gradient. Dấu gradient theo radius còn phụ thuộc đạo hàm D² theo radius, góc và các candidate khác. Không có định lý “T nhỏ kéo vào origin”.

Nếu vẫn thử degree-conditioned temperature, dùng **một T_i cho toàn bộ một hàng**, chung positive/negative; freeze từ train degree, bound rõ T_min≤T_i≤T_max và áp dụng đối xứng định nghĩa ở chiều ngược. Đây là ablation về calibration/optimization, không phải prior về hierarchy; không sử dụng trong primary.

## 5. Giả thuyết và kiến trúc C-HET điều chỉnh

### 5.1. Ba giả thuyết có thể bác bỏ

- **H1 — Edge support:** tăng tương đối trọng số cạnh modality có hỗ trợ đồng tương tác đủ mạnh cải thiện BSC so với trọng số semantic thuần. Bác bỏ nếu graph mới không thắng placebo degree-matched hoặc làm giảm held-out ranking.
- **H2 — Conservative intervention:** blend với S₀ giúp giảm nhạy cảm dataset so với thay toàn bộ graph. Bác bỏ nếu mức can thiệp nhỏ không giúp, hoặc lợi ích chỉ đến từ một vài cấu hình tuning riêng lẻ.
- **H3 — Complementarity:** graph calibration và LHC đem tín hiệu khác nhau. Kiểm chứng bằng factorial; không dùng rho>0 làm bằng chứng duy nhất. Nếu edge-only tốt hơn combined thì chọn edge-only.

Không giả định trước rằng H1–H3 đúng. Không gắn mức tăng dự kiến 3%, 6% hay SOTA vào một kiến trúc chưa chạy.

### 5.2. Luồng thông tin

```text
Raw text/image → baseline MI + candidate kNN support → W0, S0
Train-only binary R → candidate-only co-occurrence → q_ij
                       ↓
             W+ = W0 ⊙ (1 + a q)
                       ↓ normalize
             S+ = D+^(-1/2) W+ D+^(-1/2)
                       ↓ blend
             Sη = (1−η) S0 + η S+
                       ↓
             Baseline BSC Neumann update operator

Train R → unchanged FSC → Euclidean dot-product → BPR
                         ↘ optional audited LHC auxiliary loss
```

Chỉ thay **item-item graph của BSC**. Không dùng Sη làm user-item FSC adjacency; không đổi ranking scorer; không thêm geometry ở inference. User embedding update và weight decay giữ baseline. Khi LHC bật, nó góp gradient trước Adam/BSC theo cùng thứ tự của reference; không âm thầm đổi thành smooth riêng BPR rồi cộng LHC.

### 5.3. Static data và candidate-only statistics

R∈{0,1}^{U×I} chỉ chứa interaction train, deduplicate (u,i). Với item i:

\[
d_i=\sum_u R_{ui},\quad n_{ij}=\sum_u R_{ui}R_{uj},\quad
b_{ij}=\begin{cases}n_{ij}/\sqrt{d_i d_j},&d_i d_j>0\\0,&\text{otherwise}.\end{cases}
\]

Chỉ tính n_ij trên candidate support C từ baseline kNN. Không tạo dense I×I, không tính toàn bộ RᵀR. Dùng sorted user lists/intersections hoặc sparse query batched có kiểm soát intermediate memory. Degree d_i là train behavior degree, khác degree của graph item-item; không dùng lẫn trong normalization.

Đặt hệ số giảm can thiệp khi thiếu chứng cứ:

\[
\rho_{ij}=\frac{n_{ij}}{n_{ij}+t},\qquad q_{ij}=\rho_{ij}b_{ij},\qquad t>0.
\]

q∈[0,1], đối xứng; n=0 dẫn đến q=0. t=5 là **pilot heuristic**, không phải confidence interval hoặc Bayesian posterior. Cặp chỉ có một common user được giảm trọng số chứng cứ. Không đồng tương tác nghĩa là thiếu chứng cứ, không có nghĩa dislike. q không hiệu chỉnh exposure; popularity/user activity vẫn có thể gây confounding.

### 5.4. Dựng graph đúng cấp normalization

W₀ là raw nonnegative symmetric item graph **trước** symmetric degree normalization, giữ đúng modality fusion, self-loop, tie và duplicate policy baseline. S₀ là toán tử baseline thực tế.

Trong `main.py` hiện có, baseline dựng kNN từng modality với `symmetric=False`, nối các edge lists, gán weight 1, `coalesce(reduce='sum')`, rồi `to_undirected(reduce='max')`, sau đó mới `to_normalized(normalization='sym')`. **W₀ phải được lấy ngay trước bước cuối**, không phải similarity cosine raw, không phải S₀ đã chuẩn hóa và không phải một graph modality được fusion theo công thức mới. Diagonal similarity được loại trước kNN; không tự thêm self-loop. Việc một cạnh xuất hiện ở nhiều modality đã nằm trong raw weight sau coalesce.

\[
W^+_{ij}=W^0_{ij}(1+a q_{ij}),\quad a\ge0,\qquad
D^+_{ii}=\sum_j W^+_{ij},\qquad
S^+=(D^+)^{-1/2}W^+(D^+)^{-1/2}.
\]

Node degree bằng 0 có inverse-degree bằng 0, không chia 0. Nếu baseline dùng một graph khác với giả định W₀ đối xứng/nonnegative, implementation phải tái lập S₀ và kiểm tra trước; không được áp đặt điều kiện mới rồi tuyên bố baseline recovery.

\[
S_\eta=(1-\eta)S_0+\eta S^+,\qquad0\le\eta\le1.
\]

Blend ở **operator level**. Sη không nhất thiết chính là normalized adjacency của một raw graph với một degree matrix duy nhất; không gán stationary distribution của W⁺ cho Sη. Nếu a=0 hoặc η=0, trả đúng S₀ qua fast-path, không dựng lại bằng arithmetic khác.

Các cạnh q=0 giữ raw weight nhưng normalized weight có thể thay do degree đổi. Giữ support không giữ spectrum, eigenvectors hoặc “100% ngữ nghĩa”. Nếu q là hằng số trên mọi cạnh, W⁺ chỉ bị scale toàn cục và S⁺=S₀: dùng làm negative control bắt lỗi normalization.

Primary pilot: fix a=0.25, t=5; η∈{0,0.25,0.5}. Không tune đồng thời quá nhiều a và η vì ảnh hưởng gần tuyến tính dễ trùng lặp. Sau pilot mới xét a∈{0.1,0.5} theo ngân sách validation đã chốt.

| Đại lượng | Trạng thái | Nguồn/quy tắc |
|---|---|---|
| R_train, W₀, S₀, d_i, n_ij, b_ij, q_ij | Static, không autograd | Chỉ train interactions và baseline modality candidates |
| t, a, η | Hyperparameter cố định mỗi run | Pilot grid trên validation; không trainable |
| Item/user embeddings | Trainable | MI init, baseline AdamWSEvo groups/decay |
| κ, R, τ, λ schedule khi bật LHC | Hyperparameter cố định mỗi run | Reference v1 đã audit; matched Euclidean control |
| Head scale buffers | Frozen sau initialization | Initial embeddings train-only; checkpoint persistent |
| W_raw, sigmoid gate weights, radial temperature | Không có trong primary | Không để placeholder không xác định fitting |

### 5.5. BSC và ranh giới lý thuyết

Cho b_j∈[0,1) là hệ số BSC baseline tại tọa độ j, L là số bước baseline. Không thay công thức b_j bằng một heuristic mới:

\[
P_j(S)G_j=\frac{1-b_j}{1-b_j^{L+1}}
\sum_{\ell=0}^L b_j^\ell S^\ell G_j.
\]

Smoother nhận **update direction đúng vị trí baseline AdamWSEvo**, không tự chuyển sang raw gradient. Code hiện hành phải được audit để xác nhận vị trí smoothing/moment/decay; tests parity cần kiểm cả một optimizer step, không chỉ P_j.

Nếu W₀,W⁺ đối xứng nonnegative và normalization hợp lệ, spectrum S₀,S⁺ thuộc [−1,1]. Convex blend cho ‖Sη‖₂≤1. Các coefficient polynomial không âm và tổng bằng 1, nên ‖P_j(Sη)‖₂≤1. Đây là bound cho **linear operator**, không phải chứng minh convergence AdamW hay metric tăng; S có thể có eigenvalue âm và không phải PSD.

Với δ=‖Sη−S₀‖₂≤2η và hai toán tử có norm≤1:

\[
\|P_j(S_\eta)-P_j(S_0)\|_2
\le\delta\,\frac{1-b_j}{1-b_j^{L+1}}
\sum_{\ell=1}^L\ell b_j^\ell.
\]

Bound được suy ra bằng telescoping Sη^ℓ−S₀^ℓ; dùng để kiểm mức can thiệp, thường khá lỏng. Nó không bảo đảm giữ norm gradient hay lượng multimodal information. Nếu q rất thưa, đừng tăng a mạnh chỉ để “tạo hiệu ứng”.

### 5.6. LHC tùy chọn, reference kiểm soát chặt

Giữ head v1 đã audit như một nhánh riêng: cùng cap, initialization scale, β reweight, candidate dedup, multi-positive handling và denominator của reference. Không đồng thời đổi sang hybrid distance, learned curvature và degree temperature.

Định nghĩa representation rõ ràng: E∈ℝ^((U+I)×d), H⁰=E, H¹=(A_UI E)⊙β_FSC, với A_UI là adjacency user-item baseline; β_FSC=1−β₃, khác hệ số BSC b=β₃. Tách H⁰ thành U⁰/I⁰, H¹ thành U¹/I¹. Hai chiều regularization là **U⁰→I¹** và **I⁰→U¹**, không chuyển âm thầm thành item self-contrastive I⁰→I¹. Candidate là unique item/user IDs của batch theo từng chiều; positive mask lấy R_train trên các ID đó. Matrix logits có shape số unique queries × số unique candidates. Scale/cap preprocessing của head được fit một lần từ embeddings ban đầu, lưu vào checkpoint, không dùng validation/test interactions.

Cho một chiều query→candidate với M_i là tập positive train hợp lệ trong candidate set, target distribution uniform trên M_i:

\[
z_{ij}=-d_{\rm geo}^2(\psi(h_i),\psi(k_j))/(2R^2\tau),\qquad
\ell_i=-\frac1{|M_i|}\sum_{j\in M_i}
\log\frac{e^{z_{ij}}}{\sum_{k\in C_i}e^{z_{ik}}}.
\]

Average trên query có positive hợp lệ, rồi average hai chiều theo reference. Duplicate item IDs được deduplicate và remap trước; known positives không được coi là negative chỉ vì nằm ngoài diagonal. Đây là multi-positive CE regularizer; không tự tuyên bố MI lower-bound nếu sampling/label structure không thỏa điều kiện InfoNCE cổ điển.

\[
\mathcal L=\mathcal L_{\rm BPR}+\lambda(e)\mathcal L_{\rm LHC},\quad
\lambda(e)=\lambda_{\max}\operatorname{clip}\left(\frac{e-20}{30},0,1\right).
\]

Giữ regularization/decay của baseline đúng source. λ=0 bỏ tính auxiliary để bảo toàn chi phí/RNG baseline. BPR ranking vẫn là dot-product FSC embeddings. Stop-gradient không được thêm tự động: nó đổi objective và gradient flow, không chứng minh loại bỏ tuyệt đối interference. Không cần learnable θ, projector, gate hoặc auxiliary optimizer cho primary C-HET.

## 6. Hợp đồng triển khai dự kiến — chưa viết code

| File dự kiến | Trách nhiệm | Bất biến bắt buộc |
|---|---|---|
| `models/stair5_v2_graph.py` | Extract raw W₀; candidate intersections; q; normalize và blend; cache metadata | Train-only, symmetric/nonnegative; không dense I²; fast-path S₀ |
| `models/stair5_v2.py` | Inherit baseline, giữ MI/FSC/scorer; optional audited LHC; expose diagnostics | Baseline module không bị sửa; B0 tắt mọi auxiliary |
| `optimizers/stair5_v2_smoother.py` | Matrix-free polynomial callback với Sη; baseline moment/decay ordering | Không học parameter; không giữ autograd graph; item-only |
| `main_stair5_v2.py` | Seeding; optimizer groups; manifests; checkpoint, timing, diagnostics | Param xuất hiện đúng một group; cùng eval/checkpoint rule |
| `configs/dataset_stair5_v2_{baby,sports,electronics}.yaml` | Kế thừa giá trị baseline cùng pilot overrides công khai | Không thay lr/decay/batch/γ âm thầm; no test tuning |
| `tests/test_stair5_v2.py` | Algebra, source parity, optimizer-step parity, leakage, checkpoint | So baseline thực, không chỉ so bản copy công thức |
| `notebook/P5/stair5_v2.ipynb` | Kaggle setup, preflight, isolated arms, selected datasets, logs | Import đủ; fail-fast; không force-reset làm mất artifact |

Tên file là kế hoạch, không phải file đã tồn tại. Không overwrite v1: V1-L cần giữ để truy xuất kết quả. Candidate/preprocessing cache có hash R_train, features, item mapping, kNN settings, baseline normalization, a/t/η, library versions; thay split phải invalidate cache. Static graph có thể cache, không cần snapshot dynamic gate vì primary không có gate học.

### Kiểm thử cần có trước pilot

1. a=0 và η=0 cho đúng baseline encode, full/pool rank, seen masking, BPR, một AdamWSEvo step; bitwise khi dùng cùng cached S₀ và deterministic backend, nếu không báo tolerance và nguyên nhân.
2. Toy graph chứng minh row degrees, symmetry, duplicate merge, isolated node và common-user counts đúng; validation/test edge không ảnh hưởng q.
3. q constant → normalized S⁺=S₀; q nonconstant → đổi toán tử có giới hạn; no PSD assertion.
4. Polynomial sparse callback trùng dense toy calculation và bound; không materialize S² hoặc I×I.
5. Checkpoint roundtrip chứa embeddings, optimizer moments, epoch/scheduler, graph config/hash và scale buffers; resume không reinitialize auxiliary scale.
6. LHC: self-distance≈0, nonnegative, finite near origin/cap; duplicate positives không thành false negative; no-positive/empty-context batch an toàn; finite gradients nhiều bước.
7. Nếu ablate self-return, trừ đúng A_ui của graph gốc, test user/item degree 1, weighted graph và β gần clamp; context rỗng bị mask.

## 7. Thực nghiệm để phân biệt cơ chế

### 7.1. Các arm chính

| Arm | Edge calibration | Auxiliary | Vai trò |
|---|---|---|---|
| B0 | Off | Off | Baseline tái lập ghép cặp |
| V1-L | Off | Legacy H0 | Truy xuất log lịch sử; không gán sửa bug cho v2 |
| E0-A | Off | Euclidean, cùng head scaling/sampling | Control geometry |
| H0-A | Off | Audited geodesic LHC | Reference cho hình học |
| ET | On | Off | Tác dụng edge độc lập |
| ET-H0 | On | Audited geodesic LHC | Interaction edge × geometry |
| ET-placebo | Permuted q | Off | Weight distribution/popularity control |

Factorial cốt lõi là B0/H0-A/ET/ET-H0. Nếu thay audit LHC so v1, chạy so V1-L/H0-A riêng trước khi diễn giải. Placebo shuffle q trên candidate edges theo bin degree hai endpoint, symmetry preserved; giữ phân phối boost gần tương đương. Báo cáo residual imbalance, không gọi placebo này loại hết popularity confounding.

Mở rộng sau khi có tín hiệu: remove ρ để xem sample-support shrinkage có giá trị; compare behavior-only với geometry factor; degree row-temperature riêng. Không triển khai tất cả ngay hoặc tìm cấu hình thắng bằng test.

### 7.2. Quy trình ngân sách

**Giai đoạn A — parity:** toy tests và mini train cùng batches cho B0; kiểm sampler, FSC, γ, whitening, negative sampling, Adam ordering, full ranking và checkpoint rule.

**Giai đoạn B — pilot Sports và Baby:** một seed với cùng budget 150–200 epoch cho các arm cốt lõi; η nhỏ, các giá trị khác cố định. Đây chỉ là sàng lọc, vì Sports v1 đạt best ở epoch 500 nên pilot không loại được khả năng lợi ích muộn. Ghi curve, thời gian và gradient/update diagnostics; không mở test để chọn arm.

**Giai đoạn C — confirm:** shortlist arm qua validation, chạy đủ budget baseline 500 epoch, ít nhất 3 seed ghép cặp trên hai tập. 5 seed tốt hơn nếu mức tăng quanh 1%. B0 cũng chạy lại, không lấy baseline lịch sử làm đối chứng duy nhất.

**Giai đoạn D — Electronics:** chỉ mở khi preflight đạt, runtime/memory hợp lý và có tín hiệu validation không phụ thuộc riêng Sports. Run v1 hiện có khoảng 11.39h nên kiểm Kaggle session budget, checkpoint resume và precompute riêng. Không cam kết tổng thời gian trước khi benchmark candidate intersection và epoch throughput.

Các mốc trên là ngân sách đề xuất, không phải rule chứng minh significance. Có thể thay đổi theo tài nguyên trước khi xem kết quả; mọi thay đổi sau khi xem validation phải được ghi trong manifest.

### 7.3. Metric, chọn model và bất định

- Primary: NDCG@20, checkpoint chọn bằng validation NDCG@20 như baseline. Secondary: Recall@10/20, NDCG@10; dùng đúng aggregation FreeRec và protocol full/pool.
- Cùng split, seeds, item IDs, preprocessing, sampler, epoch/eval frequency và tuning budget giữa arm. Per-dataset config baseline hợp lệ được giữ, không bắt ba tập dùng chung decay nếu baseline không như vậy.
- Báo mean/std qua seed và paired difference từng seed; paired bootstrap user metric là uncertainty theo user, không thay được uncertainty do training seed. Không dùng một p-value từ user bootstrap để chứng minh robustness qua seed.
- Báo absolute delta và relative delta; dùng metric đầy đủ precision. Không chọn checkpoint theo test; không báo “best test epoch”.
- Nếu mục tiêu “>6% đa số metric” được dùng sau này, phải predefine 4 metric × 3 dataset, ngưỡng và cách tổng hợp trước. Đây là tiêu chí tham vọng, không phải forecast kiến trúc.
- Reporting trade-off: overall, train-degree strata head/mid/tail, coverage@20, mean recommended popularity, preprocessing/epoch/total time và peak GPU/CPU memory. Strata được xác định từ train, không từ test.

### 7.4. Diagnostics có ích

| Diagnostic | Mục đích | Giới hạn |
|---|---|---|
| q nonzero rate, n histogram theo degree | Biết có đủ evidence để graph khác baseline | Không phải hiệu chuẩn xác suất |
| ‖(Sη−S₀)X‖/‖S₀X‖ trên fixed probes | Đo mức can thiệp thực tế | Không thay spectral norm certificate |
| Raw gradient cosine user/item, nhiều fixed train batches | Tách khác biệt nhóm và sampling | Không suy ra test benefit |
| λ‖g_LHC‖/‖g_BPR‖, actual Adam/BSC update cosine | Đo cường độ và ảnh hưởng optimizer | Diagnostics không thay ablation |
| Radius distribution và cap-hit rate | Phát hiện saturation, mất sensitivity | Không chứng minh hierarchy |
| Head/mid/tail ranking và popularity placebo | Kiểm tra trade-off/exposure confounding | Offline test không đo causal exposure |
| Sparse nnz, preprocessing RAM, epoch throughput | Kiểm soát Kaggle budget | “Không tăng bậc” vẫn có thể tăng thời gian đáng kể |

Không đặt ngưỡng Pearson tùy ý như một chứng chỉ hierarchy. Muốn kết luận có hierarchy cần probe độc lập (category taxonomy hoặc cấu trúc graph có kiểm định) và geometry control; taxonomy cũng không tự đồng nghĩa user preference hierarchy.

## 8. Rủi ro lý thuyết và tiêu chí dừng

1. **Association không phải preference.** Common users có thể do exposure, item popularity hoặc user activity. Ochiai và ρ không giải quyết hết. Nếu placebo cho lợi ích tương tự, giảm claim xuống reweighting effect, không gọi behavior trust được xác nhận.
2. **Tail thiếu chứng cứ.** q=0 làm raw boost neutral nhưng normalization vẫn có thể giảm relative coupling với head; theo dõi tail recall. Nếu tail giảm trong khi overall tăng, báo trade-off, không gọi “fair hơn”.
3. **Co-consumption không phải tương đương.** Hai sản phẩm complement có thể cùng được mua; smoothing ép tương tự có thể hại ranking. Giữ candidate semantic support hạn chế nhưng không loại bỏ rủi ro.
4. **Can thiệp BSC không phải objective mới có guarantee.** Bounds linear operator không chứng minh tăng NDCG. Nếu ET không vượt B0 và placebo sau confirm, dừng edge branch; không thêm module để cứu narrative.
5. **LHC có thể thừa.** Nếu ET thắng ET-H0, dùng ET; nếu E0-A ngang H0-A, không gán lợi ích cho curvature.
6. **V1-A có thể khác v1 đáng kể.** Source fixes và head-scale changes phải được đo riêng; không tính tất cả là đóng góp v2.
7. **Small gains dễ do noise/tuning.** Nếu khoảng seed-difference rộng hơn gain hoặc dấu đổi theo dataset, báo chưa đủ bằng chứng. Không tuyên bố SOTA chỉ từ một seed thắng baseline.
8. **Graph pipeline không thật sự raw.** Nếu chỉ có S₀ normalized, nhân boost lên nó rồi normalize lại không bằng boost W₀. Phải tìm đúng builder hoặc ghi rõ đây là một toán tử khác; baseline off path vẫn phải giữ S₀.
9. **Preprocessing vẫn có chi phí.** kNN baseline có thể O(I²) nếu exact all-pairs. v2 không giải quyết phần đó; candidate count không đủ để suy ra toàn pipeline linear. Intersection cost phụ thuộc list lengths, không O(1) mỗi cặp.

Chỉ mở thêm hyperbolic edge factor khi cho thấy nó cung cấp thông tin vượt cosine với đúng norm setting. Không học W_raw offline rồi bỏ mất objective huấn luyện: nếu thêm learned graph encoder, đó là một nhánh kiến trúc khác cần train-only supervision, optimizer, freeze lifecycle và matched-budget control riêng.

## 9. Kết luận reviewer và mức sẵn sàng

**Đề xuất STAIR-HET nguyên bản: cần major revision trước khi triển khai.** Các blocker là công thức khoảng cách sai dấu, inference về radius không có chứng minh, normalization graph chưa xác định, baseline bị trộn và thiếu thiết kế để phân biệt tác dụng các module. Bản phản biện thứ hai sửa được một phần nhưng cũng chứa suy luận sai; không nên chuyển nguyên văn thành specification.

**C-HET điều chỉnh: đủ rõ để triển khai prototype và kiểm thử parity**, chưa đủ để kết luận hiệu quả. Ưu tiên edge-only bảo thủ, factorial với LHC đã audit, và confirm đa seed. Không có căn cứ để cam kết phục hồi Baby/Electronics hoặc vượt 6% trước thực nghiệm.

Quy trình review dùng academic-research-suite, với hai vai trò methodology và editorial tách pha lập tiêu chí trước khi xem proposal. Đây là review hỗ trợ bằng AI cùng họ mô hình, không phải hai chuyên gia độc lập hay peer review từ hội nghị. Artifact tiêu chí và review nằm trong `scratch/g5v2_review/`; các kiểm tra conformance chỉ xác minh hợp đồng review, không chứng minh kiến trúc đúng hay đã chạy.

Các file code, log và báo cáo v1 được giữ nguyên trong nhiệm vụ này. Bước tiếp theo hợp lý là triển khai prototype C-HET theo mục 6 và chạy parity/pilot; không đưa radial module của proposal vào pipeline production ngay.
