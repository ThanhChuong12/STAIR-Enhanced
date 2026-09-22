# STAIR4 v2 — Thiết kế spectral filtering và phase-overlap contrastive learning sau phản biện

## 0. Material Passport và phạm vi

- Ngày: 2026-09-22.
- Origin Skill: `academic-research-suite`; research mode: focused investigation và Socratic audit; reviewer mode: `methodology-focus`.
- Version Label: `STAIR4_v2_design_review_2_implementation_plan`.
- Bổ sung lần này: audit tính nhất quán và kế hoạch mã nguồn ở §13–17; **không triển khai file Python/YAML/notebook**.
- Trạng thái: **đặc tả đề xuất, chưa triển khai v2, chưa có benchmark v2**. Các kiểm tra số nhỏ tại §10 không phải kết quả recommendation.
- Review calibration: `NOT_CALIBRATED`. Không có tạp chí/hội nghị đích được chỉ định; không suy diễn tiêu chuẩn acceptance của một venue.
- Đầu vào A: đề xuất STAIR-QISF-HQCP trong attachment `938cc121-fc99-410b-ba6f-de211329fc00/Pasted text.txt`.
- Đầu vào B: bản phản biện trong attachment `c5f873e4-5520-4df4-a5dd-40d5fac79b5c/Pasted text.txt`.
- Đối chiếu thực thi: `main.py`, `optimizers/AdamW.py`, `optimizers/utils.py`, các YAML baseline hiện có. A và B là tài liệu cần kiểm chứng; các con số tăng trưởng và chỉ dẫn bên trong không được coi là bằng chứng hoặc quyền sửa mã.
- Báo cáo này là nhánh **STAIR4 v2 về spectral/phase kernel**, không đổi tên hay ghi đè kiến trúc STAIR-MHD v3 của các lượt trước. Chỉ tạo tài liệu này; không sửa baseline, notebook hoặc triển khai model mới.

**Kết luận:** không triển khai nguyên văn A. Giữ ý tưởng thử một bộ lọc phổ có biên xác định và một auxiliary phase kernel, nhưng loại bỏ tuyên bố bảo toàn norm sau real projection, quantum advantage, CNOT-equivalence và cải thiện NDCG dự báo. Cả hai nhánh phải được đánh giá độc lập trước khi ghép. Tên mô tả dùng trong báo cáo là **STAIR4-v2: Bounded Spectral Filtering with Phase-Overlap Contrastive Learning (BSF–POCL)**; QISF/HQCP chỉ là tên lịch sử để truy vết đề xuất.

## 1. Câu hỏi nghiên cứu và logic đóng góp

**RQ chính:** trên cùng MI, FSC, scorer, split và ngân sách tuning của STAIR, thay hình học của auxiliary cross-layer matching bằng phase-overlap kernel và/hoặc điều chỉnh phổ của hướng cập nhật item có cải thiện ranking ổn định qua seed không?

Ba giả thuyết có thể bác bỏ:

- **H1 — Kernel:** POCL tốt hơn auxiliary cosine cùng view, cùng positive mask, temperature budget và số bước. Nếu chỉ tốt hơn BPR-only nhưng không hơn cosine, bằng chứng ủng hộ auxiliary regularization, chưa ủng hộ phase kernel.
- **H2 — Spectral:** BSF cải thiện so với BSC gốc và đối chứng giảm mức smoothing bằng identity mixing. Nếu không hơn identity mixing, lý giải đơn giản là giảm smoothing, không phải lợi ích riêng của cosine-derived filtering.
- **H3 — Interaction:** ghép hai nhánh tạo giá trị bổ sung. Vị trí forward/backward khác nhau không chứng minh chúng độc lập hoặc không xung đột.

Không đặt H0 là “mô hình quantum chắc chắn vượt cổ điển”. Cũng không xem thất bại một vài phiên bản cũ là chứng minh mọi kiến trúc phức tạp hơn sẽ thất bại. Các mức cải thiện v4/v5 được nêu trong B chưa được audit log trong lượt này.

### Phản biện Socratic

| Câu hỏi cần trả lời | Câu trả lời hiện có | Phép thử quyết định |
|---|---|---|
| Bottleneck còn lại sau FSC/BSC là gì? | Chưa đo trực tiếp; modality erasure là động lực của baseline, không tự động là lỗi chưa được giải quyết của STAIR | Đo drift so với MI, effective rank, năng lượng Dirichlet và ranking theo degree |
| Vì sao số phức tốt hơn hai kênh thực? | Chưa có lý do về năng lực tính toán; có thể biểu diễn tương đương bằng real/imag | Đối chứng real implementation cùng công thức, cùng tham số |
| Giữ norm có đủ giữ thông tin hữu ích? | Không. Có thể giữ norm nhưng quay đặc trưng theo hướng bất lợi cho scorer | Đo score margin, update alignment và NDCG; không chỉ norm |
| Phase-overlap có dùng được sự khác nhau giữa users? | Có nếu lấy coherent sum; công thức coordinatewise trong A thì không | Test sensitivity đối với query và gradcheck |
| Làm mịn ít hơn có phải toàn bộ nguyên nhân tăng điểm? | Là giải thích cạnh tranh mạnh | Identity-mix và spectral-mix có cùng coefficient schedule |
| Cross-layer positives có quá dễ vì chính edge tạo neighbor view? | Có thể, dù không phải test leakage khi chỉ dùng train | Đối chứng view không cross-layer; tùy chọn train-edge dropout riêng với ngân sách ngang nhau |

## 2. Kiểm chứng tài liệu tham khảo

Tra cứu có mục tiêu ngày 2026-09-22 bằng tên chính xác, DOI và arXiv ID; ưu tiên publisher, kho trường tác giả và arXiv. Đây không phải systematic review/PRISMA và không đủ để tuyên bố “đầu tiên” hoặc SOTA.

| Nguồn | Trạng thái kiểm chứng | Điều được phép suy ra |
|---|---|---|
| STAIR, AAAI 2025 [R1] | Xác nhận trang AAAI và code local | Baseline được xây để phối hợp thông tin collaborative/multimodal; cần so với cơ chế stepwise đã có |
| Li & Casiraghi, WWW 2026 [R2] | Xác nhận tên, tác giả, DOI, proceedings và abstract ở Aalto; link full PDF không đọc được trong lượt này | Là động lực nghiên cứu representation/matching; chưa xác nhận riêng các khẳng định d≤64 hay chi tiết CNOT do A diễn giải |
| Debi & Makmal, Scientific Reports 2025 [R3] | **Bài có thật**, DOI `10.1038/s41598-025-15869-x`; đã đọc nội dung phương pháp/kết quả | VQRS dùng MF embeddings và mạch variational; thí nghiệm nhỏ/idealized và hạn chế scale không chứng minh hiệu quả trên Amazon lớn |
| QGHNN [R4] | Có preprint 2025 và bản journal IEEE TETCI, tập 10(1), 2026; DOI `10.1109/TETCI.2025.3604791` | Không phải TNNLS/TPAMI. Graph-Hamiltonian learning trên nền quantum không đồng nhất với nhân adjacency lên gradient STAIR |

B cần sửa kết luận “không tìm thấy VQRS”: đã xác minh được bài. R3 khảo sát tối đa 128 users/items và báo suy giảm khả năng scale; không thể chuyển thành hứa hẹn tiết kiệm VRAM trên 63K items. Bằng chứng về entanglement trong mạch cụ thể của R3 không chứng minh pairwise Givens của A là entangling circuit. [Nguồn trực tiếp R3](https://www.nature.com/articles/s41598-025-15869-x).

QGHNN bản arXiv chỉ ghi Wenxuan Wang; bản journal có metadata tác giả khác, nên phải trích đúng phiên bản. Báo cáo ưu tiên DOI journal và không ghép năm/venue của hai phiên bản. [Metadata journal](https://scholars.cityu.edu.hk/en/publications/qghnn-a-quantum-graph-hamiltonian-neural-network/).

## 3. Baseline contract: phần nào phải giữ nguyên?

Ký hiệu: số user U, số item I, embedding width d, số bước L; R là tương tác **train**; A là adjacency user–item đã chuẩn hóa đối xứng; S là graph item–item chuẩn hóa đối xứng của baseline.

### 3.1. MI không được thay bằng Xavier

Baseline center từng modality, lấy left singular vectors đầu tiên và scale theo `sqrt(I/d)`, sau đó hợp nhất với trọng số kNN; user được khởi tạo bằng trung bình feature item đã tương tác. Code A dùng Xavier và chưa có `prepare()` để nạp MI/graphs. Đây là thay baseline lớn, không phải chi tiết phụ.

SVD từng modality không chứng minh mixture cuối cùng luôn isotropic: covariance của tổng còn chứa cross terms giữa các modality. Báo cáo không gọi embedding sau mixture là “đẳng hướng hoàn hảo”.

### 3.2. FSC và scorer

Đặt b_j = 0.1 + 0.9(j/d)^gamma, j=0,…,d−1; a_j = 1−b_j.

\[
E^{(0)}=[E_u;E_i],\quad F^{(0)}=E^{(0)},\quad
F^{(l+1)}=A F^{(l)}\operatorname{diag}(a),\quad
Z_{:,j}=\frac{1-a_j}{1-a_j^{L+1}}\sum_{l=0}^{L}F^{(l)}_{:,j}.
\]

Giữ score `s(u,i)=Z_u^T Z_i` và BPR của baseline; không đưa phase scorer vào inference trong phiên bản chính. Full/pool ranking, seen-item masking, validation NDCG@20 selection và early-stop semantics kế thừa cùng FreeRec runtime. Không thêm `reg_loss` lên Z như A vì baseline dùng optimizer weight decay; thêm term này gây nhiễu đối chứng.

### 3.3. BSC nằm sau Adam moments

`optimizers/AdamW.py` tạo hướng Adam đã bias-correct và chia second moment:

\[
\Delta_t=\widehat m_t/(\sqrt{\widehat v_t}+\epsilon),
\]

sau đó mới gọi `smoother(Delta_t)` cho item group. BSC không chỉ tác động raw BPR gradient; gradient tạo moments bao gồm mọi loss đang bật. Vì vậy v2 phải thay callback smoother đúng vị trí, không dùng hook lên `.grad` rồi gọi Adam lần nữa.

Baseline BSC cho cột j:

\[
P_{b_j,L}(S)\Delta_{:,j}
=\frac{\sum_{l=0}^{L}b_j^l S^l\Delta_{:,j}}{\sum_{l=0}^{L}b_j^l}.
\]

Đây là polynomial Neumann hữu hạn, không phải một lần nhân Laplacian. Gamma, batch size, decay phải kế thừa YAML riêng Baby/Sports/Electronics.

## 4. Audit toán học của đề xuất A và bản phản biện B

### 4.1. Real projection phá bảo toàn norm

Với H thực đối xứng, g thực:

\[
\Re[(I-itH)g]=g,\qquad
\Re[e^{-itH}g]=\cos(tH)g.
\]

Do đó phương trình Euler trong A sai ngay cả trước vấn đề hệ số Taylor. `g−t²H²g` không bằng phần thực của một bước Euler. Taylor cosine đúng là `g−(t²/2)H²g+O(t⁴)`.

Unitary evolution giữ norm **trạng thái phức đầy đủ**, còn phần thực có thể mất norm. Với eigenvalue lambda, norm gain là `|cos(t lambda)|`. Phần bậc hai có gain `|1−t² lambda²/2|`, không luôn ≤1 khi t hoặc phổ H không bị chặn. Vì thế câu “luôn nhỏ hơn 1” trong B chỉ đúng trong miền tham số thích hợp.

Ví dụ H=diag(0,1), g=(0,1), t=0.15: norm input và unitary đầy đủ đều 1; real projection ≈0.988771; công thức A bằng 0.9775; Taylor đúng bằng 0.98875. Không có bảo toàn 100%.

### 4.2. Adjacency không phải Laplacian

Nếu H=S có phổ trong [−1,1], `1−c lambda²` giảm cả mode lambda≈+1 và lambda≈−1, ưu tiên mode gần 0. Nó không phải low-pass tiêu chuẩn theo graph Laplacian frequency `1−lambda`. Chỉ thay hệ số 1/2 như B khuyến nghị chưa sửa vấn đề này.

Một real symmetric normalized adjacency có thể dùng làm Hamiltonian toán học, nhưng không đồng nhất với spin Hamiltonian bằng Pauli trên nhiều qubit. Không có mapping subsystem/circuit thì không được chuyển chứng minh của QGHNN sang code A.

### 4.3. Hai similarity khác nhau hoàn toàn

Với z_k=exp(i phi_k), mọi |z_k|=1. Biểu thức trong A:

\[
\sum_k |\overline z_{u,k}z'_{i,k}|^2=\sum_k|z'_{i,k}|^2
\]

không phụ thuộc user; nếu lấy toàn bộ d chiều và rotation unitary thì nó là hằng d. Lấy d/2 chiều vẫn không khôi phục query dependence. Đây là lỗi mất thông tin pha do lấy modulus trước khi tổng.

Coherent normalized fidelity phải là:

\[
K(x,y)=|\psi(x)^\dagger G\psi(y)|^2,\quad
\psi_k(x)=e^{i\phi_k(x)}/\sqrt d.
\]

Code A dùng `abs(sum)/d`, tức amplitude, không phải squared fidelity. Cần chọn một định nghĩa thống nhất. Báo cáo chọn squared fidelity để tránh đạo hàm modulus tại 0 và bảo đảm score trong [0,1] khi G unitary.

B đúng rằng split cùng input làm mất cross-layer distinction, nhưng nhận định “luôn similarity=1, không còn tín hiệu học” quá mạnh: các cặp vẫn là user–item, không nhất thiết là một vector, và G có thể khác identity. Phải phân biệt lỗi layer với sự suy biến thật của coordinatewise overlap.

### 4.4. Givens không tự chứng minh entanglement hoặc lợi thế lượng tử

Givens thực 2×2 trên các cặp tọa độ là phép trộn tuyến tính bảo toàn norm. Không có tensor-product subsystem và kiểm tra separability thì chưa có cơ sở gọi nó CNOT hoặc entangler. Cũng không nên khẳng định mọi Givens đều không bao giờ tạo entanglement dưới bất kỳ encoding nào; kết luận ở đây giới hạn cho mapping chưa được đặc tả trong A.

Một vector d complex có 2d thành phần thực để lưu, không cung cấp miễn phí không gian 2^d. Inner product học được không giả định độc lập xác suất giữa user và item; nó chỉ giới hạn dạng hàm score. Squared fidelity vẫn có classical feature representation `Tr(rho_x G rho_y G†)` với `rho_x=psi_x psi_x†`; không chứng minh matching không thể tách bằng đặc trưng riêng.

Nếu áp **cùng** unitary G lên cả hai nhánh, `(G psi_x)†(G psi_y)=psi_x†psi_y`: tham số rotation triệt tiêu khỏi fidelity. V2 áp rotation trên branch layer-1, giữ layer-0 không xoay.

### 4.5. Gradient conflict vẫn tồn tại

`g_total=g_BPR+lambda*g_CL`; Adam moments và BSC nhận ảnh hưởng cả hai. POCL không tối ưu một không gian hoàn toàn tách khỏi BPR vì gradients quay về cùng embeddings. Việc hai module chạy ở hai giai đoạn không chứng minh gradient conflict bằng 0. Phải đo cosine giữa hai gradients và giữa hướng cập nhật cuối với gradient BPR, đồng thời xét decay riêng.

## 5. Audit code đính kèm

| Mức | Vị trí trong A | Hậu quả | Sửa tối thiểu |
|---|---|---|---|
| Critical | `torch.split(layer_embeds, ...)` | Input thực tế là list; kiểm tra local ném AttributeError | Dùng tensor layer 0 và layer 1 riêng, kiểm tra shape |
| Critical | `masked_fill(mask, matrix)` | Value tensor 2D không được API chấp nhận | Dùng CE trực tiếp từ logits; không cần ghi lại đường chéo |
| Critical | `Adj=None`, không có prepare; QISF chỉ được định nghĩa | encode không chạy nếu chưa gán graph; smoother chưa nối optimizer | Tái dùng baseline graph preparation, attach đúng item group |
| Major | Hai lần split cùng một biến | Mất ý nghĩa layer 0 → layer 1 | Test bằng graph mà hai layers khác nhau |
| Major | `[B,1,d]*[1,B,d]` | Materialize `[B,B,d]` complex | Dùng complex GEMM hoặc hai-kênh thực |
| Major | Xavier, thêm L2, `nn.Module` riêng | Không phục hồi baseline khi lambda=0 | Giữ FreeRec interface, MI, BPR, optimizer và evaluation |
| Major | `warmup_epochs=50` nhưng lambda tăng từ epoch đầu | Là ramp, không phải 50 epoch CL-off | Đặc tả warmup và ramp riêng |
| Major | Nhiều lần xuất hiện user/item trong batch | Diagonal-only positives tạo false negatives | Unique candidates và multi-positive train mask |
| Minor | `eps_noise`, `beta` không dùng; angle atan/tanh không thống nhất | Tài liệu không khớp thực thi | Loại tham số chết, chọn duy nhất một phase map |

## 6. Kiến trúc v2 đã hiệu chỉnh

```text
Text/Visual + R_train
        │
        ├── baseline MI ── E_user, E_item ── baseline FSC ── Z ── BPR
        │                        │
        │                        └── X0, X1 ── phase map / Givens ── POCL
        │
        └── baseline S ── H=(I−S)/2 ── BSF callback (fixed operator)

BPR + lambda(t)*POCL ── backward ── Adam moments
                                       ├── users: ordinary AdamWSEvo
                                       ├── items: blended BSC/BSF direction
                                       └── auxiliary angles: separate unsmoothed group
Inference: baseline Z-user · Z-item; no quantum head or pairwise circuit.
```

### 6.1. Trụ cột A — bounded spectral filter (BSF)

Giữ S của baseline, có symmetry, nonnegative edge weights và spectral norm ≤1. Đặt normalized Laplacian đã scale:

\[
H=(I-S)/2,\quad \sigma(H)\subseteq[0,1],\quad
Q_t(H)=I-\frac{t^2}{2}H^2,\quad 0\le t\le1.
\]

Nếu các điều kiện trên giữ, eigenvalues của Q nằm trong `[1−t²/2,1]`, nên Q SPD, contraction, không đổi dấu mode. Với mọi X:

\[
(1-t^2/2)\|X\|_F\le\|Q_tX\|_F\le\|X\|_F.
\]

Đây là **giới hạn suy hao mỗi lần áp dụng**, không bảo toàn norm và không chống oversmoothing tuyệt đối khi lặp nhiều lần. Taylor error đối với cosine bị chặn bởi `t⁴/24` trong operator norm khi phổ H⊂[0,1].

Không dựng I, H hay H² dạng dense. Tính:

```python
def apply_H(x):
    return 0.5 * (x - sparse_mm(S, x))

def apply_Q(x, spectral_time):
    return x - 0.5 * spectral_time**2 * apply_H(apply_H(x))
```

Để khôi phục baseline chính xác, trộn **output hai smoother**, không thay S bằng Q rồi vô tình lồng thêm Neumann:

\[
T_{j,t}=(1-\zeta)P_{b_j,L}(S)+\zeta Q_t((I-S)/2).
\]

Vì hai toán tử là polynomial cùng S nên dùng chung eigenbasis. Với b_j∈(0,1), polynomial P dương trên [−1,1], không vượt 1; T cũng SPD và không khuếch đại Frobenius norm. Điều này không chứng minh convergence của AdamW: `g^T T Delta` không mặc nhiên dương vì Delta là hướng adaptive có momentum, không phải g.

Baseline recovery là `zeta=0`, **không phải `spectral_time=0`**: khi t=0, Q=I và nhánh đó trở thành identity-mix. Không học t/zeta trong phiên bản chính; detached callback không tạo gradient hữu ích cho các scalar này nếu chỉ khai báo `nn.Parameter`.

### 6.2. Trụ cột B — phase-overlap contrastive learning (POCL)

Lấy auxiliary views `X0=E^(0)` và `X1=A E^(0)` trước beta attenuation. Có thể lấy X1 từ FSC step đầu trước nhân a để không thêm graph multiplication. Đây là lựa chọn rõ ràng cho auxiliary view; không thay công thức Z. Ablation riêng so với view đã nhân a nếu cần đối chiếu NLGCL cũ.

Per-row normalization để phase không phụ thuộc tùy tiện scale embedding:

\[
\bar x=\operatorname{LayerNorm}_{affine=False,\epsilon=10^{-5}}(x),\quad
\phi(x)=\arctan(s\bar x),\quad
\psi(x)=d^{-1/2}(\cos\phi+i\sin\phi),\ s=1.
\]

s cố định trong pilot. Norm của psi bằng 1; LayerNorm dùng chung quy tắc, không fit statistics từ validation/test. Với d chẵn, G(theta) là block-diagonal gồm d/2 Givens rotation; theta khởi tạo 0. Cùng G dùng cho target layer-1 ở hai hướng, không áp vào cả query và target.

Cho B_u unique users và B_i unique positive item IDs từ minibatch:

\[
K^{u\to i}=|\Psi(U_0)^*\,[G\Psi(I_1)]^T|^2,\qquad
K^{i\to u}=|\Psi(I_0)^*\,[G\Psi(U_1)]^T|^2.
\]

Trong ký hiệu trên G tác động lên từng vector cột; implementation lưu row vectors phải dùng transpose phù hợp (`z @ G.T`). Hai score matrices kích thước B_u×B_i và B_i×B_u. Dùng `conj(query) @ target.T`, sau đó `real²+imag²`; không tạo tensor `[B_u,B_i,d]`.

**Multi-positive supervision:** tạo M_ui=1 nếu cặp (u,i) thuộc R_train và cả hai ID có trong batch. Không dùng valid/test để mask. Mỗi row/column có ít nhất một positive từ batch gốc; kiểm tra invariant này. Target distributions:

\[
Y^{u\to i}_{ui}=M_{ui}/\sum_jM_{uj},\qquad
Y^{i\to u}_{iu}=M_{ui}/\sum_vM_{vi}.
\]

\[
L_{POCL}=\tfrac12\,\operatorname{mean}_u[-\sum_iY_{ui}\log\operatorname{softmax}_i(K^{u\to i}/\tau_c)]
+\tfrac12\,\operatorname{mean}_i[-\sum_uY_{iu}\log\operatorname{softmax}_u(K^{i\to u}/\tau_c)].
\]

Đây là supervised in-batch contrastive CE, không tự động là mutual-information bound của InfoNCE với i.i.d. negatives. Unique-ID reduction thay trọng số sampling so với raw interaction batch; mọi cosine control phải dùng **cùng reduction/mask** để cô lập kernel. Không xem các item chưa quan sát là negative chắc chắn; exposure bias vẫn còn.

Một row chỉ có một candidate cho loss 0. Nếu toàn bộ candidates đều là positives, CE uniform vẫn là bài toán cân bằng phân phối, không phải signal tách positive/negative; log tỷ lệ row này để phát hiện batch kém thông tin.

### 6.3. Khả năng suy biến của phase head

- Không chuẩn hóa scale: embeddings nhỏ làm mọi phase gần 0 và mọi fidelity gần 1; arctan bão hòa khi magnitude lớn. LayerNorm giảm nhưng không loại hết nguy cơ.
- Global phase không quan sát được qua fidelity; nhiều biểu diễn khác nhau có thể cùng score. Phải chấp nhận đây là invariance có thể mất thông tin.
- Với 0≤K≤1, temperature 0.2 giới hạn logit range ở 5. Nếu có B candidates và một positive, CE tối thiểu ngay cả trong trường hợp lý tưởng là `log(1+(B−1)exp(−1/tau_c))`. Với B=4096 và tau_c=0.2, khoảng 3.353; CL lớn không tự động là bug.
- Learned Givens chỉ d/2 tham số nhưng không đảm bảo chúng có gradient tốt. Kiểm tra nhiều minibatch; không suy luận từ một loss value.
- Fidelity có thể giúp auxiliary geometry nhưng inference vẫn inner product. Nếu muốn thay scorer, phải đặt thành nghiên cứu khác và benchmark inference lại.

### 6.4. Loss, optimizer và lịch trình

\[
L=L_{BPR}+\lambda(e)L_{POCL},\quad
r(e)=\operatorname{clip}((e-W)/R,0,1),\quad
\lambda(e)=\lambda_*r(e),\quad\zeta(e)=\zeta_*r(e).
\]

e là epoch **1-based** theo FreeRec. Pilot W=10, R=20: epochs 1–10 không auxiliary/mixing; epoch 11 đạt 1/20 target; epoch 30 đạt target. G không được gọi khi lambda=0 để tránh overhead và RNG side effects.

| Group | lr | Weight decay | Smoother |
|---|---|---|---|
| User embeddings | baseline | baseline | None |
| Item embeddings | baseline | baseline | T ở trên |
| Givens theta | 0.1×baseline | 0 | None |

Mỗi parameter thuộc đúng một group. Fixed LayerNorm, s, t, zeta không thuộc optimizer. Loss graph phải tạo mới mỗi step; lưu diagnostics bằng detached scalar. Trong kiến trúc này S/H là static và không có learned edge gates; snapshot chỉ cần fixed coefficients trước step, không tái sử dụng graph có `grad_fn`. Không nhập thêm MHD gates hoặc hyperedges vào v2.

### 6.5. Pilot config — giả định để thử, không phải giá trị tối ưu

Cờ `enable_pocl=false` buộc lambda hiệu dụng bằng 0; `enable_bsf=false` buộc zeta hiệu dụng bằng 0, bất kể target trong config. Nếu cả hai tắt thì chạy baseline recovery.

```yaml
base_config: Amazon2014Baby_550_MMRec.yaml  # thay đúng parent cho dataset khác
enable_pocl: true
enable_bsf: false          # thử POCL riêng trước khi ghép
pocl_weight_target: 0.001
contrastive_temperature: 0.2
phase_scale: 1.0
phase_normalization: layernorm_no_affine
rotation_init: zero
positive_policy: unique_ids_train_multi_positive
spectral_hamiltonian: half_normalized_laplacian
spectral_time: 0.5
spectral_mix_target: 0.1
warmup_epochs: 10
ramp_epochs: 20
aux_lr_ratio: 0.1
aux_weight_decay: 0.0
```

Đây là schema đề xuất, **chưa phải CLI có thể chạy**. Với t=0.5, Q có gain tối thiểu 0.875; với t=0.15 chỉ có gain tối thiểu 0.98875, hiệu ứng có thể rất nhỏ. Không dùng t=0.15 vì được mô tả là “quantum timestep” trong A mà không đo sensitivity. Pilot tuning ưu tiên lambda∈{0.0001,0.001,0.01}, tau_c∈{0.05,0.1,0.2}; BSF zeta∈{0.1,0.25}, t∈{0.15,0.5,1}. Không chạy toàn tích Descartes: stagewise, cùng giới hạn số trial cho controls và lưu mọi trial.

## 7. Giao diện triển khai và pseudocode

Các file sau là **kế hoạch triển khai**, không phải artifact đã tạo trong lượt này:

Bảng này là bản tóm tắt. Danh mục file, API, schema và tiêu chí nghiệm thu có tính quy định cho lần triển khai sau nằm ở §13–17.

| File mới dự kiến | Trách nhiệm |
|---|---|
| `models/stair4_v2.py` | FreeRec GenRecArch, MI/FSC/scorer baseline, exposes X0/X1, auxiliary loss |
| `models/stair4_v2_heads.py` | Phase encoder, Givens, efficient coherent kernel, multi-positive loss |
| `optimizers/stair4_v2_smoother.py` | P_baseline/Q blending, no learned state; operates on Adam direction |
| `main_stair4_v2.py` | Three groups, config inheritance, seed/checkpoint lifecycle, baseline evaluation |
| `configs/dataset_stair4_v2_*.yaml` | Parent baseline theo dataset, explicit overrides |
| `tests/test_stair4_v2.py` | Algebra/autograd/baseline recovery/checkpoint/integration |

```python
prepare_static_train_graphs_and_baseline_MI()
initialize_auxiliary_without_changing_baseline_rng_stream()
for epoch in one_based_epochs:
    lambda_cl, zeta = schedule(epoch)
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        Z, X0, X1 = encode_with_baseline_FSC()
        loss = baseline_BPR(Z, batch)
        if lambda_cl > 0:
            U, I, M = unique_ids_and_train_positive_mask(batch)
            loss = loss + lambda_cl * pocl(X0, X1, U, I, M)
        item_smoother.arm(zeta=zeta, spectral_time=fixed_t)
        try:
            loss.backward()
            optimizer.step()  # callback receives Adam direction, not raw gradient
        finally:
            item_smoother.clear_step_snapshot()
    evaluate_and_select_checkpoint_using_baseline_protocol()
```

`prepare()` không cache auxiliary autograd states. Checkpoint cần embeddings, static graph/data hashes hoặc khả năng tái tạo xác minh được, theta, optimizer moments, epoch/schedule, RNG, config, FreeRec monitor histories. Tránh serialize callback bound vào model. Chuẩn hóa monitor mappings thành plain dict trước lưu để tránh lỗi `defaultdict`/`weights_only=True` đã xuất hiện trên Kaggle Torch 2.10; giữ schema có version và test resume trên runtime thực. Không vá `torch.load` toàn cục hoặc thay đổi baseline source.

## 8. Chi phí tính toán và giới hạn phần cứng

| Thành phần thêm | Chi phí chính | Bộ nhớ cần chú ý |
|---|---|---|
| Phase map/Givens | O((B_u+B_i)d) | Complex64 tốn 8 bytes/phần tử, không miễn phí |
| Coherent GEMM hai hướng | O(B_u B_i d) | Output complex và logits/mask O(B_u B_i) |
| Q áp lên item direction | 2 sparse S×X | O(nnz(S)d); vài tensor I×d |
| Mixed smoother | baseline L SpMM + 2 SpMM nếu tính riêng | Thường thêm overhead; không khẳng định nhanh hơn baseline |
| Mask positives | Tra cứu train edges chỉ trong candidates | Không dựng full U×I dense |

Với B=4096,d=64, tensor broadcast `[B,B,d]` complex64 của A **riêng nó** cần 8 GiB. GEMM output `[B,B]` complex64 là 128 MiB; tổng training peak còn gồm autograd, logits, optimizer, graph, embeddings và evaluation. Vì vậy bảng “185 MB” trong A không có cơ sở.

Nếu GEMM full batch vẫn không vừa T4: chunk query rows, giữ toàn candidate set và tính tổng loss có trọng số row chính xác. Để giảm backward memory thật, dùng checkpoint/recompute hoặc custom chunked backward; chỉ nối nhiều loss chunks trong cùng graph có thể vẫn giữ toàn bộ intermediates. Không âm thầm giảm negatives/batch vì sẽ đổi objective và baseline comparison.

Đo riêng prepare time, training steady-state, evaluation và end-to-end; CUDA synchronize tại ranh giới timing. Log allocated/reserved peak đúng device; nếu muốn per-epoch peak phải reset mỗi epoch và phân biệt với cumulative peak. Không lấy NVML trừ hằng số làm tensor memory. Không gán “quantum speedup” cho code classical GPU.

## 9. Kế hoạch thực nghiệm và tiêu chuẩn bác bỏ

### 9.1. Ma trận ablation bắt buộc

| ID | Auxiliary | Smoother | Mục đích |
|---|---|---|---|
| A0 | Off | Baseline BSC | Exact recovery và baseline chính |
| A1 | Cosine cross-layer, cùng mask/normalization | Baseline BSC | Đối chứng auxiliary cổ điển |
| A2 | Phase fidelity, G=I | Baseline BSC | Tác động phase map/kernel |
| A3 | Phase fidelity, learned G | Baseline BSC | Tác động rotation |
| A4 | Off | `(1−zeta)P+zeta I` | Giảm smoothing thuần |
| A5 | Off | `(1−zeta)P+zeta Q` | BSF riêng |
| A6 | A3 | BSF mix | Full v2 |
| A7 | A1 | BSF mix | Phase có hơn cosine khi cùng smoothing? |

A1 bổ sung real Givens target-only với cùng d/2 tham số là đối chứng capacity nếu A3 có lợi. Giữ cùng normalized input và mask; một NLGCL lịch sử dùng reduction khác phải được ghi thành đối chứng riêng, không coi tương đương A1.

### 9.2. Protocol

1. Unit/integration trước; CPU smoke và một GPU smoke sau warmup. “16 test v3 passed” không chứng nhận v2.
2. Pilot Baby/Sports để kiểm tra numerics, throughput, gradient và chọn hyperparameters bằng validation; Electronics là scale test sau khi phù hợp tài nguyên, không giả định khả năng scale từ VQRS.
3. Evaluation cùng ranking mode, split, seen masking, K và validation NDCG@20 checkpoint selection. Dùng test ở checkpoint đã chọn; không chọn biến thể/hyperparameters bằng test.
4. Final chạy ít nhất 5 seeds ghép cặp giữa các phương pháp nếu ngân sách cho phép; lưu mean, SD và từng run. Ba seed chỉ được gọi pilot.
5. Báo chênh lệch NDCG@20 tuyệt đối và tương đối; CI theo paired seeds, nêu CI ít seed không ổn định. User bootstrap đo uncertainty có điều kiện trên model đã train, không thay thế training-seed uncertainty. Hiệu chỉnh multiple comparisons nếu đưa nhiều kiểm định xác nhận; exploratory phải gắn nhãn.
6. Cùng budget tuning; báo cả matched epochs và wall time. Không mặc định cùng epoch đồng nghĩa cùng chi phí.
7. Dataset/feature/train hashes, exact Git SHA, package lock, seed, config resolved, trial selection và failed runs phải lưu. Không `git reset --hard origin/main` trong lúc chạy nhiều seed.

### 9.3. Diagnostics để phân biệt cơ chế

- Fidelity quantiles, off-diagonal variance, positive/negative margins; entropy softmax; theta gradients qua nhiều bước.
- Norm và cosine của `g_BPR`, `g_POCL`, gradient ratio sau nhân lambda; log trước/ sau warmup.
- `||T Delta||/||Delta||`, alignment với Delta và với g_BPR; band energies bằng Rayleigh/Dirichlet probes, không full eigendecomposition catalog.
- Cosine/CKA drift so với MI, effective rank của embeddings; tail/head NDCG theo train item degree. Đây là proxy, không tự chứng minh semantic preservation.
- Positive multiplicity, duplicate rates, rows all-positive; sự thay đổi objective khi deduplicate.
- Time/memory theo phase; failures, NaN, skipped steps. Không thay missing values bằng đường cong giả hoặc target metrics.

Nếu A5 không hơn A4 thì không kết luận spectral shape tạo ích lợi. Nếu A3 không hơn A1 với matched capacity thì không kết luận phase có ưu thế. Nếu A6 kém A3/A5, bác bỏ giả thuyết phối hợp có lợi, không đổi tên thất bại thành “synergy”. Nếu gain chỉ ở một seed hoặc chi phí vượt đáng kể, báo trade-off thay vì SOTA.

## 10. Kiểm tra đã thực hiện và test cần bổ sung

Đã thực hiện tại máy local: Python từ `.venv-mhd-v3`, Torch 2.5.1 CPU, trích class trực tiếp từ code block A vào namespace tạm; không sửa hoặc chạy training model v2.

| Kiểm tra nhỏ | Quan sát |
|---|---|
| Gọi loss A với list hai tensor layers | `AttributeError: 'list' object has no attribute 'split'` |
| `masked_fill` với value tensor 2D | `RuntimeError: ... only supports a 0-dimensional value tensor` |
| H=diag(0,1), g=(0,1), t=.15 | Norms: 1; unitary 1; real 0.988771078; A 0.9775; Taylor cosine 0.98875 |
| Phase-coordinate modulus rồi sum | Similarity matrix 2×2 toàn 1 trong fixture không rotation |
| Coherent squared overlap | Fixture cho các giá trị khác nhau, ví dụ 1 và 0.29192658 |
| Shape-memory arithmetic | Broadcast B4096,d64 complex64: 8 GiB; GEMM output: 128 MiB |

Tests bắt buộc khi triển khai:

1. G†G≈I; psi norm≈1; complex và real/imag kernels khớp, fidelity trong [0,1].
2. Gradcheck double precision cho inputs/theta; gradients hữu hạn gần overlap=0; xác minh rotation cùng hai branch triệt tiêu như dự đoán.
3. H/Q sparse vs dense trên toy graphs; spectrum, contraction bound và t=0 identity; disconnected nodes/zero degrees không NaN.
4. `zeta=0,lambda=0`: MI, loss, gradients, optimizer update, full/pool scores khớp baseline; auxiliary init không đổi RNG stream.
5. Đúng X0/X1, index spaces U/I; duplicate invariance của auxiliary; batch singleton/multi-positive chính xác.
6. Givens theta chỉ trong aux group; smoother nhận Adam direction; không double smoothing; callback cleanup khi step lỗi.
7. Save/load/resume giữ theta/moments/schedule/RNG/hashes; CPU runtime và Kaggle Torch 2.10 phải kiểm tra riêng.
8. Chunked loss và gradients khớp full GEMM trên fixture; đo memory thực để xác nhận chunking có hiệu quả.

Không có kết quả Baby/Sports/Electronics v2 trong báo cáo này. Các giá trị test đại số không được đưa vào bảng recommendation benchmark.

## 11. Phán quyết methodology-focus và roadmap

| Vai trò thực thi | Đánh giá riêng đã nhận | Giới hạn |
|---|---|---|
| Methodology reviewer | Critical: real projection không unitary, coordinatewise overlap mất query dependence; cần đúng vị trí Adam direction, baseline recovery và matched controls | Đọc A và baseline; không xác minh nguồn hay benchmark |
| Journal-Fit / Originality reviewer | Major Revision: ý tưởng có thể thử, nhưng nhãn đóng góp vượt cơ chế và bằng chứng hiện có | Không đánh giá venue cụ thể; novelty chưa xác minh đầy đủ |

Hai lượt cùng họ mô hình và có phân tách vai trò/đầu vào; không diễn giải sự đồng thuận thành xác suất đúng hoặc chứng nhận độc lập thống kê.

**Phán quyết cho A: cần sửa lớn trước triển khai.** Ý tưởng dùng phase kernel có thể kiểm định, nhưng justification norm-preserving/entangled/SOTA hiện không đứng vững. B cũng cần chỉnh: bài VQRS có thật; lỗi layers không đồng nghĩa mọi user–item similarity bằng 1; Taylor contraction cần miền tham số; sửa 1/2 không tự biến adjacency² thành low-pass hợp lý.

**Review provenance:** một lượt methodology riêng được giao đọc A và baseline, không đọc B hoặc peer output; một lượt Journal-Fit riêng đọc A, không có venue đích. Orchestrator làm source verification, kiểm tra số và thiết kế hiệu chỉnh. Đây là các vai trò cùng họ mô hình, không phải các reviewer con người độc lập, không có calibration hoặc đánh giá acceptance của hội nghị thật. Các quyết định thiết kế ở §6 là đề xuất của bước tổng hợp, không gán cho nguồn trích dẫn.

| Vấn đề | Mức | Remedy tối thiểu | Bằng chứng đóng issue |
|---|---|---|---|
| Norm/overlap/quantum claims sai | Critical | Dùng §4–6, bỏ lợi ích tuyệt đối | Proof + toy operator/kernel tests |
| Code chưa nối baseline | Critical | Giữ MI/FSC/scorer và optimizer callback | Exact recovery test |
| Chi phí broadcast và scale chưa biết | Major | GEMM, chunking khi cần | Peak/time đo thật trên GPU |
| Không cô lập nguyên nhân gain | Major | A0–A7, paired seeds và matched budget | Run artifacts, ablation, uncertainty |
| Nguồn tham khảo/venue sai | Major | R1–R4, ghi rõ mức đọc | DOI/metadata và claim-source map ở §2 |

Thứ tự triển khai hợp lý: sửa contract baseline và tests → A0 → cosine/phase kernel riêng → BSF và identity-mix riêng → full v2 → benchmark nhiều seed. Nếu từng thành phần không có bằng chứng tốt hơn đối chứng đơn giản, dừng mở rộng nhánh đó. Không bắt buộc ghép cả hai để “đủ kiến trúc”.

## 12. Tài liệu tham khảo và giới hạn nguồn

- **[R1]** Xu, C., He, Y., Wang, J., & Zhang, W. (2025). *STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation*. AAAI, 39(12), 12899–12907. [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/33407), [code tác giả](https://github.com/yhhe2004/STAIR). Đối chiếu implementation bằng code local; không lấy số liệu từ abstract làm benchmark local.
- **[R2]** Li, A., & Casiraghi, E. (2026). *Quantum-enhanced Representation Learning and Matching Learning for Recommendation*. WWW 2026, 5722–5730. DOI: [10.1145/3774904.3792086](https://doi.org/10.1145/3774904.3792086). [Kho Aalto](https://research.aalto.fi/en/publications/quantum-enhanced-representation-learning-and-matching-learning-fo/). Mức đọc: metadata/abstract, chưa full text; không dùng để xác nhận chi tiết circuit của A.
- **[R3]** Debi, S., & Makmal, A. (2025). *Variational quantum recommendation system with embedded latent vectors*. Scientific Reports, 15, 32907. [10.1038/s41598-025-15869-x](https://www.nature.com/articles/s41598-025-15869-x). Mức đọc: nội dung phương pháp/kết quả trên publisher; không kiểm chứng lại số liệu bằng chạy code tác giả.
- **[R4]** Wang, W., Shi, J., Guan, N., Zhang, S., & Li, X. (2026). *QGHNN: A Quantum Graph Hamiltonian Neural Network*. IEEE Transactions on Emerging Topics in Computational Intelligence, 10(1), 830–843 (2026). DOI: [10.1109/TETCI.2025.3604791](https://doi.org/10.1109/TETCI.2025.3604791). [Journal metadata](https://scholars.cityu.edu.hk/en/publications/qghnn-a-quantum-graph-hamiltonian-neural-network/); [preprint của W. Wang, 2025](https://arxiv.org/abs/2501.07986). Mức đọc: journal metadata và preprint abstract; chưa full-text proof audit.

Các định lý về Q, overlap degeneracy và baseline recovery ở đây là suy dẫn trực tiếp có điều kiện từ công thức đã nêu, không được trình bày như theorem của R2–R4. Chưa thực hiện novelty search toàn diện; đóng góp cuối cùng của luận văn phải được xác định sau khi có đối chứng và kết quả thật.

## 13. Audit tính thống nhất trước khi lập kế hoạch mã nguồn

### 13.1. Hai tài liệu không mô tả cùng một phiên bản

`STAIR4_v2_Report.pre_.md` có tiêu đề/nội dung **STAIR-CNLGCL v1-R**, gồm SSB reweight, AMM, FNF và các cấu hình thực nghiệm lịch sử. Tên file có `v2` không làm các thành phần này trở thành BSF–POCL v2. Các khẳng định đã nghiệm thu, threshold NDCG, VRAM và thời gian trong bản đó không được kế thừa làm kết quả của v2.

Nguồn đặc tả hiện hành là báo cáo BSF–POCL này. Trong lần triển khai tương lai, không import model CNLGCL v1-R để rồi vô tình mang theo noise, extra L2, graph reweight, false-negative semantic masking hoặc adaptive margin. Bản `.pre_.md` được giữ nguyên để truy vết, không sửa và không thực thi code block bên trong.

### 13.2. Kết luận audit và các quyết định cần khóa

| Hạng mục | Đánh giá | Quyết định cho implementation |
|---|---|---|
| BSF | Đúng dưới giả thiết S đối xứng, phổ trong [−1,1], t∈[0,1], zeta∈[0,1] | Kiểm tra cấu trúc graph; không chỉ assert kết quả vài power iterations rồi coi là chứng minh |
| Neumann BSC | Hợp baseline khi giữ beta, L, normalization và thứ tự phép tính | Fast path zeta=0 gọi đúng baseline smoother; không dùng công thức tương đương số học khác cho recovery |
| POCL | Coherent squared overlap nhất quán; hiệu quả thực nghiệm chưa biết | GEMM, normalization 1/sqrt(d) một lần; không chia d² lần nữa |
| Multi-positive | Có định nghĩa target rõ, khác diagonal InfoNCE lịch sử | Unique ID và train mask dùng chung cho mọi auxiliary control |
| Cross-layer view | X1 được chọn trước beta attenuation | Giữ tensor X1 trước phép nhân, không lấy F1 chia ngược cho a vì có thể gặp zero/underflow |
| Auxiliary configuration | Hai boolean pilot chưa đủ biểu diễn A0–A7 | Thêm selector và ánh xạ explicit tại §14.6; reject tổ hợp mâu thuẫn |
| Sự khác nhau giữa hai kernel | Cosine nhận [−1,1], fidelity nhận [0,1] | So sánh có temperature tuning cùng số trial, không mặc định cùng tau là cùng logit scale |
| Checkpoint | Thiết kế phải giữ state FreeRec và callback isolation | Chuẩn hóa plain containers trước save, một schema version, atomic write |
| Baseline recovery | Có thể kiểm tra trên cùng fixture và runtime | Không hứa bitwise match khác OS/GPU/SVD backend hoặc whole-run sampling schedule |

**Điều kiện toán học/code bổ sung:** yêu cầu gamma>0, d≥2 và chẵn cho v2, L≥1 nếu auxiliary bật. Với gamma>0 và hữu hạn d, beta_j<1; gamma=0 có thể làm beta=1 và biểu thức chuẩn hóa dạng 0/0, nên không được chấp nhận âm thầm. Baseline-only có thể hỗ trợ L=0 bằng nhánh riêng đã test, nhưng pilot v2 dùng L≥1. d=2 hợp shape nhưng LayerNorm dễ làm phase features suy biến; không dùng d=2 như bằng chứng trainability của cấu hình d=64.

Nếu node item cô lập và baseline quy ước S-row=0 thì H trên node đó bằng I/2, Q gain là `1−t²/8`. Không tự thêm self-loop hoặc đổi H để giữ node này vì sẽ đổi đặc tả; log số node cô lập. Normalized adjacency đối xứng không đồng nghĩa adjacency PSD: tính SPD trong báo cáo thuộc **polynomial smoother**, không thuộc S.

Các block Givens thao tác trực tiếp trên cặp chiều, không dựng ma trận dense d×d mỗi step. G đặt trên target layer-1 nên hai ma trận score hai hướng **không được giả định là transpose của nhau**. Không dùng một logits matrix rồi transpose để tiết kiệm compute trừ khi đã chứng minh đúng cho một control cụ thể.

Đánh giá tổng thể: **đủ cơ sở làm một kế hoạch kiểm nghiệm, chưa đủ cơ sở khẳng định một cải tiến đã hiệu quả**. Sự đúng đắn của algebra và interface không thay thế benchmark. Lần bổ sung này rà soát code local và tài liệu đã kiểm chứng ở §2; không tuyên bố đã thực hiện một vòng literature search mới.

## 14. Danh mục file dự kiến và nội dung cụ thể

Tất cả file trong phần này là **sẽ tạo sau khi triển khai được yêu cầu**, không phải đã tồn tại hoặc đã chạy. Luồng phụ thuộc: `config → utils/heads/smoother → model → engine → notebook/reporting`. Không tạo shim v2 trong module baseline.

### 14.1. `models/stair4_v2_utils.py` — static graph và batch supervision

**Nội dung:**

- `prepare_baseline_graphs(dataset, config) -> StaticGraphBundle`: xây A, S và MI theo baseline; giữ nguyên kNN, modality order/weights, coalesce và symmetrization. Reuse các hàm thuần đã audit hoặc chép logic có truy vết vào module mới; không import `main.py` vì parser/global cfg có side effects.
- `baseline_whitening(features, d)`: center + SVD đúng scaling; kiểm tra số rows/features đủ d, finite inputs. Không tự thay SVD bằng randomized/truncated algorithm trong phiên bản chuẩn.
- `build_train_positive_index(train_edges, n_users, n_items)`: unique train edges, lưu CSR integer index sorted theo user; không tạo RᵀR hoặc U×I dense. Index được dùng cho loss, không âm thầm thay weighting graph baseline khi dữ liệu có duplicate edges.
- `make_batch_candidates(users, positives) -> CandidateBatch`: flatten raw IDs, unique sorted theo từng namespace, trả IDs và mapping; negative items của BPR không tự thêm vào candidate set POCL.
- `build_positive_mask(user_ids, item_ids, index) -> Bool[B_u,B_i]`: tra cứu train-only. Bản tham chiếu dùng sorted pair keys `u*n_items+i` int64 và `searchsorted` có bounds checks; kiểm tra integer overflow. Có thể tối ưu bằng CSR sau khi chứng minh equivalence.
- `validate_static_bundle`, `hash_data_manifest`: shape, finite weights, ID bounds, symmetry; hash train split/feature content, modality order, preprocessing config và graph. Không hash chỉ tên đường dẫn.

**StaticGraphBundle:** A sparse `(U+I,U+I)` float32; S sparse `(I,I)` float32; MI item/user tensors hoặc checksum theo chế độ lưu; beta float32 `[d]`; train index; manifest. Buffer thuộc model và di chuyển theo device; CSR train index có thể ở CPU, mask batch chuyển sang device của logits. Không ép copy toàn bộ train index lên GPU khi chưa profile.

**Ràng buộc prepare:** chạy dưới `no_grad`, không giữ `grad_fn`; graph không đổi giữa epochs. Exact blocked kNN là tùy chọn giảm peak chuẩn bị, nhưng phải kiểm chứng tie handling/coalesce khớp baseline trên fixture; nếu có khác biệt floating-point hoặc ties phải ghi nhận thay vì gọi bitwise recovery. Không import các module MHD có học gates để dựng graph này.

### 14.2. `models/stair4_v2_heads.py` — phase, rotation và contrastive loss

**API dự kiến:**

| Hàm/class | Input → Output | Contract |
|---|---|---|
| `PhaseEncoder` | real `[B,d]` → complex `[B,d]` | LayerNorm no affine, atan(scale*x), norm 1; float32→complex64, float64→complex128 |
| `PairwiseGivens` | complex hoặc real `[B,d]` → cùng shape | theta real `[d/2]`, target-only, d chẵn; zero init là identity |
| `phase_fidelity(query, target)` | complex `[Bq,d]`, `[Bk,d]` → real `[Bq,Bk]` | `c=query.conj() @ target.T`; trả `c.real.square()+c.imag.square()` |
| `cosine_similarity` | normalized real query/target → real matrix | Cùng per-row LayerNorm input; sau đó L2-normalize; G nếu control yêu cầu |
| `multi_positive_ce(logits, mask)` | real matrix, bool cùng shape → scalar | `log_softmax`, target uniform trên positives; không masked_fill positives thành -inf |
| `CrossLayerContrastiveHead` | U0/I0/U1/I1, CandidateBatch → loss + detached diagnostics | Tính riêng hai hướng, rồi mean mỗi hướng và hệ số 1/2 |

Không tạo `Parameter` complex; theta luôn real. Backend chính complex64; real/imag backend phục vụ equivalence test và profiling, không được coi là kiến trúc khác. Không bọc toàn bộ head bằng `no_grad` hoặc detach X0/X1. Evaluation không gọi head.

**Edge cases:** empty batch reject; row/column không positive reject vì vi phạm candidate contract; singleton xử lý ổn định; all-positive rows vẫn dùng uniform CE và log tỷ lệ. Không bỏ row để giảm loss mà không đổi protocol. Không clamp fidelity rộng để che lỗi normalization; chỉ kiểm tra finite/range với tolerance, phân biệt roundoff nhỏ với lỗi thật. Dùng `real²+imag²` trực tiếp để gradient tại overlap=0 là xác định.

Raw loss không nhân lambda bên trong head; engine/model nhân đúng một lần. Các diagnostics bao gồm `cl_raw`, positive count, score quantiles, entropy và theta grad norm; không gọi `.item()` lặp theo từng edge.

### 14.3. `optimizers/stair4_v2_smoother.py` — callback trên Adam direction

**Class dự kiến:** `BSFDirectionSmoother(operator_S, baseline_smoother, spectral_time)`; đây là plain callable, không có trainable parameters.

- `arm_step(mode, zeta, spectral_time)`: validate và chụp immutable detached scalar coefficients trước optimizer step.
- `__call__(delta: Tensor[I,d]) -> Tensor[I,d]`: chạy dưới no-grad, kiểm tra device/dtype/shape.
- `clear_step_snapshot()`: disarm; gọi khi hoàn tất hoặc step lỗi. Call khi chưa arm phải báo lỗi, không tự dùng coefficient của step trước.
- `mode='baseline'` hoặc zeta=0: gọi baseline `Smoother` nguyên arithmetic, tránh nhân `(1−0)`/cộng tensor zero không cần thiết.
- `mode='identity_mix'`: `(1−zeta)*P(delta)+zeta*delta`.
- `mode='bsf_mix'`: `(1−zeta)*P(delta)+zeta*Q(delta)` với H=(I−S)/2; t=0 dùng identity fast path.

Callback giữ graph thông qua getter/operator từ model để không giữ tham chiếu stale sau `.to(device)`/load. Không lưu callback trong optimizer checkpoint, không recompute moments, không đụng weight decay và không gọi optimizer.step bên trong smoother.

**Không thay thứ tự optimizer:** decoupled decay và Adam moments vẫn do `AdamWSEvo` baseline xử lý, callback chỉ biến đổi Delta. Item group phải chứa đúng item table theo thứ tự catalog; nếu vô tình thêm theta hoặc user table, assert shape/group invariant trước step.

### 14.4. `models/stair4_v2.py` — FreeRec model adapter

**Class:** `STAIR4V2(GenRecArch)` với config truyền tường minh, không global cfg. Giữ `User`, `Item`, `INeg`, `IUnseen` fields như baseline.

- `__init__`: tạo embeddings đúng baseline, chuẩn bị static bundle/MI, tạo auxiliary head theo selector; khởi tạo auxiliary trong RNG context riêng với CUDA devices tường minh hoặc `devices=[]` nếu CPU.
- `prepare`: chỉ static states; kiểm tra modality row order khớp item mapping, train-only graph, không validation/test statistics.
- `encode(return_views=False)`: trả `(Z_u,Z_i)`; khi cần thêm views, trả object có `X0` và `X1` cùng graph autograd. Giữ thứ tự FSC operations của baseline. Chỉ giữ first-hop view, không lưu danh sách toàn bộ layers nếu không dùng.
- `fit(data)`: BPR nguyên batch có duplicates như baseline; chỉ auxiliary deduplicate. Trả scalar total loss; lưu diagnostic scalars, không lưu loss tensor trên model.
- `set_epoch(epoch)`/`effective_coefficients`: schedule và enable flags hợp nhất; không tăng epoch thêm lần nữa.
- `parameter_groups`: user/item/aux coverage checks; nếu rotation identity và không có aux params thì chỉ hai nonempty groups, không tạo dummy parameter. Dấu vết role/group names lưu trong manifest/checkpoint.
- `reset_ranking_buffers`, `recommend_from_full`, `recommend_from_pool`: giữ baseline scorer/semantics. Ranking buffers detach và được reset sau load/trước evaluation.
- `get_extra_state`/`set_extra_state` hoặc adapter tương đương: model schema, static hashes, resolved selectors; strict validation trước resume.

**Shape tại biên:** BPR user/positive thường `[B,1]`, negative `[B,K]`; việc flatten cho POCL không được sửa data dict mà BPR đang dùng. Global tensor X rows `[0,U)` là users, `[U,U+I)` là items; IDs trong CandidateBatch vẫn là local namespaces, không cộng offset hai lần.

### 14.5. `utils/stair4_v2_checkpoint.py` — checkpoint và run provenance

Tách module này để không chép workaround serialization rải rác trong notebook/engine. Rà soát root hiện tại chưa thấy `utils.py`/`utils/`; kế hoạch tạo thêm `utils/__init__.py` tối thiểu cùng file mới này, không sửa `optimizers/utils.py` baseline. Trước lần triển khai sau phải kiểm tra lại namespace vì repository có thể đã thay đổi.

**Nội dung dự kiến:**

- `to_plain_checkpoint_tree`: convert `defaultdict`/nested mappings thành plain dict; cho phép tensors/primitives/list/tuple cần thiết; reject arbitrary objects/callables thay vì stringify mất state.
- `checkpoint_optimizer_state` và `restore_optimizer_state`: loại smoother callbacks khỏi serialized groups; kiểm tra role/order/parameter signature; bind live callbacks sau load. Không deep-copy callback bound vào model.
- `capture_rng`/`restore_rng`: Python/NumPy/Torch CPU/CUDA, explicit device topology; CPU load phục vụ inference không đồng nghĩa exact CUDA training resume.
- `save_checkpoint_atomic(path, payload)`: write temp cùng directory rồi replace; không đọc lại một checkpoint có defaultdict để “sửa sau” vì loader có thể đã fail.
- `load_checkpoint_checked(path, expected_manifest)`: restricted load cho schema mới, validate trước mutate model/optimizer; hỗ trợ load weights-only inference riêng với full resume.

**Payload schema v1 dự kiến:** `schema_version`, `run_id`, `completed_epoch`, `next_epoch`, `global_step`, `model_state`, `optimizer_state`, `scheduler_state` nếu có, `monitor_state`, `selection_state`, `rng_state`, `resolved_config`, `data_manifest`, `runtime_manifest`. `selection_state` phải ánh xạ đủ best epoch/value và early-stop counters của FreeRec đã khóa phiên bản. Không giả định chỉ lưu model+optimizer là resume tương đương.

Checkpoint trước epoch e ghi `completed_epoch=e−1`, `next_epoch=e`; sau epoch e ghi `completed_epoch=e`, `next_epoch=e+1`. Adapter phải ánh xạ về vòng lặp FreeRec thực tế, không chỉnh checkpoint frequency/selection nhằm làm test pass. Best weights phục vụ evaluation khác latest training checkpoint phục vụ resume; metadata phân biệt hai loại.

Exact resume giai đoạn đầu chỉ cam kết **epoch boundary, CPU, num_workers=0, cùng runtime** sau test; mid-epoch/multiworker/CUDA là các cấp kiểm chứng riêng. Không fallback bỏ qua state lỗi; checkpoint cũ chưa hỗ trợ phải thông báo migration cần thiết.

### 14.6. `configs/dataset_stair4_v2_{baby,sports,electronics}.yaml` và schema config

Mỗi YAML kế thừa baseline đúng dataset; không dùng config Baby rồi chỉ override dataset name. Parser v2 resolve inheritance trước CLI overrides, reject unknown keys, cycle và cấu hình xung đột.

Tạo thêm **`stair4_v2_config.py`** ở root làm nguồn schema duy nhất: `STAIR4V2Options` (dataclass), `load_inherited_config`, `resolve_aliases`, `validate_options`, `effective_schedule`. Module này không import model hoặc chạy parser khi import. Engine dùng schema để đăng ký CLI; model nhận options đã validate; checkpoint lưu config resolved; notebook không tự tính defaults lần thứ hai. Baseline FreeRec keys và v2 keys được kiểm tra theo tập hợp hợp lệ, không chỉ reject mọi key ngoài dataclass v2.

**Selector mới bổ sung cho pilot ở §6.5:**

| Key | Giá trị | Default pilot |
|---|---|---|
| `auxiliary_kernel` | `none`, `cosine`, `phase_fidelity` | `phase_fidelity` |
| `rotation_mode` | `identity`, `learned_givens`, `frozen_random_givens` | `learned_givens` |
| `smoother_mode` | `baseline`, `identity_mix`, `bsf_mix` | `baseline` |
| `aux_view` | `raw_first_hop` (phiên bản chính) | `raw_first_hop` |
| `kernel_backend` | `complex`, `real_imag` (phase only) | `complex` |
| `contrastive_chunk_rows` | positive integer hoặc null | null, full GEMM reference |
| `diagnostic_interval_steps` | positive integer | 100, chi phí ghi vào manifest |

`enable_pocl` ở config cũ mang nghĩa **bật auxiliary cross-layer**, kể cả cosine control; tên này dễ gây hiểu nhầm nên parser chuyển sang `auxiliary_kernel` trong resolved config. Quy tắc: false→`none`; true cần selector không none. `enable_bsf=false` yêu cầu baseline; true cần selector `identity_mix` hoặc `bsf_mix`. Nếu selector và boolean trái nhau thì báo lỗi, không âm thầm ưu tiên một phía. YAML mới nên chỉ viết selectors, giữ booleans như compatibility aliases của schema kế hoạch cũ. Schedule không đổi: `none` buộc lambda=0, baseline buộc zeta=0.

Các validation khác: d chẵn; gamma>0; phase_scale>0 hữu hạn; tau_c>0 hữu hạn; t∈[0,1]; zeta_target∈[0,1]; lambda≥0; W≥0; ramp R>0; aux lr ratio=0.1 và decay=0 theo contract. Muốn thử ratio khác phải thành protocol mới có ghi chú, không coi baseline-equivalent. Boolean CLI dùng flag đúng parser, không truyền chuỗi `False` như positional value.

| Ablation | auxiliary_kernel | rotation_mode | smoother_mode |
|---|---|---|---|
| A0 | none | identity | baseline |
| A1 | cosine | identity | baseline |
| A2 | phase_fidelity | identity | baseline |
| A3 | phase_fidelity | learned_givens | baseline |
| A4 | none | identity | identity_mix |
| A5 | none | identity | bsf_mix |
| A6 | phase_fidelity | learned_givens | bsf_mix |
| A7 | cosine | identity | bsf_mix |

Cosine capacity control thêm learned Givens trên target; gọi A1-G/A7-G, không thay A1/A7 âm thầm. Fixed random rotation dùng auxiliary seed riêng và lưu buffer vào checkpoint. Theta frozen không thuộc optimizer group.

### 14.7. `main_stair4_v2.py` — training orchestration

**Nội dung:** config loader/validator; seed; dataset adapter; model creation; optimizer groups; `CoachForSTAIR4V2`; `training_step`; run manifest và resume integration.

Engine kế thừa FreeRec evaluation/check_best/seen masking. Chỉ override phần cần thiết: optimizer, train hook, checkpoint adapter và logging. Giữ sampling algorithm baseline; nếu dùng epoch seeding/canonical launcher order để resume thì ghi protocol và áp cùng schedule cho matched controls. Không nói whole-run bitwise giống entry point cũ khi đã thay RNG scheduling.

Vòng step: zero_grad → fresh forward → loss → arm fixed smoother state → backward → gradient diagnostics khi đến interval → optimizer.step → clear snapshot trong finally. Chỉ gọi step khi finite loss/grads; khi nonfinite thì fail run, không tự skip để báo thành công. Diagnostic `autograd.grad` nếu dùng phải retain/recompute đúng graph và không cộng gradients hai lần.

Log tách raw BPR, raw auxiliary, lambda-weighted auxiliary, total loss, zeta, spectral_time, grad norms và candidate counts. Mean theo epoch phải có mẫu số đúng: BPR/total theo raw batch counts; auxiliary còn ghi unique-user/item counts và reduction của nó; không diễn giải mọi mean là cùng trọng số. Zero-denominator diagnostic dùng null, không zero giả.

`run_manifest.json`: config resolved, command, Git SHA/dirty status, dataset hashes, versions, CUDA/device, seed, ablation ID, run status. Artifacts đặt trong run directory duy nhất gồm checkpoint, diagnostics và result summary. `metrics.json` phân biệt `valid_best`, `test_at_valid_best`, `best_epoch`, `checkpoint_hash`, `status=completed`; không lấy last TEST bất kỳ bằng regex làm official result.

### 14.8. `scripts/stair4_v2_preflight.py`, notebook và tài liệu thực nghiệm

- `scripts/stair4_v2_preflight.py`: subprocess import/interface check, resolved config inspection, dataset files/shapes/IDs, one-step CPU/GPU smoke tùy mode. Khi chạy GPU smoke phải qua ít nhất vài step sau warmup và thử save/resume; in import success không có nghĩa đã training thành công.
- `notebook/P4/stair4_v2.ipynb`: tạo sau engine; chỉ cài/kiểm tra runtime, chọn run config, gọi subprocess, hiển thị structured artifacts. Không chép model/loss/optimizer vào cell. Không tự xóa repo nếu git fetch lỗi; checkout commit cố định và lưu artifacts ngoài checkout.
- `scripts/stair4_v2_report.py`: đọc manifest/metrics/diagnostics theo run ID; xuất bảng/hình chỉ từ completed runs, missing→N/A. Muốn hình minh họa phải output riêng và watermark rõ. Distribution gate/phase phải lấy model state hoặc saved measurement thật, không chỉ đổi nhãn khi file checkpoint tồn tại.
- `requirements/stair4_v2-*.txt` hoặc lockfile runtime phù hợp: chỉ tạo sau khi tìm bộ phiên bản đã test; ghi Python, Torch/CUDA, FreeRec, torchdata, PyG. Không ghi `>=` và gọi đó là môi trường đã khóa. CPU và Kaggle CUDA có thể cần lock riêng.
- `docs/giai_doan_4/STAIR4_v2_Implementation_Verification.md`: lập **sau** kiểm thử; lưu command, runtime, test outcomes, limitations. Không tạo sẵn dòng “passed” trong kế hoạch.

## 15. Bộ kiểm thử dự kiến theo file và lỗi cần bắt

| Test file dự kiến | Kiểm tra có ý nghĩa | Điều kiện nghiệm thu |
|---|---|---|
| `tests/test_stair4_v2_graphs.py` | MI/S/A với fixture baseline, IDs/duplicates/isolated nodes, train-only mask, no grad_fn trong prepare | Graph/MI khớp baseline trong tolerance quy định; changing valid/test không đổi train index |
| `tests/test_stair4_v2_heads.py` | Norm, coherent fidelity vs broadcast nhỏ, real/imag equivalence, gradcheck, one-sided/shared rotation, multi-positive hand calculation | Forward và gradients khớp reference; không materialize B×B×d trên path production |
| `tests/test_stair4_v2_smoother.py` | H/Q/T dense reference, eigenvalue bounds, zeta0/t0, correct Adam-direction placement | Baseline recovery exact trên fixture cùng runtime; Q/T bounds với tolerance float |
| `tests/test_stair4_v2_model.py` | X1 trước beta, BPR duplicates giữ nguyên, aux group coverage, multiple-step theta gradients, ranking methods | Không extra L2/MI change, finite updates, aux off không chạy head |
| `tests/test_stair4_v2_checkpoint.py` | Nested monitors, callback-free serialization, epoch mapping, all states, mismatched hashes, interrupted atomic save | Restricted load roundtrip; next-step/next-epoch equivalence ở scope runtime đã định |
| `tests/test_stair4_v2_cli.py` | Real FreeRec CLI, config inheritance/override/conflict, A0/A3/A5/A6 toy runs, completed metric artifact | Process exit0 và assertions về behavior; không mock toàn bộ FreeRec để che lỗi API |
| `tests/test_stair4_v2_reporting.py` | Failed/incomplete/VALID-only logs, missing metrics, multiple run IDs, resume duplicate epochs | Không xuất failed run thành verified test, không fallback số dự kiến |

`tests/test_stair4_v2.py` ở bảng §7 được triển khai thành **nhóm file nêu trên**, tránh một file quá nhiều trách nhiệm. Gọi pytest bằng danh sách tường minh bảy đường dẫn trong bảng để không phụ thuộc shell glob; ghi command thực tế vào verification report. Config unit tests nằm trong `tests/test_stair4_v2_cli.py` và kiểm tra cả pure config functions lẫn real CLI. GPU tests có marker explicit, không tuyên bố GPU passed khi chúng skipped.

Baseline reference phải dùng đúng class/operations của `main.py` trong fixture kiểm soát parser side effects; có thể AST-extract class phục vụ test như cách đã dùng ở v3, không thực thi CLI baseline khi chỉ cần class. Không sửa baseline để dễ import. Khi chi phí test nhẹ nhưng invariant quan trọng, test trực tiếp kết quả thay vì chỉ so tên hàm.

**Fixture tối thiểu:** tiny bipartite graph bất đối xứng về degree; modality có rank đủ d; repeated users/items; nhiều train positives cùng batch; một isolated item; loss singleton. Với d nhỏ gradcheck dùng d≥4 và dữ liệu không hằng, không chỉ d=2 dễ suy biến. Dense eigenvalue tests chỉ dùng toy graph; GPU smoke kiểm tra sparse backward thật và CUDA allocator.

**Chunking:** full GEMM là reference bắt buộc. Tối ưu query chunks chỉ được bật mặc định sau khi loss/grad equivalence và actual peak memory đã đo. Không tạo mọi chunk graph rồi giữ lại và tuyên bố tiết kiệm bộ nhớ. Log số candidates không đổi; nếu giảm candidate set thì phải đổi ablation/protocol name.

## 16. Trình tự triển khai tương lai và cổng nghiệm thu

| Bước | File chính | Deliverable | Chỉ chuyển bước khi |
|---|---|---|---|
| P0 | Schema/config, graph utils, baseline test fixture | Contract baseline, dataset manifests | Có shape/MI/S/A references và config resolved rõ |
| P1 | Smoother + tests | BSF/identity/baseline callbacks | Dense algebra, arm/clear và Adam direction tests đạt |
| P2 | Heads + tests | Cosine/phase/Givens, train-positive masks | Gradcheck, duplicates, coherent fidelity đạt |
| P3 | Model + optimizer wiring | A0/A1/A2/A3/A4/A5 trên fixture | Exact A0 recovery và multi-step gradients đạt |
| P4 | Checkpoint + engine + CLI | Epoch-boundary save/resume và metrics artifacts | Real CLI/resume cùng runtime đạt; failed run không export |
| P5 | GPU preflight, lockfile, chunk optimization nếu cần | CPU/GPU verification tách biệt | Post-warmup sparse backward, finite updates, memory measured |
| P6 | Notebook/reporting | Wrapper Kaggle có thể Restart & Run All | Không phụ thuộc kernel state, đúng run IDs/selection, không số mô phỏng |
| P7 | Pilot rồi multi-seed ablation | Bằng chứng cho H1/H2/H3 | Chọn config bằng validation, ghi failed trials/cost/uncertainty |

Không dùng outcome của P0–P6 làm kết quả NDCG. Không gộp cải tiến graph, auxiliary, sampling, initialization và scorer trong một commit không thể ablate. Ưu tiên commits theo P0–P6 để mỗi thay đổi có test/contract tương ứng; chưa yêu cầu tạo commits ở lượt lập kế hoạch.

**Điều kiện dừng và xử lý:** nếu A0 không khớp, dừng tuning và sửa recovery; nếu theta không có gradient sau nhiều step, xác minh cancellation/normalization/mask trước tăng lambda; nếu OOM, profile activation/GEMM/graph preparation trước thay batch; nếu checkpoint không load restricted được, sửa payload schema trước workaround loader; nếu baseline classical control thắng, ghi nhận và không tăng claim quantum-inspired.

## 17. Definition of Done cho kế hoạch và lần triển khai sau

**Lượt hiện tại hoàn tất ở tài liệu:** đã phân biệt v1-R với v2; khóa algebra/shape/config selectors; chỉ rõ files, API, dependencies, artifact schema, tests và thứ tự thực hiện. Không có file mã nguồn/config/notebook v2 nào được tạo bởi lượt này; không benchmark và không cập nhật bằng chứng nguồn của §2.

**Lần triển khai chỉ được gọi hoàn tất khi:**

1. Các module thực hiện đúng contract §3/§6/§13–14; baseline source không bị thay đổi.
2. A0 recovery, heads/smoother autograd và group isolation đạt; mọi skip GPU được khai báo.
3. Checkpoint/save/resume và structured result selection kiểm tra trên môi trường mục tiêu; best weights và training checkpoint phân biệt rõ.
4. Notebook chạy từ kernel sạch, dùng engine duy nhất, fail rõ khi dependency/data/test lỗi, export đúng measured results.
5. Verification report có command/runtime/output thật và giới hạn; chưa có benchmark thì trạng thái effectiveness vẫn UNVERIFIED.

Các quyết định nghiên cứu có thể đổi sau pilot, nhưng phải version config/spec và ghi lý do trước final multi-seed evaluation. Không tự mang lại các module từ `.pre_.md` chỉ vì chúng từng được gọi là “đã hoàn tất”.
