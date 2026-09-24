# STAIR4-v4 — Confidence-Shrunk Behavioral Graph Calibration for STAIR

**Tên làm việc:** STAIR4-CSGC (Confidence-Shrunk Graph Calibration).  
**Ngày:** 23/09/2026.  
**Trạng thái:** đề xuất kiến trúc và kế hoạch triển khai/kiểm định; **chưa implement, chưa train, chưa có metric v4**.  
**Mục tiêu:** tăng validation-selected TEST NDCG@20 và Recall@20 với training cost gần STAIR, sau khi v2.1 cho gain không nhất quán và overhead lớn.  
**Phạm vi review:** nghiên cứu có mục tiêu từ nguồn sơ cấp và phản biện nội bộ; không phải systematic review toàn bộ lĩnh vực, không phải panel độc lập; NOT_CALIBRATED.

> “Triển vọng nhất” trong tài liệu này là lựa chọn kỹ thuật dưới ngân sách hiện có, không phải xác suất đã đo hoặc lời bảo đảm v4 thắng baseline. Đóng góp mới còn phải được kiểm chứng trước các công trình gần như EVEN và DAMGO.

## 1. Quyết định kiến trúc

Chọn **một thay đổi chính**: hiệu chỉnh trọng số trên support item–item graph của STAIR bằng thống kê train-only, với mức can thiệp giảm khi bằng chứng ít, sau đó dùng graph cố định này trong **BSC optimizer-direction smoothing**.

Giữ MI, FSC, BPR, user/item embeddings và dot-product scorer. Không bật CL, Givens, learned gate, diffusion, graph transformer hoặc spectral auxiliary loss trong v4-core.

Lý do:
- C2 v2.1 đã sửa FSC nhưng chưa chứng minh phase kernel có ích riêng.
- Thời gian tăng ngay khi bật BCCR, dù lambda chỉ 1e-4.
- Graph semantic gần nhau không nhất thiết phù hợp cho chia sẻ update hướng recommendation.
- Hiệu chỉnh graph một lần có thể kiểm định giả thuyết này mà không thêm O(B²) computation vào mỗi batch.
- Cần bảo vệ long-tail: không có co-interaction là **thiếu bằng chứng**, không phải nhãn cạnh nhiễu.

Cấu hình chính chỉ tune alpha (mức pha trộn operator) trong vòng đầu. Các hệ số thống kê còn lại khóa trước để tránh biến một heuristics pipeline thành search grid lớn.

## 2. Bài học thực nghiệm v2.1 dùng làm ràng buộc thiết kế

Theo [báo cáo đã sửa](STAIR4_v21_Experiment_Report.md):
- Baby selected R@20/N@20: 0.1030/0.0450 so reference 0.1042/0.0454.
- Sports: 0.1115/0.0501 so 0.1111/0.0500, chỉ một seed.
- BCCR-on runtime tăng khoảng 7.58×/6.19× so warm-up cùng run.
- Chưa có B1 paired control đủ để quy đóng góp cho BCCR.
- PyTorch allocated thấp không chứng minh throughput tốt.
- Baseline Gate 0 phải được kiểm chứng bằng production methods và optimizer trajectory, không bằng hai bản chép cùng công thức trong test.

Câu hỏi nghiên cứu:

**RQ1:** Giữ cùng topology kNN và optimizer, hiệu chỉnh tương đối các cạnh bằng bằng chứng hành vi có tăng ranking so với STAIR không?  
**RQ2:** Shrinkage theo độ hỗ trợ có giảm tổn hại long-tail so với reweighting không shrinkage không?  
**RQ3:** Lợi ích có đến từ thông tin hành vi, hay chỉ từ việc thay phân phối trọng số/độ mạnh smoothing?  
**RQ4:** Chi phí end-to-end, gồm preprocessing và tuning, có phù hợp hơn C2 v2.1 không?

## 3. Rà soát nghiên cứu và các hướng mới

### 3.1. Phạm vi nguồn và giới hạn so sánh

Đã tìm theo các hướng behavioral denoising, multimodal graph calibration, spectral filtering, efficient recommendation và missing modalities. Nguồn ưu tiên: paper tác giả/arXiv, publisher AAAI/IEEE, repository tác giả. Không suy ra acceptance từ nhãn tìm kiếm; nguồn chỉ có abstract được ghi riêng.

“SOTA” dưới đây chỉ là các ứng viên mạnh/gần đây cần đối chứng. Không xếp hạng xuyên paper bằng số bảng khác split/features. Baby/Sports cùng tên chưa chắc cùng ID mapping, split, negative sampler hoặc evaluation.

Ví dụ, FastMMRec mô tả split 8:1:1 trong §5.1.1; Sports local có 218409/37899/40029 tương tác train/valid/test, không thể mặc định cùng split. Không nhập trực tiếp paper metric vào bảng thắng/thua với local STAIR.

### 3.2. Ma trận tài liệu

| Nguồn và trạng thái kiểm tra | Cơ chế liên quan | Bài học cho v4 | Quyết định |
|---|---|---|---|
| [STAIR, AAAI 2025](https://arxiv.org/abs/2412.11729), abstract + local baseline code | MI, stepwise propagation, constrained updates nhằm giữ collaborative và modality information | BSC là vị trí can thiệp tự nhiên; phải giữ baseline contract | Backbone |
| [FREEDOM, ACM MM 2023](https://arxiv.org/abs/2211.06924), abstract/metadata | Frozen item graph và degree-sensitive user–item denoising | Độ phức tạp động không phải điều kiện cần để hiệu quả | Cơ sở cho static graph, không chép user–item pruning |
| [EVEN, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33358/35513), full text, §Proposed Method | Kết hợp semantic graph với behavioral co-occurrence, pruning feedback và alignment | Tiền lệ gần: behavior-driven graph không mới tự thân | Đối chứng literature bắt buộc |
| [SMORE, WSDM 2025](https://arxiv.org/html/2412.14978v1), full text §4; [author code](https://github.com/kennethorq/SMORE) xác nhận venue | FFT feature fusion/filtering và modality-aware preference | Feature-frequency filtering khác graph spectral filtering | External comparator, không thêm FFT vào core |
| [GUIDER, WSDM 2025](https://doi.org/10.1145/3701551.3703507), metadata tác giả và đoạn nguồn paper; full retrieval không thành công | Denoised ID teacher, guided calibration/KD | Teacher có thể làm sạch semantic transfer nhưng tốn thêm training | Nhánh dự phòng, chưa đủ full-method audit để tái hiện |
| [MM-GF](https://arxiv.org/html/2503.04406), full text; metadata arXiv ghi journal EAAI | Training-free graph filters, coefficients/fusion chọn bằng validation | Không phải mọi cải thiện cần thêm loss; cần tính cả search cost | Efficiency comparator |
| [FastMMRec](https://arxiv.org/html/2507.18489v1), full text §4–5; dùng trạng thái preprint trong review này | Chuyển graph convolution sang testing phase | Hướng tiết kiệm lớn nhưng thay training/inference contract | Nhánh riêng nếu STAIR full-graph cost vẫn cao |
| [Structured Spectral Reasoning](https://arxiv.org/html/2512.01372v1), full text §3; preprint | Graph-frequency bands, masking, low-rank cross-band interaction | Học band reliability là hướng mới, nhưng nhiều thành phần và chi phí spectral cần audit | Không ưu tiên trên Kaggle hiện tại |
| [DAMGO, ESWA 2026](https://doi.org/10.1016/j.eswa.2025.130899), publisher search abstract/metadata; direct full text chưa truy cập được | Complementary graph kết hợp cross-modal similarity và behavioral co-occurrence | Gần trực tiếp với v4; phải thu hẹp novelty claim | Closest-work audit trước tuyên bố novelty |
| [DGMRec, SIGIR 2025](https://arxiv.org/abs/2504.16352), abstract; [author repository](https://github.com/ptkjw1997/DGMRec) xác nhận release | Disentangling và generation khi modality missing | Hợp lý nếu missing modality thực sự là bottleneck | Không ưu tiên khi chưa thấy thiếu features |
| [GRE-MC, 2026](https://arxiv.org/abs/2605.00670), abstract; preprint | Retrieval subgraph + transformer để hoàn thiện modality | Bài toán robustness khác complete-feature benchmark | Deferred |
| [Critical analysis of multimodality, 2025](https://arxiv.org/abs/2508.05377), abstract; preprint | Đánh giá efficiency, task, stage và integration | Chọn modality/architecture theo task, không theo độ mới | Nguyên tắc đánh giá |

### 3.3. Phản biện kỹ thuật từ các nguồn gần

**EVEN:** §Denoising Item Semantic Priors tạo shared-user counts, chọn top-k behavioral items rồi trộn với semantic graph. v4 không thể nhận “đầu tiên sử dụng hành vi để denoise graph”. Khác biệt dự định là giữ support STAIR, điều chỉnh theo mức hỗ trợ và can thiệp vào update smoother; ý nghĩa thực nghiệm của khác biệt này chưa biết.

**SMORE:** FFT trên features không đồng nghĩa eigenmodes của adjacency. Không lấy kết quả SMORE làm chứng cứ cho một Hamiltonian/unitary smoother. Hơn nữa, tọa độ pretrained embedding không mặc nhiên có thứ tự không gian; nếu thử spectral feature transforms phải có real linear/control và audit invariance.

**MM-GF/FastMMRec:** mang lại hướng efficiency khác hẳn BCCR. Tuy nhiên, bỏ training hoặc dời propagation sang inference làm thay nhiều contract. Nên tái hiện như comparator hoặc nhánh riêng, không trộn vào v4-core rồi quy mọi gain cho calibration.

**DAMGO:** metadata cho thấy kết hợp modality/co-occurrence không còn là khoảng trống mới. Khi chưa đọc full methods/code, chỉ ghi overlap ở mức abstract, không khẳng định v4 khác hoàn toàn.

### 3.4. Vì sao chọn CSGC trước các hướng khác?

| Hướng | Thay đổi STAIR | Rủi ro chi phí | Khả năng tách nguyên nhân | Ưu tiên |
|---|---|---|---|---|
| Static confidence-shrunk graph calibration | Thay item smoother operator | Precompute, sparse support giữ nguyên | Cao | v4-core |
| Learnable item/user modality gate | Thay forward/parameters | Vừa | Vừa, dễ confound MI/FSC | Sau core |
| Teacher distillation | Thêm teacher/objective | Vừa đến cao | Cần tính teacher cost | Dự phòng |
| Frequency-adaptive multiband | Thêm filters/gates/loss | Vừa đến cao | Nhiều ablations | Chưa ưu tiên |
| Diffusion/generation | Thêm denoiser/sampling | Cao | Khó trong budget hiện tại | Chưa ưu tiên |
| Graph only at inference | Thay training backbone | Có thể giảm mạnh training | Khác phương pháp STAIR | Track riêng |

Đây là đánh giá engineering có điều kiện, không phải ranking thực nghiệm các papers.

## 4. Kiến trúc tổng thể và đường truyền dữ liệu

~~~mermaid
flowchart TD
    F[Raw visual/text features] --> K[Baseline kNN support and raw weights W0]
    R[Train interactions only] --> C[Candidate edge statistics and support confidence]
    K --> C
    C --> W[Bounded symmetric edge recalibration Wr]
    W --> S[Normalize Sr and mix S_alpha]
    K --> S
    F --> MI[Baseline MI]
    R --> MI
    MI --> E[Trainable user and item embeddings]
    E --> FSC[Unchanged baseline FSC on user-item adjacency]
    FSC --> BPR[BPR and unchanged dot-product scorer]
    BPR --> ADAM[AdamWSEvo moments and update direction]
    ADAM --> U[User update unchanged]
    ADAM --> BSC[Item BSC using static S_alpha]
    S --> BSC
~~~

Chỉ train split tạo C và adjacency. Features của catalog theo cùng transductive setup baseline được phép dùng; không đưa validation/test interactions vào graph hoặc thống kê.

v4-core không có auxiliary parameters. Alpha là hyperparameter, không nn.Parameter. Không có câu chuyện “gate học bằng backprop” trong thiết kế static này.

## 5. Ký hiệu và baseline contract

- R ∈ {0,1}^{U×I}: train interactions đã deduplicate.
- d_u = sum_i R_ui; d_i = sum_u R_ui.
- A: normalized symmetric user–item adjacency của baseline.
- W0: symmetric **unnormalized** item graph sau coalesce(sum), to_undirected(max) đúng thứ tự baseline.
- S0 = D0^(-1/2) W0 D0^(-1/2).
- E = [E_u; E_i], dimension d=64.
- b_j = 0.1 + 0.9(j/d)^gamma; a_j = 1 − b_j.
- L=3 mặc định; gamma/lr/weight decay/batch theo dataset gốc.

MI giữ nguyên whitening, weighting textual:visual theo số neighbors 5:1 và user initialization R_left × MI. Không align/rotate hai modality trong core; việc SVD hai spaces có thể không đồng nhất là giả thuyết khác, phải thử riêng.

FSC giữ:
F0 = E; F_(l+1) = A F_l diag(a);
Z_:j = [b_j / (1 − a_j^(L+1))] sum_(l=0)^L F_l,:j.

BSC dùng **b**, khác FSC recurrence dùng **a**:
P_(b_j,L)(S) = [(1−b_j)/(1−b_j^(L+1))] sum_(l=0)^L b_j^l S^l.

Không đảo b và 1−b. Khi L=0, cả hai polynomial bằng identity, với tolerance floating point thích hợp.

## 6. Static calibration: đặc tả triển khai chi tiết

### 6.1. Candidate support và tính tái lập

C = {(i,j): i<j, W0_ij>0}. Chỉ tính statistics trên C; không tạo R^T R hoặc I×I dense matrix.

- Dùng cùng kNN candidate set cho B0/B1/v4.
- Cache raw neighbors, directed multiplicity và W0.
- Graph cache key gồm hash train split, item mapping, features, kNN k/backend/tie policy.
- Blockwise exact kNN chỉ đảm bảo cùng top-k khi không có numerical/tie ambiguity; kiểm tra neighbor hashes, không tự nhận bitwise baseline parity.
- Alpha=0 bỏ qua toàn bộ calibration và gọi trực tiếp baseline S0, giữ RNG consumption.

### 6.2. Hỗ trợ hành vi có giảm ảnh hưởng user hoạt động nhiều

Cho mỗi user:
w_u = 1 / max(1, d_u).

Cho candidate pair:
c_ij = sum_(u∈N(i)∩N(j)) w_u;
v_ij = sum_(u∈N(i)∩N(j)) w_u²;
q_i = sum_(u∈N(i)) w_u.

Weighted behavioral cosine:
s_ij = c_ij / sqrt(q_i q_j), nếu q_i q_j>0; ngược lại s_ij=0.

Do s là cosine giữa vectors sqrt(w_u)R_ui, nên 0≤s_ij≤1. Đây là association score, **không phải xác suất cạnh đúng**, không phải causal debiasing. Activity weighting giảm đóng góp từng heavy user, không loại bỏ mọi popularity bias.

Cách tính:
- Build sorted item→user CSR lists từ train.
- Intersect lists cho từng candidate unordered pair; đồng thời accumulate c và v.
- q_i/d_i tính bằng sparse reductions.
- Không enumerate mọi cặp trong lịch sử của từng user, tránh chi phí sum_u d_u².
- Implement compiled two-pointer hoặc vectorized sparse backend; Python loop từng edge chỉ dùng oracle nhỏ, không dùng production Electronics.
- Có thể hash-intersect danh sách nhỏ với membership của danh sách lớn; benchmark và xác minh kết quả.

### 6.3. Mức hỗ trợ hiệu dụng và shrinkage

n_eff,ij = c_ij² / v_ij nếu v_ij>0, ngược lại 0.

Với m shared users dương, 1≤n_eff≤m; nếu chỉ một user chi phối, n_eff nhỏ. Đó là effective-support heuristic, không coi các user độc lập hoặc gọi đây là posterior confidence.

r_ij = [n_eff,ij / (n_eff,ij + tau_c)]
       × sqrt([d_i/(d_i+tau_d)] [d_j/(d_j+tau_d)]).

Pilot khóa tau_c=5, tau_d=10. Nếu c_ij=0 thì r_ij=0. Nếu item không có train history, r=0.

### 6.4. Hiệu chỉnh tương đối trong strata mức phổ biến

Một cosine nhỏ có ý nghĩa khác ở head/tail. Xây item degree bins bằng quantiles của log(1+d_i), mặc định 4 bins; biên quantile trùng thì merge, không chia ngẫu nhiên các degrees bằng nhau.

Mỗi unordered candidate pair thuộc stratum (min(bin_i,bin_j), max(...)). Tính midrank empirical CDF F_g trên s của **mọi candidate edges** trong stratum:
F_g(s) = [count(s_e<s) + 0.5 count(s_e=s)] / |C_g|.

Nếu stratum <200 edges, dùng pooled CDF toàn C. Tie comparator và float precision cố định; logging fallback count.

z_ij = 2F_g(s_ij) − 1 ∈ [−1,1].
h_ij = r_ij z_ij.

Nếu C rỗng, trả S0. Nếu c=0, h=0 dù CDF cho rank thấp. Nếu mọi score trong stratum giống nhau thì z=0.

**Ý nghĩa:** s cao/thấp tương đối chỉ điều chỉnh mạnh khi có đủ support. Không có chứng cứ thì raw edge weight giữ nguyên. Degree stratification là heuristic chống scale differences, không bảo đảm fairness.

### 6.5. Raw graph và bounded multiplier

W_r,ij = W0_ij (1 + epsilon h_ij), với epsilon=0.5 cố định.

Mirror cùng weight cho j,i; diagonal giữ như baseline. Vì |h|≤1:
(1−epsilon)W0_ij ≤ W_r,ij ≤ (1+epsilon)W0_ij.

Không threshold/prune/add cạnh trong core. Support giữ nguyên, weights dương khi W0 dương. Không làm item cô lập mới.

Lưu ý: giữ raw weight khi r=0 **không bảo đảm normalized row giữ nguyên**, vì degree của neighbor có thể thay đổi. Chỉ alpha=0 bảo đảm baseline toàn bộ. Phải đo tail behavior sau normalization.

### 6.6. Normalize và pha trộn operator

D_r,ii = sum_j W_r,ij;
S_r = D_r^(-1/2) W_r D_r^(-1/2), inverse degree=0 khi degree=0.

S_alpha = (1−alpha)S0 + alpha S_r, alpha∈[0,1].

Pilot alpha∈{0.10,0.25,0.50}; primary initial candidate alpha=0.25. Không đồng thời tune epsilon và alpha trong Stage 1.

- Đây là mixture của **hai normalized operators**, không phải normalize mixture W.
- Không row-renormalize S_alpha sau đó.
- Hai graph cùng support; merge values thành **một CSR** trong prepare().
- Mỗi BSC hop chỉ cần một SpMM, không tính S0x và S_rx riêng mỗi step.
- Zero-degree row và self-loop semantics giữ đúng baseline.
- Graph static trong toàn run; không warm-up alpha và không cập nhật graph mỗi 5 epochs.

## 7. Lý thuyết: điều được bảo đảm và điều không được bảo đảm

### 7.1. Đối xứng và chuẩn phổ

W0,Wr không âm và đối xứng ⇒ normalized adjacency S0,Sr đối xứng và có spectrum trong [−1,1], theo quy ước zero-degree ở trên.

S_alpha cũng đối xứng:
||S_alpha||2 ≤ (1−alpha)||S0||2 + alpha||Sr||2 ≤ 1.

Không gọi S_alpha là PSD: adjacency chuẩn hóa có thể có eigenvalues âm. Mixture có thể không có eigenvalue đúng 1 vì hai operators có eigenvectors chủ đạo khác nhau.

### 7.2. BSC polynomial và perturbation bound

Đặt t_l,j = [(1−b_j)/(1−b_j^(L+1))]b_j^l.
Với 0≤b_j<1: t_l,j≥0 và sum_l t_l,j=1.

Do đó ||P_j(S_alpha)||2≤1; applied theo từng feature column không tăng Frobenius norm của update đầu vào. Đây là bound của smoother, không bound toàn bộ AdamW hoặc training loss.

Từ telescoping và ||S||≤1:
||S_alpha^l−S0^l||2 ≤ l ||S_alpha−S0||2.
Suy ra:
||P_j(S_alpha)−P_j(S0)||2
≤ alpha ||Sr−S0||2 sum_l l t_l,j
≤ 2alpha sum_l l t_l,j.

Bound thô này giúp giới hạn can thiệp, nhưng có thể lỏng. Không suy ra gradient descent, generalization hoặc NDCG tốt hơn. Adam direction không đồng nhất raw gradient, và adaptive optimizer có dynamics riêng.

### 7.3. Exact recovery và giới hạn shrinkage

- Alpha=0: operator, forward, loss và optimizer update phải khớp baseline khi cùng inputs/RNG.
- Epsilon=0: Wr=W0; về toán học Sr=S0, nhưng production alpha=0 fast path vẫn là chuẩn parity.
- c=0: raw multiplier=1, không phạt trực tiếp missing evidence.
- n_eff nhỏ: h bị shrink về 0, nhưng không có guarantee tail recall tăng.
- Reweighting đồng đều mọi cạnh có thể bị normalization triệt tiêu. Vì vậy cần đo variance(h), ||Sr−S0|| và spectral/action differences thay vì chỉ báo distribution multipliers.

## 8. Forward, loss, optimizer và evaluation

Loss duy nhất: mean BPR, negative sampling giống baseline. Không thêm explicit L2 loss ngoài weight decay nếu baseline không có.

Parameter groups:
- Users: AdamWSEvo, smoother=None, baseline lr/weight_decay.
- Items: AdamWSEvo, static calibrated BSC, baseline lr/weight_decay.
- Không auxiliary group trong core; assert mọi trainable parameter xuất hiện đúng một lần.

BSC biến đổi **Adam update direction ở đúng vị trí baseline**, không áp trực tiếp lên raw gradient rồi gọi Adam như cũ. Giữ moment update, bias correction và weight-decay ordering.

prepare() chỉ tạo buffers/static statistics, no active grad_fn. Smoother không giữ trainable parameters. Nếu reuse step-scoped API, arm trước optimizer.step và clear trong finally; static graph không cần gate snapshot mới mỗi step.

Scorer = Z_u^T Z_i. Giữ nguyên:
- full/pool ranking;
- seen-item masking;
- candidate pool construction;
- NDCG/Recall implementation;
- chọn best bằng validation NDCG@20;
- TEST report sau Load best model.
Không rerank test bằng semantic graph hoặc tune alpha trên test.

## 9. Complexity và kế hoạch hiệu năng

Ký hiệu M = |C| unordered candidates, E_R = nnz(R).

- CSR histories: O(E_R+I).
- Candidate intersections: O(sum_(i,j)∈C (d_i+d_j)) với two-pointer; có thể nặng với hubs, phải đo preprocessing.
- Rank calibration: O(M log M) tổng khi sort từng stratum; memory O(M).
- Normalize/merge: O(M+I).
- Training BSC: O(L·nnz(S0)·d), cùng sparsity và số hop baseline.
- FSC full graph vẫn trả chi phí baseline. v4 không giải quyết toàn bộ full-graph bottleneck.
- Cache S0, static statistics, hashes trên disk; khi train chỉ cần S_alpha trên device nếu không diagnostic.
- Không I², không B² tensor, không R^T R materialized.

**Target engineering, chưa phải kết quả:** epoch throughput ≤1.2× baseline runtime; GPU allocated ≤1.2× baseline trong matched environment; tổng preprocessing+tuning phải báo riêng. Nếu vượt, profile trước, không đổi metric hoặc giấu startup.

Nếu candidate intersections quá chậm trên Electronics: tối ưu exact backend trước. Approximation/sampling là biến thể có tên riêng, không tự thay bằng xấp xỉ trong core.

## 10. Diagnostic preflight trước GPU training dài

Tạo thống kê train-only:
- fraction candidates có c>0 và n_eff≥2/5;
- distribution r,h,multiplier theo modality provenance và item-degree bins;
- variation của raw/normalized weights;
- relative action difference ||(Sr−S0)X||F / max(||S0X||F,eps) trên fixed random probes;
- graph symmetry, finite values, degrees, nnz, connected components nếu khả thi;
- fraction tail edges thay đổi normalized weight;
- thời gian chuẩn bị và peak host RAM.

Không dùng validation/test để quyết định cạnh “đúng”. Có thể đánh giá graph retention trên một inner-train holdout, nhưng đó là thí nghiệm phụ; không đưa holdout labels vào graph vừa đánh giá.

Nếu support gần như bằng 0 hoặc Sr≈S0, **dừng mở rộng**: calibration không có đủ tín hiệu. Không tuyên bố phương pháp thất bại do noise khi nó thực chất không thay đổi operator.

## 11. Kế hoạch thực nghiệm và go/no-go

### 11.1. Gate 0

1. Tiny deterministic graph, CPU float64 oracle: FSC/BSC production vs baseline.
2. Reuse exactly static tensors; compare nhiều steps BPR, gradients, Adam moments, embeddings, scores.
3. Alpha=0 phải bypass preprocessing ngẫu nhiên và không đổi RNG consumption.
4. Full/pool masking và best checkpoint semantics giống baseline.
5. Checkpoint roundtrip/resume: optimizer, RNG, sampler, static hashes, config và source version.
6. FP32 tolerance đăng ký trước, không nới để che bug; sparse nondeterminism/ties được ghi.

### 11.2. Ablation IDs riêng v4

| ID | Thay đổi | Câu hỏi |
|---|---|---|
| V4-B0 | Baseline main.py | Mốc matched rerun |
| V4-B1 | v4 alpha=0 | Engine parity |
| V4-C | Core shrinkage + activity weighting + degree strata | Phương pháp đề xuất |
| V4-NS | Core nhưng r=1 nếu c>0, r=0 nếu c=0 | Support shrinkage có ích? |
| V4-ND | Core dùng pooled CDF, bỏ degree strata | Stratification có ích? |
| V4-NA | Core w_u=1; giữ các phần khác | Activity weighting có ích? |
| V4-SH | Shuffle h giữa edges trong cùng degree stratum, seed cố định | Thông tin hành vi hay chỉ phân phối weights? |
| V4-SM | Thay smoother bằng (1−delta)P(S0)+delta I | Chỉ giảm smoothing có giải thích gain? |

V4-SM là control riêng ở output smoother, không đồng nhất S_alpha. Delta và ngân sách search công khai; không dùng test chọn control yếu.

### 11.3. Stage budget có giới hạn

**Stage A: CPU/static audit.** Chạy Gate 0 và diagnostics trên Baby, Sports, Electronics nếu data sẵn. Không cần training 500 epochs.

**Stage B: Baby pilot seed=1, 100 epochs.** V4-B0/B1, V4-C alpha {0.10,0.25,0.50}; tối đa 5 runs. B0/B1 có thể rút về integration check ngắn nếu parity đã đạt, nhưng vẫn giữ B0 full pilot để so curve. Chọn tối đa một alpha bằng validation cuối pilot theo rule cố định; không xem test. Short-horizon bias phải được ghi.

**Stage C: confirmation.** Baseline và core đã khóa, Baby+Sports, 500 epochs, paired seeds {1,2,3}. Nếu có điều kiện dùng {1,...,5}. Run cũ seed=1 chỉ thay paired run khi manifest/source/environment đủ khớp; mặc định không.

**Stage D: ablation.** Chỉ nếu core có tín hiệu: NS/ND/NA/SH/SM, Baby rồi Sports; cùng alpha hoặc cùng budget tuning đã công khai. Báo cả failed trials.

**Stage E: Electronics.** Benchmark throughput/prepare trước, sau đó cùng recipe đã khóa, cùng budget baseline; ít nhất seed=1 exploratory, thêm paired seeds nếu đủ. Không điều chỉnh recipe bằng Electronics test.

### 11.4. Ngưỡng quyết định đề xuất

Các ngưỡng sau là **quy tắc ưu tiên ngân sách**, không phải kết quả dự đoán hoặc ngưỡng significance:
- Core cần mean validation NDCG@20 tăng ≥0.5% trên ít nhất một dataset, dataset còn lại không giảm quá 0.5%, và chiều gain lặp ở ≥2/3 paired seeds.
- Throughput target ≤1.2× baseline, báo cả total time; nếu accuracy gain rõ nhưng runtime vượt, đánh giá Pareto công khai.
- Không có gain hoặc SH đạt tương tự ⇒ chưa chứng minh information-driven calibration; không mở thêm modules để cứu narrative.
- Nếu CI chênh lệch bao gồm 0, ghi exploratory; ba seeds không đủ kết luận mạnh.
- TEST chỉ dùng xác nhận recipe đã khóa, không dùng gate để tiếp tục tuning.

Không đặt mục tiêu số như “NDCG chắc chắn 0.052” hoặc “tăng 3%”. Không có dữ liệu v4 hỗ trợ các con số đó.

### 11.5. Reporting

Mean/std theo seed; paired deltas; user-level bootstrap nếu lưu per-user metrics, nhưng không coi users là thay thế cho seed replication. Report số trials, total GPU-hours, preprocessing, peak RAM/VRAM, epoch latency, validation latency.

Phân tầng head/mid/tail theo train degree, cố định bins trước evaluation. Tail recall tính trên relevant tail items với denominator rõ; không tự đổi full ranking thành tail-only ranking. Báo cả overall NDCG để không đánh đổi âm thầm.

External comparator ưu tiên EVEN và SMORE trên cùng split/features; nếu chưa reproduce được thì chỉ để related work, không đưa paper values vào bảng ranking nội bộ.

## 12. Phản biện Socratic và failure modes

| Câu hỏi khó | Rủi ro thực tế | Thử nghiệm/đáp ứng |
|---|---|---|
| Có gì mới so EVEN/DAMGO? | Semantic + behavior graph là tiền lệ rõ | Thu hẹp claim vào support-preserving, confidence-shrunk BSC calibration; đọc DAMGO full trước novelty claim |
| Shared users có thể chỉ là popularity? | Association vẫn bias | Activity weighting, degree strata, shuffled control, tail metrics |
| Item thay thế nhau ít co-click? | Co-occurrence bỏ sót substitutes | Zero evidence không downweight; giữ semantic topology; audit vẫn có thể thất bại |
| Vì sao n_eff là confidence? | User không độc lập, trọng số không xác suất | Gọi support heuristic; no calibration-probability claim |
| Rank thấp có chắc là noise? | Không; chỉ thấp tương đối | Bound multiplier và alpha; NS/SH controls; tránh diễn giải causal |
| Shrinkage có bảo vệ tail tuyệt đối? | Degree renormalization tác động gián tiếp | Đo normalized action và tail metrics; không hứa guarantee |
| Không thêm edges có bỏ lỡ quan hệ tốt? | Có; core không tăng recall của graph support | Chấp nhận giới hạn để isolate weights; topology expansion là v4.1 riêng |
| Chỉ hiệu chỉnh BSC có đủ tạo gain? | Có thể MI/FSC đã giải quyết phần lớn | B1/C và operator-action diagnostics; nếu không gain thì bác bỏ H1 |
| Bound norm có bảo đảm ranking? | Không | Chỉ dùng cho stability contract |
| Static graph có lỗi thời? | Có thể trong temporal/dynamic data | Scope batch offline; không mở rộng claim tới online recommendation |
| Có thể đơn giản tune gamma tốt hơn? | Gain có thể chỉ phản ánh tuning budget | Baseline tuning budget tương đương, SM control |
| So với v3 có vượt? | Chưa biết; graph pairwise thiếu group structure | Matched v3 comparator nếu có artifacts; không suy từ theory |
| Precompute có đắt hơn tiết kiệm training? | Hubs làm intersection nặng | Report amortized và one-run total cost, cache validity |
| Current graph tốt rồi thì sao? | CSGC chỉ thêm noise | Alpha=0 được phép thắng; công bố negative result, không ép chọn v4 |

## 13. Kế hoạch triển khai mã nguồn

**Cập nhật triển khai 2026-09-23:** các module CSGC, training engine, ba YAML, tests, notebook và scripts dưới đây đã được bổ sung. Chi tiết API thực tế, lệnh chạy, giới hạn resume và kiểm thử nằm trong [STAIR4_v4_Implementation.md](STAIR4_v4_Implementation.md); đối chiếu code tác giả tại [STAIR4_v4_Source_Audit.md](STAIR4_v4_Source_Audit.md). Các kết quả test không phải bằng chứng tăng Recall/NDCG. Phần pseudocode bên dưới mô tả ý tưởng, không thay thế API trong mã nguồn.

| File dự kiến | Trách nhiệm chính |
|---|---|
| models/stair4_v4_utils.py | Baseline raw support, sorted train CSR, exact candidate intersection, support statistics, strata midranks, hash/cache validation |
| models/stair4_v4_graph.py | W_r construction, normalization, CSR mixture, static diagnostics; immutable graph bundle |
| models/stair4_v4.py | Baseline MI/FSC/scorer/BPR; register S_alpha buffer; parameter groups chỉ users/items |
| optimizers/stair4_v4_smoother.py | Thin parameter-free BSC wrapper hoặc reuse MHDSmoother callback; đúng b coefficient và clear lifecycle |
| main_stair4_v4.py | CLI validated, YAML inheritance, seed, manifests, evaluation baseline, telemetry và checkpoint/resume |
| configs/dataset_stair4_v4_{baby,sports,electronics}.yaml | Kế thừa baseline; chỉ thêm calibration config |
| tests/test_stair4_v4.py | Algebraic oracle, production Gate 0, randomized sparse checks, optimizer parity, checkpoint/resume, no-leakage |
| notebook/P4/stair4_v4.ipynb | CPU preflight → source hash → benchmark → pilot → training; không force reset |
| scripts/audit_stair4_v4_graph.py | Diagnostic trước training; JSON artifact để quyết định có đủ signal |
| scripts/summarize_stair4_runs.py | Parse selected-checkpoint metrics; không trộn epoch 500 và best |

API contracts đề xuất:
- build_candidate_statistics(train_csr, candidate_pairs) -> c, v, q, degrees.
- calibrate_raw_graph(W0, statistics, config) -> Wr, diagnostics.
- build_operator(W0, Wr, alpha) -> detached CSR S_alpha.
- apply_bsc(direction, operator, b, L) -> detached direction, same dtype/device/shape.
- train/evaluation dùng cùng scorer; không calibration trong recommend_from_full/pool.

Pseudocode:
~~~python
# All graph preparation is static and train-only.
W0, S0 = build_baseline_item_graph(features, baseline_neighbors)
if config.alpha == 0.0:
    S = S0
else:
    stats = build_candidate_statistics(train_csr, unordered_support(W0))
    reliability = support_shrinkage(stats, tau_c=5.0, tau_d=10.0)
    association = degree_stratified_midrank(stats)
    Wr = reweight_symmetric(W0, 1.0 + 0.5 * reliability * association)
    Sr = symmetric_normalize(Wr)
    S = merge_same_support_csr(S0, Sr, alpha=config.alpha)

# MI, forward and ranking follow STAIR.
loss = baseline_bpr(model.encode(), batch)
optimizer.zero_grad()
loss.backward()
# AdamWSEvo applies item BSC using S at the baseline update location.
try:
    smoother.arm_static(S)
    optimizer.step()
finally:
    smoother.clear_step_snapshot()
~~~
Pseudocode là interface dự kiến, không phải callable code đã có.

## 14. Config pilot và kiểm thử bắt buộc

~~~yaml
# Minimal executable example for main_stair4_v4.py.
base_config: Amazon2014Baby_550_MMRec.yaml
alpha: 0.25
edge_epsilon: 0.5
support_tau: 5.0
degree_tau: 10.0
degree_bins: 4
min_stratum_edges: 200
activity_weighting: inverse_degree
ablation_id: V4-C
~~~

Baseline inherited:
| Dataset | Batch | LR | Weight decay | Gamma | Epochs | d / L |
|---|---:|---:|---:|---:|---:|---|
| Baby | 1024 | 1e-3 | 0.3 | 0.1 | 500 | 64 / 3 |
| Sports | 1024 | 1e-3 | 0.1 | 0.2 | 500 | 64 / 3 |
| Electronics | 4096 | 1e-3 | 0.1 | 0.4 | 500 | 64 / 3 |

Tests phải gọi production paths:
- Candidate c/v/q so với dense tiny oracle; duplicate interactions và unordered symmetry.
- Zero overlap, empty support, isolated items, shared-user dominated edge, ties và merged bins.
- Multipliers trong bounds; no new edges; finite symmetric normalized graph.
- Small dense eigensolver kiểm tra spectrum; không dùng eigendecomposition production.
- Core alpha=0 nhiều steps khớp baseline embeddings/moments/scores.
- Không test PSD adjacency sai; test contraction của polynomial ở toy cases.
- Thêm/sửa valid/test interactions không đổi graph/train statistics.
- Shared-user statistics cache invalidated khi train/features/mapping thay đổi.
- Parameter coverage/disjointness; không static buffers có grad_fn.
- End-to-end train/eval/resume smoke test trên toy dataset có real sampler contract.
- Profiler benchmark không chứa full-training diagnostic overhead mặc định.

## 15. Nguồn, claim boundaries và kết luận thiết kế

Nguồn chính và locators đã liên kết ở §3: STAIR abstract/local main.py; EVEN Proposed Method, equations (1)–(3); SMORE §4.1; FastMMRec §4–5.1; MM-GF §4.2–4.3; Structured Spectral Reasoning §3.1–3.2. GUIDER/DAMGO chỉ đạt partial-source access, không xem như full-method replication.

Không có paper nào trong danh sách chứng minh công thức shrinkage/CDF/multiplier được đề xuất ở đây sẽ vượt STAIR. Các công thức đó là thiết kế mới **trong phạm vi project**, còn originality với toàn bộ literature chưa được xác lập.

**Quyết định:** ưu tiên v4-core CSGC vì isolate được giả thuyết semantic-edge reliability và giữ chi phí online gần baseline theo số sparse operations. Triển khai theo Gate 0 → static signal audit → Baby pilot → paired confirmation → Electronics. Nếu không có signal hoặc alpha=0 thắng, chấp nhận bác bỏ hướng này; không gắn thêm module để tạo vẻ phức tạp.
