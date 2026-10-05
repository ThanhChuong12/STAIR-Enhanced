# STAIR5-v5.2: NLGCL-BPE — Budgeted Path Expansion for Item Update Smoothing

**Ngày đối soát:** 05/10/2026. **Trạng thái:** đề xuất nghiên cứu và đặc tả triển khai; chưa huấn luyện v5.2. **Comparator trực tiếp:** STAIR5-v4/NLGCL-CSE của giai đoạn 5, không phải STAIR4-v4 ở giai đoạn 4. **Mục tiêu:** tăng trên 5% tương đối so với GD5-v4, kiểm chứng bằng thực nghiệm; không phải kết quả đã đạt.

## 1. Quyết định kiến trúc cuối cùng

Chọn **NLGCL-BPE**: giữ MI, FSC, BPR, NLGCL gốc và optimizer của GD5-v4; mở rộng **raw CF graph** bằng một tập cạnh `item → item trung gian → item` có ngân sách; chuẩn hóa đối xứng lại CF graph rồi trộn với semantic operator theo đúng hệ số v4. Graph được xây một lần từ train và cache. Không thêm mạng gating, loss alignment, graph học qua validation, hay eigen-decomposition vào primary.

Điểm thay đổi là **support và phân bố trọng số của CF graph**, không phải tăng hệ số CF, tăng số lớp FSC hay thay objective. Chốt:

\[
W_\star=W_1+W_{\mathrm{add}},\qquad
S_\star=\operatorname{SymNormIso}(W_\star),\qquad
S_{5.2}=0.9S_0+0.1S_\star.
\]

`SymNormIso` là chuẩn hóa degree đối xứng của graph không âm, cộng identity **chỉ tại node có degree bằng 0**. Tắt expansion phải gọi trực tiếp đường v4 và khôi phục đúng operator, optimizer groups, loss và ranking của v4. Phiên bản này kế thừa v4, không mặc định kế thừa overhead KPE của v5.1.

**Giả thuyết có thể bác bỏ:** một số đường đi CF hữu ích đang bị ảnh hưởng bởi sparsification và độ dài đường truyền trong BSC. Nén có chọn lọc các đường đi này thành cạnh trực tiếp có thể cải thiện hình học của hướng cập nhật, trong khi giới hạn số cạnh và khối lượng trọng số ngăn can thiệp quá mạnh. Giả thuyết sai nếu mở rộng không thắng v4, không thắng mở rộng direct CF cùng ngân sách, hoặc chỉ tạo lợi ích tương đương tăng số bước BSC.

Đây là lựa chọn ưu tiên dưới ràng buộc giữ backbone và sparse training. Không có định lý rằng nó là kiến trúc tốt nhất, rằng đường đi dài luôn mang tín hiệu đúng, hay rằng tăng hơn 5% chắc chắn xảy ra. Nếu muốn giải quyết basis dependence hoặc unimodal erasure trong `EStair.md`, cần một nghiên cứu riêng thay đổi biểu diễn; BPE không tuyên bố đã giải quyết chúng.

## 2. Đối soát kết quả v5.1 trước khi suy luận nguyên nhân

### 2.1. Phạm vi bằng chứng

Nguồn chính là log, manifest và telemetry trong `logs/GD5/stair5_v51_complete_artifacts/`, đối chiếu `stair5_v4_complete_artifacts/`. Bảng dưới dùng kết quả **test của checkpoint chọn bằng validation NDCG@20**, theo manifest tổng hợp; độ chính xác hiển thị bốn chữ số thập phân.

| Dataset | Model | R@10 | R@20 | N@10 | N@20 | Epoch chọn |
|---|---|---:|---:|---:|---:|---:|
| Baby | GD5-v4 | 0.0678 | 0.1056 | 0.0364 | 0.0461 | 480 |
| Baby | v5.1 KPE | 0.0676 | 0.1054 | 0.0364 | 0.0461 | 480 |
| Sports | GD5-v4 | 0.0765 | 0.1143 | 0.0419 | 0.0517 | 485 |
| Sports | v5.1 KPE | 0.0758 | 0.1136 | 0.0418 | 0.0516 | 500 |
| Electronics | GD5-v4 | 0.0458 | 0.0678 | 0.0260 | 0.0316 | 495 |
| Electronics | v5.1 KPE | — | — | — | — | — |

Electronics trong manifest v5.1 có `test_metrics={}`, `best_epoch=null`, `train_time_min=null`. Vì vậy nhận định “v5.1 Electronics N@10 = 0.0260, parity với v4” trong bản đề xuất không được artifact này xác nhận: 0.0260 hiện là số của v4.

Trên Sports, thay đổi từ v4 sang v5.1 lần lượt khoảng **−0.92%, −0.61%, −0.24%, −0.19%** theo số đã làm tròn. Baby R@10/R@20 khoảng **−0.29%/−0.19%**, còn hai NDCG bằng nhau ở độ chính xác hiển thị. Đây là **gần ngang**, không phải chứng minh tương đương tuyệt đối. Một seed không xác lập equivalence; muốn kết luận tương đương thống kê cần biên tương đương đăng ký trước và nhiều run.

Sports log có test cuối checkpoint chọn ở độ chính xác cao hơn: `(0.075831, 0.113562, 0.041824, 0.051588)`. Baby summary trước `Load best model @Epoch: 480` có `(0.067748, 0.105644, 0.036642, 0.046351)` của epoch cuối 500; không được thay số checkpoint 480 bằng summary này. Bộ log được kiểm tra chỉ in bốn chữ số cho test sau khi load checkpoint Baby tốt nhất. Các số sáu chữ số được gán cho checkpoint 480 trong báo cáo phải có evaluator artifact riêng trước khi sử dụng.

### 2.2. Thời gian và VRAM: giữ đúng ranh giới phép đo

| Dataset | v4 `Coach.fit` | v5.1 `Coach.fit` | Overhead fit | v5.1 max training peak allocated |
|---|---:|---:|---:|---:|
| Baby | 1,193.516 s = 19.89 phút | 1,359.394 s = 22.66 phút | +13.90% | 169.63 MiB |
| Sports | 2,532.349 s = 42.21 phút | 3,110.679 s = 51.84 phút | +22.84% | 238.10 MiB |

Manifest tổng hợp ghi thời gian process khoảng 20.54/23.44 phút Baby và 43.04/52.97 phút Sports cho v4/v5.1; đây không phải cùng ranh giới với `Coach.fit`. Training peak lấy **max trên 500 dòng telemetry**, không lấy epoch cuối. Không đồng nhất allocated với reserved hoặc NVML whole-process VRAM. Bộ artifact đã kiểm tra chưa cung cấp phép đo raw xác nhận hai giá trị full-process 520/727 MB trong báo cáo; không cộng một hằng số CUDA context để suy ra chúng.

Train fingerprint và graph fingerprint v4/v5.1 **trùng nhau trên từng Baby/Sports**. Điều này hỗ trợ việc graph không đổi trong đối chứng KPE. Nó không chứng minh mọi source hash, RNG draw, sampler state và kernel đều giống nhau. Chi phí search/mask trên GPU cũng không phải bằng chứng dùng Tensor Cores. Không có profiler ở mức kernel để xác nhận nguyên nhân cụ thể của toàn bộ overhead.

Telemetry có **191 lần loss tăng giữa hai epoch liên tiếp trên Baby và 183 trên Sports**. Loss giảm về xu hướng tổng thể không đồng nghĩa giảm đơn điệu. Validation NDCG không được so với baseline **test** NDCG để tuyên bố vượt baseline ở một epoch trung gian.

### 2.3. V5.1 đã bác bỏ điều gì, chưa bác bỏ điều gì?

Kết quả hiện tại không cho thấy lợi ích ranking của **cấu hình KPE này** trên hai dataset/seed đã chạy. Nó làm giảm ưu tiên KPE trong primary v5.2, đặc biệt với chi phí Sports vượt 20%. Nó chưa chứng minh quy luật “topology luôn quan trọng hơn objective”, chưa phủ định mọi positive-aware loss, và chưa chứng minh false negatives là không đáng kể.

Density **train** Baby là `118551/(19445*7050) ≈ 0.000865`, Sports là `218409/(35598*18357) ≈ 0.000334`; tương ứng khoảng 0.0865%/0.0334%. Không dùng mật độ của toàn bộ train+valid+test để phân tích collision của train batches. Một pair probability cũng khác probability có ít nhất một collision trong batch.

Gọi \(r_a\) là phần softmax mass của các cột bị loại khỏi denominator, target vẫn được giữ. Với logits cố định:

\[
\ell_{\rm before}-\ell_{\rm after}=-\log(1-r_a).
\]

Số collision ít vẫn có thể có mass lớn. Với normalized cosine, \(\tau=0.2\), một cột có logit 5 và 1.023 cột có logit −5 nhận mass khoảng **0.956**, không phải \(1/1024\). Đây là phản ví dụ đại số, không phải mô tả phân bố thực tế STAIR.

Với shuffled interaction pairs, tần suất item key liên quan popularity. Xấp xỉ sampling độc lập có hoàn lại, số **ID dương khác target** kỳ vọng xuất hiện trong keys của user \(u\) là:

\[
\sum_{i\in\mathcal N_{\rm train}(u)\setminus\{p\}}
\left[1-\left(1-\frac{n_i}{E}\right)^{B-1}\right].
\]

Đây chỉ là xấp xỉ; batch shuffle không hoàn lại cần phân bố tương ứng. Density \(E/(UI)\) và \(B\times density\) không thay thế được collision đo theo sampler, unique keys, repeated users và degree. Overhead mỗi batch không tự làm giảm diversity nếu số epoch/batch thực hiện vẫn bằng nhau; điều đó chỉ có thể xảy ra với ngân sách wall time cố định hoặc thay sampler.

Để chẩn đoán KPE, run đối chứng sau phải log: tỷ lệ query có extra known-positive, số key bị mask, removed softmax mass, chênh lệch loss và cosine/norm giữa gradient NLGCL có/không mask. Log thưa trên cùng batch trước update; không thay sampler để đo. Chưa có các giá trị này nên nguyên nhân parity vẫn là một tập giả thuyết: mass nhỏ, auxiliary gradient yếu, hard-negative removal gây bất lợi, hoặc thay đổi gradient không chuyển thành thứ hạng top-K.

## 3. Phản biện ba bản đề xuất được cung cấp

| Ý tưởng/khẳng định | Quyết định | Lý do và sửa đổi |
|---|---|---|
| `i → u → j` là CF mới hai-hop | Sửa định nghĩa | Đây là \(R^TR\), đã dùng để xây W1 của v4; đổi degree-user weighting là reweighting co-occurrence, không phải thêm chiều quan hệ độc lập. |
| `i → k → j`, k là item | Giữ làm giả thuyết chính | Là đường dài 2 trên item graph, tương ứng một quan hệ gián tiếp; phải loại self-loop và cạnh đã có để đo expansion. |
| Tăng CF seed từ k=5 lên 10–15 đồng thời thêm path | Tách đối chứng | Đổi direct support và thêm path cùng lúc khiến không biết thành phần nào giúp. Primary giữ W1 v4. |
| Chỉ giữ cặp có ít nhất hai item trung gian | Chuyển thành ablation | Trên Baby offline, điều kiện này thu hẹp ứng viên rất mạnh; không đủ cơ sở gọi nó là “nhiễu đã được khử”. |
| NCER bảo đảm loại nhiễu phổ | Không dùng như định lý cho STAIR | Neighborhood overlap không xác định clean/noisy edge; không chuyển bảo đảm có điều kiện của paper sang graph đã pruning và max-union. |
| Mean text-pair cosine + image-pair cosine cần shared coordinates | Bác bỏ yêu cầu đó | Mỗi cosine so hai item **trong cùng modality**; khác với cosine trực tiếp text-image của cùng item. |
| Semantic hard threshold giải quyết substitute/complement | Không mặc định | Tương đồng thấp có thể là complementary; tương đồng cao có thể là substitute. Không có relation labels để suy ra dấu cập nhật. |
| Relation MLP dùng timestamp/price/category để phân loại | Trì hoãn | Cần audit dữ liệu và supervision. Có TIMESTAMP trong Baby train không đồng nghĩa có price, impressions hoặc nhãn complement/substitute. |
| Learned edge gate học BPR trong BSC offline | Không hợp lệ như mô tả | BSC/optimizer step chạy `no_grad`; gate chỉ ở đó không nhận ordinary backprop từ BPR. Forward graph hoặc hypergradient là kiến trúc khác. |
| Loại “edge có λmax lớn” | Loại bỏ | Eigenvalue thuộc operator/subgraph được định nghĩa, không phải thuộc một cạnh riêng chưa định nghĩa toán tử. |
| 5–10 power iterations chứng nhận norm≤1 | Thay bằng chứng minh chuẩn hóa | Rayleigh estimate thường là lower estimate; không phải upper certificate để chia operator. |
| Zero additional compute/VRAM vì preprocess một lần | Sửa | Không thêm module per batch nhưng CSR lớn hơn vẫn tăng traffic/SpMM; preprocess và cache có chi phí. |
| +5–10%/khôi phục chắc chắn | Đổi thành target | Không có run v5.2. Tăng so với STAIR gốc khác tăng so với v4. |

Công thức hinge alignment trong đề xuất thứ nhất có dấu ngược: chuẩn margin yêu cầu \([d_{pos}-d_{neg}+m]_+\), không phải \([d_{neg}-d_{pos}+m]_+\). Khi dương đã gần và âm đã xa, biểu thức thứ hai vẫn phạt mạnh; sửa dấu cũng chưa làm alignment phù hợp với bài toán ranking. Vì vậy primary không thêm loss này.

Đề xuất hub penalty \(d^{-1/2}\exp(-d/P95)\) không chỉ có exponential factor: với d=10, P95=200, hệ số đầy đủ khoảng **0.301**, không phải 0.95. Đây là heuristic degree, không phải “spectral protection”. Nó có thể bỏ trung gian phổ biến nhưng hữu ích. BPE giới hạn degree của **seed graph** và dùng path weighting minh bạch thay vì gọi penalty này là bảo đảm loại hub noise.

Một bảng dự đoán cũ ghi Sports N@20 0.0532 và Baby 0.0478: so với GD5-v4 lần lượt chỉ **+2.90%** và **+3.69%**. Điểm Electronics N@10 0.0278 là **+6.92%** so với 0.0260 của v4. Bảng điều chỉnh 0.054/0.048 cho Sports/Baby cũng chỉ +4.45%/+4.12%. Các giá trị này vừa chưa đo, vừa không đồng loạt đáp ứng target trên 5% so với comparator yêu cầu.

## 4. Nghiên cứu nguồn sơ cấp và giới hạn chuyển giao

Tra cứu tập trung vào các claim quyết định kiến trúc: collaborative paths, graph refinement, relation sign và norm/efficiency. Không cộng phần trăm của các paper thành dự báo cho STAIR. Các tóm tắt sau cố ý ngắn; phần chứng minh BPE ở §7 là lập luận riêng cho operator được đặc tả ở đây.

| Nguồn đã kiểm tra | Nội dung liên quan được xác nhận | Hàm ý cho quyết định v5.2 |
|---|---|---|
| [STAIR, AAAI 2025](https://arxiv.org/html/2412.11729v1) | BSC sửa hướng cập nhật item qua semantic graph đối xứng; FSC xử lý UI graph. | Giữ ranh giới forward/optimizer; graph chỉ ở BSC không phải differentiable forward gate. |
| [Have We Really Understood Collaborative Information?, WSDM 2026](https://arxiv.org/html/2511.06905v2) | Nghiên cứu sequential; quy ước 0-hop là direct co-occurrence, 1-hop có một item trung gian. Khoảng 90% là thống kê quan hệ trên các benchmark của bài. | Không đồng nhất hop với UI graph; không suy ra “90% tín hiệu Amazon STAIR chưa được học”. |
| [GSPRec, v3/RecSys 2026](https://arxiv.org/html/2505.11552v3) | Có ordering-derived proximity, diffusion, bandpass/lowpass; abstract xác nhận trung bình +5.12% N@10. Bỏ bandpass làm biến thể dưới các GSP baseline. | Không dùng gain đó để dự báo BPE hoặc tuyên bố BSC thay thế bandpass tương đương. |
| [IIMRec, arXiv 2607.24607v1](https://arxiv.org/html/2607.24607v1) | NCER trên fused graph dùng common-neighbor/k; Theorem 5.2 có giả định clean/noise overlap và tỷ lệ Frobenius. Toàn model còn UI expansion/gating/objective. | Chỉ mượn giả thuyết structural redundancy; không gán toàn gain hay noise theorem cho path extension của STAIR. |
| [Edge Spectrum, arXiv 2608.29578v1](https://arxiv.org/html/2608.29578v1) | Sign mismatch xét choice-derived in-slate competitors; paper phân biệt rõ với co-click graphs. | Không kết luận shared-user Amazon edges nhất thiết là cạnh cạnh tranh có dấu âm. BSC update compatibility cần đo riêng. |
| [MLLMRec, v2](https://arxiv.org/html/2508.15304v2) | Refinement cùng MLLM-generated preferences. “12%” là tỷ lệ Baby edges similarity dưới 0.5 trong hình phân tích. | Không diễn giải là ground-truth noise 12% đã loại, hay như gain của riêng edge pruning. |
| [EVEN, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33358) | Joint structure evaluation/denoising cho multimodal recommendation. | Hướng đo task relevance đáng nghiên cứu; không đưa learned forward denoising vào BSC bằng một gate thiếu gradient path. |
| [MMGACL, KBS 2025](https://www.sciencedirect.com/science/article/abs/pii/S0950705125000826) | Modality complementation và modality-aware UI fusion trong missing-modality setting. | Setting khác; không chứng minh cosine guard giải quyết substitute/complement trong dữ liệu đầy đủ. |
| [MHRec, ESWA 2026](https://www.sciencedirect.com/science/article/abs/pii/S0957417426012248) | Publisher abstract mô tả modality-aware hypergraph edge diffusion. | Xác nhận hướng nhưng không audit đủ toàn method/code; không ghi thành paper 2025 đã tái lập hoặc chi phí tương đương STAIR. |
| [Resource allocation, Zhou et al., 2009](https://arxiv.org/abs/0901.0553) | Local link prediction với common-neighbor/degree weighting. | Nguồn ý tưởng RA; weighted confidence và budgets BPE là lựa chọn thiết kế, không định lý ranking. |
| [Graph Diffusion Convolution, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html) | Preprocessing diffusion/sparsification để đổi graph propagation. | Diffusion preprocessing có tiền lệ; BPE khác ở support CF và optimizer channel, không tuyên bố phát minh graph diffusion. |

**Audit code availability:** [CI_Investigation](https://github.com/Zhang-xiaokun/CI_Investigation) và [Edge-Spectrum-in-CF](https://github.com/kyomusso/Edge-Spectrum-in-CF) có repository đọc được. Không chạy tái lập benchmark của chúng trong lần review này. Link IIMRec mà paper nêu, `github.com/Jinfeng-Xu/IIMRec`, trả 404 khi truy cập công khai; đây là giới hạn truy cập, không chứng minh repository không tồn tại/private. Chưa xác nhận author repo GSPRec từ full text đã đọc. Không ghi “đã audit implementation” cho các code không truy cập được.

GAL-Rec được nhận diện qua [arXiv 2406.13235](https://arxiv.org/abs/2406.13235), thuộc hướng language-model recommendation; tồn tại paper không tự xác nhận toàn metadata TKDE 2025 trong bản gửi. Các tên FGCM, PEARL, AdaMixNorm, SSAGF và GraphBooster thiếu định danh/nguồn sơ cấp đủ rõ trong tài liệu hoặc lượt tra cứu tập trung này. Giữ chúng là citation gaps; không kết luận giả, cũng không dùng headline gain chưa đối soát làm tiền đề thiết kế.

### 4.1. Kiểm tra riêng phép suy luận NCER

Điều kiện “mean consistency của signal lớn hơn noise” không đủ cho mọi bất đẳng thức norm khi reweighting bình phương. Ví dụ đại số với năng lượng signal/noise ban đầu bằng nhau: noise consistency `(0,1)` có mean 0.5; signal đều 0.51. Với multiplier `1+c`, noise amplification theo RMS là \(\sqrt{2.5}\), lớn hơn 1.51 của signal. Noise fraction có thể tăng từ 0.7071 lên 0.7232 mặc dù mean signal cao hơn.

Đây là cảnh báo về việc dùng trung bình thay cho moment bậc hai, không phải một benchmark refutation của IIMRec. Appendix của paper cũng có variance/remainder term. Để có theorem áp dụng cho STAIR cần các giả định và đại lượng đúng, hoặc một kiểm chứng riêng; không bỏ variance rồi gọi overlap là certificate khử nhiễu. BPE không dùng theorem này làm bảo đảm chính.

## 5. Tín hiệu mới nằm ở đâu — và giới hạn quan trọng

### 5.1. Hai graph, hai quy ước đường đi

Đặt R là binary train UI matrix. \(C=R^TR\) đo số user cùng tương tác. V4 đã sử dụng C để xây W1. Do đó:

- `item → user → item`: direct CF evidence của v4; không phải đóng góp mới của v5.2.
- `item → item k → item`: hai cạnh trong seed II graph; là loại path được BPE xét.
- W1 bình phương chứa path hai bước, nhưng không đồng nghĩa nên materialize toàn bộ \(W_1^2\).

BSC v4 vốn có polynomial \(P(S_4)\), L=3. Các hạng \(S_4^2\) đã chứa CF-CF, semantic-CF và CF-semantic paths. Vì vậy BPE không tạo nguồn thông tin ngoài train và features; nó **chọn, nén, thay trọng số đường truyền** trong polynomial optimizer. Nếu không pruning/normalization và chỉ thay S bằng aS+bS² thì đây vẫn là một polynomial của operator cũ. Control về radius BSC và direct expansion là bắt buộc để phân biệt đóng góp.

### 5.2. Không thể “cứu mọi isolated node” bằng bình phương CF

Nếu row i của W1 bằng 0 thì row i của \(W_1^2\) cũng bằng 0. Các đường đi không nối hai connected components khác nhau của seed CF. BPE không thể phục hồi CF evidence cho node không có đường trong seed, cũng không khẳng định làm giảm CF-isolated fraction.

Metadata v4/v5.1: Baby có **2.448/7.050 CF-isolated nodes (34.72%)**; Sports **8.262/18.357 (45.01%)**. Phần này vẫn dựa semantic branch như v4. Một hỗ trợ riêng dùng weak direct co-occurrence hoặc mixed semantic-CF paths có thể nghiên cứu sau, nhưng cần giả thuyết và controls mới; không âm thầm đưa vào primary hiện tại.

### 5.3. Offline feasibility đã thực hiện trên Baby

Dùng binary train Baby hiện có, gọi builder W1 v4 nguyên trạng trên CPU. Không training, không đọc valid/test để build graph. Kết quả:

| Đại lượng | Kết quả |
|---|---:|
| W1 entries có hướng, không self-loop | 23.054 |
| Max degree W1 sau max-union | 185 |
| Mutual seed K=10: entries / max degree | 14.560 / 10 |
| Pure II two-step candidates ngoài W1, m≥1 | 64.466 entries |
| Candidates ngoài W1, m≥2 | 1.510 entries |
| Nodes có candidate m≥2 | 678, khoảng 9.62% catalog |
| Trial m≥1, confidence shrinkage, mutual k_add=3 | 4.906 added entries, 2.695 endpoint nodes |
| Trial added entries / W1 entries | 21.28% |

Trial đầu dùng row mass cap ν=0.2 và kiểm tra budget pass. Trial thứ hai thêm median scale calibration (§6.5), ν=0.5: κ≈110.16; trên 2.695 active endpoints, median added/base mass≈0.2604, P90/max=0.5. Row budget vẫn pass. **Chưa loại support S0** vì không có run-specific semantic graph cache trong phép thử này; các số candidate/endpoint là upper bounds cho pipeline primary có semantic exclusion. Trial cũng chưa phải kết quả của operator S5.2 cuối cùng hoặc một phép đo GPU. Seed lấy từ W1 hiện hữu, không tăng direct CF k của v4.

Phát hiện này khiến điều kiện m≥2 không phù hợp làm mặc định nếu mục đích là coverage rộng. Primary giữ m≥1 nhưng dùng confidence shrinkage, mutual selection, count/weight budgets. “Nhiều đường” không đồng nghĩa “nhiều bằng chứng độc lập”: chúng có thể cùng xuất phát từ một nhóm user hoặc một hub.

## 6. Đặc tả thuật toán NLGCL-BPE

### 6.1. Các object bất biến của v4

Giữ train pairs và deduplication chuẩn v4, W1 có c_min=2, k_cf=5, t_shrinkage=5. Với \(c_{ij}\) binary co-occurrence, \(n_i\) binary item interaction degree:

\[
q_{ij}=\frac{c_{ij}}{c_{ij}+5}\frac{c_{ij}}{\sqrt{n_in_j}}.
\]

Builder giữ row top-5 rồi max-union hai chiều. Không giả định degree cuối bằng 5. S0 là **baseline normalized semantic graph nguyên trạng**, cùng MI whitening, modal kNN, feature files và normalization. Không denoise S0 trong primary.

### 6.2. Seed có giới hạn bậc để kiểm soát preprocessing

Từ W1, mỗi node lấy top `K_seed=10` theo raw q, tie-break bằng item ID tăng. Chỉ giữ cạnh được chọn **ở cả hai đầu** (mutual). Gọi seed này T:

\[
T_{ij}=W_{1,ij}\,\mathbf1[j\in\operatorname{Top}_{10}(i)]\,
\mathbf1[i\in\operatorname{Top}_{10}(j)].
\]

T đối xứng, không âm, không diagonal, degree-count \(d_k^T\le10\). Mutual policy chỉ dùng để tìm path; **không thay W1** đang được retain trong final CF graph. Nó có thể bỏ nhiều bridge quan trọng; log số component/isolates tăng trong seed và coi đó là hạn chế.

### 6.3. Score path, không dùng eigenvalue của cạnh

Với hai node có chung intermediate k trong T:

\[
m_{ij}=\sum_k\mathbf1[T_{ik}>0]\mathbf1[T_{kj}>0],\qquad
p_{ij}=\sum_{k:d_k^T>0}\frac{T_{ik}T_{kj}}{d_k^T},
\]

\[
a_{ij}=\frac{m_{ij}}{m_{ij}+t_{path}}p_{ij},\qquad t_{path}=1.
\]

Denominator là **count degree trong T**, không user degree của item, không weighted degree, không P95 và không spectral radius. Đây là weighted RA-inspired heuristic. Score đối xứng và không âm; confidence term giảm mức tin cậy single-intermediate mà vẫn cho nó cơ hội. Không diễn giải a là xác suất cạnh đúng.

Primary candidate set:

\[
\mathcal C=\{(i,j):i\ne j,\ m_{ij}\ge1,\ W_{1,ij}=0,\ S_{0,ij}=0\}.
\]

Exclusion S0 làm primary xét cạnh mới đối với cả **union support v4**, tránh gọi tăng trọng số cạnh semantic sẵn có là mở rộng support. Cho phép overlap S0 chỉ là control riêng có tên rõ ràng. Không loại cặp chỉ vì chúng có **một** train shared user; c_min=2 của W1 có thể đã pruning chúng. Log tỷ lệ candidate direct co-occurrence 0/1/≥2 để phân biệt khôi phục direct evidence đã pruning với quan hệ gián tiếp thực sự.

Không dùng matrix N×N dense. Enumerate bounded wedges `i-k-j` theo block; tích lũy score và số distinct k, loại diagonal/existing support bằng sorted CSR lookup. Dùng float64 để cộng score preprocessing, int64 cho IDs/counts, rồi cast FP32 khi finalize graph. Không truncate candidates trước khi cộng đủ mọi intermediate trong seed.

### 6.4. Ngân sách số cạnh và lựa chọn hai chiều

Mỗi row candidate lấy top `k_add=3`, tie `(score descending, item ID ascending)`. Giữ **mutual** row selection. Gọi raw candidate matrix sau bước này Q. Chỉ xét cặp vô hướng i<j, sort `(-a_ij,i,j)` và giữ tối đa:

\[
B_{pairs}=\left\lfloor\frac{\beta_{edges}\,\operatorname{nnz}(W_1)}{2}\right\rfloor,
\qquad \beta_{edges}=0.25.
\]

Sau đó lưu cả hai chiều. Do vậy `nnz(Q) ≤ 2 B_pairs ≤ 0.25 nnz(W1)` và mỗi node có thêm tối đa 3 láng giềng. Tất cả `nnz` ở đây đếm entries có hướng; `pairs` đếm cạnh vô hướng. W1 không chứa diagonal. Nếu primary candidates rỗng, không tự nới threshold hoặc chuyển arm: chạy reference và báo expansion inactive.

Global budget có thể thiên về head nodes; phải báo theo binary train-degree deciles. Không gọi mutual top-k là capacity-neutral: nó thay cấu trúc được chọn. Placebo và direct-support controls đo chính phần này.

### 6.5. Ngân sách khối lượng theo node

Path score có scale khác q của W1: tích hai q thường nhỏ hơn một q, nên cộng trực tiếp có thể tạo arm gần như không can thiệp. Hiệu chỉnh scale bằng **train graph**, sau candidate selection, trước mass limiter:

\[
\kappa=\frac{\operatorname{median}\{W_{1,ij}:W_{1,ij}>0\}}
{\operatorname{median}\{Q_{ij}:Q_{ij}>0\}},\qquad \widehat Q=\kappa Q.
\]

Median ở đây lấy **values** dương, không lấy boolean predicates; mỗi entry hai chiều được tính cùng cách ở numerator và denominator. Empty Q delegate v4. Tính bằng float64, yêu cầu finite positive medians/scale; không chia epsilon rồi silently chấp nhận graph lỗi. Đây là scale calibration heuristic, không phải learned edge utility. Global positive scale không đổi candidate ranking; nó tránh việc đơn vị score quyết định mức can thiệp ngoài ý muốn. Đặt:

\[
d_i=\sum_jW_{1,ij},\quad r_i=\sum_j\widehat Q_{ij},\quad
u_i=\begin{cases}\min(1,\nu d_i/r_i),&r_i>0,\\1,&r_i=0.\end{cases}
\]

\[
(W_{add})_{ij}=\widehat Q_{ij}\min(u_i,u_j).
\]

Primary **ν=0.5**: added raw mass mỗi node không quá 50% mass W1. Conservative arm ν=0.2, cùng candidates. Min endpoint multiplier bảo toàn symmetry và giới hạn mass ở **cả hai đầu**. Nó không phải v5 endpoint gate trên normalized operators; graph sẽ được chuẩn hóa lại sau mixing raw weights.

Độ mạnh thực tế vẫn có thể **thấp hơn** budget vì ít candidates và min endpoint caps. Log medians, κ, `added_mass/base_mass` distribution và tỷ lệ cap active; nếu hầu hết gần 0, không gọi arm là “50% intervention”. Calibration không chứng minh paths đáng tin; phải có no-calibration control khi nghiên cứu contribution weighting. Không rescale bằng validation để ép sử dụng hết budget.

### 6.6. Normalize một lần và merge sparse operator

\[
W_\star=W_1+W_{add},\quad d_i^\star=\sum_jW_{\star,ij},
\]

\[
S_\star=D_\star^{-1/2}W_\star D_\star^{-1/2}
+\operatorname{diag}(\mathbf1[d_i^\star=0]).
\]

Inverse sqrt bằng 0 tại degree 0. Không thêm identity cho mọi node. Giữ **η=0.1** như v4, merge `0.9*S0 + 0.1*S_star` thành **một CSR graph** trên GPU. Không lưu T,Q,path matrices trên GPU sau prepare. Không dùng row stochastic normalization rồi giả định singular norm≤1; symmetric degree normalization mới là certificate ở đây.

### 6.7. Training và objective

FSC vẫn dùng normalized **train UI adjacency gốc**, cùng beta schedule và L=3. NLGCL dùng đúng hai hướng heterogeneous của v4:

- User-side: query `I^(g+1)[positive item]`, keys `U^g[batch users]`.
- Item-side: query `U^(g+1)[user]`, keys `I^g[batch positive items]`.

Giữ normalization, denominator, duplicate semantics và reduction của `models/stair5_v4_objectives.py`; không đảo thành user-ego→item-neighbor hoặc dedup keys bằng sự suy đoán. Primary objective:

\[
\mathcal L=\mathcal L_{BPR}+0.01\mathcal L_{NLGCL},
\quad \tau=0.2,\ \alpha=0.5,\ G=1.
\]

Optimizer/smoother nhận **Adam-normalized update direction**, không riêng raw BPR gradient. Gradient từ BPR và NLGCL đều đóng góp hướng item trước BSC. User embedding group `smoother=None`; item group dùng S5.2; mọi trainable parameter thuộc đúng một group. Không thêm regularization loss ngoài baseline hoặc đổi weight decay khi mô tả “giữ BPR”. LR, weight decay, batch size, epochs và các hyperparameter còn lại kế thừa từng YAML v4.

BSC coordinate j giữ chính xác công thức hiện tại:

\[
P_j(S)V_{:,j}=\frac{1-b_j}{1-b_j^{L+1}}
\sum_{\ell=0}^{L}b_j^\ell S^\ell V_{:,j},\quad b_j=\texttt{beta3[j]},\quad L=3.
\]

Không đổi `beta3` thành `1-beta3` ở smoother do suy luận từ tên ký hiệu paper. FSC có convention riêng; source v4 là reference cho baseline recovery. Graph static không cần snapshot động hoặc refresh mỗi K epoch; `prepare()` dưới no_grad, không giữ tensor có grad_fn.

## 7. Cơ sở toán học và giới hạn của bảo đảm

### 7.1. Row mass, preservation và attenuation

Vì min(u_i,u_j)≤u_i:

\[
\sum_jW_{add,ij}\le u_ir_i\le\nu d_i.
\]

Do đó \(d_i\le d_i^\star\le(1+\nu)d_i\) với d_i>0. Raw direct CF edges không bị xóa hoặc giảm. **Normalized** direct edge có thể giảm do degree tăng:

\[
\frac{(S_\star)_{ij}}{(S_1)_{ij}}
=\sqrt{\frac{d_id_j}{d_i^\star d_j^\star}}
\in\left[\frac1{1+\nu},1\right]\quad(W_{1,ij}>0).
\]

ν=0.5 bảo đảm multiplier≥2/3 trên old CF edges; ν=0.2 cho ≥5/6. Vì vậy không tuyên bố giữ nguyên CF row. **Semantic term `0.9*S0` giữ nguyên từng entry** trong primary, nhưng total operator và update có thay đổi. Rows không có added edge cũng có thể đổi do neighbor degree; không quảng cáo “node không can thiệp luôn giống v4”.

### 7.2. Operator contraction

W★ đối xứng, không âm. Trên active nodes, S★ similar với random-walk matrix có eigenvalues trong [−1,1]; S★ đối xứng nên singular norm là max absolute eigenvalue≤1. Isolated identity block cũng có norm 1. S0 có cùng symmetric normalized nonnegative reference contract. Vì thế:

\[
\|S_{5.2}\|_2\le0.9\|S_0\|_2+0.1\|S_\star\|_2\le1.
\]

Chứng minh không áp dụng cho graph directed chưa symmetrize, signed weights, normalization degree trước pruning, hay gate arbitrary trên normalized edges. Norm≤1 không có nghĩa positive-semidefinite: adjacency có thể có eigenvalues âm. Không cần power iteration để scale toàn graph.

### 7.3. Polynomial BSC bounded

Coefficients của P_j không âm và tổng bằng 1 với b_j∈[0,1). Vì ∥S5.2∥≤1 nên ∥P_j(S5.2)∥≤1. Eigen-response:

\[
p_j(\lambda)=\frac{1-b_j}{1-b_j^{L+1}}
\frac{1-(b_j\lambda)^{L+1}}{1-b_j\lambda}>0
\quad(\lambda\in[-1,1]).
\]

Đây là tính bounded và positive response của **smoother với operator static**. Không phải chứng minh hội tụ global của AdamWSEvo trên nonconvex BPR+NLGCL, không chứng minh energy preservation, và không suy ra Recall/NDCG tăng.

### 7.4. Reference recovery

`expansion_enabled=False`, ν=0, β_edges=0 hoặc Q rỗng đều phải delegate exact v4 graph path; S5.2=S4, không rebuild một bản mathematically-equal với rounding khác rồi gọi bitwise recovery. η=0 riêng biệt khôi phục N0 semantic-only khi NLGCL bật. λ_NLGCL=0, η=0 khôi phục B0. Hai recovery này khác comparator v4.

### 7.5. Update compatibility vẫn phải đo

Cosine modality và common-neighbor không xác lập hai item nên có cùng Adam update direction. Theo bottleneck `EStair.md`, diagnostic thưa nên đo cosine giữa unsmoothed item directions V_i,V_j trên sampled added edges và matched direct edges; đo weighted fraction cosine âm, head/tail distribution, mức đổi direction sau BSC. Direction lấy trước smoother, cùng step, detach; không gọi đây là causal edge utility hay dùng test metric để chọn edge.

Nếu added edges chủ yếu nối updates ngược chiều và validation suy giảm, đó là evidence chống giả thuyết BPE. Không tự lật dấu graph, vì signed graph cần lý thuyết và degree policy khác. Static CF expansion chỉ giải quyết một bottleneck support; modality-coordinate dependence và exposure confounding vẫn còn.

## 8. Tốc độ, bộ nhớ và GPU trên dataset lớn

### 8.1. Không để preprocessing bùng nổ

Max-union W1 không có bound degree=k_cf; Baby thực tế max 185. Nếu enumerate trực tiếp W1², số wedge events liên quan \(\sum_k(d_k^{W1})^2\), có thể lớn. Mutual seed đảm bảo:

\[
\#\text{ordered non-self wedge events}
=\sum_kd_k^T(d_k^T-1)\le N K_{seed}(K_{seed}-1).
\]

K_seed=10, N=63.001 cho Electronics: upper bound **5.670.090 ordered events**, không phải 5.67M retained edges. Enumerate theo row block; mỗi row có tối đa K_seed(K_seed−1) path events trước dedup. Không xây W1²/full BBᵀ dense và không làm all-pairs feature cosine cho guard.

W1 builder v4 vẫn dùng blocked sparse RᵀR. Phép giới hạn path không giải quyết chi phí gốc của exact co-occurrence hoặc MI/SVD/kNN; các cache đó phải giữ và đo riêng. Count product workspace budget không bao gồm toàn bộ R/CSR/Python allocator overhead.

### 8.2. Online budget

Một CSR operator merged; số SpMM ở BSC vẫn L=3, FSC vẫn như reference. Vì candidates primary loại cả support S0/W1:

\[
\operatorname{nnz}(S_{5.2})\le\operatorname{nnz}(S_4)+
\beta_{edges}\operatorname{nnz}(W_1).
\]

Không có relation MLP, dynamic graph rebuild hoặc KPE mask ở primary. Với nnz trong manifest v4 Baby/Sports, count budget này cho upper bound tăng operator entries khoảng **6.85%/5.25%**, trước khi xét thiếu candidates. Đây là bound entries, **không phải bound wall time**: SpMM phụ thuộc cache locality, bandwidth, degree và kernel. Với values FP32, col IDs int64, thêm e directed entries khoảng 12e bytes trong CSR; COO hoặc conversion workspaces tốn thêm. Đừng quảng cáo zero additional VRAM hoặc 1ms không đo.

Float32 CSR SpMM giữ reference semantics; không tự bật AMP/TF32 trên preprocessing kNN hoặc training chỉ để nhanh hơn. Nếu benchmark AMP/TF32 phải có arm riêng và kiểm tra ranking drift. Tránh `.item()`/CPU transfer trong hot path, giữ graph buffers trên device và không log từng batch full graph. Diagnostic direction sampling thưa, ghi riêng thời gian; tắt trong throughput benchmark chính.

### 8.3. Resource gates đăng ký trước

- Cold preprocessing: ghi wall time/RSS theo từng stage và peak GPU nếu dùng GPU kNN; warm-cache measurement ghi riêng.
- Training: benchmark cùng T4, versions, batch size, eval frequency; warmup ít nhất 20 steps, 100+ steps measured với CUDA synchronization ở boundaries. Không synchronize từng op trong run thường.
- Target fit overhead≤20%, training allocated peak≤1.2×v4, process NVML peak trong giới hạn Kaggle. Các ngưỡng là mục tiêu nghiệm thu, không guarantee.
- Nếu throughput overhead>20%: giảm β_edges từ 0.25 xuống 0.10 theo resource fallback đăng ký trước, rebuild có arm ID mới. Không âm thầm đổi batch size/eval frequency để tạo bảng thời gian tốt hơn.
- Electronics chỉ chạy full sau topology/resource preflight và evidence chọn arm trên Baby/Sports. Không cần chạy v5.1 Electronics để “hoàn thành parity” khi không có lợi ích đo và overhead đã tăng.

## 9. Thiết kế thực nghiệm phân biệt nguyên nhân

### 9.1. Controls bắt buộc và controls điều kiện

| Arm | Thay đổi duy nhất/ý nghĩa | Ưu tiên |
|---|---|---|
| C-V4 | Original GD5-v4 N-CSE: S4, NLGCL gốc | Bắt buộc |
| C-KPE | v5.1 trên đúng graph v4 | Sử dụng evidence hiện có; thêm seeds nếu tuyên bố equivalence |
| P-BPE | Primary BPE ν=0.5, β=.25 | Bắt buộc |
| P-BPE-low | Cùng candidates, ν=.2 | Conservative strength control |
| C-Direct | Thêm CF edges từ co-occurrence trực tiếp ngoài top-5, cùng c_min/shrinkage, cùng edge/mass caps | Bắt buộc cho đóng góp path |
| C-Radius | V4, tăng **BSC** L từ 3 lên 4; FSC vẫn 3, objective giữ nguyên | Phân biệt path compression với tăng propagation radius; chi phí báo riêng |
| C-Placebo | Thay added endpoints bằng random/degree-stratified feasible edges, cùng counts và mass-budget policy | Kiểm tra ý nghĩa structural score |
| C-M2 | m≥2 thay m≥1, giữ các budget còn lại | Khi path coverage đủ; có thể ít cạnh hơn, phải ghi rõ |
| C-NoScale | BPE với κ=1, còn lại giống primary | Chẩn đoán scale weighting; báo realized mass khác nhau |
| C-Overlap | Cho candidate overlap S0 nhưng không overlap W1 | Tách new support với semantic-edge reinforcement |
| C-SemGuard | Same pipeline; raw within-modal cosine guard | Điều kiện, không primary |

Direct control tái dùng blocked RᵀR computation và lấy danh sách reserve top candidates ngoài W1/S0. Không đổi c_min hoặc tạo weak-user evidence rồi gọi “cùng dữ liệu weighting”. Apply same mutual selection, budget, median scale calibration và mass limiter. Có thể không đạt đúng count bằng BPE nếu pool thiếu; phải công khai realized count và so sánh matched subbudget, không claim exact capacity match khi chưa làm được.

Placebo không mặc nhiên bảo toàn weighted degree từng node. Degree-stratified permutation/rejection cần log realized counts, binary/weighted degree differences và tỷ lệ rewired; exact degree-preserving swap chỉ được ghi nếu thật sự enforce cả constraints. Random cạnh vi phạm existing-support exclusions cần resample hoặc báo không khả thi.

Semantic guard điều kiện dùng features **raw cùng modality** đã row-normalize, score `(cos_text(i,j)+cos_image(i,j))/2`; không whiten lại MI. Chọn threshold bằng train candidate distribution đăng ký trước hoặc nested validation arm rõ ràng. Không coi threshold 0.25/0.35 là universal và không gán relation labels từ cosine. Không thêm alignment network. Candidate-only dot products chunked một lần offline; báo thời gian/RSS thêm.

NCER không nằm trong bảng primary. Nếu sau này thử, định nghĩa B là graph nào, degree bound và score normalization trước; áp dụng trên candidate-restricted pairs, không tạo unrestricted BBᵀ. Common-middle m đã có trong confidence nên NCER cùng nguồn có thể chỉ double-count. Phải có control matched-degree/common-path count và không viện dẫn theorem chưa thỏa giả định.

### 9.2. Data roles và tránh leakage

Train: mọi co-occurrence, seed, candidate, budgets, degree bins, feature hashes và diagnostics. Validation: chọn checkpoint/arm theo cùng NDCG@20, eval cadence như comparator. Test: đánh giá cuối cấu hình khóa; không dùng test growth để chọn architecture/hyperparameter.

Không train edge MLP bằng validation ranking rồi dùng cùng validation để chọn checkpoint mà không khai báo bilevel/auxiliary split. Primary không cần gate hoặc label temporal/substitution. Các side features của item vẫn ở transductive catalog như baseline; không claim cold-start generalization nếu chưa có held-out-item protocol.

Seen-item masking, full/pool ranking, aggregation Recall/NDCG, metric cutoffs và tie behavior dùng **cùng evaluator baseline/v4**. Chỉ so metric cùng split/mode. Test metric của nhiều checkpoint trong FreeRec log nếu có không được dùng để reselect checkpoint.

### 9.3. Lộ trình có giới hạn ngân sách

1. **Gate 0 — CPU/math:** tests §11, hash/shape contract, candidate count, realized mass, topology components. Nếu no support hoặc mass gần 0, ghi inactive/weak intervention; không hứa gain.
2. **Gate 1 — hệ thống:** same-seed v4/BPE ngắn trên Baby/Sports và resource preflight Electronics. Không kết luận ranking từ vài epoch.
3. **Gate 2 — Baby diagnosis:** tối đa bốn full-run seed1 arms ban đầu: C-V4, P-BPE, C-Direct, C-Radius; ν=.2 chỉ thêm nếu primary có dấu hiệu over-intervention. Chỉ chọn bằng validation. Save nguyên artifacts, không vừa chạy vừa thay score.
4. **Gate 3 — Sports confirmation:** comparator và arm thắng Baby, cộng C-Direct nếu path contribution chưa rõ. Không gọi pilot 100 epoch không tăng là definitive NO-GO vì reference peaks gần epoch 480–500. Có thể dừng vì NaN/resource failure hoặc regression validation lớn lặp lại theo tiêu chí đăng ký, nhưng ghi đó là screening.
5. **Gate 4 — Electronics:** arm khóa + matched C-V4 mới, cùng epochs/budget. Nếu cả Baby/Sports không có validation gain, dừng mở rộng full Electronics cho kiến trúc này và báo negative result; không thêm hàng loạt module để cứu test.
6. **Gate 5 — robustness:** ít nhất ba seeds ghép `(1,2,3)` cho comparator/final arm trên dataset được claim; mở rộng năm seeds khi có ngân sách. Single-seed screening không nằm trong bảng final significance.

Ngân sách full runs là đáng kể vì v4 Electronics fit khoảng 347.76 phút/seed trong raw log. Ghi total GPU-hours của cả tuning và reporting, không chỉ run tốt nhất. Không cần làm mọi optional arm trên cả ba datasets nếu chỉ muốn đánh giá primary; phải làm control liên quan trước khi claim cơ chế của nó.

### 9.4. Success criteria, không có bảng “kết quả dự kiến”

\[
\operatorname{Growth}(m,d)=100\frac{m_{5.2,d}-m_{v4,d}}{m_{v4,d}}.
\]

Ngưỡng minh họa **bằng đúng +5%** từ v4 seed1 đã làm tròn; để đạt **trên** 5% phải lớn hơn ngưỡng. Final assessment dùng unrounded matched-seed results, không bảng rounded này.

| Dataset | R@10 > | R@20 > | N@10 > | N@20 > |
|---|---:|---:|---:|---:|
| Baby | 0.071190 | 0.110880 | 0.038220 | 0.048405 |
| Sports | 0.080325 | 0.120015 | 0.043995 | 0.054285 |
| Electronics | 0.048090 | 0.071190 | 0.027300 | 0.033180 |

Primary research goal: N@20 tăng>5% trên cả ba datasets. Secondary ambitious goal: ít nhất 3/4 metrics tăng>5% trên **mỗi** dataset. Intermediate useful outcome: N@20 tăng nhất quán nhưng chưa đạt5%; phải gọi đúng partial gain. Không dataset nào regression>2% là robustness target; efficiency phải đạt §8.3. Các tiêu chí không thay đổi sau khi nhìn test.

Báo mean/std và paired seed differences. Bootstrap user-level NDCG differences trên cùng evaluation users trả lời uncertainty của một checkpoint; không thay thế variance giữa training seeds. Với ít seeds, ghi CI rộng và phương pháp tính; không viện dẫn p-value chưa tính. Không claim all-dataset success nếu Electronics chưa chạy hoặc chỉ dùng metric của v4.

## 10. Kế hoạch tổ chức mã nguồn — chưa triển khai trong yêu cầu này

| File dự kiến | Nội dung và trách nhiệm |
|---|---|
| `models/stair5_v52_graph.py` | Reuse W1 v4; bounded mutual seed, blocked wedge accumulator, existing-support exclusion, deterministic mutual selection/global pair budget, median scale calibration, mass limiter, symmetric normalization; trả state/metadata/hashes. |
| `models/stair5_v52_utils.py` | Canonical config/cache key, artifact validation, atomic save, CSR finite/symmetry/count checks, fingerprint. Reuse utilities đủ contract, tránh copy đổi hành vi. |
| `models/stair5_v52.py` | Thin v4-derived model giữ MI/FSC/scorer/train sampler/NLGCL; thay graph prepare bằng BPE; expose arm metadata, exact off path, item-only smoother groups. Không thêm learned parameters. |
| `optimizers/stair5_v52_smoother.py` | Thin named adapter hoặc reuse v4 static callback smoother; beta/L và recurrence nguyên trạng, no_grad/no parameters. |
| `main_stair5_v52.py` | Engine/evaluator/checkpoint selection cùng v4; CLI architecture contract, seed, source/data/config hashes, cold/warm cache, telemetry/raw selected metrics. |
| `configs/Amazon2014{Baby,Sports,Electronics}_STAIR5_v52.yaml` | Kế thừa hyperparameters từng dataset v4; explicit primary defaults §6 và named arms. Không silently short-run50/100 thay full epochs. |
| `tests/test_stair5_v52_graph.py` | Algebra/sparse/toy graph, exact counts/mass, deterministic support, isolated recovery, budget and overflow validations. |
| `tests/test_stair5_v52_pipeline.py` | Reference recovery, identical parameter membership, real backward/update/checkpoint/resume/config contract, evaluator agreement. |
| `notebook/P5/stair5_v52.ipynb` | Fresh subprocess per dataset, selectable Baby/Sports/Electronics, fail-fast training return code; match CLI; plot imports self-contained, manifest/metric export; no forced reset losing uncommitted work. |

Cache key tối thiểu: version, W1 fingerprint, S0 fingerprint, train hash/shape, every path/budget parameter, score/calibration formula IDs, tie policy, normalization/isolation policy, preprocessing source hash và dtype. Guard arm thêm features/missing-feature policy hashes. Invalid cache phải rebuild hoặc fail rõ; không reuse cache khác arm vì dataset name giống nhau.

Checkpoint contract gồm graph fingerprint/config/data/source hashes và optimizer state. Resume mismatch phải báo lỗi; không resume weights v4 với optimizer state từ một graph khác và gọi đó là fair scratch run. Warm-start là arm riêng. `prepare()` cache tensors static/no grad_fn; dynamic embeddings chỉ nằm trong training computation như v4.

Telemetry thêm: input W1 degree quantiles/max; seed retention; candidate count m bins; direct co-occurrence bins; excluded S0/W1/self counts; final pairs/nnz and endpoint coverage; nodes/components changed; score medians/κ; `added_mass/base_mass` quantiles; cap-active fraction; semantic-vs-CF provenance; actual operator nnz; cold/warm time; measured fit/process times và peak allocated/reserved/NVML riêng. Final evaluator export JSON metrics unrounded **sau load selected checkpoint** với split, epoch và checkpoint hash.

## 11. Verification và phép thử bác bỏ

### 11.1. Unit/integration tests cần trước training

1. Dedup train interactions; correct integer co-occurrence; no valid/test edges; IDs ngoài shape/overflow bị từ chối.
2. Chain/diamond toy: count distinct intermediate đúng, diagonal/existing support bị loại; m=1 và m=2 khác nhau đúng, không count duplicate wedge như độc lập.
3. Max-union hub toy chứng minh k row-topk không bound symmetric degree; mutual seed bound≤K_seed.
4. Sparse accumulator đối chiếu dense T D_count^-1 T và binary T² trên toy; block sizes khác cho cùng support/score tolerance.
5. Mutual k_add/global pair budget bảo toàn symmetry, no diagonal và no-existing support; tie-ID order deterministic.
6. Scale calibration/mass limiter: medians đúng, global scaling giữ score ordering; mọi row added_mass≤ν base_mass, d_new bounds; empty/d_zero paths không NaN; invalid ν/β/K và nonfinite scale bị từ chối.
7. Normalization: eigenvalue/norm trên toy≤1; matrix-free recurrence trùng dense polynomial; no assumption PSD của adjacency.
8. Off/ν0/β0/empty candidates: exact v4 delegation, embeddings/loss/gradients/updates/optimizer states giống trên cùng batch và RNG.
9. Item-only smoother; group uniqueness/completeness; backward và multiple optimizer steps đúng; graph buffers không grad_fn, no extra learned params.
10. Checkpoint atomic save/load/resume roundtrip; graph/config mismatch detection; selected-checkpoint JSON đúng; full/pool evaluator cùng model embeddings cho cùng metrics.

Không viết một test chỉ kiểm tra tensor shape rồi gọi rigorous equivalence. Các tests phải có reference độc lập hoặc phản ví dụ. Bitwise requirements chỉ cho delegated path/cùng deterministic environment; reordered sparse sums so tolerance và báo kernel limitations.

### 11.2. Những gì đã kiểm tra trong lần thiết kế này

Script **`scratch/g5v52_review/audit_design.py`** và receipt **`design_receipt.json`** chứa read-only raw artifact audit, CPU toy checks và Baby train-only topology feasibility. Đã chạy thành công:

- Row mass budget, normalization norm trên toy (1.0), off recovery và CF-isolated square-zero identity.
- Hard-positive softmax mass counterexample và NCER average-only amplification counterexample.
- V4/v5.1 matching train/graph hashes; fit overhead; training peaks; Electronics evidence missing.
- Baby bounded seed/candidate counts và tentative no-semantic-exclusion path budget trial.

Các phép thử này không phải unit suite của v5.2 production, không chứng minh ranking improvement và không benchmark GPU. Chưa triển khai các file §10, chưa chạy Baby/Sports/Electronics v5.2.

### 11.3. Điều gì sẽ khiến phải bỏ hoặc sửa kiến trúc?

- Sau exclusion S0, candidates/mass quá ít: graph mới gần v4; báo weak intervention và cân nhắc direct-support control, không hứa +5%.
- C-Direct bằng hoặc tốt hơn BPE: lợi ích có thể là sparsification budget, không bằng chứng path scoring tốt hơn.
- C-Radius cho lợi ích tương đương với cost chấp nhận được: compression chưa có đóng góp riêng; cần báo efficiency/capacity đúng.
- Added direction alignment âm và ranking giảm: structural transitivity không tương thích update; bỏ giả thuyết phổ quát “co-occurrence kéo gần luôn đúng”.
- Head metrics tăng nhưng tail giảm: global ranking mean che tradeoff; không gọi cải thiện tất cả nhóm.
- BPE hiệu quả Baby nhưng không Sports/Electronics: không chuyển gain xuyên dataset bằng lý luận độ thưa; báo dataset dependence.
- Một arm thắng một seed rồi thua các seeds khác hoặc chỉ thắng do chọn nhiều configs: chưa có robust improvement.
- Need forward learned graph để tiến xa: đó là nghiên cứu tiếp theo với autograd/hypergradient và efficiency riêng; không gọi là sửa nhỏ của primary BPE.

## 12. Methodology-focus review và trạng thái học thuật

Thực hiện quy trình `academic-research-suite`: hai vai trò methodology và editorial lập Phase1 commitments trước khi đọc tài liệu; Phase2 review độc lập của **ba bản đề xuất và rationale v5.1 được gửi**, không phải phê duyệt bản thiết kế BPE này. D1 methodology và D2 exposition đều đánh dấu **repairable/block**, synthesis `major_revision` cho material gốc. Hồ sơ trong `scratch/g5v52_review/` gồm contract, metadata, source materials, cards và synthesis; canonical checker kiểm tra conformance về commitment/schema, không xác nhận khoa học đúng hoặc chấp nhận hội nghị.

Các vấn đề trung tâm được sửa ở báo cáo này: comparator/result scope, UI/II hop convention, static/learned gradient path, norm certificate, cost bounds, uncertainty, forecast attribution và ablation. Lựa chọn BPE cùng các tham số mặc định là quyết định thiết kế của coordinator sau khi đối soát nguồn và toy evidence. Không ghi thành kiến trúc “đã được peer review chấp nhận”. Review là **AI role-separated, NOT_CALIBRATED**, thiếu binding tiêu chí venue; không thay human review.

**Kết luận nghiên cứu:** v5.1 hiện không có measured improvement đáng kể so với GD5-v4 trên hai run đã có và có overhead. V5.2 chuyển phép thử sang budgeted CF path expansion, giữ backbone mạnh và đưa ra cơ chế/cost có thể kiểm tra. Contraction và sparse bounds là bảo đảm toán học hữu hạn; target hơn5% vẫn cần matched training, selected-checkpoint evaluation và nhiều seeds mới có thể kết luận.
