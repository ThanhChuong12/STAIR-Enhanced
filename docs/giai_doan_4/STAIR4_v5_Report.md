# STAIR4-v5: Residual Adaptive Multimodal Ranking (STAIR-RAM)

**Trạng thái: đề xuất nghiên cứu, chưa triển khai và chưa có kết quả v5.**  
**Ngày hoàn tất đối chiếu: 26/09/2026.**  
**Mục tiêu:** tăng tương đối **trên 6% ở ít nhất 3/4 chỉ số R@10, R@20, NDCG@10, NDCG@20 trên từng dataset Baby, Sports và Electronics**, so với STAIR chạy cùng điều kiện. Đây là tiêu chí thành công tương lai, không phải dự báo đã được chứng minh.

## 1. Material Passport và kết luận thiết kế

| Thuộc tính | Phạm vi thực hiện |
|---|---|
| Workflow | academic-research-suite: deep-research + academic-paper-reviewer, methodology-focus |
| Đầu vào | Ba báo cáo thực nghiệm GĐ4 v1/v3/v4; bốn chương LaTeX GĐ1–GĐ4; baseline và cấu hình; log gốc, JSONL và manifest hiện có; tài liệu tác giả/nhà xuất bản |
| Phân biệt nguồn | Các phát biểu trong báo cáo cũ được xem là giả thuyết cần đối chiếu, không phải chỉ dẫn hay sự thật mặc định |
| Verification | Đã kiểm tra nguồn và tính nhất quán của số liệu được dẫn; không chạy lại thí nghiệm lịch sử; chưa xác nhận hiệu quả kiến trúc mới |
| Review | Có đối chiếu bằng chứng và hai góc phản biện phương pháp/đóng góp trong cùng hệ thống; không phải phản biện độc lập khác họ mô hình |
| Calibration | NOT_CALIBRATED; criteria_binding_unavailable: chưa có venue/rubric, không kết luận khả năng được nhận bài |
| Phạm vi thay đổi | Chỉ tài liệu này; không sửa baseline, các báo cáo nguồn, notebook hay mã huấn luyện |
| Tên phiên bản | GĐ4 v5 / STAIR-RAM; khác “STAIR-v5” BSC-Reweight của GĐ3 |

**Quyết định:** chuyển vị trí can thiệp từ làm mịn gradient/ép đồng nhất representation sang **học phần hiệu chỉnh trực tiếp cho điểm xếp hạng**, bằng hai nhánh text và image riêng biệt. Giữ nguyên STAIR trong Stage A; đóng băng checkpoint được validation chọn; Stage B học một residual có giới hạn biên độ từ nội dung item và lịch sử train của user. Loss chính so sánh positive với nhiều negative, không thêm contrastive loss phụ lên backbone.

Đây là hướng có triển vọng nhất **trong các phương án đã xét dưới ràng buộc baseline mạnh, full ranking và Kaggle**, theo đánh giá phương pháp; chưa có bằng chứng xác suất thành công cao nhất. Lý do chọn là thay đổi trực tiếp lớp hàm chấm điểm và lượng tín hiệu xếp hạng, trong khi vẫn kiểm soát được chi phí và quy nguyên nhân. Nếu lợi ích chỉ đến từ loss hoặc thêm capacity, phải báo cáo đúng điều đó.

## 2. Kiểm toán các kết quả đã đạt được

### 2.1. Quy tắc đọc và so sánh

Thứ tự bốn số trong các bảng là **R@10 / R@20 / N@10 / N@20**. Mỗi hàng chính phải lấy cả bốn chỉ số tại **cùng checkpoint chọn bởi validation NDCG@20**. Epoch cuối, epoch test tốt nhất và epoch validation tốt nhất không được tráo đổi.

\[
\Delta_{d,m}=100\left(\frac{M_{\mathrm{variant},d,m}}
{M_{\mathrm{baseline},d,m}}-1\right).
\]

Phân cấp bằng chứng:

- **A:** test của selected checkpoint trong JSONL + manifest đối chiếu control/treatment. Hiện chỉ một seed nên chưa có suy luận về độ ổn định.
- **B:** log text có bước load best model và test sau đó. So sánh lịch sử, chưa bảo đảm cùng fingerprint/runtime.
- **C:** chỉ có bảng hoặc bảng mâu thuẫn với log/protocol. Dùng để nhận diện vấn đề, không làm bằng chứng tăng trưởng chắc chắn.

### 2.2. Baseline lịch sử có log selected checkpoint

| Dataset | Epoch | R@10 | R@20 | N@10 | N@20 | Nguồn |
|---|---:|---:|---:|---:|---:|---|
| Baby | 455 | .0674 | .1042 | .0359 | .0454 | [baby.log:2222](../../logs/paper/baby.log#L2222) |
| Sports | 500 | .0743 | .1111 | .0405 | .0500 | [sports.log:2250](../../logs/paper/sports.log#L2250) |
| Electronics | 490 | .0442 | .0665 | .0246 | .0303 | [electronics.log:2248](../../logs/paper/electronics.log#L2248) |

Các mẫu số này được dùng thống nhất để mô tả lịch sử, **không thay thế baseline ghép cặp của v5**. Đặc biệt, Electronics không dùng .0440/.0663/.0245/.0302 của bảng LaTeX cũ.

### 2.3. Ma trận kết quả lịch sử đã chỉnh cách chọn checkpoint

| Phiên bản | Dataset / epoch chọn | Test R10 / R20 / N10 / N20 | Tăng tương đối so với bảng 2.2 (%) |
|---|---|---|---|
| GĐ2 NLGCL | Baby / 365 | .0666 / .1028 / .0360 / .0453 | −1.19 / −1.34 / +0.28 / −0.22 |
| | Sports / 500 | .0761 / .1110 / .0417 / .0507 | +2.42 / −0.09 / +2.96 / +1.40 |
| | Electronics / 440 | .0458 / .0676 / .0258 / .0314 | +3.62 / +1.65 / +4.88 / +3.63 |
| GĐ3 NE-NLGCL+ | Baby / 260 | .0674 / .1024 / .0359 / .0448 | 0.00 / −1.73 / 0.00 / −1.32 |
| | Sports / 500 | .0753 / .1118 / .0414 / .0508 | +1.35 / +0.63 / +2.22 / +1.60 |
| | Electronics / 490 | .0457 / .0680 / .0257 / .0314 | +3.39 / +2.26 / +4.47 / +3.63 |
| GĐ3 BSC-Reweight | Baby / 455 | .0675 / .1041 / .0360 / .0454 | +0.15 / −0.10 / +0.28 / 0.00 |
| | Sports / 500 | .0744 / .1115 / .0406 / .0502 | +0.13 / +0.36 / +0.25 / +0.40 |
| | Electronics / 490 | .0443 / .0666 / .0246 / .0303 | +0.23 / +0.15 / 0.00 / 0.00 |
| GĐ4 CNLGCL v1-R | Baby / 335 | .0674 / .1027 / .0360 / .0451 | 0.00 / −1.44 / +0.28 / −0.66 |
| | Sports / 500 | .0747 / .1124 / .0411 / .0509 | +0.54 / +1.17 / +1.48 / +1.80 |
| | Electronics / 395 | .0449 / .0671 / .0251 / .0309 | +1.58 / +0.90 / +2.03 / +1.98 |
| GĐ4 MHD v3 | Baby / 300 | .0670 / .1033 / .0355 / .0448 | −0.59 / −0.86 / −1.11 / −1.32 |
| | Sports / 485 | .0748 / .1129 / .0409 / .0508 | +0.67 / +1.62 / +0.99 / +1.60 |
| | Electronics | Chưa có selected-checkpoint test | Không ước lượng |

Các hàng trên đạt mức B; phần trăm tính từ số đã làm tròn trong log, không có CI. Nguồn selected test:

| Nhóm | Baby | Sports | Electronics |
|---|---|---|---|
| NLGCL | [log:2233](../../logs/v4_STAIR-NLGCL/baby.log#L2233) | [log:2245](../../logs/v4_STAIR-NLGCL/sports.log#L2245) | [log:2583](../../logs/v4_STAIR-NLGCL/electronics.log#L2583) |
| NE-NLGCL+ | [log:2278](../../logs/GD3/baby_v5_plus.log#L2278) | [log:2305](../../logs/GD3/sports_v5_plus.log#L2305) | [log:2307](../../logs/GD3/electronics_v5_plus.log#L2307) |
| BSC-Reweight | [log:2257](../../logs/GD3/Amazon2014Baby_550_MMRec_v5.log#L2257) | [log:2279](../../logs/GD3/Amazon2014Sports_550_MMRec_v5.log#L2279) | [log:2284](../../logs/GD3/Amazon2014Electronics_550_MMRec_v5.log#L2284) |
| CNLGCL | [log:2322](../../logs/GD4/baby_v1_r.log#L2322) | [log:2346](../../logs/GD4/sports_v1_r.log#L2346) | [log:2347](../../logs/GD4/electronics_v1_r.log#L2347) |
| MHD | [log:1862](../../logs/GD4/mhd_v3/baby_mhd_v3.log#L1862) | [log:1881](../../logs/GD4/mhd_v3/sports_mhd_v3.log#L1881) | [log chưa hoàn tất](../../logs/GD4/mhd_v3/electronics_mhd_v3.log) |

### 2.4. v4: so với control cùng runtime mới là phép so sánh quyết định

| Dataset | Arm / epoch | R@10 | R@20 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|
| Baby | B1 / 215 | .066066605 | .102992486 | .035194509 | .044669301 |
| | C / 325 | .066019953 | .102370262 | .035334364 | .044713175 |
| | **C/B1 − 1 (%)** | **−0.0706** | **−0.6041** | **+0.3974** | **+0.0982** |
| Sports | B1 / 500 | .074499003 | .113029002 | .040595918 | .050525369 |
| | C / 500 | .074719052 | .112874499 | .040665590 | .050499555 |
| | **C/B1 − 1 (%)** | **+0.2954** | **−0.1367** | **+0.1716** | **−0.0511** |
| Electronics | — | Chưa có | selected | test v4 | trong audit |

Nguồn A: [Baby B1](../../logs/GD4/V4/v4_20260925_094013/baby/full/V4-B1_seed1/evaluation.jsonl), [Baby C](../../logs/GD4/V4/v4_20260925_094013/baby/full/V4-C_seed1/evaluation.jsonl), [Sports B1](../../logs/GD4/V4/sport/V4-B1_seed1/evaluation.jsonl), [Sports C](../../logs/GD4/V4/sport/V4-C_seed1/evaluation.jsonl), các record selected test ở dòng 104 và run_manifest.json cùng thư mục. Source/runtime và các fingerprint graph/train/features tương ứng khớp giữa hai arm.

**Kết luận có thể bảo vệ:** v4 giữ chi phí sparse thấp và không lặp lại suy giảm lớn ở hai run này. **Chưa chứng minh cải thiện accuracy có ý nghĩa hoặc nhất quán.** Sports B1 vượt baseline lịch sử không thể quy cho CSGC vì B1 tắt CSGC và còn có R@20 cao hơn C.

### 2.5. Các mâu thuẫn không được kế thừa sang v5

| Nguồn đang có | Vấn đề | Cách xử lý trong báo cáo này |
|---|---|---|
| [GĐ1 LaTeX:675](../../report/chapters_v1/03_stair.tex#L675) | GCL Baby .0695/.1047/.0365/.0455 trộn final 500 với selected 380 | Raw selected là .0673/.1047/.0355/.0451 tại [log:2230](../../logs/log_stair_v1_baby.txt#L2230); không gộp bảng GĐ1 vào ma trận tăng trưởng chung |
| [GĐ2 LaTeX:786](../../report/chapters_v2/03_stair.tex#L786), [GĐ3 LaTeX](../../report/chapters_v3/03_stair.tex) | Baseline Electronics thay đổi mẫu số | NLGCL N@10 tăng 4.88% theo raw baseline, không phải 5.31% |
| [GĐ4 LaTeX:164](../../report/chapters_v4/03_stair.tex#L164) | CNLGCL Baby ghi epoch 455 và giá trị khác raw log | Ưu tiên selected 335; chưa tìm thấy raw chứng minh hàng 455 đó |
| [v1 report:100](STAIR4_v1_Experiment_Report.md#L100) | CNLGCL Electronics .0450/.0675/.0252/.0310 là final 500 | Selected 395 thấp hơn; dùng hàng ở §2.3 |
| [v3 report:90](STAIR4_v3_Experiment_Report.md#L90) | Nhấn mạnh MHD Baby final 500 .0672/.1038/.0361/.0455 | Selected 300 mới là kết quả chính; cả bốn chỉ số đều giảm |
| [v3 report:100](STAIR4_v3_Experiment_Report.md#L100) | So validation Electronics R20 .0671 với baseline test .0665 | Không có kết luận test-growth; log dừng sau train epoch 451 |
| [v4 report, kết luận mở đầu](STAIR4_v4_Experiment_Report.md) | “SOTA mới”, “tổng quát hóa vững chắc”, “chứng minh hoàn toàn” | Hạ thành quan sát một seed, hiệu ứng nhỏ và trái dấu |
| v4 report, thời gian Sports | 8040.8/2375.55 được ghi 4.28× | Phép chia là 3.38×; dùng manifest C total_seconds 2361.4467 thì khoảng 3.41×, nhưng timer khác run vẫn cần chuẩn hóa |
| v4 report, VRAM/phổ | High-water mark phẳng được diễn giải là không leak; cạnh không boost được coi là toán tử không đổi | Peak tích lũy không đo instantaneous memory; normalization có thể đổi cả cạnh raw giữ nguyên |

v2/v2.1 là các cảnh báo thiết kế từ báo cáo trước, không được đưa tỷ lệ suy giảm hoặc cơ chế nhân quả chưa kiểm toán vào bảng tăng trưởng lần này. “Gradient conflict”, “oversmoothing” hay “saturation” phải có phép đo riêng; loss xấu không đủ chứng minh từng nguyên nhân.

## 3. Câu hỏi Socratic: baseline còn thiếu gì?

Đối chiếu [main.py](../../main.py), đặc biệt whitening, prepare, encode, fit và recommend_from_full, với [STAIR AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33407) và [mã tác giả](https://github.com/yhhe2004/STAIR):

| Câu hỏi phản biện | Quan sát chắc chắn / giới hạn suy luận | Giả thuyết cần kiểm tra |
|---|---|---|
| Nội dung item còn đường học trực tiếp đến score sau initialization không? | MI dùng feature để khởi tạo, BSC dùng graph modality; scorer cuối là tích vô hướng embedding đã học, không có projector nội dung được tối ưu trực tiếp | Residual nội dung có thể giải thích lỗi xếp hạng còn lại |
| Hai modality được giữ riêng đến thời điểm matching không? | Local MI lấy U từ SVD từng modality rồi cộng với trọng số 5:1; không có bước căn chỉnh hai cơ sở riêng | Tách subspace có thể giữ thông tin bổ sung; chưa chứng minh phép cộng hiện tại là nguyên nhân thất bại |
| Sửa các cạnh cũ có đủ thay đổi mục tiêu lớn không? | v4 Baby chỉ khoảng 10.45% candidate pairs có evidence; sai khác Frobenius tương đối toán tử chuẩn hóa khoảng .00395 | Can thiệp nhỏ có thể thiếu lực; không suy ra theorem rằng tăng trưởng ranking cũng nhỏ |
| BPR một negative có đang khai thác tốt top-K không? | Baseline lấy một negative; BPR không trực tiếp tối ưu NDCG | Nhiều candidate có thể giúp; cần loss-only control, không quy hết lợi ích cho architecture |
| Gắn thêm loss buộc các view giống nhau có cần thiết không? | Lịch sử cho thấy lợi ích không cộng tuyến; Baby thường suy giảm | Học trực tiếp điểm final có thể tránh một nguồn xung đột mục tiêu, không bảo đảm complementarity |
| Frozen backbone có bảo vệ ranking không? | Bảo vệ tham số backbone; score vẫn đổi | Chỉ có bound mức sửa score và baseline checkpoint dự phòng, không có bảo đảm test không giảm |

**RQ chính:** Sau khi STAIR đã học cấu trúc cộng tác, liệu hồ sơ nội dung từ lịch sử train còn dự đoán được lỗi ranking, và có thể khai thác phần tín hiệu đó bằng một residual nhỏ, tách modality, có chi phí chấp nhận được không?

## 4. Nghiên cứu nguồn: hướng SOTA và cơ chế có thể chuyển giao

Đây là **targeted review**, không phải systematic review đầy đủ. Tìm theo bốn trục: multimodal fusion/personalization, graph structure, loss/negative sampling và tách giai đoạn tối ưu. Ưu tiên paper nhà xuất bản, arXiv tác giả và repository chính chủ; phân biệt đọc paper với đọc code và với tái lập thực nghiệm. Một paper báo cáo SOTA trên protocol của họ không chứng minh vượt STAIR ở split hiện tại.

| Nguồn / trạng thái | Cơ chế liên quan và mức kiểm tra | Quyết định cho v5 |
|---|---|---|
| [STAIR, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33407) | MI/FSC/BSC; đối chiếu paper, repo và local implementation | Giữ nguyên Stage A; baseline bắt buộc |
| [VBPR, AAAI 2016](https://arxiv.org/abs/1510.01784) | Học điểm visual preference từ feature; paper nền tảng | Ghi nhận prior art cho direct modality scoring; không nhận novelty của ý tưởng này |
| [MGCN, ACM MM 2023](https://arxiv.org/abs/2308.03588), [code](https://github.com/demonph10/MGCN) | Purification theo hành vi và fusion đa view; đã đọc model source | Học cách phân biệt tín hiệu modality/hành vi; không mang toàn bộ graph và CL sang |
| [FREEDOM, ACM MM 2023](https://github.com/enoche/FREEDOM) | Freezing và denoising graph; kiểm tra repository/README tác giả | Ủng hộ kiểm soát chi phí graph; freezing graph của paper khác freezing backbone của v5 |
| [LGMRec, AAAI 2024](https://arxiv.org/abs/2312.16400) | Local graph + global hypergraph; kiểm tra paper | Hướng mở rộng receptive field đáng so sánh, nhưng thêm global propagation trước khi chứng minh thiếu tín hiệu sẽ tăng confounding |
| [SMORE, WSDM 2025](https://doi.org/10.1145/3701551.3703561), [code](https://github.com/kennethorq/SMORE) | Fusion trong miền tần số; đã đọc model source | Xem xét giữ modality riêng; FFT trên chiều feature không đồng nghĩa graph-Laplacian spectrum hoặc lượng tử |
| [PGL, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33429/35584) | Principal Graph Learning nhấn mạnh thông tin cá thể; paper-level | Động lực đo preservation của thông tin; không nhập con số tăng trưởng của paper thành kỳ vọng STAIR |
| [SimpleX, CIKM 2021](https://arxiv.org/abs/2109.12613), [repo](https://github.com/66chenl/SimpleX) | Làm rõ vai trò loss và số negative; đọc paper/repo, không tái lập | Bắt buộc loss/negative-budget controls; không sao chép nguyên protocol example |
| [Sampled Softmax, TOIS 2024](https://wujcan.github.io/papers/tois-ssm.pdf), [repo](https://github.com/wujcan/SSM-Torch) | Phân tích sampled-softmax, temperature và normalization; paper kiểm tra, source chưa audit toàn bộ | Dùng objective có nhiều candidate; giữ rõ bias do sampling và phân phối negative |
| [SimCE, arXiv 2024](https://arxiv.org/abs/2406.16170) | Simplified sampled CE; đã xem pseudocode trong paper | Làm control/hướng phụ, không chọn hardest-negative max làm mặc định khi chưa đo false negatives |
| [FDRec, CIKM 2025](https://eprints.gla.ac.uk/361950/1/361950.pdf) | Teacher/student và frequency-decoupled distillation; paper đã đọc | Động lực tách tối ưu/chi phí; v5 không dùng distillation nên không gọi là tái hiện FDRec |
| [MAGNET, arXiv v3, 14/09/2026](https://arxiv.org/html/2602.20723v3), [repo](https://github.com/jidaivita/MAGNET) | History-induced modality cues, structured experts, routing theo interaction; paper và repo/README đã kiểm tra | Dùng personalization đơn giản hơn để giữ scorer phân tách user/item; không copy chín expert hay pseudo-edge dual graph |
| [Towards Trustworthy Multimodal Recommendation, arXiv 2026](https://arxiv.org/abs/2602.00730) | Abstract nêu modality rectification và pseudo-edge có thể giúp hoặc hại khi nhiễu | Nguồn bối cảnh cho robustness tests; chưa đủ audit để nhập Sinkhorn vào core |

### 4.1. Source audit cụ thể và giới hạn chuyển giao

- [MGCN model source](https://raw.githubusercontent.com/demonph10/MGCN/main/src/models/mgcn.py): forward có modality purification, item graph propagation, user aggregation và fusion; calculate_loss kết hợp BPR và CL. v5 chỉ lấy động lực về nguồn thông tin, không tuyên bố tương đương MGCN.
- [SMORE model source](https://raw.githubusercontent.com/kennethorq/SMORE/main/src/models/smore.py): spectrum_convolution dùng rFFT theo chiều feature, learned complex filters và fusion. Gọi mọi phép này là “bảo toàn phổ graph” là sai loại toán tử.
- [MAGNET v3](https://arxiv.org/html/2602.20723v3): router theo cặp user–item và expert phi tuyến có thể biểu đạt tốt hơn scorer phân tách. v5 chấp nhận giảm khả năng ấy để full-ranking bằng GEMM. Chưa xác nhận toàn bộ model source và commit triển khai; không dùng README làm bằng chứng tái lập. Trạng thái xác minh ở đây là **preprint**, không coi dòng journal trong template là chứng nhận chấp nhận.
- [FDRec repository được institutional record trỏ tới](https://github.com/Suehn/FDRec_) đang trống khi kiểm tra. Liên kết FDRec khác trong paper không truy cập được trong phiên audit. Vì vậy chỉ dùng bằng chứng paper, không nói đã audit implementation.
- Các repo ở nhánh mặc định có thể thay đổi. Trước khi tái lập external baseline phải chốt commit SHA, license, feature mapping và evaluator. Không import code ngoài vào proposal này.

### 4.2. Vì sao không chọn các hướng khác làm core?

| Phương án | Lợi ích có thể có | Lý do chưa chọn |
|---|---|---|
| Tiếp tục chỉ sửa BSC/CSGC | Giữ inference rẻ | Tác động accuracy hiện nhỏ, không mở đường nội dung trực tiếp đến score |
| Hypergraph/graph augmentation lớn | Thêm kết nối, global context | Pseudo-edges có thể sai; thêm propagation và tuning; chưa chứng minh thiếu cạnh là bottleneck chính |
| Pair-conditioned MoE | Fusion linh hoạt | Full-catalog scoring có thể rất đắt; khó tách lợi ích expert/capacity/routing |
| Quantum-inspired head/unitary smoother | Một lớp tham số hóa khác | Không có bằng chứng địa phương đủ mạnh; norm preservation không suy ra cải thiện ranking |
| Full end-to-end MGCN/SMORE stack | Tăng biểu đạt | Làm thay đổi đồng thời backbone, fusion, graph, loss; khó giữ baseline recovery |
| **Frozen STAIR + bounded modality residual + sampled ranking** | Thêm tín hiệu và objective trực tiếp, có controls rõ, scorer phân tách | **Chọn làm v5 đầu tiên**, đồng thời chấp nhận rủi ro không đủ vượt 6% |

Đóng góp tiềm năng là **thiết kế và kiểm chứng lợi ích có điều kiện của modality còn lại sau STAIR**, không phải phát minh residual, PCA, history pooling, gating hoặc sampled softmax.

## 5. Kiến trúc STAIR-RAM v5

### 5.1. Sơ đồ và hợp đồng hai giai đoạn

~~~mermaid
flowchart TD
    R[Train interactions] --> A[Stage A: unchanged STAIR MI / FSC / BSC / BPR]
    X[Original item text and image features] --> A
    A --> H[Validation-selected frozen encoded user and item vectors]
    H --> S0[Baseline dot-product score]
    X --> P[Separate fixed PCA and normalization per modality]
    R --> T[Binary train-only user histories]
    P --> T
    T --> Q[LOO profile in training; full train profile in evaluation]
    H --> Q
    Q --> PROJ[Trainable query projectors]
    P --> ITEM[Trainable item projectors]
    H --> G[User-only modality gate]
    PROJ --> RES[Bounded modality residual]
    ITEM --> RES
    G --> RES
    S0 --> SUM[Final score = baseline + residual]
    RES --> SUM
    SUM --> LOSS[Stage B: sampled candidate ranking loss]
    SUM --> EVAL[Exact full or pool ranking with baseline masks]
~~~

**Stage A:** MI, FSC, BSC, BPR, optimizer, evaluator và criterion chọn checkpoint giữ baseline. Có thể dùng đường B1 của v4 với alpha=0 và graph preparation theo block nếu đã qua parity audit; không mặc nhiên xem runtime thay đổi là đồng nhất. Không dùng CSGC làm backbone mặc định.

**Stage B:** load checkpoint chọn theo validation NDCG@20; gọi encode của baseline dưới no_grad; lưu hai ma trận **đầu ra FSC cuối cùng**, không nhầm với embedding layer-0. Đặt toàn bộ backbone requires_grad=False, eval mode, không đưa vào optimizer mới. Học residual tối đa 100 epochs.

Checkpoint “B epoch 0” là baseline thật với residual bị disable. Nó luôn là ứng viên validation. Trước bước học đầu tiên bật residual và dùng initialization nhỏ khác 0 như §5.6. Hai trạng thái này phải được đặt tên riêng.

### 5.2. Ký hiệu và thông tin được phép sử dụng

| Ký hiệu | Nghĩa / kích thước |
|---|---|
| U, I; D=64 | Số user, item; chiều baseline |
| R ∈ {0,1}^(U×I) | Train CSR đã deduplicate các cặp user–item |
| I_u; d_u | Tập item train của u; số phần tử |
| h_u^0, h_i^0 | Frozen final FSC vectors, chiều D |
| m ∈ {t,v}; P=128 | Text/image; chiều PCA mỗi modality |
| x_i^m, t_u^m | Fixed feature của item; tổng feature train của user |
| r=32 | Chiều mỗi residual branch |
| A_m, B_m, C_m | Projectors lần lượt r×D, r×P, r×P |
| π_u^m | User-only modality weights; không phụ thuộc candidate |
| η; a | Giới hạn biên độ cố định theo run; biên độ thực học được |

Không dùng interaction validation/test trong graph, profile, degree, negative-exclusion hay scale calibration. Feature của toàn bộ catalog đã có sẵn được dùng theo **transductive protocol** của benchmark; không gọi đây là đánh giá cold-start item mới. File mapping item ID phải được kiểm tra trước mọi phép nhân.

### 5.3. Separate fixed modality features

Với mỗi modality độc lập, normalize raw rows hợp lệ, trừ mean và fit PCA:

\[
\tilde f_i^m=\operatorname{norm}_{\epsilon}(f_i^m),\quad
\mu_m=\operatorname{mean}_{i\in\mathrm{valid}_m}\tilde f_i^m,\quad
x_i^m=\operatorname{norm}_{\epsilon}
\left((\tilde f_i^m-\mu_m)V_m\right)\in\mathbb R^P.
\tag{1}
\]

Ở đây norm_epsilon(y)=y/max(||y||_2,epsilon), epsilon=10^−8 trong float32. V_m gồm P principal directions, **không whitening**, không cộng text-PCA với image-PCA. Chọn P=min(128, số chiều raw, số row hợp lệ−1), log chiều thực; nếu cần tensor thống nhất thì zero-pad và lưu rank thực.

Chốt preprocessing trước pilot: PCA trên feature catalog, không dùng nhãn held-out; lưu mean/components, explained variance, algorithm, seed, input/output hashes và mask missing. Dùng truncated/randomized SVD có seed cố định trong **nhánh mới** nếu full SVD quá đắt; lưu oversampling=16, power iterations=4 và canonical sign theo loading có trị tuyệt đối lớn nhất. Không thay SVD của MI baseline.

Raw row bị missing/non-finite/zero phải được phát hiện trước fit. Với row missing, x_i^m=0 sau transform; không biến nó thành vector −mean. Modality hoàn toàn không hợp lệ: dừng preflight hoặc chạy arm single-modality được ghi rõ, không âm thầm đổi thí nghiệm. Missingness và explained variance phải được báo cáo vì PCA128 không bảo đảm giữ thông tin preference.

### 5.4. User profiles và leave-positive-out (LOO)

Cache chỉ các tổng feature cố định:

\[
t_u^m=\sum_{j\in I_u}x_j^m,\qquad
\bar x_u^m=t_u^m/\max(d_u,1).
\tag{2}
\]

Với **mỗi training row (u,i⁺)**:

\[
\bar x_{u,-i^+}^m=
\begin{cases}
(t_u^m-x_{i^+}^m)/(d_u-1),&d_u>1,\\
0,&d_u=1.
\end{cases}
\tag{3}
\]

Row không thuộc train phải bị reject. Profile của user có d_u=0 là vector 0 nhưng benchmark warm-start cần báo riêng số user này. Missing feature được giữ bằng 0; mẫu số vẫn là số item train, nhất quán cả training và evaluation.

**Quan trọng:** trong một row, dùng cùng profile LOO để chấm positive và mọi negative. Không loại từng candidate khỏi profile theo cách khác nhau. Hai row cùng user nhưng khác positive có hai query khác nhau; không deduplicate user rồi tái dùng query sai. Deduplicate item candidates chỉ khi giữ inverse indices để khôi phục đầy đủ row/slot.

LOO loại đường tắt positive xuất hiện trực tiếp trong auxiliary feature mean. h_u^0 đã học mọi train edge, gồm positive này; do đó đây **không phải cross-fitting hay target-independent teacher**. Evaluation dùng full train profile, tạo một train/eval context shift cần đo theo độ dài lịch sử.

### 5.5. Queries, item keys và user gate

Đặt b_u=norm_epsilon(h_u^0). Với context c là profile LOO khi train hoặc full train profile khi eval:

\[
q_{u,c}^m=\operatorname{norm}_{\epsilon}
(A_mb_u+B_m\bar x_{u,c}^m),\qquad
z_j^m=\operatorname{norm}_{\epsilon}(C_mx_j^m).
\tag{4}
\]

Không thêm bias vào ba projector; item thiếu modality giữ z=0. Gate affine nhỏ, chỉ đọc user:

\[
\ell_u=\frac{\log(1+d_u)}{\max_v\log(1+d_v)+\epsilon},\qquad
\boldsymbol\pi_u=\operatorname{softmax}
\left(W_g[b_u;\ell_u]+b_g\right).
\tag{5}
\]

W_g có shape 2×(D+1). Gate không dùng item, held-out metadata hoặc trainable per-user table. Không thêm entropy loss mặc định: nếu gate collapse, trước hết so với fixed global mixture; có thể modality đó thực sự hữu ích hơn. Gate weight cũng không tự động là giải thích nhân quả về preference.

### 5.6. Bounded residual và khởi tạo tránh zero-gradient trap

\[
s^0(u,j)=(h_u^0)^\top h_j^0,\qquad
a=\eta\tanh(\theta),\qquad
r(u,j\mid c)=a\sum_m\pi_u^m(q_{u,c}^m)^\top z_j^m,
\tag{6}
\]
\[
\boxed{s(u,j\mid c)=s^0(u,j)+r(u,j\mid c).}
\tag{7}
\]

Biên độ phải có đơn vị của score baseline. Dùng tập tối đa 100,000 train pairs được lấy mẫu bằng RNG riêng và một train-unseen uniform negative mỗi pair:

\[
\sigma_0=\max\left(10^{-3},
\operatorname{median}_{(u,i,j)}
|s^0(u,i)-s^0(u,j)|\right),\qquad \eta=c_\eta\sigma_0.
\tag{8}
\]

Cố định calibration sample/hash trong mỗi seed; không update sigma0 theo bước. Pilot c_eta ∈ {0.25,0.5,1.0}, mặc định 0.5. Đây là heuristic scale calibration cần validation, không có “biên độ tối ưu” theo định lý.

Initialization Stage B:

1. C_m khởi tạo orthogonal rows nếu shape cho phép; B_m copy **giá trị** C_m ban đầu nhưng là parameter riêng; A_m=0.1×Xavier. Không zero cả branch.
2. W_g=0, b_g=0 cho gate .5/.5; gate chỉ một affine layer nên không có hidden layer bị khóa bởi last-layer zero.
3. theta=atanh(0.05), tức a_init=0.05 eta. Scalar và projector/gate đều có đường gradient ngay từ đầu; dữ liệu vẫn có thể khiến gradient nhỏ, cần kiểm tra nhiều bước.
4. **Gate-0 là mode riêng:** disable residual, return baseline scorer trực tiếp. Không dùng theta=0 như initialization chính.

Tại a=0, ∂L/∂A_m, ∂L/∂B_m, ∂L/∂C_m và ∂L/∂W_g đều bằng 0. Chỉ scalar amplitude có thể mở branch, và gradient của nó cũng có thể rất nhỏ. Khởi tạo khác 0 giải quyết cấu trúc chặn gradient này; không bảo đảm optimization thành công.

### 5.7. Ba tính chất có thể chứng minh, và điều không thể suy ra

**Bound score.** Vì ||q||,||z||≤1 và π nằm trên simplex:

\[
|r(u,j)|\le |a|\le\eta.
\tag{9}
\]

Do đó một cặp có baseline margin s0(u,i)−s0(u,j)>2eta sẽ không bị residual đảo thứ tự. Mệnh đề này cũng chỉ ra hạn chế: cap quá nhỏ có thể khiến mô hình không sửa được lỗi đáng kể.

**Frozen gradient.** Trong Stage B, ∇_(theta_STAIR)L=0 theo hợp đồng autograd. Không suy ra score tốt hơn, chống overfit tuyệt đối, hoặc không còn mọi dạng xung đột giữa hai modality.

**Separable full ranking.** Khi eval, mỗi user có một profile cố định. Đặt:

\[
\tilde h_u=[h_u^0;\ a\pi_u^t q_u^t;\ a\pi_u^v q_u^v],
\qquad
\tilde h_i=[h_i^0;\ z_i^t;\ z_i^v].
\tag{10}
\]

Khi đó s(u,i)=tilde_h_u^T tilde_h_i. a có thể âm; không dùng sqrt(a). Phép phân tách vẫn đúng. Chiều scorer là D+2r=128, không phải 64.

Tương đương đại số không bảo đảm bitwise bằng hai GEMM khác thứ tự cộng. **Gate-0 gọi nguyên scorer baseline** để bảo vệ tie behavior. Với head bật, test full/pool đối chiếu cùng công thức trong sai số float32 đã định. Không suy ra NDCG tăng từ bất kỳ bound nào ở trên.

## 6. Objective, sampler và vòng đời huấn luyện

### 6.1. Loss chính trên score cuối

Với row (u,i⁺), lấy K=32 item khác nhau từ catalog không thuộc I_u:

\[
\mathcal L_{\mathrm{rank}}=
\frac1{|\mathcal B|}\sum_{(u,i)\in\mathcal B}
\left[
\operatorname{logsumexp}
\left(\frac{s(u,i\mid -i)}{T},
\left\{\frac{s(u,j\mid-i)}{T}\right\}_{j\in N_u}\right)
-\frac{s(u,i\mid-i)}{T}
\right].
\tag{11}
\]

T=1.0 mặc định; không normalize tổng score vì cần giữ s0 và đơn vị eta. Không thêm POCL/InfoNCE trên embedding hoặc BPR thứ hai trong arm chính. Đây là **sampled candidate classification/ranking loss**, không được gọi là mutual-information bound hay estimator không chệch của full softmax. Động lực lấy từ [nghiên cứu SSM](https://wujcan.github.io/papers/tois-ssm.pdf); objective và score normalization cụ thể của v5 là lựa chọn cần kiểm chứng.

Với K=1,T=1, Eq.11 bằng softplus(s_neg−s_pos)=−log sigmoid(s_pos−s_neg), tức BPR. Đây là identity để test; không chứng minh K lớn luôn tốt hơn.

Với logits ℓ=s/T, gradient theo score là (p_j−y_j)/T. Frozen base chỉ cung cấp offset; head học sửa các candidate còn cạnh tranh. Nếu base phân tách gần hết uniform negatives, gradient có thể nhỏ dù K tăng.

### 6.2. Negative sampling và false negatives

- Uniform without replacement theo từng row trên I\I_u; positive đã bị loại bởi train mask. CSR membership đọc **train only**.
- Candidate có thể lặp giữa các row. Không lặp trong cùng row; không dùng in-batch item làm negative mặc định.
- Nếu số unseen <K, lấy tất cả và dùng mask/count thực trong logsumexp; nếu không có unseen, bỏ row và log số lượng.
- Không lọc validation/test positives để “làm sạch negative”: điều đó dùng nhãn held-out. Unobserved có thể là positive chưa biết; đây là hạn chế của implicit feedback.
- RNG sampler tách khỏi model/PCA/calibration, lưu state để resume. Một effective batch phải giữ nguyên candidates khi so microbatch/full-batch.
- Primary arm không dùng hard mining. Chỉ mở ablation 50% uniform + 50% semi-hard từ ranking của frozen s0 nếu pilot cho thấy gradient chết do negative quá dễ. Chốt rank band trước run; không chọn theo test; báo đúng rằng sampler đã đổi.

### 6.3. Optimizer, cache và snapshot

Stage A dùng AdamWSEvo baseline: user không smoother, item BSC gốc. Stage B dùng AdamW thông thường **chỉ cho head**:

| Group | Parameters | LR khởi điểm | Weight decay | Smoother |
|---|---|---:|---:|---|
| Projection | A_m, B_m, C_m | 1e−3 | 1e−4 | Không |
| Gate | W_g, b_g | 1e−3 | 0 | Không |
| Amplitude | theta | 1e−4 | 0 | Không |
| Backbone | h0 xuất từ selected STAIR | Frozen | — | Không chạy optimizer ở Stage B |

Mỗi trainable parameter xuất hiện đúng một group; backbone không có trong optimizer. Không truyền head parameters vào BSC smoother.

Cache hợp lệ: frozen h0, fixed PCA features, R, d, t, scale sigma0. **Không cache z_j hoặc q_u có graph autograd qua các bước**. Recompute projectors/gate mỗi step; deduplicate item IDs trong batch rồi gather lại bằng inverse map là hợp lệ. Evaluation cache được tạo dưới no_grad và invalidate sau mọi update/load checkpoint.

Không tạo ma trận U×I logits hoặc I×I similarity ở Stage B. Full catalog embedding cache chỉ được dùng cho evaluation; không tái dùng tensor detached đó làm trainable keys.

### 6.4. Training pseudocode: hợp đồng dự kiến, chưa phải implementation

~~~python
# Stage A: preserve STAIR training and validation checkpoint selection.
teacher = load_selected_stair_checkpoint_with_manifest()
teacher.eval().requires_grad_(False)
with torch.no_grad():
    h_user, h_item = teacher.encode()  # Final FSC output, not layer zero.
features, train_csr, sums, degrees = load_verified_static_inputs()

best = evaluate_baseline_as_stage_b_epoch_zero()
head = initialize_nonzero_residual_head()
optimizer = build_disjoint_head_parameter_groups(head)

for epoch in range(1, max_head_epochs + 1):
    for pairs in shuffled_train_pairs:
        candidates = sample_uniform_train_unseen(pairs, k=32)
        optimizer.zero_grad(set_to_none=True)
        for micro in split_preserving_rows_and_candidates(pairs, candidates):
            profiles = leave_positive_out(sums, degrees, micro.positive)
            query, keys, gate = head.encode_fresh(micro, profiles, features)
            score = frozen_baseline_scores(micro) + head.residual(query, keys, gate)
            loss = masked_candidate_cross_entropy(score, temperature=1.0)
            (loss * micro.num_rows / pairs.num_rows).backward()
        record_gradient_and_score_diagnostics()
        clip_grad_norm_(head.parameters(), max_norm=5.0)
        optimizer.step()
        invalidate_evaluation_cache()
    if should_validate(epoch):
        rebuild_eval_vectors_from_full_train_profiles()
        select_checkpoint_using_validation_ndcg20_only()

# Reload exactly the selected epoch, including epoch zero if it won.
# Test once after configuration and selection are frozen.
~~~

LOO implementation phải nhận cả user IDs và positive IDs; pseudocode lược argument để minh họa luồng. Gradient clipping thực hiện sau accumulation, không sau từng microbatch. Không BatchNorm/dropout trong core để microbatch không thay đổi định nghĩa score.

## 7. Cấu hình dự kiến và tính khả thi trên Kaggle

### 7.1. Giữ cấu hình backbone theo từng dataset

| Stage A | Baby | Sports | Electronics |
|---|---:|---:|---:|
| Embedding D / FSC layers | 64 / 3 | 64 / 3 | 64 / 3 |
| Epoch tối đa | 500 | 500 | 500 |
| Batch size | 1024 | 1024 | 4096 |
| AdamWSEvo LR | .001 | .001 | .001 |
| Weight decay | .3 | .1 | .1 |
| Gamma | .1 | .2 | .4 |
| Text/image kNN | 5 / 1 | 5 / 1 | 5 / 1 |
| Checkpoint metric | valid NDCG@20 | valid NDCG@20 | valid NDCG@20 |

Nguồn: [Baby YAML](../../configs/Amazon2014Baby_550_MMRec.yaml), [Sports YAML](../../configs/Amazon2014Sports_550_MMRec.yaml), [Electronics YAML](../../configs/Amazon2014Electronics_550_MMRec.yaml). Không dùng một weight decay/gamma chung rồi gọi là phục hồi baseline.

| Stage B | Mặc định pilot | Cho phép mở rộng có điều kiện |
|---|---|---|
| PCA P / branch r | 128 / 32 | P=64; r=16 cho capacity/efficiency controls |
| Head epochs | 100 tối đa | Validation mỗi 5 epochs; patience 6 lần đánh giá |
| Effective batch | Theo Stage A | Microbatch 512 nếu cần; không đổi số update tùy tiện |
| K / T | 32 / 1.0 | K=1 cho BPR control; K=8/64 khi phân tích sensitivity |
| c_eta | .5 | {.25,.5,1.0}, chỉ validation |
| Projection LR | .001 | {.0003,.001}, pilot tối đa 6 cấu hình cùng c_eta |
| Feature/gate dropout | 0 | Không thêm khi chưa có evidence overfit |
| Precision | Float32 | AMP chỉ arm riêng sau parity/numerical checks |
| Early stopping | Valid N@20, strictly greater; ties giữ checkpoint cũ | Epoch 0 phải được xét |
| Gradient clipping | Global norm 5 | Log tỷ lệ step bị clip, không chỉ log threshold |

Các số trên là starting configuration, **không phải hyperparameters đã tối ưu**. So sánh primary dùng cùng training budget cho các head controls. Early stopping Stage B khác Stage A được ghi rõ.

### 7.2. Parameter count, FLOPs và memory

Với D=64,P=128,r=32 và hai modality:

\[
N_{\mathrm{head}}=2r(D+2P)+2(D+1)+2+1=20{,}613.
\tag{12}
\]

Không thêm embedding table U×r hoặc I×r được học. Mỗi step dự kiến có chi phí projection O(B·r(D+P)+n_unique·P·r) trên mỗi modality và matching O(B(K+1)·2r). Stage B không chạy sparse graph propagation mỗi step.

Chi phí full-ranking vẫn là O(B_eval·I·(D+2r)); với r=32, **GEMM rộng gấp đôi baseline**. “Không thêm graph training” không đồng nghĩa “inference miễn phí”.

Ví dụ kích thước Electronics trong raw log: U=192,403, I=63,001. Ước lượng float32 dưới đây chỉ tính tensor được nêu, không phải VRAM đo thực:

| Thành phần | Công thức | MiB xấp xỉ |
|---|---|---:|
| h0 + hai item-feature matrices + hai user-profile sums | 4(U+I)(D+2P) | 311.77 |
| Cache vector ranking concatenated | 4(U+I)(D+2r) | 124.71 |
| Score block 256 user × toàn catalog | 4·256·I | 61.52 |
| Candidate branch vectors, B4096,K32 | 4·4096·33·2r | 33.00 |

Autograd intermediates, candidate input features, CSR, raw features, CUDA allocator và library workspace còn thêm bộ nhớ. Không cộng các dòng rồi tuyên bố đó là peak. Đọc raw modality/PCA theo modality trên CPU khi cần; không giữ raw high-dimensional tensors đồng thời trên GPU chỉ để tránh I/O.

Stage A gốc có dense kNN I×I trong prepare. Với Electronics đây có thể là bottleneck memory lớn hơn head. Dùng blockwise exact top-k đã kiểm toán ở v4, ghi neighbor/graph hashes, tie policy và baseline parity; không thay bằng ANN âm thầm. Chi phí PCA, graph preparation, Stage A, Stage B, validation và selected test phải xuất riêng và tổng hợp.

Ngân sách khởi điểm để quyết định chạy dài: head + preprocessing bổ sung không quá 35% wall-clock Stage A trên cùng máy, eval không quá 2.5× baseline, giữ tối thiểu 20% GPU memory headroom. Đây là **engineering stop criteria có thể điều chỉnh công khai**, không phải kết quả đo hay bằng chứng đảm bảo trên T4.

## 8. Kế hoạch triển khai mã nguồn — chưa thực hiện

Không ghi đè các file v2/v3/v4. Giữ main.py và optimizer baseline. Mã mới, docstrings và tests dùng tiếng Anh.

| File dự kiến | Nội dung / hợp đồng |
|---|---|
| models/stair4_v5.py | Wrapper GenRecArch/STAIR adapter; Stage A delegation; load/freeze encoded teacher; Gate-0; full/pool scorer; không parse CLI khi import |
| models/stair4_v5_features.py | PCA fit/transform, missing masks, ID alignment, hashes, static cache, binary CSR train sums, LOO theo từng row |
| models/stair4_v5_heads.py | Query/item projectors, affine user gate, bounded amplitude, residual scoring và concatenated evaluation vectors |
| models/stair4_v5_sampling.py | Uniform distinct train-unseen negatives, masks cho catalog nhỏ, RNG save/load; optional mining nằm ở arm riêng |
| models/stair4_v5_objectives.py | Masked sampled CE, BPR identity, diagnostics trên final score; không dùng .item() trước backward |
| main_stair4_v5.py | stage=a/b/all; teacher path + manifest validation; optimizer isolation; accumulation; checkpoint lựa chọn; JSONL telemetry |
| configs/dataset_stair4_v5_{baby,sports,electronics}.yaml | Kế thừa YAML baseline; cấu hình riêng Stage B và ablation_id; không override ngầm monitor |
| tests/test_stair4_v5.py | Algebra, sampler, LOO, autograd, parameter isolation, scoring parity, checkpoint/resume contracts |
| scripts/analyze_stair4_v5.py | Chỉ đọc selected test; merge theo dataset/seed/control hash; compute paired deltas, uncertainty và missing-run report |
| notebook/P4/stair4_v5.ipynb | Kaggle setup, dataset selection, preflight, pilot/confirm arms, process return-code check, parser và plots |

Notebook phải có DATASETS_TO_RUN hiển thị rõ, cho chạy riêng sports/electronics; gọi dataset explicit không bị chặn bởi danh sách baby mặc định. Tách pilot 50/100 khỏi Stage A 500; in command/config/resolved epochs trước launch. Plot lỗi không được làm mất trạng thái training success; training process exit khác 0 phải dừng arm và giữ stderr.

Checkpoint Stage A cần lưu model + optimizer/scheduler/RNG tại checkpoint được chọn nếu muốn continuation đúng state. Nếu checkpoint lịch sử chỉ có model, không gọi việc reset optimizer là exact continuation: ghi reset, áp dụng cùng reset cho các control tương ứng hoặc tạo lại Stage A.

Checkpoint Stage B lưu teacher checksum/path, feature transform/hash, head, optimizer, scheduler nếu có, RNG, epoch/step, selected validation metric, stage, Gate-0 flag và config. Không phụ thuộc Python globals trong notebook. Load phải reject teacher/data/dimension mismatch, không tự động fallback sang file cùng tên.

## 9. Bộ kiểm thử trước khi chạy thực nghiệm

| Contract | Kiểm tra bắt buộc |
|---|---|
| Baseline parity | Stage A fixed seed cùng inputs và graph; so encode, BPR, một optimizer update, ranking và selected-checkpoint behavior |
| Gate-0 | Gọi scorer baseline nguyên bản; full/pool, seen-item masking, NDCG@20 giống baseline; không tạo head RNG làm đổi Stage A |
| FSC cache | Cached vectors bằng encode cuối cùng tại selected checkpoint, khác layer-0 khi graph không identity |
| Frozen isolation | Backbone requires_grad=False, grad=None và hash/values không đổi sau nhiều Stage B steps |
| Parameter groups | Tập hợp params đúng head, không overlap, không thiếu và không chứa backbone |
| LOO algebra | Dense toy oracle, duplicate edges, degree 1/0, positive membership; hai positive cùng user tạo query đúng |
| Candidate alignment | Duplicate items giữa rows; unique/inverse khôi phục logits/gradients đúng |
| Sampler | Không có train positive, không lặp trong row, K lớn hơn unseen xử lý finite, không đọc held-out mask |
| Normalization/missing | Zero vectors finite; missing item branch score 0; NaN features fail hoặc mask theo policy đã chốt |
| Residual bound | Với cả a âm/dương, abs(residual)≤eta trong tolerance |
| Scoring algebra | Eq.7 bằng Eq.10; full/pool tương ứng; mọi negative trong row dùng cùng LOO query |
| Loss identity | K1,T1 sampled CE bằng BPR, cả value và gradient; masked padding không góp denominator |
| Gradient flow | Scalar, A/B/C và gate có finite gradient ở nhiều nondegenerate steps; initialization nhỏ khác 0; test zero-amplitude trap có chủ đích |
| Microbatch | Effective batch/candidates giữ nguyên; accumulated gradient gần full-batch trong tolerance đã chốt |
| No stale graph | Backward liên tiếp ≥3 steps không retain_graph; train keys fresh; eval cache invalidation sau update |
| Checkpoint roundtrip | Teacher/head/config/hashes đúng; logits và metrics khớp; resume một update với RNG/state giống uninterrupted |
| Selection contract | Validation duy nhất chọn; baseline epoch0 có thể thắng; không lấy từng metric ở epoch riêng |
| Memory | Bounded batch tensors, không allocate U×I/I×I trong Stage B; peak và current memory được phân biệt |

Unit tests pass chỉ xác nhận hợp đồng phần mềm/toán học; không chứng minh chất lượng recommendation. Cần synthetic integration test và real dataset smoke run nhỏ trước khi đánh giá accuracy.

## 10. Thí nghiệm phân rã: làm sao biết vì sao tăng?

### 10.1. Các arm bắt buộc trước khi nhận đóng góp kiến trúc

Tất cả Stage B arms dùng **cùng selected teacher theo dataset/seed**. Baseline B0 được đánh giá một lần rồi tái sử dụng artifact; không train lại ngẫu nhiên riêng cho mỗi arm.

| Arm | Thiết kế | Câu hỏi giải đáp |
|---|---|---|
| B0 | STAIR nguyên bản 500 epochs | Baseline chính cho mục tiêu >6% |
| B-long | STAIR liên tục 600 epochs, BPR; giữ best validation toàn bộ 600 | Chỉ thêm ngân sách có đủ không? |
| B-restart | Từ selected teacher, thêm ≤100 epochs BPR, optimizer state như đã khai báo | Restart tại best checkpoint có giải thích lợi ích không? |
| B-SS | Như B-restart nhưng K32 sampled CE, không modality residual; item optimizer vẫn BSC baseline | Loss và negative count có đủ không? |
| R-ID | Frozen teacher + head cùng chiều/count tham số, K32; inputs chỉ từ frozen ID vectors | Thêm capacity/second stage có giải thích lợi ích không? |
| R-MM-BPR | Head đề xuất, K1,T1 BPR | Giá trị của nội dung khi giữ objective quen thuộc |
| **R-MM-SS** | **STAIR-RAM hoàn chỉnh, K32** | Hiệu quả tổng thể của đề xuất |

B-long tiếp tục từ epoch500/optimizer500; B-restart và B-SS bắt đầu từ **selected epoch** chứ không epoch500. Báo tổng thời gian gồm 500 epochs tìm teacher, không trừ các epoch “sau best”. Giữ optimizer moments tương ứng nếu đã lưu; trường hợp reset phải ghi và áp dụng đối xứng.

Định nghĩa R-ID cụ thể để tránh control quá yếu: tạo hai fixed semi-orthogonal lifts J_m: R^64→R^128 của normalized frozen item vectors, x_i^(ID,m)=J_m norm(h_i^0). Dùng cùng history/LOO/projector/gate/residual và training schedule. J_m không học; head vẫn 20,613 tham số và chiều ranking 128. Đây là control capacity có ý nghĩa, dù phân phối input không thể hoàn toàn đồng nhất với nội dung.

Ngoài epoch/update matching, báo matching theo wall-clock và số candidate scores. K32 xét nhiều negative hơn K1; không gọi hai run “same compute” chỉ vì cùng epochs.

### 10.2. Mechanism ablations sau khi core có tín hiệu

| Ablation | Điều được kiểm tra |
|---|---|
| Fixed mixture .5/.5 và learned global 2-way gate | Personalization theo user có cần thiết không? |
| No-history: B_m=0, giữ hidden width; báo parameter count thay đổi | History bổ sung gì ngoài frozen user embedding? |
| No-LOO train profile | Có shortcut và train/eval shift không? Không lấy train loss thấp làm thành công |
| Text-only / image-only | Nguồn nào tạo gain; gate có phù hợp với kết quả ablation không? |
| Same head với modality row-permutation cố định | Content–item alignment có tạo giá trị không? Giữ mapping permutation cố định, train lại |
| Separate PCA64 vs PCA128 | Gain do giữ modality hay chỉ tăng raw feature capacity? |
| Matched-input early fusion | Ghép x_t,x_v rồi fixed projection về tổng dimension tương đương; cân head capacity, không dùng phép cộng basis ngẫu nhiên làm strawman |
| Eta cap và r16/r32 | Expressivity–stability–runtime tradeoff |

Không chạy mọi tổ hợp Cartesian. Chọn ablation theo câu hỏi cụ thể sau core pilot; mọi thay đổi configuration vẫn chỉ dựa validation.

### 10.3. Diagnostics để bác bỏ giả thuyết, không chỉ minh họa

- Gradient norm từng branch, gate, theta; phân vị |residual| và ratio với |s0|; tanh saturation; tỷ lệ step bị clip.
- Histogram baseline positive–negative margins và gradient magnitude của Eq.11; tỷ lệ row gần như không còn tín hiệu. Loss giảm mà validation không tăng là dấu hiệu không đủ.
- Validation-only top-20 boundary audit: với relevant item bị miss, tính khoảng cách đến item ở boundary baseline; tỷ lệ gap≤2eta là **điều kiện cần đơn giản cho việc sửa từng cặp**, không phải upper bound chính xác của recall toàn hệ thống.
- NDCG/Recall theo train-degree quantiles của user và popularity item; cohort cutoffs được tạo từ train, không chọn cohort sau khi xem test.
- Rank displacement, thay đổi top20 overlap; gain/loss theo text/image ablation. Gate entropy một mình không chứng minh expert học đúng ý nghĩa.
- Cost: PCA/graph prep, mỗi stage, mỗi validation, selected test, peak allocated/reserved, CPU RAM, current allocated sau epoch; report cả cold-cache và warm-cache khi tái dùng teacher.

## 11. Tiêu chí >6%, thống kê và kế hoạch chạy có điểm dừng

### 11.1. Định nghĩa chính xác thành công

Với S paired seeds:

\[
\bar M_{d,m}^{(v)}=\frac1S\sum_sM_{d,m,s}^{(v)},\qquad
G_{d,m}=100\left(\frac{\bar M_{d,m}^{(v5)}}
{\bar M_{d,m}^{(B0)}}-1\right).
\tag{13}
\]

Mục tiêu đề bài được coi là đạt ở mức **point estimate** khi:

\[
\forall d\in\{\mathrm{Baby,Sports,Electronics}\}:
\sum_{m\in\{R10,R20,N10,N20\}}\mathbf1[G_{d,m}>6]\ge3.
\tag{14}
\]

Đề xuất thêm guardrail: mean R@20 và NDCG@20 không giảm trên dataset nào. Chín ô pass gộp trên 12 ô **không đủ** nếu một dataset chỉ pass một metric. R@1 là secondary diagnostic, không thay một metric khó bằng R@1 sau khi xem kết quả.

Ngưỡng minh họa từ baseline lịch sử, cần **vượt** các giá trị sau để tăng trên 6%:

| Dataset | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| Baby | .071444 | .110452 | .038054 | .048124 |
| Sports | .078758 | .117766 | .042930 | .053000 |
| Electronics | .046852 | .070490 | .026076 | .032118 |

Ngưỡng thực tế phải tính lại từ **mean B0 mới ghép cặp**, không cherry-pick baseline lịch sử thấp hơn.

### 11.2. Uncertainty và test disclosure

Seeds pilot dự kiến {1,2,3}; confirmation ít nhất {1,2,3,4,5} với cấu hình đã khóa, báo rõ ba seed pilot đã tham gia chọn hyperparameters. Nếu có ngân sách, thêm seed mới 6–10 làm tập xác nhận ít thích nghi hơn. Cùng seed không bảo đảm cùng random stream khi architecture khác; pairing dựa cùng split, teacher, data order/candidates trong các head arms và cùng điều kiện phần cứng.

Báo mean±SD từng metric, paired absolute differences và gain Eq.13; bootstrap theo **seed pairs** chỉ là CI mô tả ở S nhỏ. User-level paired bootstrap conditional trên model có thể bổ sung nhưng không thay seed uncertainty; users chung graph cũng không độc lập hoàn toàn.

Với 5 seed, exact two-sided sign-flip test chỉ có 2^5 cấu hình dấu; p nhỏ nhất là 2/32=.0625 nếu không có ties. Vì vậy không hứa p<.05 bằng phép kiểm định này. Paired t-test ở n nhỏ có giả định mạnh; phải nói rõ nếu sử dụng. Muốn kết luận “gain ít nhất 6% được xác nhận thống kê”, cần CI/kiểm định cho M_v5−1.06 M_B0 và ngân sách seed phù hợp, khác với chỉ point estimate>6%.

NDCG@20 mỗi dataset là metric chính cho chọn mô hình; family confirmatory gồm ba dataset cần điều chỉnh multiplicity phù hợp như Holm. Nếu đưa kết luận về cả 12 metric thì xác định family 12 từ trước; không chọn phép kiểm định có p đẹp nhất.

Test chỉ mở sau khi configuration và checkpoint selection đã khóa. Các tập test lịch sử đã nhiều lần được xem và ảnh hưởng quyết định nghiên cứu, nên benchmark không còn là test hoàn toàn “chưa từng nhìn”. Muốn khẳng định generalization mạnh hơn cần split/dataset bổ sung được chốt trước; không reset lịch sử bằng cách đổi tên v5.

### 11.3. Trình tự chạy tiết kiệm và quy tắc dừng

1. **Gate kỹ thuật:** synthetic contracts + smoke một epoch từng dataset + Gate-0. Không chạy 500 epochs nếu hash/protocol/sampler sai.
2. **Teacher audit:** tái dùng checkpoint chỉ khi source/data/evaluator hashes tương thích; nếu không, tạo B0 mới. Baby và Sports trước để phát hiện hướng không có lợi.
3. **Pilot Baby:** tối đa sáu cấu hình c_eta×LR, một seed, ≤50 head epochs; chọn theo validation N@20. Test chưa mở. Sau đó kiểm tra cấu hình tốt nhất trên ba seed và các controls quan trọng.
4. **Pilot Sports:** cùng core configuration, điều chỉnh tối đa cùng search budget công khai; chạy R-MM-SS, B-SS và R-ID ít nhất. Không chọn cấu hình chỉ vì test cũ thuận lợi.
5. **Electronics có điều kiện:** smoke/memory có thể làm sớm; full accuracy runs sau khi hai pilot cho tín hiệu. Nếu Baby/Sports không có validation gain ổn định, chưa tiêu tốn toàn bộ budget Electronics.
6. **Confirmation:** configuration khóa; B0 và primary v5 trên cả ba dataset, 5 seed tối thiểu. Gộp dữ liệu chỉ khi run manifests đủ và cùng protocol; run lỗi là missing, không ghi 0.

Heuristic ưu tiên tài nguyên: tiếp tục xác nhận khi pilot có mean validation N@20 gain khoảng ≥2% trên Baby và Sports, cùng dấu ≥2/3 seed, không có R20 degradation >1%, và head không thua rõ B-SS/R-ID. Đây là **decision rule**, không ngưỡng ý nghĩa thống kê hay dự báo sẽ đạt 6%. Nếu không đạt, dừng tuning mù và dùng diagnostics để quyết định:

| Quan sát | Kết luận/điều chỉnh hợp lý |
|---|---|
| B-SS bằng hoặc hơn R-MM-SS | Ưu tiên objective-only upgrade; chưa có đóng góp modality residual |
| R-ID bằng R-MM-SS | Extra capacity/second-stage optimization có thể giải thích gain |
| Fixed gate bằng learned gate | Chọn fusion đơn giản; không nhận thành công personalization |
| Eta chạm cap, nhiều boundary gaps>2eta | Core có thể thiếu biên độ; thử grid đã đăng ký trước, không bỏ bound âm thầm |
| Gate/branch gradients gần 0, negatives quá dễ | Kiểm tra implementation và sampler; semi-hard chỉ là ablation mới |
| Validation tăng nhưng held-out confirmation/test giảm | Overfitting/adaptation plausible; báo thất bại, không chọn epoch theo test |
| Chỉ Electronics tăng | Có kết quả theo dataset, nhưng không đạt mục tiêu cả ba |
| Tăng 2–5% nhất quán, chi phí thấp | Có thể là cải tiến thực dụng; vẫn phải ghi chưa đạt >6% |

Không có bảng “v5 dự kiến R20=...” hay thời gian “15 phút chắc chắn”. BPR/CE loss thấp không thay cho Recall/NDCG ở checkpoint được validation chọn.

## 12. Methodology-focus review và quyết định sau phản biện

| Mức độ | Phản biện mạnh nhất | Cách xử lý trong thiết kế | Điều còn chưa biết |
|---|---|---|---|
| Major | Đây có thể chỉ là thêm feature head và nhiều negative | B-SS, R-ID, R-MM-BPR bắt buộc; prior art VBPR/MGCN ghi rõ | Novelty và causal attribution phụ thuộc ablations |
| Major | Frozen backbone không bảo vệ ranking | Score bound + epoch0 candidate; bỏ mọi claim non-regression guarantee | Test có thể giảm dù validation tăng |
| Major | Cap khiến 6% bất khả thi | Calibrate theo baseline margins, audit boundary, c_eta grid | Giới hạn expressive thực tế chưa đo |
| Major | Khởi tạo amplitude 0 khóa projectors | Nonzero init cho training, Gate-0 riêng, multistep gradients | Gradient vẫn có thể yếu nếu base đã phân tách negative |
| Major | LOO bị quảng bá quá mức | Chỉ bỏ trực tiếp positive khỏi auxiliary history; thừa nhận teacher đã học positive | Train/eval shift, history đa sở thích |
| Major | Nhiều modality có thể thêm nhiễu | Single-modality, permutation, missingness/cohort audit | Có thể raw feature không chứa signal bổ sung |
| Major | Stage B và chiều ranking tăng chi phí | Báo tổng A+B+prep+eval; width 128; matched controls và budgets | Không có wall-clock/VRAM v5 đo thực |
| Major | So sánh checkpoint/mẫu số cũ sai | Ma trận selected-only, paired B0 mới, manifest lock | Lịch sử thiếu provenance chưa thể sửa bằng suy luận |
| Major | 5 seeds bị dùng để khẳng định quá mạnh | Nêu min exact p=.0625, uncertainty và multiplicity | Chưa đủ statistical power cho claim >6% mạnh |
| Minor | User-only gate thiếu interaction-level flexibility | Chấp nhận đánh đổi để exact separable ranking | Có thể pair-conditioned head tốt hơn nhưng đắt hơn |

**Verdict:** có thể chuyển sang implementation/pilot có kiểm soát sau khi các contract được cài đặt, **chưa đủ bằng chứng để gọi SOTA hoặc cam kết vượt 6%**. Thiết kế này đáng thử hơn việc cộng thêm cơ chế không được chẩn đoán, vì giả thuyết, phương trình, control và điều kiện bác bỏ đều cụ thể.

### Arithmetic receipts và claim ledger

- Phần trăm lịch sử được tính lại bằng công thức tại §2.1 từ cùng selected checkpoint; số làm tròn có thể lệch vài phần trăm của một điểm phần trăm so với full precision.
- Bảng paired v4 dùng JSONL precision cao; không gán CI hay p-value cho một seed.
- Eq.9, Eq.10, K1→BPR và giới hạn sign-flip là mệnh đề đại số; chúng không phải kết quả thực nghiệm v5.
- Kiểm tra tensor tổng hợp float64 ngày 26/09/2026: Eq.7 so với Eq.10 sai khác lớn nhất 4.44×10^−16 trong fixture có amplitude âm; residual/margin bounds pass; K1→BPR khớp cả value và gradient; fixture amplitude 0 chặn branch gradient, initialization khác 0 mở gradient. Đây là kiểm tra đại số nhỏ, không phải implementation v5 hoàn chỉnh hay recommendation run.
- **no_recomputable_statistics:** chưa có thống kê suy luận của v5 để kiểm tra lại; chưa có test statistic/df/p hoặc effect CI được báo như kết quả thực.
- **Observed:** kết quả selected-checkpoint lịch sử và các hạn chế provenance ở §2.
- **Established by construction:** frozen gradient, residual bound, score separability, exact disabled path nếu implementation tuân thủ.
- **Hypothesized:** modality profiles bổ sung tín hiệu, personalization có ích, K32 cải thiện ranking, tổng gain đạt >6%.
- **Not claimed:** quantum advantage, chống oversmoothing đã chứng minh, tối ưu toàn cục, cold-start generalization, test non-regression hoặc vượt mọi SOTA.

**Kết quả đầu ra mong muốn của giai đoạn tiếp theo** là một bảng so sánh ghép cặp có thể tái kiểm tra cho cả ba dataset, không phải chỉ một run loss đẹp. Nếu v5 thất bại, bộ controls vẫn phải trả lời được giới hạn nằm ở content signal, sampling, capacity, biên độ residual hay budget.
