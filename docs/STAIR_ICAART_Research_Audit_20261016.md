# STAIR — Audit kết quả giai đoạn 5–6 và quyết định nghiên cứu cho ICAART

**Ngày đối soát:** 16/10/2026, Asia/Saigon.  
**Quyết định:** chọn **STAIR5-v8.1 / BSCV, 64D** làm ứng viên paper chính hiện tại; giữ V8 làm đối chứng bắt buộc. Chưa chốt learned gate là thành phần cần thiết, chưa chốt 256D là cấu hình tốt hơn, chưa đủ bằng chứng gọi kết quả là SOTA hoặc có ý nghĩa thống kê.

> **Bổ sung sau khi tác giả xác nhận CoGate đã nộp và đang chờ kết quả:** quyết định chọn model ở đây không phải green light cho một archival submission độc lập song song. CoGate cùng nhóm đã chứa phần lớn auxiliary consensus-gating route. Xem [audit overlap CoGate–V8.1](giai_doan_5/STAIR5_v81_CoGate_Overlap_Audit.md) trước khi chốt hình thức công bố; substantial independent contribution chưa được xác lập.

## 1. Phạm vi và mức độ kiểm chứng

Áp dụng `academic-research-suite`: targeted deep research và methodology-focus review. Đây là phân tích cùng ngữ cảnh do một lượt nghiên cứu tổng hợp, **NOT_CALIBRATED**, không phải nhiều reviewer độc lập và không dự báo xác suất được nhận bài.

- Index 27 tài liệu kiến trúc/triển khai/thực nghiệm thuộc giai đoạn5–6; xem trajectory, protocol và các bảng kết quả liên quan. Không tuyên bố đọc toàn bộ từng đoạn của cả27 tài liệu dài.
- Đọc chương `report/chapters_v6/03_stair.tex`; đối chiếu báo cáo V8.1 với raw log, manifests và 500 dòng telemetry mỗi run.
- Trích TEST **sau `Load best model`**, không lấy bảng summary trước đó thay cho selected TEST. Bổ sung Electronics của GĐ6-v2 mà audit14/10 còn thiếu.
- Kiểm tra nguồn chính thức ICAART/ICORE và các paper gần cơ chế hiện tại. Không chạy lại training, không sửa source hoặc nội dung lịch sử của experiment reports/LaTeX trong lượt này.

Thứ tự bằng chứng: selected TEST log → manifest cùng run → telemetry → báo cáo diễn giải. Report và CSV tổng hợp có thể chép lại số sai; không dùng chúng để xác nhận chính chúng.

Các receipt ở `reports/stair_icaart_audit_20261016/`: `selected_test.csv` có12 dòng được đối soát; `run_evidence.json` lưu provenance/telemetry; `comparability_checks.json` kiểm tra V8/V8.1; `material_inventory.json` lưu danh mục và SHA256 tài liệu. Các đường dẫn source và dòng log nằm trong CSV.

Nhãn: **FACT** = trực tiếp từ source/log; **OBSERVATION** = số đo; **INTERPRETATION** = cách giải thích có thể khác; **HYPOTHESIS** = cần thí nghiệm phân biệt.

## 2. Hiện tại phiên bản nào tốt nhất?

### 2.1 Selected TEST mới nhất, dimension64, seed1

| Giai đoạn / phiên bản | Dataset | R@10 | R@20 | N@10 | N@20 | Selected epoch |
|---|---|---:|---:|---:|---:|---:|
| GĐ5 V8 | Baby | .0675 | .1054 | .0365 | .0462 | 430 |
| GĐ5 V8.1 BSCV | Baby | **.0679** | .1054 | **.0366** | **.0463** | 430 |
| GĐ6 V1 LCAE | Baby | .0674 | .1050 | .0365 | .0462 | 430 |
| GĐ6 V2 AKP | Baby | .0674 | .1054 | .0365 | **.0463** | 430 |
| GĐ5 V8 | Sports | .0762 | .1142 | .0420 | .0518 | 500 |
| GĐ5 V8.1 BSCV | Sports | **.0767** | **.1149** | **.0423** | **.0521** | 500 |
| GĐ6 V1 LCAE | Sports | .0760 | .1142 | .0420 | .0519 | 500 |
| GĐ6 V2 AKP | Sports | .0759 | .1141 | .0419 | .0518 | 500 |
| GĐ5 V8 | Electronics | .0460 | .0680 | .0260 | .0317 | 500 |
| GĐ5 V8.1 BSCV | Electronics | **.0462** | **.0683** | **.0261** | **.0318** | 500 |
| GĐ6 V1 LCAE | Electronics | .0461 | .0680 | .0260 | .0317 | 500 |
| GĐ6 V2 AKP | Electronics | .0460 | .0678 | .0260 | .0317 | **495** |

**FACT:** GĐ6-v2 đã hoàn tất Electronics; artifact nằm trong thư mục lồng `logs/GD6/stair6_v2_artifacts/stair6_v2_artifacts/`. Selected epoch495, không phải500. Vì vậy nhận xét cũ “V2 thiếu Electronics” đã lỗi thời.

**INTERPRETATION:** V8.1 có bộ kết quả cân bằng nhất hiện tại: cùng mức N20 cao nhất Baby với AKP và cao hơn hai hướng GĐ6 trên Sports/Electronics, đồng thời có incremental comparison khá sạch với V8. Đây là **best observed paper candidate**, chưa phải winner thống kê.

Không có winner mọi metric: GĐ5-v4/v6 vẫn khoảng R20=.1056 Baby, nhỉnh hơn .1054 của V8.1; R@1 Sports của V8.1=.0151 thấp hơn một số run .0152. Không viết “mọi chỉ số đều cao nhất”. Các headline V7 .0520/.0318 ở bảng tổng hợp vẫn không thay được selected ledger đã audit trước đó.

### 2.2 Incremental gain thực sự của consensus

Tính `100(new/old−1)` từ **cùng precision selected TEST bốn chữ số**:

| Dataset | ΔR10 vs V8 | ΔR20 vs V8 | ΔN10 vs V8 | ΔN20 vs V8 |
|---|---:|---:|---:|---:|
| Baby | +0.593% | 0.000% | +0.274% | +0.216% |
| Sports | +0.656% | +0.613% | +0.714% | +0.579% |
| Electronics | +0.435% | +0.441% | +0.385% | +0.315% |

Các chênh lệch N20 tuyệt đối là .0001/.0003/.0001. Đây là cải thiện nhỏ có hướng tích cực; số làm tròn và một seed không cho phép khẳng định vượt random-seed variance. Không biến chúng thành “bước nhảy5–7% của consensus”.

Baseline công bố và baseline tái lập là hai comparator khác nhau:

| Dataset | STAIR paper N20 | STAIR tái lập lịch sử N20 | V8.1 N20 | Δ vs paper, mô tả | Δ vs tái lập lịch sử, mô tả |
|---|---:|---:|---:|---:|---:|
| Baby | .0453 | .0454 | .0463 | +2.21% | +1.98% |
| Sports | .0503 | .0500 | .0521 | +3.58% | +4.20% |
| Electronics | .0302 | .0303 | .0318 | +5.30% | +4.95% |

Paper STAIR Table2 có kết quả qua5 independent runs, trong khi V8.1 hiện là seed1. Do đó bảng trên chỉ là đối chiếu mô tả, không phải một kiểm định head-to-head. Rerun STAIR cùng pipeline/runtime/seeds là đối chứng chính để viết bài. [STAIR, bản chính thức AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/33407/35562).

### 2.3 V8 → V8.1 có phải phép so sánh đáng tin hơn các đời trước?

**FACT:** ba cặp manifests có cùng `v8_config`, PyTorch2.11.0+cu128, TeslaT4, FP32/TF32 policy, data/split/feature fingerprints, candidate IDs và BSC graph fingerprint. V8.1 dùng commit `3e6cefafa817de4a935a8d03875211d541d2795d`, soft consensus, learned gate, rho=.2, warmup/ramp30/30.

Maximum absolute BPR-loss difference trong epochs1–30: Baby5.66e-9, Sports1.54e-9, Electronics6.33e-10. Điều này hỗ trợ recovery trước activation; không phải bitwise equality trên GPU và không thay được multi-seed.

**INTERPRETATION:** bằng chứng hiện có phù hợp với tác động của auxiliary route, khó giải thích đơn thuần bằng đổi graph support hoặc embedding dimension. Tuy nhiên chưa xác định lợi ích đến từ semantic conditioning, norm cap hay learned gate.

## 3. Bài học từ các phiên bản giai đoạn5–6

| Hướng | Bằng chứng/giới hạn | Quyết định hiện tại |
|---|---|---|
| GĐ5-v1 LHC | Phức tạp hơn, chi phí lớn; selected kết quả không ổn định; không dùng last epoch để cứu headline | Không quay lại làm primary |
| GĐ5-v2 C-HET, v3 DP-PC-BSC | Nhiều run gần control; tối ưu compute không đồng nghĩa tăng chất lượng | Chỉ giữ bài học engineering |
| GĐ5-v4 NLGCL-CSE | Cột mốc tăng metric rõ hơn các refinement sau | Comparator/ablation bắt buộc |
| GĐ5-v5.1 KPE | Không có incremental improvement thuyết phục; report cũ có chọn lọc precision/comparator | Không mặc định false-negative filtering là bottleneck chính |
| GĐ5-v5.2 BPE | Tăng topology rồi regression; thiếu controls nên semantic dilution chỉ là giả thuyết | Không mở rộng hop/graph một cách mặc định |
| GĐ5-v6 BCSR | Operator dosage rất nhỏ, gần parity | Chứng minh normalization không đủ chứng minh utility |
| GĐ5-v7 UCR-D | Chi phí động; headline selected có sai khác; không chứng minh superiority qua seed | Không thêm dynamic snapshots vào primary |
| GĐ5-v8 WMSG-CSE | Nền static, sparse, 3 datasets; WMSG dùng `.05+.95*rectcos²` | Nền đối chứng chính |
| GĐ5-v8.1 BSCV | Incremental gain nhỏ trên cả3 N20; giữ graph/scorer V8; có diagnostics | Ứng viên paper ưu tiên |
| GĐ6-v1 LCAE | Tự sinh clusters thay metadata thiếu; không tăng primary metrics nhất quán | Không gộp vào BSCV hiện tại |
| GĐ6-v2 AKP | KKT chính xác nhưng ranking gần parity; Electronics selected R20 giảm | Backup về sparsification, không là paper primary |

Các diễn giải nguyên nhân ở đây không phải causal proof. Không có ablation phù hợp thì “semantic dilution”, “over-smoothing” hoặc “loss interference” chỉ là alternative hypotheses.

## 4. Review V8.1: điều gì đã chứng minh và điều gì chưa?

### 4.1 Các bảo đảm đúng

Với static symmetric nonnegative `W_C`, đúng row degrees và isolated rows0, normalized `C` có `||C||₂≤1`. Với cap và gate trong[0,1], residual thỏa `||J_i−I_i¹||≤rho_e||I_i¹||`. Đây là **forward perturbation bounds**; không chứng minh embedding trajectory không drift, BPR/CL không xung đột, optimizer hội tụ hoặc ranking tốt hơn.

Auxiliary scalar context có rotation-invariant forward values; module equivariant khi cùng quay embeddings. Toàn STAIR vẫn coordinate-dependent do FSC/BSC. MLP không tự giải quyết basis dependence hay independent modality whitening.

### 4.2 Telemetry thực tế

Mean dưới đây là **unweighted mean theo470 active epochs**, mỗi epoch đã có sampled unique-row summaries. Không phải mean toàn catalog hoặc energy decomposition.

| Diagnostic | Baby | Sports | Electronics |
|---|---:|---:|---:|
| Mean gate, active epochs | .93895 | .93786 | .98414 |
| Gate cuối epoch500 | .95800 | .95456 | .99494 |
| Gate saturation fraction cuối, g<.05 hoặc g>.95 | 87.13% | 78.92% | 100.00% |
| Cap-active fraction, mean | 80.18% | 41.57% | 17.19% |
| Residual norm ratio, mean | .16501 | .13026 | .10321 |
| Angular shift, mean radians | .13003 | .09033 | .07273 |
| Perpendicular norm fraction, mean | .75192 | .61393 | .66293 |
| **Zero-message fraction, mean** | **12.86%** | **21.09%** | **15.65%** |
| Static C covered rows / catalog | 4,602/7,050 | 10,095/18,357 | 34,536/63,001 |

**OBSERVATION:** Electronics gần như chọn maximum gate; last-batch gate gradient cuối≈1.22e-10. Điều này có thể là dose tối ưu nằm sát trần, sigmoid saturation, hoặc learned adaptation không cần thiết. Không đủ để gọi gradient collapse của toàn mô hình; càng không chứng minh MLP học reliability chính xác.

**HYPOTHESIS ưu tiên kiểm định:** một **fixed strength** gần1 có thể đạt hiệu quả như learned gate. Nếu đúng, bỏ MLP làm final model sẽ đơn giản và dễ bảo vệ hơn; không thêm entropy regularization chỉ để ép gate trông đa dạng.

**FACT:** floor=.05 là mức nền của edge weight, không phải hard semantic threshold. `zero_message_fraction` không đếm trực tiếp item thiếu semantic cosine≥.05. Nó đếm sampled rows có message norm<eps; có thể do thiếu CF support hoặc vector cancellation. Catalog coverage và sampled-batch zero rate có mẫu số khác nhau. Report đang ghi ~0% Baby/Sports là sai.

Perpendicular fraction là `||m−proj_x m||/||m||`, không phải tỷ lệ “tri thức mới”. Nhiễu cũng có thể trực giao. Energy fraction phải bình phương từng row rồi aggregate; không lấy square của mean norm fraction để thay. Cần semantic-placebo, CF-only và ranking-margin ablations mới suy luận được utility.

### 4.3 Không có quy luật “dataset càng lớn/càng dày thì cap càng ít” đã được xác nhận

| Dataset | Train interactions | Train density | Mean item degree |
|---|---:|---:|---:|
| Baby | 118,551 | .08648% | 16.82 |
| Sports | 218,409 | .03342% | 11.90 |
| Electronics | 1,254,441 | .01035% | 19.91 |

Scale, density và degree khác nhau. Sports có average item degree thấp hơn Baby, trong khi cap cũng thấp hơn. Norm ratios còn chịu MI whitening, gamma, operator normalization, degree distribution và embedding trajectory. Phân tích theo within-dataset degree quintile/norm bins mới phân biệt được các giả thuyết.

## 5. Chi phí thực: không gọi zero training overhead

| Dataset | V8 mean train s/epoch | V8.1 mean train s/epoch | Δtrain | V8 max allocated MiB | V8.1 max allocated MiB | V8.1 max reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| Baby | 2.1990 | 2.5135 | +14.30% | 145.17 | 148.23 | 246 |
| Sports | 4.6783 | 5.5123 | +17.83% | 223.71 | 224.61 | 462 |
| Electronics | 59.0762 | 68.2917 | +15.60% | 800.67 | 804.22 | 1,584 |

Native `Coach.fit` wall time V8.1: Baby1316.30s≈21.94min, Sports2918.16s≈48.64min, Electronics36326.20s≈10.09h. Mean training counters và whole fit là hai scope khác nhau. Gate41 parameters rất nhỏ nhưng full SpMM/backward vẫn tăng training cost.

**Giữ được claim:** không thêm route vào inference graph/scorer. **Không giữ:** zero training cost, total VRAM<805MiB, hoặc giảm bộ nhớ X% so với paper dùng RTX3090/cách đo khác. Allocated khác reserved, NVML/process, preprocessing và evaluation peaks. Một lần chạy không chứng minh timing causality so với GĐ6.

## 6. Review trực tiếp `report/chapters_v6/03_stair.tex`

| Vị trí | Vấn đề | Sửa trước khi dùng cho paper |
|---|---|---|
| Dòng3–7 và captions | BSCV được đổi tên STAIR6-v1, trong khi code/report hiện có STAIR6-v1=LCAE | Dùng tên phương pháp BSCV độc lập version hoặc thêm explicit mapping; không gộp hai kiến trúc |
| Dòng105 | Mũi tên S8→FSC dễ hiểu thành FSC dùng item graph S8 | FSC dùng normalized user–item Adj; S8 đi vào item BSC ở optimizer |
| Dòng144–200 | Baseline (Paper) khác baseline tái lập của MD | Các số paper đã được Table2 xác nhận; giữ cột riêng, bổ sung rerun cùng điều kiện |
| Bảng/phần text về Baby | Precision cao từ summary có thể thuộc epoch500 | Selected430 hiện chỉ có log4 decimals; không gán .012979/.067961/.036799 của last500 cho selected430 |
| Dòng214 | Cap activations bị diễn giải thành đã ngăn collapse/tail drift | Giữ observation; cần no-cap, degree-stratum và effective-rank controls |
| Dòng215 | “61–75% tri thức mới”, “energy”, suy ra nonredundancy/utility | Đổi thành observed perpendicular norm fraction; bỏ causal knowledge claim |
| Dòng216 | Zero messages do cosine dưới.05; “item ngách” chưa đo | Đổi thành sampled zero-message fraction, nêu thiếu CF coverage/cancellation; đo tail trước khi gán nhãn |
| Dòng233–247 | Mean allocated được so như peak; so hardware paper khác run | Báo scope/cùng hardware, max/mean/reserved riêng; không khẳng định tiết kiệm causal |
| Toàn đoạn | “vượt bậc”, “chứng minh”, “toàn diện” | Thay bằng measured single-seed small improvement, exceptions và uncertainty |

Công thức cap, query orientation và việc không thêm inference route cơ bản khớp implementation. Gate nhận gradient trực tiếp từ **CL route**, không từ BPR trong cùng forward vì BPR không phụ thuộc gate; BPR có thể tác động gián tiếp qua shared embeddings ở những bước sau.

FreeRec `fit()` gọi last-epoch test → summary → `eval_at_best()`. Bảng summary có thể chứa high-precision **last TEST**. Baby minh họa rõ: last500 N20=.046184/R20=.104502, selected430 N20=.0463/R20=.1054. Sports/Electronics selected500 nên khác biệt này không đổi checkpoint, nhưng không được trộn precision khi tính relative gain.

## 7. Literature: đóng góp nào còn có thể bảo vệ?

Đây là targeted review, không phải systematic survey. Chỉ dùng nguồn sơ cấp; không chuyển relative gains của paper khác thành dự báo cho STAIR.

| Nguồn đã xác minh | Authors / venue / year | Cơ chế liên quan và implication |
|---|---|---|
| [STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33407) | Cong Xu, Yunhang He, Jun Wang, Wei Zhang; AAAI2025; DOI10.1609/aaai.v39i12.33407 | MI/FSC/BSC là borrowed backbone; không claim mới |
| [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://arxiv.org/abs/2211.06924) | Xin Zhou, Zhiqi Shen; ACM MM2023 | Static graph là lựa chọn có tiền lệ; dynamic không mặc nhiên tốt hơn |
| [Multi-View Graph Convolutional Network for Multimedia Recommendation](https://arxiv.org/abs/2308.03588) | Penghang Yu, Zhiyi Tan, Guanming Lu, Bing-Kun Bao; ACM MM2023; DOI10.1145/3581783.3613915 | Behavior-aware purification/fusion và auxiliary learning đã có; “adaptive multimodal gate” riêng lẻ không mới |
| [AlignRec: Aligning and Training in Multimodal Recommendations](https://arxiv.org/abs/2403.12384) | Yifan Liu, Kangning Zhang, Xiangyuan Ren, Yanhua Huang, Jiarui Jin, Yingjie Qin, Ruilong Su, Ruiwen Xu, Yong Yu, Weinan Zhang; CIKM2024 | Content/content-ID/user-item alignment; thêm alignment vào BSCV cần problem-specific evidence |
| [MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33408) | Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Hewei Wang, Edith C.H. Ngai; AAAI2025; DOI10.1609/aaai.v39i12.33408 | Alignment phải giữ interaction information; làm baseline gần cơ chế auxiliary supervision |
| [Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation](https://arxiv.org/abs/2412.14978) | Rongqing Kenneth Ong, Andy W.H. Khong; WSDM2025; DOI10.1145/3701551.3703561 | Fusion/noise suppression trong frequency domain; không đồng nhất FFT feature channels với graph spectrum |
| [Unveiling the Impact of Multi-modal Content in Multi-modal Recommender Systems](https://xmudm.github.io/files/Xv25MMUnveiling.pdf) | Guipeng Xv, Xinyu Li, Yi Liu, Chen Lin, Xiaoli Wang; ACM MM2025; DOI10.1145/3746027.3755300 | ISOLATOR chỉ ra user-side content bias; đánh giá content-consistent/diverse groups thay vì mặc định semantic agreement hữu ích cho mọi user |
| [One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation](https://arxiv.org/abs/2607.24607) | Jinfeng Xu, Zheyu Chen, Ziyue Peng, Shuo Yang, Jinze Li, Zewei Liu, Shujie Li, Yipeng Du, Edith C.H. Ngai; arXiv2026, metadata ghi accepted ACM MM2026 | IIMRec có semantic+co-occurrence, NCER, residual gate, graph/BPR augmentation: **related work gần nhất phải so sánh** |
| [MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning](https://arxiv.org/abs/2609.04574) | Ahmad Mousavi, Majid Alikhani, Yeon-Chang Lee, Roberto Corizzo, Yeganeh Abdollahinejad; arXiv2026, chưa xác nhận venue peer review | Adaptive topology và uncertainty fusion là hướng khác, không phải lý do bắt buộc thêm dynamic module |

Code tác giả được xác minh: [IIMRec](https://github.com/Jinfeng-Xu/IIMRec), [AlignRec](https://github.com/sjtulyf123/AlignRec_CIKM24), [SMORE](https://github.com/kennethorq/SMORE), [STAIR](https://github.com/yhhe2004/STAIR). IIMRec công bố Baby/Sports/Clothing/TikTok; phải port đúng preprocessing/evaluator trước khi đối chiếu với Electronics của đề tài.

**Integrity note:** arXiv [DA-MRS 2406.12501](https://arxiv.org/abs/2406.12501) hiện đánh dấu withdrawn ngày02/04/2025. Không lấy preprint đó làm bằng chứng đã qua review hoặc tự suy ra mọi bản xuất bản cùng tên đã bị rút. Bản publisher/journal tiếp nối cần xác minh riêng. Không sử dụng các con số gain từ citation chưa xác minh.

### 7.1 Novelty matrix

| Thành phần/claim | Trạng thái |
|---|---|
| MI/FSC/BSC, BPR, InfoNCE, graph normalization, scalar MLP, norm cap | Borrowed/standard; không đủ component novelty |
| CF + semantic confidence | Đã có tiền lệ; “consensus” không phải một concept mới |
| Giữ BSC/scorer cố định, chỉ bounded item-query CL residual trên existing support | Problem/combination positioning có thể bảo vệ; phải so sánh với IIMRec/MGCN/MENTOR và CoGate |
| Không thêm inference route | Engineering property rõ, không đồng nghĩa zero train overhead |
| Adaptive gate tạo gain | Chưa nhận dạng; saturation làm fixed-strength baseline cấp thiết |
| Topology expansion hoặc thêm auxiliary regularizers | Không cần chỉ để tạo novelty; tăng attribution burden |

Có triển vọng ICAART nếu phần hiệu quả/ablation/reproducibility hoàn chỉnh, nhưng không có bằng chứng cho xác suất được nhận. Tăng từ64 lên256 là capacity study, không tạo novelty riêng.

## 8. Có nên chạy V8.1-256D?

**Có — sau hoặc song song với paired replication64D, không chạy một mình rồi chốt model dựa trên TEST.**

| Arm | d | Consensus | Câu hỏi |
|---|---:|---|---|
| A | 64 | Off, V8 control | Nền đã có |
| B | 64 | Soft learned, V8.1 | Incremental auxiliary effect |
| C | 256 | Off, V8 control | Capacity + dimension-dependent STAIR schedule effect |
| D | 256 | Soft learned, V8.1 | Incremental consensus ở256D |

Với mỗi seed/dataset, đặt `Δ64=M(B)−M(A)`, `Δ256=M(D)−M(C)`, interaction `I=Δ256−Δ64`. D>B chứng minh hệ thống256 tốt hơn64 ở điều kiện đó; **không** chứng minh consensus giúp scaling. D>C và I>0 lặp lại mới hỗ trợ capacity–consensus synergy. Nếu C vàD cùng tăng nhưng gần nhau, phần gain chủ yếu là scaling.

CoGate manuscript nội bộ báo STAIR64→256 N20 Baby .0453→.0460, Sports .0503→.0542, Electronics .0302→.0350. Đây là **tín hiệu nội bộ để đặt giả thuyết**, chưa phải prediction hoặc validated result của V8.1. CoGate còn thay graph/gate/loss/negatives/tuning. Không sao chép headline của CoGate sang V8.1.

**Vì sao scaling có thể giúp:** MI giữ thêm SVD directions, user/item tables có thêm capacity, message chứa thêm phân biệt không gian. **Vì sao có thể hại:** whitening khuếch đại low-singular-value noise, dot-product/norm/regularization đổi scale, overfitting và redundancy. Native FSC/BSC schedule cũng được đánh giá lại trên256 coordinates; đây không phải tăng params trong một hàm số hoàn toàn giữ nguyên.

Giữ cùng static candidate IDs/S8/C; recompute MI và đúng beta3 theo d, không pad64 lên256. Đo singular-value tail/retained variance, embedding norms/effective rank, train–valid gap, gate/cap/angle diagnostics ở cả hai dimension. D256 có 4× table parameters, không được chỉ so với baseline64.

### 8.1 Thứ tự dataset và tài nguyên

Sports là pilot256 ưu tiên: signal consensus64 rõ nhất, runtime chưa quá lớn. Baby là confirmatory rẻ nhưng dễ saturation/overfit; Electronics chỉ sau khi đo cost và có validation trend. Không dùng early pilot100 epochs để dự báo chắc chắn thứ hạng checkpoint500.

FP32 embedding+grad+2 Adam moments riêng Electronics256: `16*(192403+63001)*256` bytes≈997.67MiB. Một full consensus message≈61.52MiB, chưa tính backward/workspace/loss/evaluation. Không áp mục tiêu800MiB của64D cho256D. Profile2–5 active epochs và full ranking trên GPU thật; không giả định wall time đúng4×.

## 9. Thực nghiệm nào nên làm trước để tăng sức thuyết phục?

### 9.1 Minimal sufficient ablation64D

| Arm | Thay đổi duy nhất | Kết quả có ý nghĩa |
|---|---|---|
| V8 off | Không auxiliary | Comparator |
| V8.1 learned | Bản hiện tại rho=.2 | Primary |
| Fixed-half | g=.5, rho=.2 | Kiểm tra learned vs fixed với dose thấp hơn; **chưa** là dose-matched |
| **Fixed-full strength** | g=1, effective rho=.2, cùng cap/C | Phân biệt benefit của adaptation với maximum dose |
| CF-only | Raw CF normalized, bỏ semantic reweighting | Semantic consensus có giá trị thật không? |
| Degree-stratified weight placebo | Cùng support/multiplier distribution, đảo semantic assignment | Agreement pattern hay generic perturbation tạo gain? |

Existing implementation có fixed gate=.5. Có thể chạy fixed-full strength **không thêm code** bằng `--consensus-gate fixed --consensus-rho 0.4`: effective coefficient=.4*.5=.2 với cùng warmup/ramp. Ghi rõ nominalrho=.4, effective bound=.2; không gọi đây là learned rho=.4 hay bỏ cap. Muốn code hỗ trợ hằng1 trực tiếp thì thêm ablation riêng sau, không âm thầm sửa primary.

Nếu fixed-full strength tương đương hoặc tốt hơn learned qua paired seeds, ưu tiên bounded consensus không MLP. Nếu CF-only bằng soft consensus, bỏ claim semantic conditioning. Nếu placebo bằng primary, core hypothesis về agreement pattern chưa được hỗ trợ. Không tránh các negative controls chỉ vì sợ paper yếu.

No-cap ablation có thể chạy riêng để kiểm định safety/benefit, bắt đầu nhỏ và kiểm tra finite loss; primary giữ cap. Không đồng thời đổi batch, early stopping, CL coefficient hoặc negative sampler trong ablation.

### 9.2 Hai nhánh nghiên cứu tiếp theo — chỉ nếu ablation yêu cầu

**Ưu tiên1: calibrated fixed-dose/simplified consensus.** Evidence hiện tại là gate saturation. Thử effective max dose {.10,.20,.30} trên cùng graph/cap, cùng validation budget; không search thêm dozens knobs. Causal chain: learned gate chủ yếu tăng amplitude → constant dose tái tạo message → query geometry tương tự với ít params → nếu ranking giữ nguyên, model đơn giản hơn và explanation mạnh hơn. Đây là simplification/hyperparameter study, chưa là breakthrough mechanism.

**Ưu tiên2 có điều kiện: coverage-aware auxiliary-only fallback.** Static C chỉ phủ~65.28%/54.99%/54.82% catalog; sampled zero message vẫn~12.86%/21.09%/15.65%. Chỉ nghiên cứu nếu uncovered/tail strata bị tụt hoặc nhận ít gain. Trên rows C cô lập, thử một-hop semantic message từ existing V8 semantic support, vẫn chỉ vào CL query:

\[
m_i'=m_i+\mathbf1[d_i^C=0]h_i(S_{sem,8}V^0)_i,
\quad 0\le h_i\le1,
\quad J_i=x_i+\rho g_i\operatorname{stopgrad}\min(1,\|x_i\|/(\|m_i'\|+\epsilon))m_i'.
\]

Ban đầu `h_i` là confidence cố định dựa trên existing raw edge scores, không thêm learned network. Đây là **HYPOTHESIS**, chưa được chọn làm final architecture. Fallback hợp nhất có thể directed và không có `||C'||≤1` như normalized C; forward cap bound vẫn áp dụng nhưng không chuyển spectral theorem cũ sang operator mới. Rủi ro: thiếu behavioral support, substitute noise, tăng sparse pass. Falsification: uncovered R20/N20 không tăng, overall regression hoặc shuffled fallback không kém. Không sửa BSC/topology scorer để giữ attribution; nếu phải đổi thêm nhiều chỗ thì dừng.

Không khuyến nghị ngay lúc này: channel gates, full CoGate stack, dynamic topology/EMA, multi-hop expansion, GMM category, learned hard negatives hoặc spectral fusion. Chưa có evidence rằng chúng là bottleneck còn lại trên V8.1, và sẽ tạo thêm một vòng nghiên cứu thay vì hoàn tất bài.

## 10. Diagnostics và thống kê để quyết định

1. **Gate necessity:** compare learned với fixed-full; log gradient và parameter movement, không chỉ mean gate.
2. **Dose utility:** angular shift và `||J−x||/||x||` theo degree/coverage bins; không chỉ residual mean toàn dataset.
3. **Ranking mechanism:** frozen validation positive-vs-sampled-negative margins với cùng negative IDs, delta per-user N20/R20; gate cao mà margins xấu bác bỏ reliability interpretation.
4. **Long-tail/coverage:** chọn item degree bins từ train; báo per-user và target-item-group metrics với mẫu số rõ ràng, catalog coverage, recommendation popularity.
5. **Content bias:** dùng history/content similarity từ train để phân nhóm; nếu dùng held-out outcomes cho descriptive stratification thì không đưa nhóm/labels vào tuning hoặc graph build. Kiểm tra consistent/diverse users theo gợi ý ISOLATOR.
6. **Norm cap:** measured ratio/bound compliance có FP32 tolerance; zero queries không đưa vào angular statistics như valid directions.

Chạy tối thiểu seeds{1,2,3}; nếu đủ budget thêm{4,5}, pairs dùng cùng splits, configs, sampler policy, epoch budget và evaluator. Report mean±SD, từng seed, absolute/relative deltas và CI của paired deltas. Không coi1500 epochs là1500 independent replications.

Với n=3, kiểm định seed-level rất yếu; exact two-sided sign-flip test có minimum p=.25. Với5 pairs, exact signed-rank/sign-flip minimum p=.0625 nếu không tie; không hứa Wilcoxon p<.05 từ5 runs. Paired t-test cần nêu giả định/phạm vi và uncertainty, không dùng như tem bảo đảm. Per-user bootstrap là uncertainty conditional on trained models, **không thay thế training-seed variance**. Dùng paired hierarchical analysis nếu có đủ runs; điều chỉnh multiplicity cho nhiều primary comparisons.

Primary frozen: N20 và R20, same-dimension V8 vs final consensus; không đổi sang R1 khi R20 không tăng. Validate lựa chọn config, chốt trước Test. Vì test đã được xem qua nhiều versions, lịch sử có selection bias; multi-seed trên cùng test không hoàn toàn xóa nó. Nếu budget cho phép, thêm một dataset/split confirmatory đã khóa trước để kiểm tra generalization.

GO khi paired mean positive, uncertainty hỗ trợ và không có regression nghiêm trọng ở dataset/tail; WEAK-GO nếu trend nằm trong variance; NO-GO nếu controlled seeds/placebo không hỗ trợ cơ chế. Không đặt một ngưỡng+5% tùy ý để cứu hoặc loại model.

## 11. ICAART: fit, deadline và gói bài cần có

**FACT:** ICORE2026 xếp ICAART **B**. Rank là một nguồn xếp hạng cụ thể, không quy đổi thành CCF-B, journal quartile hoặc cam kết accept. [ICORE portal](https://portal.core.edu.au/conf-ranks/?search=ICAART&by=all&source=all&sort=atitle&page=1).

Theo trang chính thức ngày16/10: ICAART2027 tổ chức23–26/02/2027; regular stage1 đã hết29/09; **stage2 regular/position deadline22/10/2026, AOE**, notification04/12, camera-ready18/12. Vì vậy còn khoảng6 ngày, không nên lên lịch một vòng redesign nhiều tuần cho chính đợt nộp này. [Important dates](https://icaart.scitevents.org/ImportantDates.aspx), [CFP](https://icaart.scitevents.org/CallForPapers.aspx).

Fit phù hợp Area2 Artificial Intelligence: ML/DL/data mining. Regular paper cần nghiên cứu hoàn tất; position paper dành cho work-in-progress. Guidelines hiện dùng10,000–50,000 ký tự không kể khoảng trắng cho regular submission, double-blind; publication full/short tương ứng12/8 trang. Kiểm tra template và chính sách AI disclosure/anonymity trước khi nộp; không suy ra indexation chắc chắn từ câu “submitted for indexation”. [Submission guidelines](https://icaart.scitevents.org/Guidelines.aspx).

### 11.1 Định vị paper nên chọn

Ứng viên thông điệp: **bounded auxiliary semantic–behavioral views improve a fixed sparse recommender without changing its ranking propagation or serving path**. Đây là claim cần ablation, chưa phải claim cuối cùng đã kiểm định. Không lấy “nhiều phiên bản”, “41 params”, “256D” hoặc “triple SOTA” làm novelty chính.

Related-work distinction bắt buộc: IIMRec có residual item gate + graph/optimization augmentation; BSCV đặt bounded residual **chỉ vào item query của CL**, giữ ranking/BSC backbone fixed. MGCN/MENTOR/AlignRec đã có adaptive fusion/alignment. Do đó paper cần chứng minh ích lợi của **vị trí can thiệp và bound**, thay vì chỉ nói dùng multimodal consensus.

Baseline package tối thiểu: STAIR rerun, V4, V8, V8.1; external strong recent controls ưu tiên MGCN/MENTOR/SMORE và IIMRec, cộng LightGCN/MF hoặc BM3 làm reference. Không copy paper numbers từ split/encoder khác vào một bảng “fair rerun”. Nếu chưa rerun được một method, phân cột published/reference rõ ràng và không dùng nó làm causal superiority claim.

### 11.2 Lộ trình thực tế16–22/10

| Công việc | Ưu tiên | Quyết định |
|---|---|---|
| Freeze source/splits/configs, sửa result extraction và claims | Ngay | Không mất thêm GPU |
| V8/V8.1 seeds2–3, cả3 datasets | P0 | Ước lượng~42.1 GPU-hours bổ sung theo fit64 lịch sử, chưa gồm queue/preprocessing; không là SLA |
| Fixed-full, CF-only, placebo trên Sports/Baby | P0/P1 | Hoàn tất mechanism tests; promote finalists sang Electronics khi có evidence |
| V8-256/V8.1-256 Sports một paired seed, active smoke trước | P1 | Là capacity pilot, không thay core64 replication |
| Baby256 confirmatory rồi Electronics256 | P2 | Chỉ nếu đủ thời gian; giữ64 làm main nếu256 chưa đủ controls |
| External baselines và viết paper song song | P0 | Nếu không đủ evidence trước22/10, giảm claim hoặc chọn kỳ sau; không hạ chuẩn vì deadline |

Ước lượng42.1h =2 seeds bổ sung × tổng fit một seed V8+V8.1 trên3 datasets≈21.06h. Không giả định25–40 full runs256 đều kịp. Nếu chỉ có một GPU, tránh dùng toàn bộ6 ngày cho Electronics256 rồi thiếu replicates/ablations.

## 12. Phán quyết cuối cùng

**IMPLEMENT WITH MODIFICATIONS TO THE EXPERIMENTAL EVIDENCE, NOT AN IMMEDIATE ARCHITECTURAL REDESIGN.**

- **Chọn V8.1-64D làm ứng viên paper hiện tại**, vì incremental evidence tốt hơn hướng LCAE/AKP và các ancestor refinement, không vì version mới nhất.
- **Chưa chốt learned gate**: fixed-full strength là ablation cấp thiết nhất từ saturation evidence.
- **Nên thử256D theo factorial có V8-256 control**; chưa có cơ sở hứa gain hay synergy. Chạy Sports trước; không bỏ multi-seed64 để chạy duy nhất D256.
- Nếu controlled tests xác nhận consensus, đóng băng kiến trúc và viết bài. Nếu fixed gate đủ, đơn giản hóa final. Nếu placebo/CF-only phủ nhận hypothesis, sửa hoặc bỏ claim thay vì thêm module để cứu metric.
- Hướng cải tiến dài hơn, nếu cần, là coverage-aware auxiliary-only fallback có diagnostics và falsification rõ; không ghép ngay category/adaptiveK/dynamic graph vào primary.

Hiện có một kết quả nghiên cứu đáng theo đuổi và đủ để bắt đầu manuscript; **chưa có cơ sở khoa học cho “SOTA trên cả ba tập”, “gate đã học đúng reliability”, hoặc bảo đảm được ICAART nhận**.
