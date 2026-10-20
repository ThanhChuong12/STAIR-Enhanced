# ECLIPSE — Evidence readiness và định hướng trình bày paper

**Audit ngày 2026-10-20.** Phạm vi: artifacts đã có, bản thảo hiện tại và định hướng Experiments/Ablation. Không chạy mô hình, không viết thay phần Experiments. Không có kế hoạch nào bảo đảm bài được nhận; mục tiêu là kết luận đúng bằng chứng và xử lý các phản biện có thể tránh được.

## 1. Quyết định hiện tại

**Giữ kiến trúc ECLIPSE đã chốt. A0 và A1 có thể dùng cho paper sau các đính chính.** Chưa nên gọi gói thực nghiệm đã hoàn thiện: benchmark ngoài nhóm, evidence controls và kết quả A2/A3 từ Hiếu vẫn cần hợp nhất/đối chiếu. Không mở thêm kiến trúc hoặc 256D trong đợt chốt này.

ECLIPSE là phương pháp cuối của nhóm. MI/FSC/BSC/AdamW/scorer có nguồn kế thừa STAIR; WMSG/CSE/LC là các phần mở rộng trong framework; PEEC là calibration trung tâm. Trình bày một phương pháp độc lập không có nghĩa bỏ dẫn nguồn STAIR hoặc nhận thành phần kế thừa là mới. Những version nội bộ không cần xuất hiện trong bảng external benchmark; dùng tên ablation cho các cấu hình rút gọn.

## 2. Bằng chứng nào đã có thật?

| Nội dung | Trạng thái tại workspace | Phạm vi sử dụng |
| --- | --- | --- |
| A0 Full, ba tập, seeds 1–3 | 9 receipts/telemetry; source/config đã đối chiếu; checkpoint s2/s3 có, s1 thiếu local | Mean ± SD, từng seed, selected epoch, tài nguyên có giới hạn đo |
| A1 w/o EC, ba tập, seeds 1–3 | 9 receipts/checkpoints; s1 recovered historical run | Paired analysis; công khai provenance khác revision ở s1 |
| A2 w/o LC / A3 w/o CSE | **HUMAN_REPORTED_COMPLETE**, artifacts ở máy Hiếu; Hiếu phân tích riêng | Chờ bảng/receipts; không tự suy phạm vi hoặc điền metric |
| A5 Count-only | Hoàn tất 100% phạm vi (7 runs): Baby seeds 1–3, Sports seeds 1–3, Electronics seed 1 | [Audit A5](ECLIPSE_Ablation_Study_Count_Only.md); Baby/Sports n=3, Electronics n=1; 7 runs COMPLETED |
| Binary / Shuffle / Global-full | Chưa tiếp nhận receipts mới của các arm này trong audit | Notebook READY không phải kết quả; ghi actual n khi nhận |
| STAIR tái lập | Có nguồn lịch sử, ngoài phạm vi 18-run audit này | Chỉ dùng số/protocol được kiểm tra, ghi n thực tế |
| Published baselines khác | Chưa lập source/protocol matrix trong lần này | Không nhập số khác splits như cùng một protocol |
| Per-user statistics | Chưa có arrays trong gói audit | Không tạo từ aggregate metrics |

[Audit 18 runs](../../reports/eclipse_multiseed_paper_audit_20261020/audit_acceptance.json): 676/676 đối chiếu đạt. Đây là artifact analysis, không phải 676 model tests hay độc lập tái lập metric. Ba checkpoint A0 s1 cần thu hồi từ nơi chạy gốc; chưa cần retrain chỉ vì thiếu file lưu trữ.

## 3. Kết quả định vị câu chuyện paper

| Dataset | Full NDCG@20, n=3 | w/o EC NDCG@20, n=3 | Relative mean gain EC | Wins | Raw paired t p | Holm3 p |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | 0.046340 ± 0.000241 | 0.046070 ± 0.000162 | +0.585% | 3/3 | 0.2745 | 0.5490 |
| Sports | 0.051956 ± 0.000156 | 0.051703 ± 0.000229 | +0.490% | 3/3 | 0.0446 | 0.1339 |
| Electronics | 0.031759 ± 0.000093 | 0.031715 ± 0.000030 | +0.138% | 2/3 | 0.5734 | 0.5734 |

Cần tách hai câu hỏi:

1. **Toàn bộ ECLIPSE có tốt so với published baselines không?** Cần external benchmark cùng protocol hoặc phân biệt rõ nguồn. A0–A1 không trả lời thay.
2. **Calibration có ích ngoài framework còn lại không?** A0–A1 cho mean gain nhỏ, dương nhưng Electronics không nhất quán qua seed; chưa có family-wise significance ở endpoint chính. Count/Shuffle kiểm tra tiếp tính đặc thù của estimator và vị trí evidence.

Không mô tả riêng EC là tạo toàn bộ gain ECLIPSE–STAIR. Không gọi “significant trên cả ba tập”, “loại trừ ngẫu nhiên”, “đạt SOTA” hoặc “parity đã chứng minh” từ bảng này. Không giấu kết quả âm Electronics s3. Gain nhỏ không buộc bỏ phương pháp; nó đòi hỏi claim vừa mức và controls thuyết phục hơn.

## 4. Bảng nên có trong paper

### Table 1 — Dataset statistics and protocol

Users, items, TRAIN/VALID/TEST interactions và số eligible evaluation users. Nếu quá rộng, đưa split details vào appendix. Số trong manifest:

| Dataset | Users | Items | TRAIN | Unique semantic pairs | Final operator nnz |
| --- | --- | --- | --- | --- | --- |
| Baby | 19,445 | 7,050 | 118,551 | 29,926 | 84,138 |
| Sports | 35,598 | 18,357 | 218,409 | 78,872 | 206,296 |
| Electronics | 192,403 | 63,001 | 1,254,441 | 271,356 | 739,381 |

Semantic pairs là cạnh vô hướng duy nhất; final nnz theo builder sau CF union/xử lý graph, không cùng quy ước đếm. VALID/TEST counts chưa được xác nhận từ raw files trong audit này: không điền ước lượng.

Mô tả split origin/filtering/dedup, features/extractors/dimensions, full-ranking, eligibility/masking/ties, VALID NDCG@20 selection, 64D/L3/500 epochs/evaluation mỗi 5 epochs, search budget và actual seeds. Hashes đưa vào artifact appendix; matching hash không chứng minh split construction đúng.

### Table 2 — Main external benchmark

- ECLIPSE: bốn metrics × ba tập, mean ± sample SD, n=3.
- STAIR là đối chiếu trực tiếp cần có và được attribution đúng. Thêm published methods khác khi có citation, protocol và provenance đủ rõ.
- Tách **Reported** và **Reproduced** bằng hai panel hoặc hai bảng. Không tính gain như matched experiment khi số published dùng split/feature/evaluator khác.
- Nếu STAIR tái lập chỉ n=1 theo quyết định nhóm: ghi giá trị đơn, n=1 và giới hạn; không SD giả, không paired-seed p-value. Không bắt buộc rerun ba seed STAIR. So mean ba seed ECLIPSE với một seed STAIR chỉ là chênh lệch mô tả, độ chắc chắn không ngang nhau.
- Không đưa v4/v8/v8.1/REEC vào external benchmark. Bold/underline theo quy tắc công khai, không coi rounding tie là chiến thắng.
- Không dùng significance stars từ convergence-window exports cũ.

### Table 3 — Ablation and evidence controls

| Panel | Hàng | Câu hỏi |
| --- | --- | --- |
| Components | Full, w/o EC, w/o LC, w/o CSE; Binary nếu có | Thành phần có ích conditional on phần còn lại? |
| Evidence | Full, Count-only, Shuffled evidence; Global-full nếu có | Excess subtraction, placement và activity conditioning có ích ngoài raw reinforcement? |

Main table ưu tiên NDCG@20 và Recall@20; appendix giữ bốn metrics và từng seed. Ghi actual n từng ô. Electronics control n=1 chỉ có point estimate, không “±0.000”. Không tăng n bằng cách coi epochs là mẫu.

### Table 4 — Resources and reproducibility

Preprocessing, `Coach.fit`, training peak allocated **và** reserved, parameters thêm so với w/o EC cùng framework, hardware/runtime và phạm vi timer. A0 mean fit Baby1180.35s, Sports2496.83s, Electronics32492.36s; lấy SD từ CSV khi typeset.

Preprocessing chịu cache/CPU load. So khác revision/máy phải công khai. Electronics800.67MiB allocated/1578MiB reserved là training telemetry, không phải toàn bộ GPU process. Không gọi <800MiB. Không cần dựng efficiency leaderboard nếu chưa có cùng timer/hardware cho external baselines; có thể đưa bảng này vào appendix nếu thiếu diện tích.

## 5. Hình nên chèn

### Figure 1 — Architecture (bắt buộc)

Ba vùng offline/training/inference; màu phân biệt inherited STAIR, framework additions và PEEC. R/features → WMSG/CSE → PEEC normalization/blend; static S_PEEC chỉ đi vào **item-BSC trên Adam directions**, không đi vào user bypass hay forward FSC.

MI initialize once; FSC trên TRAIN UI graph; LC dùng cross-entity adjacent-layer pairs; BPR dùng aggregated representation. Inference: selected checkpoint → FSC → dot product → seen masking/top-K. TRAIN UI graph vẫn cần cho FSC inference. Hình tổng quan không cần mọi công thức/hyperparameter.

Caption nêu PEEC không thêm inference branch và BSC được kế thừa STAIR. Tránh mũi tên BPR→LC, loss→inference hoặc static operator→Adam moments/user update. Đọc thử figure trong final PDF ở kích thước xuất bản.

### Figure 2 — Component/evidence ablation (khi đủ controls)

Ba dataset panels; x là actual arms, y là NDCG@20 hoặc relative difference với reference khai báo. Mean points + seed points; SD error bars phải ghi “sample SD”, không nhầm CI. Arm n=1 chỉ point. Recall@20 có thể ở companion figure appendix; không vẽ lại nguyên Table2 thành bar chart.

Nếu A2/A3/Count/Shuffle chưa hợp nhất, chỉ xuất Full–w/o EC và ghi phạm vi. Không vẽ placeholder như measurement. Dùng cùng arm names trong ledger và receipt parser.

### Figure 3 — Calibration mechanism (ưu tiên)

Panel trái: coverage r>0 và weighted coverage. Panel phải: relative operator Frobenius dose và finite-polynomial probe dose.

| Dataset | Coverage | Weighted coverage | Operator dose | Polynomial probe dose |
| --- | --- | --- | --- | --- |
| Baby | 10.379% | 12.175% | 7.129% | 2.365% |
| Sports | 10.009% | 11.754% | 7.106% | 2.174% |
| Electronics | 7.241% | 8.493% | 6.207% | 1.656% |

Probe coordinate32, dimension8, seed20261016, L3; không gọi là actual Adam-update change. Chỉ có ba static graphs; không tạo error bars từ ba training seeds lặp lại cùng graph/probe.

Thông điệp: calibration có độ phủ và tác động sau normalization/filter. Hình không chứng minh causal evidence, đúng/sai của cạnh hay quan hệ dose→metric. Spillover ở r=0 có thể đưa appendix: median coefficient ratios .986281/.986214/.992607 cho thấy normalization ảnh hưởng cả raw weights không đổi.

### Figure 4 — Paired EC effect hoặc efficiency (tùy diện tích)

Ưu tiên paired ΔNDCG@20: ba seed points/dataset, zero line, mean; nếu có CI thì ghi unadjusted paired t-CI, df2. Hình này cho thấy Electronics s3 âm và magnitude nhỏ mà grouped bars dễ che. Nếu Figure2 đã thể hiện đủ seed pairs thì chuyển hình này vào appendix.

Efficiency–accuracy chỉ khi hardware/timer scopes matched. Không suy “nhanh hơn baseline” khi chưa đo; bảng resources thường đủ.

### Appendix figures và chuẩn export

VALID curves ba seed/dataset + selected epoch; BPR/raw LC/weighted LC losses; allocated/reserved training memory; optional score quantiles và sensitivity. Không cần mọi run một hình trong main paper.

PDF vector cho diagram/charts, PNG300dpi preview, kèm CSV/JSON và caption. Ghi actual n, units, error-bar method, reference/dose scope. Không chọn seed dễ nhìn để che số âm, không thay missing bằng0.

Tên file thống nhất:

- `eclipse_architecture.pdf`
- `eclipse_component_ablation.pdf`
- `eclipse_evidence_controls.pdf`
- `eclipse_calibration_diagnostics.pdf`
- `eclipse_paired_ec_effect.pdf`
- Per-run: `eclipse_<dataset>_<arm>_seed<s>_<kind>`.

## 6. Định hướng viết Ablation Study — chỉ outline

1. **Setup:** drop-one component, giữ protocol Full, named-arm contract, actual n và selected TEST. Caveat historical revision A1 s1 đưa artifact appendix.
2. **EC:** báo paired mean của bốn metrics, nhấn N@20; gain nhỏ và Electronics mixed. Không gọi tất cả components indispensable nếu số liệu không ủng hộ. A0–A1 chưa đủ để nói calibration giải quyết noise.
3. **LC/CSE/Binary:** dùng bảng Hiếu đã audit. w/o CSE η0 phải giữ calibration trên semantic support, không tự đổi LC/hyperparameters. Binary giữ candidates và thứ tự sum→max-symmetrize→calibration→normalization. Drop-one effect là conditional; không cộng các Δ thành tổng contribution.
4. **Count:** c/(c+4) thay excess. Nếu Count≥Full, specific null-subtraction claim chưa thuyết phục. Count có thể có liều lớn hơn: đo operator/probe dose; bằng số parameters không tự loại dose confound.
5. **Shuffle:** randomize placement với fixed placebo seed, giữ support/score distribution theo contract; ghi rõ nhóm endpoint-degree nào được dùng để permute theo implementation thực tế. Nếu Shuffle≈Full, vị trí evidence chưa được attribution. Kiểm degree/dose confound; không tune placebo seed đến khi nó thua.
6. **Activity:** Global-full μ=n_i n_j/|U_active| phải cùng full population và regularizer; pooled-half không phải global-null. Nếu không chạy, nêu activity conditioning là design choice chưa isolate, không claim đóng góp riêng đã chứng minh.
7. **Limitations:** small n, user uncertainty conditional on checkpoint, implicit feedback không causal, normalization spillover và phụ thuộc feature/split. Không ép mọi ablation phải giảm.

## 7. Statistics được phép dùng

- Native selected TEST → mean ± sample SD, paired Δ và wins/losses; N@20 endpoint chính, R@20 phụ quan trọng.
- Paired t-CI/p với n3 chỉ thăm dò; normality của difference khó kiểm tra. Holm3 N@20 và Holm12 exploratory trong audit này là hồi cứu, không phải preregistration.
- Không significance sau correction không chứng minh equality. Exact sign-flip n3 all-positive cũng có p hai phía tối thiểu .25. Không đổi sang one-sided hoặc chọn test có p nhỏ nhất sau khi xem dữ liệu.
- Không pool dataset/metric/epochs để tăng n. Per-user bootstrap cần cùng eligible users, mean tái tạo aggregate, checkpoint VALID-selected và kết quả riêng từng seed; không thay thế seed variance.
- **Thu hồi convergence-window/synthetic epoch p-values cũ**, kể cả khi đã export CSV.

## 8. Review Example.tex — nơi cần sửa

Line numbers thuộc snapshot 2026-10-20, trước khi nhóm chỉnh manuscript. Không sửa `Example.tex` ở audit này vì người dùng yêu cầu định hướng Experiments/Ablation.

| Vị trí | Vấn đề | Hướng chỉnh |
| --- | --- | --- |
| L32 abstract | Single-seed, V8/gated variant, nhiều seed như future work | Dùng actual n3; không biến gain A0–A1 nhỏ thành strong significance |
| L33 | `onecolumn` | Đối chiếu official author kit; chưa xác nhận layout đúng vì chưa compile |
| L42,L56,L72 | Split-wise bottleneck là lịch sử nhánh nội bộ | Không gán là điểm yếu chung của SOTA; cần control/appendix hoặc bỏ origin story |
| L51,L59 | Design controls được kể như contribution hoàn tất | Viết theo actual findings; kế hoạch test chưa phải kết quả nghiên cứu |
| L66 | Never co-occur ⇒ spurious; zero test-time overhead | Không co-occur chưa chứng minh noise; overhead chỉ relative cùng framework/scorer |
| L69 | NLGCL/citation | Kiểm exact source và cross-entity pairs; không nhận InfoNCE là hoàn toàn mới |
| Method L95–273 | Attribution và fixed-operator bound đã khá rõ | Giữ attribution; bound không Adam convergence; score không p-value/causal confidence |
| L126 | `ECLIPSE_architecture.png` | Chưa có trong ICAART folder; đưa final asset đúng path, ưu tiên vector |
| L287 | “Only four narrative files, no code/raw receipts, single seed” | Không còn đúng; cập nhật actual receipts n3 và thiếu A0s1 checkpoints |
| L297–303 | TRAIN stats có, VALID/TEST/eligibility thiếu | Điền raw counts đã xác nhận; không suy từ tên550 |
| L306,L321–358 | Main table dùng versions nội bộ và rounded seed1 | Tách external benchmark; own configurations thành ablations; Full lấy CSV n3 |
| L316–317 | Seed1-only/protocol TODO cũ | Dùng actual cfg/n/selection; mỗi seed có selected epoch riêng |
| L360–382 | Ablation TODO cũ | A1 đủ n3; A2/A3 chờ Hiếu; Count/Shuffle chưa audit tại đây |
| L390 | Sensitivity grid | Ưu tiên controls gắn claim; không tune TEST hoặc đổi Full sau khi xem số |
| L428,L432,L437 | Resource/scope còn nói thiếu receipts/nhiều seed | Electronics s1 fit533.32min; n3 đã có; cập nhật Conclusion đúng bằng chứng |
| L447 | `bibliography{example}` | `example.bib` chưa có ở folder; có `method_references.bib`. Kiểm project paths/cite keys |
| L457 | `input{STAIR/method_details}` | File đang ở cùng folder `method_details.tex`; kiểm root/main compile |

`report/chapters_v6/03_stair.tex` còn lịch sử versions nội bộ; không lấy nguyên số làm tròn/claims đó vào paper. `STAIR_versionICAART_Report.md` là design reference, không thay receipts. Cần kiểm bibliography và compile project đúng template trước nộp; audit này không xác nhận compilation hoặc novelty trên toàn bộ literature.

## 9. Bàn giao từ Hiếu cho A2/A3 và controls

Hiếu phân tích riêng; bảng paper chung vẫn cần schema thống nhất:

1. Mỗi dataset/arm/seed: native selected metrics, epoch, selection criterion, COMPLETED/epochs, effective config và stdout; không suy actual n từ filename.
2. Split/feature/candidate hashes, commit+dirty diff/source hashes, environment/precision. Arm đổi đúng thành phần; nêu invariants và allowed graph changes.
3. A2 λLC0, A3η0; giữ protocol còn lại như Full. Đọc effective manifest, không chỉ CLI. Validator hiện tại `stair_icaart_config.py:121,130` vẫn khóa λLC=.01 và semantic_mode=weighted; vì vậy Hiếu cần bàn giao source/patch thực tế cho w/o LC/Binary và bằng chứng effective setting, không chỉ YAML đã đổi. Đây là yêu cầu provenance, chưa phải kết luận run của Hiếu sai.
4. Checkpoint/checksum, graph fingerprint/reference, preprocessing/Coach.fit và training peak allocated/reserved đúng scope. Không p-value theo epoch.
5. Bốn metrics từng seed + mean/sample SD/actual n. Missing=NOT_MEASURED; giữ kết quả âm/null.
6. Figures PDF/PNG/captions/source data; ZIP inventory/checksum; một người đọc chéo ký nghiệm thu. Thông báo đã chạy xong không thay artifact audit.

## 10. Thứ tự việc cần hoàn thành

| Ưu tiên | Việc | Tiêu chí |
| --- | --- | --- |
| P0 | Dùng A0/A1 đã đính chính, bỏ significance cũ | Bảng paper khớp native CSV, không all-significant claim |
| P0 | Hợp nhất A2/A3 với Hiếu | Metrics/config/selection/source đọc chéo được; ghi actual coverage |
| P0 | External benchmark source/protocol matrix | Reported/reproduced/n/split rõ; không bắt buộc retrain STAIR3seeds |
| P0 | Thu/hoàn thành Count và Shuffle theo plan | Baby/Sports matched seeds ưu tiên; Electronics n1 nếu hạn chế, ghi đúng n |
| P0 | Sửa Example.tex và paths/assets | Abstract/Intro/Conclusion đúng actual findings; compile bằng official template |
| P1 | Thu ba checkpoint Fulls1, per-user nếu khả thi | Khôi phục file gốc; không tự retrain vì thiếu local |
| P1 | Binary/Global-full nếu đủ thời gian | Cần nếu muốn riêng claim weighted-vs-binary/activity conditioning; không có thì giảm claim |
| P1 | Xuất figures/tables/appendix | Một câu hỏi mỗi hình, đọc được ở final size |
| P2 | Stress/noise/long-tail/sensitivity mở rộng | Không chặn submission nếu không claim robustness tương ứng |

Chốt code/architecture và ngân sách trước TEST mới; không chọn seed thuận lợi hoặc dừng vì kết quả xấu. Nếu Count/Shuffle không ủng hộ excess-evidence claim, báo trung thực; không đổi kiến trúc rồi trộn results dưới cùng một phương pháp. Câu chuyện phù hợp hiện tại: **framework có attribution rõ, calibration tĩnh có phân tích toán tử và thực nghiệm kiểm soát**, không hứa gain lớn chưa quan sát được.

## 11. Tài liệu chuẩn sau audit

- [A0 Full](ECLIPSE_Full_Model_Multi_Seed_Analysis.md), [A1 ablation](ECLIPSE_Ablation_Study_wo_EC.md).
- [Selected runs và resources/provenance](../../reports/eclipse_multiseed_paper_audit_20261020/selected_test_runs.csv).
- [Paired statistics](../../reports/eclipse_multiseed_paper_audit_20261020/paired_statistics.csv), [LaTeX data snippet](../../reports/eclipse_multiseed_paper_audit_20261020/a0_a1_metrics_table.tex).
- [Execution plan](ECLIPSE_Experiment_Execution_Plan.md), actual status ở mục19.

Báo cáo seed1 cũ đã đánh dấu historical/superseded; không dùng làm nguồn significance cho paper.
