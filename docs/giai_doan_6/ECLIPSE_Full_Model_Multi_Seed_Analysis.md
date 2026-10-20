# ECLIPSE Full Model (A0) — Audit ba seed và mức sẵn sàng cho paper

**Cập nhật 2026-10-20 — ANALYZED_ARTIFACTS_ONLY.** Báo cáo này thay thế kết luận của bản trước. Phân tích JSON/log/config, đối chiếu source ở Git commit đã ghi nhận và hash các checkpoint đang có; không training, không chạy test mô hình, không nạp checkpoint để đánh giá lại. 676/676 đối chiếu dữ liệu/provenance đạt trong [audit machine-readable](../../reports/eclipse_multiseed_paper_audit_20261020/audit_acceptance.json). Điều này không đồng nghĩa độc lập tái lập metric hay chứng minh không leakage.

## 1. Kết luận chính

- [OBSERVATION] Có đủ **9 native selected-TEST receipts**, tương ứng 3 training seeds × 3 datasets của A0 Full ECLIPSE; cả 9 có telemetry 500 epochs và status COMPLETED.
- [OBSERVATION] Mean NDCG@20 là **0.046340 / 0.051956 / 0.031759** trên Baby/Sports/Electronics. Đây là thống kê thực nghiệm có thể đưa vào bảng paper, kèm sample SD và n=3.
- [OBSERVATION] So với A1 w/o EC matched seeds, NDCG@20 tăng trung bình **0.585% / 0.490% / 0.138%**. Có 8/9 chênh lệch seed–dataset dương; Electronics seed 3 âm. Đây là xu hướng nhỏ, không phải bằng chứng tăng mạnh hoặc có ý nghĩa thống kê trên mọi tập.
- [FACT về hồ sơ] **Ba best_model.pt của A0 seed 1 chưa có ở workspace**. Metric receipts/log vẫn đọc được; không cần tự chạy lại chỉ vì thiếu file. Cần thu hồi checkpoint gốc để hoàn chỉnh gói tái lập hoặc đánh giá per-user. Sáu checkpoint A0 seeds2–3 đã được hash bytes.
- A2/A3: Hiếu thông báo đã chạy xong, artifacts còn trên máy Hiếu và Hiếu phân tích riêng. Báo cáo này không xác nhận phạm vi/metric hai arms đó.
- **Quyết định:** A0 đã đủ cho một hàng kết quả n=3; chưa đủ để kết luận toàn bộ paper sẵn sàng nộp. Cần benchmark bên ngoài có protocol rõ, ablations đã kiểm toán, Count/Shuffle để bảo vệ vai trò excess evidence. Không cần mở kiến trúc mới vì audit này.

## 2. Nguồn và phạm vi kiểm toán

| Arm/seed | Nguồn | Trạng thái chấp nhận |
| --- | --- | --- |
| A0 s1 | `logs/GD6/STAIR-PEEC/stair_icaart/`; Electronics dùng nhánh `electronics/stair_icaart/` | Selected metrics/log/telemetry/source kiểm tra được; checkpoint thiếu local |
| A0 s2 | `logs/ICAART/eclipse_seed2/` | Receipts, config SHA, source SHA và checkpoint bytes đã đối chiếu |
| A0 s3 | `logs/ICAART/eclipse_seed3/` | Receipts, config SHA, source SHA và checkpoint bytes đã đối chiếu |
| A1 s1 | Gói `eclipse_wo_ec_seed1_reuse_20261018T152158Z_69603726` | Checkpoint uncalibrated tái sử dụng; xem báo cáo recovery, không phải rerun cùng commit |
| A1 s2/s3 | `logs/ICAART/eclipse_wo_ec_seed2/`, `eclipse_wo_ec_seed3/` | Native receipts + config/source/checkpoint đối chiếu được |

Electronics Full seed 1 có hai bản sao thư mục với selected receipt giống nhau; chỉ tính **một** run. Các chỉ số nằm trong `selected_metrics.json`, không lấy từ bảng benchmark cũ hoặc TEST terminal-epoch nếu khác checkpoint được VALID chọn.

Source commit A0 cả 9 run: `5212263e87c5c1cfe6e48c91afa90201a48f5929`. Runtime manifests ghi Python3.13.15, PyTorch2.11.0+cu128, FreeRec0.9.7, TeslaT4. Đối chiếu source SHA theo Git objects tại commit đã ghi, không giả current HEAD là mã đã chạy. File hash chỉ kiểm tra correspondence với manifest; chưa đọc lại feature/split bytes gốc ở môi trường training.

## 3. Kết quả TEST tại checkpoint VALID chọn

| Dataset | Seed | Selected epoch | RECALL@10 | RECALL@20 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | 1 | 480 | 0.06861877 | 0.10610716 | 0.03679847 | 0.04639788 |
| Baby | 2 | 450 | 0.06816122 | 0.10440073 | 0.03724547 | 0.04654667 |
| Baby | 3 | 480 | 0.06828927 | 0.10315081 | 0.03712785 | 0.04607514 |
| Sports | 1 | 500 | 0.07624275 | 0.11450039 | 0.04218513 | 0.05208228 |
| Sports | 2 | 470 | 0.07713924 | 0.11482047 | 0.04203995 | 0.05178147 |
| Sports | 3 | 500 | 0.07699431 | 0.11460018 | 0.04226106 | 0.05200424 |
| Electronics | 1 | 485 | 0.04640054 | 0.06811273 | 0.02621387 | 0.03183061 |
| Electronics | 2 | 475 | 0.04614907 | 0.06835274 | 0.02605182 | 0.03179316 |
| Electronics | 3 | 340 | 0.04598017 | 0.06827527 | 0.02588534 | 0.03165399 |

Giữ toàn bộ precision gốc trong [selected_test_runs.csv](../../reports/eclipse_multiseed_paper_audit_20261020/selected_test_runs.csv). Bảng hiển thị 8 chữ số; thống kê tính trên số gốc, không tính lại từ số làm tròn.

### 3.1 Mean ± sample standard deviation

| Dataset | Arm | n | RECALL@10 | RECALL@20 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | full | 3 | 0.068356 ± 0.000236 | 0.104553 ± 0.001484 | 0.037057 ± 0.000232 | 0.046340 ± 0.000241 |
| Sports | full | 3 | 0.076792 ± 0.000481 | 0.114640 ± 0.000164 | 0.042162 ± 0.000112 | 0.051956 ± 0.000156 |
| Electronics | full | 3 | 0.046177 ± 0.000212 | 0.068247 ± 0.000122 | 0.026050 ± 0.000164 | 0.031759 ± 0.000093 |

Với mỗi dataset/metric: mean = ΣM_s/3; sample SD = sqrt(Σ(M_s−mean)²/(3−1)). Sample SD không phải confidence interval và bản thân SD không phải estimator không chệch. Không dùng epoch như các quan sát độc lập. SD nhỏ tuyệt đối không tự chứng minh gain vượt seed variance; phải xét paired differences. Không gộp ba dataset thành chín training replications của cùng một population.

## 4. Protocol thực tế và comparability

- 64D, FSC L=3, 500 training epochs/run; evaluation mỗi 5 epochs; chọn max VALID NDCG@20 rồi báo TEST. Native VALID events xác nhận selected epoch A0.
- WMSG floor-power: f(a)=0.05+0.95[max(0,a)]² trên candidates text k=5, visual k=1. Cộng modality trước max-symmetrization. Không có inverse-frequency hub formula trong arm này.
- CSE: co-occurrence c≥2; Shrunk Ochiai q=c/(c+5)·c/sqrt(n_i n_j); top-5 outgoing rồi max-union; η=.1. LC λ=.01. MI/FSC/BSC và dot-product scorer kế thừa STAIR; ECLIPSE gồm các mở rộng đồ thị/objective đã nêu trong Method, PEEC là cơ chế calibration trung tâm.
- PEEC `pooled_activity_excess`, activity bins log2, halves0, δ1, τ2 ⇒ denominator c+μ+4. μ từ toàn bộ tầng activity, chỉ TRAIN. **Ghi chú:** đây là expected-overlap reference có giả định, không phải calibrated probability hay causal significance.
- A0 và A1 cùng feature/split/candidate/data fingerprints và backbone config sau khi bỏ seed. A1 modern s2–3 có cùng raw semantic/count/null/score/CF hashes với A0, khác calibration delta và final operator. A1 s1 có source revisions khác; xem audit recovery về equivalence, không khẳng định RNG trajectory giống nhau.
- Full graph fingerprint giống nhau qua ba seed của mỗi dataset. Đây là recorded static operator identity; không chứng minh toàn bộ training deterministic hoặc không có leakage.

## 5. Đóng góp riêng của calibration: A0 vs A1

| Dataset | Full N@20 | w/o EC N@20 | Δ absolute | Δ relative | Seed wins | p t-test raw | p Holm (3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baby | 0.046340 ± 0.000241 | 0.046070 ± 0.000162 | +0.00026963 | +0.585% | 3/3 | 0.2745 | 0.5490 |
| Sports | 0.051956 ± 0.000156 | 0.051703 ± 0.000229 | +0.00025326 | +0.490% | 3/3 | 0.0446 | 0.1339 |
| Electronics | 0.031759 ± 0.000093 | 0.031715 ± 0.000030 | +0.00004377 | +0.138% | 2/3 | 0.5734 | 0.5734 |

CI và cả12 metric comparisons: [A1 report](ECLIPSE_Ablation_Study_wo_EC.md), [statistics CSV](../../reports/eclipse_multiseed_paper_audit_20261020/paired_statistics.csv). Paired t-tests n=3, df=2 chỉ thăm dò, giả định các seed differences độc lập và gần normal. Holm cho ba N@20 comparisons là family định nghĩa trong audit hồi cứu, không preregistration. **Không dataset nào đạt p_Holm<.05**. Sports unadjusted t-CI loại 0 nhưng chưa có family-wise significance. Exact sign-flip hai phía p=.25/.25/.75; độ phân giải n=3 thấp.

Không gọi Electronics “equivalent/parity đã chứng minh”: chưa có equivalence margin/test. 32/36 metric cells thắng không phải32 thí nghiệm độc lập. A0 có stability mô tả qua seed; incremental EC benefit còn nhỏ và bất định. Comparison với published baselines phải được đánh giá riêng.

## 6. Chi phí: đo đúng phạm vi

| Dataset | Seed | Preprocess (s) | Coach.fit (s) | Σ epoch train (s) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | 1 | 25.70 | 1214.19 | 1156.66 | 145.17 | 244.00 |
| Baby | 2 | 23.66 | 1160.48 | 1105.06 | 145.17 | 244.00 |
| Baby | 3 | 22.86 | 1166.38 | 1111.82 | 145.17 | 244.00 |
| Sports | 1 | 51.32 | 2520.87 | 2371.07 | 223.71 | 458.00 |
| Sports | 2 | 46.64 | 2469.78 | 2320.93 | 223.71 | 458.00 |
| Sports | 3 | 46.16 | 2499.85 | 2353.33 | 223.71 | 458.00 |
| Electronics | 1 | 270.76 | 31999.03 | 30071.28 | 800.67 | 1578.00 |
| Electronics | 2 | 287.60 | 32006.65 | 30099.64 | 800.67 | 1578.00 |
| Electronics | 3 | 279.32 | 33471.41 | 31594.48 | 800.67 | 1578.00 |

### 6.1 Tóm tắt ba seed

| Dataset | Preprocess mean ± SD (s) | Fit mean ± SD (s) | Allocated peak (MiB) | Reserved peak (MiB) |
| --- | --- | --- | --- | --- |
| Baby | 24.07 ± 1.47 | 1180.35 ± 29.46 | 145.17 | 244.00 |
| Sports | 48.04 ± 2.85 | 2496.83 ± 25.68 | 223.71 | 458.00 |
| Electronics | 279.23 ± 8.42 | 32492.36 ± 847.89 | 800.67 | 1578.00 |

- `Coach.fit` lấy từ stdout original; tổng epoch-train seconds từ telemetry là timer khác. Electronics s1: fit31999.03s (=533.32phút), epoch sum30071.28s (=501.19phút). Không dùng501.19phút làm toàn bộ fit.
- Allocated/reserved lấy max training telemetry. Electronics **800.67MiB allocated**, 1578MiB reserved; không được gọi “<800MiB” hoặc toàn bộ process chỉ800MiB. Chưa bao gồm đầy đủ driver/evaluation/inference peaks.
- Equal allocator peaks qua seed không chứng minh không memory leak; không có slope/RSS/process-wide trace trong audit này. Có thể nói các run hoàn tất và reported training allocator peaks như bảng.
- Preprocessing bao gồm các bước trong timer manifest, chịu ảnh hưởng CPU/cache; chưa đo overhead EC thuần bằng cùng máy, cùng cache và repeated profiling. Không tuyên bố zero preprocessing cost hoặc chính xác cùng fit time.
- “0 additional trainable parameters” chỉ so với A1 cùng framework, không có nghĩa mô hình không có embedding parameters. Không suy chi phí toàn ECLIPSE bằng STAIR gốc từ ablation này.

## 7. Diagnostics cơ chế từ manifest

| Dataset | Unique candidate pairs | r>0 pairs | Coverage | Weighted coverage | max r | Operator dose | Polynomial probe dose | Median coeff ratio on r=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baby | 29926 | 3106 | 10.379% | 12.175% | 0.790917 | 7.129% | 2.365% | 0.986281 |
| Sports | 78872 | 7894 | 10.009% | 11.754% | 0.870763 | 7.106% | 2.174% | 0.986214 |
| Electronics | 271356 | 19648 | 7.241% | 8.493% | 0.924703 | 6.207% | 1.656% | 0.992607 |

Operator dose = ||S_PEEC−S_noEC||_F/||S_noEC||_F. Polynomial dose là fixed Rademacher probe tại coordinate32, dimension8, seed20261016, L3; **không đo actual Adam-direction perturbation**. Số lặp qua ba seeds vì cùng graph/probe, không phải ba graph samples độc lập. [CSV](../../reports/eclipse_multiseed_paper_audit_20261020/mechanism_diagnostics.csv).

[OBSERVATION] Calibration thay đổi toán tử và polynomial probe khác 0; r=0 vẫn bị ảnh hưởng gián tiếp qua degree normalization. [INTERPRETATION] Electronics có coverage và probe dose nhỏ hơn, đồng thời EC mean gain nhỏ hơn. Ba dataset không đủ chứng minh quan hệ nhân quả hoặc quy luật dose→metric. [HYPOTHESIS] Excess evidence đặt đúng cạnh hữu ích hơn raw count/placebo; **cần Count-only và Shuffled evidence** để kiểm tra, A0–A1 chưa trả lời.

## 8. Kiểm tra learning curves và checkpoint selection

Tất cả A0 có500 telemetry rows, losses hữu hạn và loss identity khớp tolerance1e−7. Đây là sanity check, không chứng minh gradients không collapse hoặc đã hội tụ. Sports s1/s3 selected epoch500 chạm budget, không chứng minh optimum đã đạt. Electronics s3 chọn340, s1/s2 chọn485/475: thể hiện selection variability, chưa đủ kết luận overfit.

Learning curves ưu tiên VALID và selected epoch từng seed. Không dùng TEST để chọn epoch/hyperparameter hoặc dùng terminal TEST thay selected TEST. Loss plots tách BPR/raw LC/weighted LC với scale đọc được. Không bootstrap epochs để tạo p-values.

### 8.1 Learning-dynamics summary từ native telemetry

| Dataset | Seed | VALID selected N@20 | VALID epoch500 N@20 | Loss epoch1 | BPR epoch500 | λLC epoch500 | Total loss epoch500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baby | 1 | 0.044176 | 0.043783 | 0.686816 | 0.135588 | 0.052792 | 0.188380 |
| Baby | 2 | 0.044040 | 0.043864 | 0.686830 | 0.135434 | 0.052788 | 0.188222 |
| Baby | 3 | 0.044157 | 0.043620 | 0.686820 | 0.136038 | 0.052785 | 0.188823 |
| Sports | 1 | 0.050221 | 0.050221 | 0.667559 | 0.023708 | 0.046966 | 0.070674 |
| Sports | 2 | 0.050312 | 0.050143 | 0.667688 | 0.023730 | 0.046970 | 0.070700 |
| Sports | 3 | 0.050519 | 0.050519 | 0.667525 | 0.023663 | 0.046970 | 0.070633 |
| Electronics | 1 | 0.031175 | 0.031115 | 0.686983 | 0.033227 | 0.063916 | 0.097143 |
| Electronics | 2 | 0.031151 | 0.031094 | 0.686985 | 0.033205 | 0.063917 | 0.097122 |
| Electronics | 3 | 0.031064 | 0.030996 | 0.687001 | 0.033140 | 0.063914 | 0.097054 |

Bảng này là **VALID và training losses**, không thay bảng selected TEST. Epoch500 là event VALID trước restore, không phải event reevaluation cuối file ở selected checkpoint. VALID giảm từ đỉnh đến epoch500 ở một số seed là quan sát model selection, không tự chứng minh overfit hoặc cơ chế EC. Loss hữu hạn không chứng minh gradient không collapse khi chưa đo gradient. Giá trị đầy đủ và source hashes ở CSV.

## 9. Đính chính so với báo cáo trước

1. Bỏ nhãn VERIFIED theo nghĩa tái lập mô hình: ba checkpoint A0s1 còn thiếu, không thể claim hash đủ chín checkpoints.
2. Giữ các mean/SD cơ bản đúng, bỏ “record variance”, “loại trừ may mắn”, “graph lớn làm SD nhỏ”. SD không đủ chứng minh cơ chế.
3. Sửa preprocessing/fit và tách epoch-sum khỏi fit. Không dùng một seed để lập bảng nguồn lực như số trung bình.
4. Sửa Sports/Electronics rmax và dose theo manifest; Electronics rmax=.924703, operator dose6.207%, không suy từ narrative.
5. BSC lọc **Adam-preconditioned total-loss item directions**, không phải trực tiếp raw gradient hay embeddings forward. Bound fixed operator không bảo đảm Adam convergence.
6. S3 A1 đã có thật, bỏ ghi chú “đang training”. Thu hồi bảng p-values từ convergence windows trong báo cáo seed1.

Bản trước được giữ ở [before_revision](../../reports/eclipse_multiseed_paper_audit_20261020/before_revision/). Không thay đổi raw logs/artifacts hoặc code training.

## 10. Chuyển giao cho paper

Đọc [kế hoạch evidence/figures/tables và review manuscript](ECLIPSE_Paper_Experimental_Readiness_and_Presentation_Plan.md). Không đặt các version nội bộ làm external baselines. Dùng **ECLIPSE** là phương pháp cuối, A1/A2/A3 và các controls là ablations; STAIR được dẫn nguồn cho MI/FSC/BSC/scorer. Main table cần external benchmark evidence phân biệt published/reproduced, protocol, actual n; không bắt buộc chạy lại ba seed STAIR nếu nhóm đã quyết định giữ n=1, nhưng phải công khai giới hạn thống kê đó.

### Dữ liệu đầu ra có thể sử dụng

- [Script tái tính](../../reports/audit_eclipse_full_multiseed.py): chỉ hậu xử lý dữ liệu, không model/test/training.
- [18 selected runs + resources/provenance](../../reports/eclipse_multiseed_paper_audit_20261020/selected_test_runs.csv).
- [Mean/SD](../../reports/eclipse_multiseed_paper_audit_20261020/mean_std.csv); [paired differences](../../reports/eclipse_multiseed_paper_audit_20261020/paired_differences.csv); [statistics](../../reports/eclipse_multiseed_paper_audit_20261020/paired_statistics.csv).
- [Acceptance checks](../../reports/eclipse_multiseed_paper_audit_20261020/artifact_checks.csv). Đạt các đối chiếu không thay thế human cross-read và independent evaluator reproduction.
