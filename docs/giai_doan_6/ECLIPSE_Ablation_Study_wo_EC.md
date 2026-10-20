# ECLIPSE — Ablation A1 w/o EC, seeds 1–3

**Audit cập nhật 2026-10-20. Status: ANALYZED_ARTIFACTS_ONLY.** Có 9 run A1 với receipts/telemetry và checkpoints local. Phân tích cả 18 run A0/A1, không training hoặc đánh giá lại mô hình. Bản này thay kết luận “significance cao/loại trừ ngẫu nhiên” của bản trước; [backup trước sửa](../../reports/eclipse_multiseed_paper_audit_20261020/before_revision/ECLIPSE_Ablation_Study_wo_EC.md).

## 1. Research question và arm contract

**A0 Full ECLIPSE**: WMSG+CSE+LC+MI/FSC/BSC, PEEC `pooled_activity_excess`, δ1. **A1 w/o EC**: cùng framework, δ0. Scientific question: calibration có cải thiện ngoài phần còn lại của ECLIPSE không? A1 là ablation backbone của chính nhóm, **không phải STAIR gốc hoặc published external baseline**.

WMSG f(a)=.05+.95[max(0,a)]² trên cached candidates; cộng modalities rồi max-symmetrize. CSE η=.1, co-occurrence≥2, top5 union, shrinkage5 vẫn bật; λLC=.01, 64D, L3, 500 epochs. δ=0 ⇒ W̃=W và operator là uncalibrated reference. “w/o EC” không có nghĩa bỏ mọi behavioral information vì CSE vẫn dùng TRAIN.

## 2. Audit source, selection và provenance

- A1 s2/s3: source `5212263e87c5c1cfe6e48c91afa90201a48f5929`, native calibration bypass `delta_zero`, operator/poly dose 0; config SHA và source SHA đối chiếu Git objects.
- A1 s1 dùng checkpoint uncalibrated gốc tái sử dụng: Baby/Sports commit `014886e7e7cd03096c724ba67815907e87e05d7c`; Electronics `d401a27511a3f77b13c082d68c7f2359b7e4e61c`. **Không ghi tất cả 18 run cùng commit**. Xem [seed1 audit](ECLIPSE_wo_EC_Seed1_Artifact_Audit.md) và [recovery script](../../reports/audit_eclipse_wo_ec_seed1_recovery.py). Bytes checkpoint khớp SHA của recovery receipt. Runtime recovery và training provenance lưu riêng; không gán thời gian recovery evaluation thành fit time.
- Feature/split/candidate/data fingerprints, MI và backbone config khớp giữa seeds/arms, trừ seed. s2/s3 có counts/null/raw-semantic/CF/score hashes khớp; delta và final operator khác.
- Native runs hiện đại: selected epoch khớp argmax VALID NDCG@20, selected TEST JSONL khớp receipt. Legacy s1 có TEST recovery JSONL; original VALID nằm trong `training.log`, không có native VALID JSONL đầy đủ như run mới.
- Mỗi run có 500 telemetry epochs và COMPLETED. A0 seed 1 thiếu checkpoint local nhưng metrics/logs còn. Đây là paired-seed design với historical-source caveat ở seed1, không cam kết cùng stochastic trajectory bitwise.

## 3. Kết quả A1 ở checkpoint được VALID chọn

| Dataset | Seed | Selected epoch | RECALL@10 | RECALL@20 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | 1 | 430 | 0.06748052 | 0.10543584 | 0.03647647 | 0.04624349 |
| Baby | 2 | 420 | 0.06732502 | 0.10360957 | 0.03662976 | 0.04592250 |
| Baby | 3 | 485 | 0.06755901 | 0.10322828 | 0.03688588 | 0.04604480 |
| Sports | 1 | 500 | 0.07618512 | 0.11424804 | 0.04196987 | 0.05181572 |
| Sports | 2 | 470 | 0.07614414 | 0.11432372 | 0.04154703 | 0.05143965 |
| Sports | 3 | 500 | 0.07637790 | 0.11441525 | 0.04199272 | 0.05185285 |
| Electronics | 1 | 500 | 0.04603368 | 0.06796098 | 0.02600378 | 0.03168096 |
| Electronics | 2 | 480 | 0.04604753 | 0.06816746 | 0.02601074 | 0.03173518 |
| Electronics | 3 | 375 | 0.04580556 | 0.06837226 | 0.02589744 | 0.03173031 |

Cả4metrics lấy uppercase native keys; missing không điền0. Extra Recall@1 chỉ có một số legacyruns không đưa vào bảng đủn3. Selected epoch là model selection, không phải sốepochsđãtrain; tất cảbudget500.

## 4. Mean ± sample SD: A0 và A1

| Dataset | Arm | n | RECALL@10 | RECALL@20 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | full | 3 | 0.068356 ± 0.000236 | 0.104553 ± 0.001484 | 0.037057 ± 0.000232 | 0.046340 ± 0.000241 |
| Baby | wo_ec | 3 | 0.067455 ± 0.000119 | 0.104091 ± 0.001180 | 0.036664 ± 0.000207 | 0.046070 ± 0.000162 |
| Sports | full | 3 | 0.076792 ± 0.000481 | 0.114640 ± 0.000164 | 0.042162 ± 0.000112 | 0.051956 ± 0.000156 |
| Sports | wo_ec | 3 | 0.076236 ± 0.000125 | 0.114329 ± 0.000084 | 0.041837 ± 0.000251 | 0.051703 ± 0.000229 |
| Electronics | full | 3 | 0.046177 ± 0.000212 | 0.068247 ± 0.000122 | 0.026050 ± 0.000164 | 0.031759 ± 0.000093 |
| Electronics | wo_ec | 3 | 0.045962 ± 0.000136 | 0.068167 ± 0.000206 | 0.025971 ± 0.000063 | 0.031715 ± 0.000030 |

Sample SD ddof1; không phải95%CI. Không xếp một seed1A1 cạnhmean 3seedA0 như matched estimate.

## 5. Paired differences theo seed

Δ_s = M_A0,s − M_A1,s. Δ dương nghĩa Full cao hơn. Các số dưới là chênh lệch metric, không phải phần trăm.

| Dataset | Metric | Δ seed1 | Δ seed2 | Δ seed3 | Mean Δ | Relative mean gain | Wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baby | RECALL@10 | +0.00113825 | +0.00083620 | +0.00073026 | +0.00090157 | +1.337% | 3/3 |
| Baby | RECALL@20 | +0.00067131 | +0.00079116 | -0.00007747 | +0.00046167 | +0.444% | 2/3 |
| Baby | NDCG@10 | +0.00032200 | +0.00061571 | +0.00024197 | +0.00039323 | +1.073% | 3/3 |
| Baby | NDCG@20 | +0.00015439 | +0.00062417 | +0.00003033 | +0.00026963 | +0.585% | 3/3 |
| Sports | RECALL@10 | +0.00005763 | +0.00099511 | +0.00061641 | +0.00055638 | +0.730% | 3/3 |
| Sports | RECALL@20 | +0.00025235 | +0.00049675 | +0.00018494 | +0.00031135 | +0.272% | 3/3 |
| Sports | NDCG@10 | +0.00021526 | +0.00049292 | +0.00026834 | +0.00032551 | +0.778% | 3/3 |
| Sports | NDCG@20 | +0.00026656 | +0.00034182 | +0.00015139 | +0.00025326 | +0.490% | 3/3 |
| Electronics | RECALL@10 | +0.00036686 | +0.00010154 | +0.00017461 | +0.00021434 | +0.466% | 3/3 |
| Electronics | RECALL@20 | +0.00015175 | +0.00018528 | -0.00009698 | +0.00008002 | +0.117% | 2/3 |
| Electronics | NDCG@10 | +0.00021009 | +0.00004108 | -0.00001209 | +0.00007969 | +0.307% | 2/3 |
| Electronics | NDCG@20 | +0.00014965 | +0.00005798 | -0.00007632 | +0.00004377 | +0.138% | 2/3 |

[OBSERVATION] N@20: Baby3/3, Sports3/3, Electronics2/3 seed differences dương. Electronics seed3 giảm khoảng0.00007614; Baby seed3 R@20 cũng giảm, Electronics seed3 R@20/N@10 giảm. Có32/36 metric cells dương, R@10 là9/9; **chỉ là mô tả vì các metrics phụ thuộc nhau**, không dùng 32 như sample size.

[INTERPRETATION] EC có incremental mean gain nhỏ, rõ hơn trên Baby/Sports và yếu hơn Electronics. Không có cơ sở gọi thành phần này là nguồn duy nhất của chênh lệch ECLIPSE–STAIR. Kết quả nhỏ không tự phủ nhận phương pháp tổng thể nhưng phải giảm cường độ claim về riêng calibration.

## 6. Statistical interpretation đã hiệu chỉnh

| Dataset | Full N@20 | w/o EC N@20 | Δ absolute | Δ relative | Seed wins | p t-test raw | p Holm (3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baby | 0.046340 ± 0.000241 | 0.046070 ± 0.000162 | +0.00026963 | +0.585% | 3/3 | 0.2745 | 0.5490 |
| Sports | 0.051956 ± 0.000156 | 0.051703 ± 0.000229 | +0.00025326 | +0.490% | 3/3 | 0.0446 | 0.1339 |
| Electronics | 0.031759 ± 0.000093 | 0.031715 ± 0.000030 | +0.00004377 | +0.138% | 2/3 | 0.5734 | 0.5734 |

| Dataset | Mean Δ N@20 | 95% t-CI (unadjusted) | Exact two-sided sign-flip p |
| --- | --- | --- | --- |
| Baby | +0.00026963 | [-0.00050851, +0.00104777] | 0.25 |
| Sports | +0.00025326 | [+0.00001501, +0.00049150] | 0.25 |
| Electronics | +0.00004377 | [-0.00023856, +0.00032610] | 0.75 |

### 6.1 Cách tính và giới hạn

Với n=3, t = mean(Δ)/(SD(Δ)/sqrt(3)), df=2; CI95% = mean(Δ) ± t_.975,2·SD(Δ)/sqrt(3). Paired t-test giả định các seed differences độc lập và normal model phù hợp; n=3 không đủ để xác nhận bằng normality test. Paired test không yêu cầu variance hai arms bằng nhau. Các CI trên **chưa hiệu chỉnh multiplicity**.

Family chính hồi cứu: ba NDCG@20 comparisons theo dataset, hiệu chỉnh Holm step-down. Không dataset nào còn p<.05. Family exploratory gồm12 dataset×metric comparisons cũng không có p_Holm<.05; xem [CSV](../../reports/eclipse_multiseed_paper_audit_20261020/paired_statistics.csv). Baby R@10 raw p=.017888, Holm12=.214656. Không chọn chỉ hai ô raw p nhỏ rồi gọi “high significance”; p=.06/.08 không phải “gần chứng minh”.

Exact sign-flip liệt kê 2³ sign assignments của paired differences, giả định sign exchangeability dưới null. Ba pairs cùng dương chỉ có p hai phía tối thiểu .25. Điều này cho thấy độ phân giải thấp ở n=3, không chứng minh không có hiệu quả. Không chuyển sang one-sided sau khi nhìn dấu; không gộp dataset/metric/epoch để tăng n.

**Kết luận được phép:** “Calibration improves mean NDCG@20 on all three datasets in three paired-seed comparisons, with small gains and limited statistical resolution; the Electronics effect is inconsistent across seeds.” Không ghi “significantly improves across all datasets” hoặc“proven equivalent”.

### 6.2 Nếu có per-user metrics

Chỉ dùng cùng eligible users và checkpoint VALID-selected của hai arms; kiểm mean per-user khớp aggregate evaluator. Paired bootstrap/randomization theo user trả lời uncertainty conditional on checkpoint, **không thay thế training-seed uncertainty**. Cần thu hồi checkpoint Full s1 còn thiếu nếu đánh giá lại seed này. Không tạo user p-value từ bốn aggregate metrics hoặc losses. Không bắt buộc thêm ba seed STAIR để hoàn thiện A1; nếu STAIR n=1 thì ghi n=1, không paired-seed p-value.

## 7. Calibration diagnostics và placebo interpretation

- A1 s2/s3 vẫn tính evidence coverage dương khi build; δ=0 làm calibration operator/poly dose đúng0. Coverage không thành0 chỉ vì arm w/o EC.
- A1 s1 legacy không đo native PEEC evidence coverage: **NOT_MEASURED**, không thay bằng0. Zero dose suy giải tích từ operator equivalence đã audit, không phải measured perturbation trên Adam directions.
- Operator/probe difference khác 0 ở A0 cho thấy tác động lên toán tử, không chứng minh các cạnh được tăng là đúng/causal. Count-only tách excess subtraction khỏi raw-count reinforcement; Shuffle tách placement khỏi distribution của reweighting.
- Shuffle giữ histogram score nhưng degree normalization có thể tạo final dose khác. Log dose/degree/candidates, không mặc định dose matched. Global-full null cần implementation đúng nếu muốn claim riêng activity conditioning; pooled-half không thay thế arm này.

## 8. Resources A1: original training, không lẫn recovery evaluation

| Dataset | Seed | Preprocess (s) | Coach.fit (s) | Σ epoch train (s) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Baby | 1 | 24.11 | 1158.38 | 1099.48 | 145.17 | 244.00 |
| Baby | 2 | 23.42 | 1194.30 | 1138.11 | 145.17 | 244.00 |
| Baby | 3 | 21.80 | 1148.81 | 1094.73 | 145.17 | 244.00 |
| Sports | 1 | 28.93 | 2499.69 | 2339.15 | 223.71 | 458.00 |
| Sports | 2 | 48.84 | 2548.35 | 2399.91 | 223.71 | 458.00 |
| Sports | 3 | 46.03 | 2510.47 | 2362.69 | 223.71 | 458.00 |
| Electronics | 1 | 280.47 | 31602.47 | 29538.10 | 800.67 | 1578.00 |
| Electronics | 2 | 285.44 | 32302.49 | 30454.18 | 800.67 | 1578.00 |
| Electronics | 3 | 285.31 | 33683.37 | 31786.63 | 800.67 | 1578.00 |

| Dataset | Preprocess mean ± SD (s) | Fit mean ± SD (s) | Allocated peak (MiB) | Reserved peak (MiB) |
| --- | --- | --- | --- | --- |
| Baby | 23.11 ± 1.18 | 1167.16 ± 23.98 | 145.17 | 244.00 |
| Sports | 41.27 ± 10.77 | 2519.50 ± 25.56 | 223.71 | 458.00 |
| Electronics | 283.74 ± 2.83 | 32529.44 ± 1058.85 | 800.67 | 1578.00 |

Fit lấy từ `Coach.fit`; tổng epoch-train seconds là timer khác. Peaks là max training telemetry, không phải whole-process/evaluation memory. Paired timing vẫn khác CPU load/cache và legacy revision, không diễn giải dao động wall-clock thành overhead EC thuần. Bỏ claim “mọi cặp chênh<1%”: Babyseed1 Full1214.19s so với A11158.38s đã chênh khoảng4.82%.

## 9. Đính chính và tình trạng những arm khác

- A1 đủ ba seed × ba datasets; bỏ TODO seed3 còn training. Đủ source/hashes/logs không đồng nghĩa independent evaluator reproduction.
- Thu hồi p-values convergence-window/synthetic exports của seed1. Không dùng significant stars cũ trong paper.
- A2 w/o LC và A3 w/o CSE: **HUMAN_REPORTED_COMPLETE** ở máy Hiếu, chưa chuyển artifacts vào workspace. Không gán metric/scope/hardware đã audit. Hiếu bàn giao native per-seed table và provenance/selection/hashes; xem roadmap.
- Binary/Count/Shuffle/Global-full chưa có receipts mới được đưa vào audit này. Notebook READY không tương đương run COMPLETED; không tự điền số hoặc tăng n.

## 10. Định hướng viết ablation — không phải bản thảo Experiments

1. Bảng component ablation: Full, w/o EC, w/o LC, w/o CSE, Binary nếu có. Ghi actual n/seed/dataset; A0/A1 đã được audit ở đây.
2. Bảng evidence controls: Full, Count-only, Shuffle, Global-full nếu đủ. Tách attribution riêng PEEC khỏi external benchmark.
3. Đoạn A1: effect nhỏ, xu hướng dương Baby/Sports, Electronics mixed; bốn metrics và endpoint chính N@20. Không giấu seed 3 âm.
4. Hình paired ΔN@20: ba seeds × ba datasets, zero line, mean/CI có nhãn unadjusted. Không dùng learning curve làm replication.
5. Drop-one ablation chỉ cho incremental usefulness conditional on Full; không tạo additive causal contribution analysis.

Xem [kế hoạch paper](ECLIPSE_Paper_Experimental_Readiness_and_Presentation_Plan.md), [Full report](ECLIPSE_Full_Model_Multi_Seed_Analysis.md) và [bảng dữ liệu](../../reports/eclipse_multiseed_paper_audit_20261020/). ECLIPSE là phương pháp cuối; không đặt versions nội bộ thành external SOTA.
