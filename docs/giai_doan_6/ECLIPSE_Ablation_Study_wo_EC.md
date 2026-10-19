# ECLIPSE w/o EC — Ablation Study, Seeds 1–2 and Seed-3 Handoff

## Material Passport

- **Date:** 2026-10-19, Asia/Saigon.
- **Verification Status: ANALYZED.** Kiểm tra JSON/CSV/logs, cấu hình, digests của source trong Git và bytes checkpoint; **không** load checkpoint, chạy lại evaluator hoặc training trên máy local.
- **Artifact acceptance:** seed 1 giữ `REUSED_VERIFIED` theo audit recovery trước; seed 2 đạt `VERIFIED_ARTIFACT` trên cả ba dataset sau **151 kiểm tra nhất quán**. Nhãn nghiệm thu artifact không đồng nghĩa tái lập metric bằng model rerun, significance hoặc xác nhận độc lập bởi Hiếu.
- **Mẫu hiện có:** w/o EC có hai training seeds (1, 2) trên mỗi dataset. Trong thư mục logs local hiện chỉ có **một matched seed (1)** giữa Full ECLIPSE và w/o EC. Chưa tìm thấy native Full seed-2/3 receipts; không suy diễn từ việc user cho biết các run đang chạy.
- **Primary endpoint:** NDCG@20. Secondary: Recall@20; đồng thời báo Recall@10 và NDCG@10.
- **Sources:** [seed-2 artifacts](../../logs/ICAART/eclipse_wo_ec_seed2/), [seed-1 artifact audit](ECLIPSE_wo_EC_Seed1_Artifact_Audit.md), [execution plan](ECLIPSE_Experiment_Execution_Plan.md).
- **Machine-readable audit:** [acceptance.json](../../reports/eclipse_wo_ec_seed2_audit/acceptance.json), [recomputed selected TEST](../../reports/eclipse_wo_ec_seed2_audit/selected_test_by_seed.csv), [mean/SD](../../reports/eclipse_wo_ec_seed2_audit/mean_std.csv), [matched seed-1 differences](../../reports/eclipse_wo_ec_seed2_audit/matched_seed1_differences.csv). Recompute with `python reports/audit_eclipse_wo_ec_seed2.py`; script does not execute models.

## 1. Kết luận và những diễn giải đã sửa

**GO sang w/o EC seed 3, giữ nguyên cấu hình.** Ba run seed 2 hoàn tất 500 epochs và có selected TEST receipts hợp lệ. Không có dấu hiệu cần đổi kiến trúc hoặc tune hyperparameter từ các kết quả này.

Phần số liệu seed 2 và mean ± SD của bản báo cáo trước khớp native artifacts. Phần diễn giải có các lỗi đã sửa:

1. **Full seed 1 vs w/o EC seed 2 hoặc mean w/o EC n=2 không phải matched comparison.** Không được gọi đó là “EC thắng qua cả hai seed”. Bảng so sánh chính dưới đây chỉ dùng seed 1 ghép cặp đúng.
2. **Hai seed gần nhau không chứng minh hội tụ ổn định tuyệt đối, giảm phương sai do quy mô graph hoặc loại bỏ stochastic fluctuation.** Chỉ mô tả biến thiên đã quan sát.
3. **Runtime fields khớp không chứng minh toàn bộ môi trường bitwise giống nhau.** Các trường Python/Torch/FreeRec/GPU và precision đã khớp; các phiên Kaggle độc lập vẫn có thể khác tải, driver và trạng thái hệ thống.
4. **Không có quy tắc toán học “n<3 thì cấm p-value/CI, n=3 thì kiểm định chính thức chắc chắn hợp lệ”.** Hiện chưa tính vì chỉ có một matched pair cho so sánh Full–w/o EC; n=3 sau này vẫn rất ít mẫu.
5. **Peak training allocator không phải tổng VRAM và không bảo đảm không OOM.** Phải giữ phạm vi đo và phân biệt allocated/reserved/evaluation/driver.
6. **Không gọi checkpoint được chọn là “checkpoint epoch 500” nếu epoch khác.** Budget là 500; Baby s1 chọn 430, Baby s2 chọn 420, Sports s2 chọn 470 và Electronics s2 chọn 480.
7. Bỏ claim “cả hai seed đều khóa tại cùng commit”: seed 1 tái sử dụng source revisions cũ đã audit tương đương; seed 2 dùng pinned PEEC revision.

## 2. Cấu hình ablation và provenance

w/o EC giữ MI/FSC, BPR, LC, WMSG, CSE và item-side BSC; chỉ đặt `evidence_delta=0.0`. PEEC là **Pooled Excess-Evidence Calibration**; CSE là **Candidate Support Expansion**, triển khai qua CF support expansion. LC ở cấu hình này dùng first adjacent-layer gap, không phải mọi cặp tầng.

\[
\widetilde W_{ij}=W_{ij}\exp(\delta r_{ij}),\qquad
S_\delta=(1-\eta)\mathcal N(\widetilde W)+\eta\bar S_{CF}.
\]

Với $\delta=0$, $\eta=0.1$:

\[
S_{\mathrm{woEC}}=0.9\mathcal N(W)+0.1\bar S_{CF}.
\]

Implementation dùng exact cached native operator (`exact_v8_bypass=true`, `inactive_reason=delta_zero`). BSC tiếp tục lọc **Adam-preconditioned item directions của tổng loss**, không phải “làm phẳng phổ” đã được chứng minh hoặc một forward consensus view. User directions không được lọc bằng item graph. w/o EC không đồng nghĩa original STAIR vì WMSG/CSE/LC vẫn bật.

Seed 2 dùng commit `5212263e87c5c1cfe6e48c91afa90201a48f5929`. Seed 1 dùng historical commits `014886e7e7cd03096c724ba67815907e87e05d7c` (Baby/Sports) và `d401a27511a3f77b13c082d68c7f2359b7e4e61c` (Electronics), đã được kiểm tra operator/data/backbone equivalence trong audit riêng. Điều đó không chứng minh hai implementations có stochastic trajectory bitwise-identical.

Seed-2 manifest/contract khớp seed-1 reference về splits, feature digests, semantic candidates, TRAIN multiplicity, backbone hyperparameters và precision. Uncalibrated graph hashes bằng seed-1 w/o EC trên từng dataset. Ba YAML có byte SHA-256 khớp run manifest. Source digests khớp Git objects tại pinned revision. Có `best_model.pt` và đã hash bytes; không deserialize để khẳng định weights hoặc metrics đã được rerun.

## 3. Selected TEST — nguồn số liệu chính

Các số dưới đây lấy từ native `selected_metrics.json`. Tính bằng số chưa làm tròn; bảng chỉ định dạng để đọc. R@1 không có trong seed-2 receipts, nên **NOT_MEASURED**, không điền 0 hoặc suy từ seed 1.

| Dataset | Seed | VALID-selected epoch | TEST R@10 | TEST R@20 | TEST N@10 | TEST N@20 |
|---|---|---|---|---|---|---|
| Baby | 1 | 430 | 0.06748052 | 0.10543584 | 0.03647647 | 0.04624349 |
| Baby | 2 | 420 | 0.06732502 | 0.10360957 | 0.03662976 | 0.04592250 |
| Sports | 1 | 500 | 0.07618512 | 0.11424804 | 0.04196987 | 0.05181572 |
| Sports | 2 | 470 | 0.07614414 | 0.11432372 | 0.04154703 | 0.05143965 |
| Electronics | 1 | 500 | 0.04603368 | 0.06796098 | 0.02600378 | 0.03168096 |
| Electronics | 2 | 480 | 0.04604753 | 0.06816746 | 0.02601074 | 0.03173518 |

Mỗi seed chọn checkpoint riêng bằng cùng quy tắc VALID NDCG@20. Seed-2 selected epochs đều khớp argmax của native VALID history. `run_status.json` đều `COMPLETED`; telemetry có chính xác epochs 1–500.

**Lưu ý native evaluator:** mỗi seed-2 dataset có **hai TEST events**: terminal epoch 500 không được đánh dấu selected, rồi TEST tại restored selected checkpoint. Cụ thể: Baby `[500,420]`, Sports `[500,470]`, Electronics `[500,480]`. Báo cáo paper chỉ lấy selected receipt. Không được viết rằng pipeline chưa từng tính TEST nào khác; cũng không được dùng terminal TEST để chọn epoch/arm. Các artifacts hiện tại xác nhận selected epoch theo VALID, không phải argmax TEST.

## 4. Biến thiên riêng của w/o EC: n=2

| Dataset | n | R@10 mean ± SD | R@20 mean ± SD | N@10 mean ± SD | N@20 mean ± SD |
|---|---|---|---|---|---|
| Baby | 2 | 0.06740277 ± 0.00010996 | 0.10452271 ± 0.00129137 | 0.03655311 ± 0.00010839 | 0.04608300 ± 0.00022697 |
| Sports | 2 | 0.07616463 ± 0.00002898 | 0.11428588 ± 0.00005351 | 0.04175845 ± 0.00029900 | 0.05162768 ± 0.00026592 |
| Electronics | 2 | 0.04604060 ± 0.00000979 | 0.06806422 ± 0.00014601 | 0.02600726 ± 0.00000492 | 0.03170807 ± 0.00003834 |

\[
\bar M=\frac{M_1+M_2}{2},\qquad
s=\sqrt{\frac{\sum_{s=1}^2(M_s-\bar M)^2}{2-1}}.
\]

Đây là **sample SD với ddof=1**, không phải SE hoặc 95% CI. Sample variance dùng mẫu số n−1 là unbiased trong mô hình IID phù hợp; bản thân sample SD không nên gọi là “unbiased SD”. Các bảng notebook mean/std khớp recomputation.

[OBSERVATION] Baby và Sports có NDCG@20 seed 2 thấp hơn seed 1; Electronics seed 2 cao hơn seed 1. Baby Recall@20 dao động đáng chú ý (SD khoảng 0.001291), dù một số metric Electronics có SD nhỏ. Chưa thể suy ra quy mô catalog làm giảm phương sai hoặc toàn pipeline “cực kỳ ổn định” từ hai observations.

[INTERPRETATION] Những biến thiên này đủ để yêu cầu hoàn thành seed 3 và matched Full seed 2/3 trước khi kết luận incremental EC. Không có lý do thay hyperparameters chỉ vì một seed thấp hơn seed khác.

## 5. So sánh Full vs w/o EC — chỉ matched seed 1

| Dataset | Metric | Full s1 | w/o EC s1 | Full − w/o EC | Relative gain (%) |
|---|---|---|---|---|---|
| Baby | RECALL@10 | 0.06861877 | 0.06748052 | +0.00113825 | +1.6868 |
| Baby | RECALL@20 | 0.10610716 | 0.10543584 | +0.00067131 | +0.6367 |
| Baby | NDCG@10 | 0.03679847 | 0.03647647 | +0.00032200 | +0.8828 |
| Baby | NDCG@20 | 0.04639788 | 0.04624349 | +0.00015439 | +0.3339 |
| Sports | RECALL@10 | 0.07624275 | 0.07618512 | +0.00005763 | +0.0756 |
| Sports | RECALL@20 | 0.11450039 | 0.11424804 | +0.00025235 | +0.2209 |
| Sports | NDCG@10 | 0.04218513 | 0.04196987 | +0.00021526 | +0.5129 |
| Sports | NDCG@20 | 0.05208228 | 0.05181572 | +0.00026656 | +0.5144 |
| Electronics | RECALL@10 | 0.04640054 | 0.04603368 | +0.00036686 | +0.7969 |
| Electronics | RECALL@20 | 0.06811273 | 0.06796098 | +0.00015175 | +0.2233 |
| Electronics | NDCG@10 | 0.02621387 | 0.02600378 | +0.00021009 | +0.8079 |
| Electronics | NDCG@20 | 0.03183061 | 0.03168096 | +0.00014965 | +0.4724 |

Công thức: $\Delta_s=M_{\mathrm{Full},s}-M_{\mathrm{woEC},s}$; relative gain là $100(M_{\mathrm{Full},s}/M_{\mathrm{woEC},s}-1)$, không làm tròn trước khi tính.

[OBSERVATION] Seed 1 có chênh lệch dương ở cả 12 ô (4 metrics × 3 datasets). NDCG@20 tăng tương đối khoảng **0.334% Baby, 0.514% Sports, 0.472% Electronics**. Đây là kết quả mô tả **một paired training seed**, không phải bằng chứng significance hay robust gain qua hai seeds.

**Không đưa Full s1 vs w/o EC s2/mean n=2 thành bảng superiority.** Để thấy rủi ro của cách so đó: Electronics w/o EC s2 R@20=0.06816746 cao hơn Full s1 R@20=0.06811273; đây cũng không phải bằng chứng matched rằng tắt EC tốt hơn. Training RNG và calibration đang thay đổi đồng thời.

| Dataset | w/o EC seeds đã có | Full seeds có native receipts local cho comparison | Matched pair hiện dùng |
|---|---|---|---|
| Baby | 1, 2 | 1 | 1 |
| Sports | 1, 2 | 1 | 1 |
| Electronics | 1, 2 | 1 | 1 |

## 6. Loss và kiểm tra cơ chế

| Dataset | BPR epoch 1 → 500 | Raw LC epoch 1 → 500 | Interpretation |
|---|---|---|---|
| Baby | 0.629709 → 0.135911 | 5.717718 → 5.283269 | Training loss decreases; does not prove ranking convergence |
| Sports | 0.612982 → 0.023779 | 5.482507 → 4.698714 | Training loss decreases; does not prove ranking convergence |
| Electronics | 0.618115 → 0.033256 | 6.904704 → 6.393362 | Training loss decreases; does not prove ranking convergence |

Tổng loss khớp BPR + 0.01 × raw LC trong sai số phù hợp FP32; residual lớn nhất chỉ khoảng $4.26\times10^{-9}$, nhỏ hơn audit tolerance $10^{-7}$. Loss giảm không chứng minh ranking margin, hội tụ gradient, chống oversmoothing hoặc khử nhiễu semantic. Checkpoint chọn trước epoch 500 cũng không có nghĩa training bị dừng sớm.

### Evidence có thể được tính nhưng không được áp dụng

| Dataset | Candidate edges with r>0 (%) | Raw-weight coverage (%) | Relative operator dose | Polynomial probe dose |
|---|---|---|---|---|
| Baby | 10.3789 | 12.1749 | 0.0 | 0.0 |
| Sports | 10.0086 | 11.7541 | 0.0 | 0.0 |
| Electronics | 7.2407 | 8.4928 | 0.0 | 0.0 |

Seed 2 vẫn tính candidate statistics offline, nên $r>0$ có coverage khác 0, nhưng multiplier với $\delta=0$ vẫn là 1. Vì vậy **zero calibration dose không có nghĩa zero evidence coverage**. Seed-1 historical coverage/quantiles không được đo thì vẫn NOT_MEASURED; không lấp bằng 0.

Polynomial diagnostics dùng fixed Rademacher probes tại các tọa độ; không phải actual Adam-direction perturbation. Exact operator identity giải thích zero perturbation về đại số. Chưa có phép đo chứng minh positive-evidence edges là noise-free hoặc ranking-useful. Count/Shuffle và activity-null controls vẫn cần cho các claim cơ chế đó.

## 7. Thời gian và bộ nhớ

| Dataset | Preprocessing s | Coach.fit s | Fit hours | Training peak allocated MiB | Training peak reserved MiB |
|---|---|---|---|---|---|
| Baby | 23.42 | 1194.30 | 0.332 | 145.17 | 244.00 |
| Sports | 48.84 | 2548.35 | 0.708 | 223.71 | 458.00 |
| Electronics | 285.44 | 32302.49 | 8.973 | 800.67 | 1578.00 |

Tổng Coach.fit seed 2 = **36,045.14 s = 10.0125 giờ**; đây là tổng fit ba datasets, không phải end-to-end session time hoặc một benchmark song song. Preprocessing và packaging nằm ngoài tổng này.

Historical seed-1 fit cộng lại khoảng **9.7946 giờ**. Reusing checkpoints tránh lặp lại lượng fit work này, không phải phương pháp chạy nhanh hơn 9.79 giờ. Native reevaluation khoảng 30.81 s được ghi riêng trong recovery artifacts; graph preparation và các overhead khác không biến mất.

Fit seed 2 cao hơn historical seed 1 khoảng 3.10% Baby, 1.95% Sports và 2.22% Electronics. Đây là so sánh mô tả giữa các phiên có cùng GPU model, không chứng minh throughput hoặc môi trường Kaggle “ổn định tuyệt đối”. Cache/preprocessing state và runtime revisions của hai lịch sử cũng khác.

Allocated peaks có cùng giá trị ghi nhận với historical seed 1, nhưng chỉ thuộc **child-process per-training-epoch CUDA allocator peaks**; không bao gồm evaluation peaks, driver, process khác hoặc toàn bộ host RAM. Reserved không đồng nghĩa used tensors. Không được gọi các con số này là tổng VRAM hoặc bảo đảm tuyệt đối không OOM. Các run đã cung cấp hoàn tất là bằng chứng vận hành trên cấu hình đã đo, không phải mọi cấu hình/phần cứng.

## 8. Thống kê và giới hạn

- Chưa tính p-value, significance hoặc paired multi-seed CI vì hiện chỉ có **một matched Full–w/o EC pair** trên mỗi dataset.
- Không coi epochs là independent samples; không gộp datasets thành training seeds hoặc nhân bản Full s1 để ghép với w/o EC s2.
- n=2 đủ để tính sample SD; không có lệnh cấm toán học chung đối với CI/t-test ở n=2. Nhưng các giả định rất mạnh và uncertainty rất lớn. n=3 cũng không tự biến kiểm định thành đáng tin cậy.
- Khi đủ paired seeds 1–3: báo từng $\Delta_s$, mean±SD mỗi arm và win/tie/loss. Paired t-test nếu dùng chỉ có df=2 và phải nêu giới hạn. Exact two-sided sign-flip với ba nonzero pairs có min p=2/8=0.25; không đổi test để săn p<0.05.
- Nếu khai báo family Full vs w/o EC trên ba datasets cho NDCG@20, xử lý multiplicity theo kế hoạch. Không tuyên bố mọi ablation bắt buộc có significance.
- Per-user paired bootstrap/randomization là tùy chọn nếu có native per-user metrics khớp aggregate; nó không thay thế biến thiên training seeds.
- Đây là controlled component ablation trong benchmark đã khóa, chưa chứng minh nhân quả hành vi, exposure correction hoặc popularity debiasing.

## 9. Hình và dữ liệu cho paper

Các output seed 2 đã có PDF vector/PNG và source CSV/JSON. Giữ nguyên raw artifacts; không sửa số liệu để làm đẹp hình.

- `eclipse_wo_ec_<dataset>_seed2_learning_curves`: VALID curves và loss; marker tại selected epoch, không chọn bằng TEST.
- `eclipse_wo_ec_<dataset>_seed2_training_memory`: training allocator và epoch time, ghi đúng scope.
- `eclipse_wo_ec_<dataset>_seed2_calibration_diagnostics`: measured evidence coverage/quantiles bên cạnh zero operator/polynomial dose. Caption hiện đã nêu bằng chứng có thể tồn tại khi dose bằng 0.
- `eclipse_wo_ec_seed2_resource_profile`: mô tả tài nguyên của run này.
- `eclipse_wo_ec_seed_stability_ndcg20` và `...recall20`: hiện đúng n=2, sample SD, seed 3 để trống; không dùng làm hình chứng minh EC thắng Full.
- Hình ablation chính Full vs w/o EC nên đợi cùng matched seed set. Không dùng Full n=1 đối diện w/o EC n=2 mà gắn chung error bar hoặc cùng n.

Có thể dùng các hình theo run trong supplement. Architecture figure của notebook là schematic ablation, không thay Figure 1 full-method đã rà soát riêng.

## 10. Notebook và bước tiếp theo

Đã tạo [eclipse_wo_ec_seed3.ipynb](../../notebook/P6/eclipse_wo_ec_seed3.ipynb) từ notebook seed 2:

- SEED=3; cả ba datasets, 64D, 500 epochs; delta=0; không resume và không tải weights seed khác.
- Giữ nguyên training config/CLI checks, zero-bypass validator, runner và plot functions.
- Giữ seed-1 recovered measurements và thêm **audited seed-2 w/o EC receipts** chỉ để aggregate. Snapshots có native file hashes, canonical payload checks và contract/runtime/source validation. Full-model reference metrics không đi vào arm này.
- Tên hình tự đổi sang `eclipse_wo_ec_<dataset>_seed3_*`; PDF vector + PNG 300 dpi + CSV/JSON/captions. Khi run hoàn tất, seed-stability/mean±SD dùng actual measured seeds 1–3; thiếu run thì vẫn ghi NOT_MEASURED.
- Data-preflight dùng repair đã kiểm tra ở full seed-3 notebook: attach processed dataset bằng **Add Input trước Save & Run All**; không gọi attach API trong Kaggle batch. Việc này không thay algorithm hay fingerprints.
- Đóng gói configs, provenance, native receipts, checkpoints, logs, telemetry, tables và figures thành ZIP.

**GO** ở đây nghĩa là artifact/config đủ điều kiện tiếp tục replication, không nghĩa phương pháp đã được xác nhận thắng. Không rerun seed 2 hoặc đổi cấu hình từ TEST. Full seed 2/3 còn thiếu ở local cần được bàn giao cùng receipts để matched comparison. Human cross-read bởi Hiếu chưa được tự động suy từ tên người phụ trách trong ledger.

## 11. Fallacy scan — 11/11 mục đã xem xét

| Mục | Kết quả và giới hạn |
|---|---|
| Simpson's paradox | Giữ từng dataset; chưa có head/tail hoặc per-user subgroup để loại trừ đảo dấu phân nhóm. |
| Ecological fallacy | Không suy từ aggregate gain sang mọi user/item đều hưởng lợi. |
| Berkson/selection bias | Chỉ ngoại suy trong filtered benchmark và split hiện có, không toàn bộ thị trường. |
| Collider bias | Không có covariate adjustment ở đây; VALID selection không nhận dạng causal user mechanism. |
| Base-rate neglect | Recall/NDCG không được diễn giải thành xác suất mua hàng. |
| Regression to mean | Không chọn seed/cấu hình bằng TEST; kết quả seed 1 dương không bảo đảm seed sau dương. |
| Survivorship bias | Ba run seed 2 được cung cấp đều hoàn tất; audit này không chứng minh chưa có attempt khác không được bàn giao. |
| Look-elsewhere | NDCG@20 primary; báo đủ bốn metrics, không chuyển endpoint theo kết quả. |
| Forking paths | Giữ protocol; không tune lại ablation sau khi xem TEST. Không giả định một file plan là preregistration có timestamp độc lập. |
| Correlation/causation | Một matched seed chỉ cho mô tả incremental component effect trong run đó; không chứng minh khử nhiễu/popularity mechanism. |
| Reverse causality | Evidence dùng TRAIN; đồng xuất hiện vẫn không phải causal preference hoặc exposure-adjusted utility. |
