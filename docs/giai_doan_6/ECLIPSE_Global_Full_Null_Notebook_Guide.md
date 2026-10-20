# ECL-global_full_null — hướng dẫn sáu run ablation

**Cập nhật 2026-10-20. Trạng thái: Seeds 1 & 2 COMPLETED ($n=2$ per dataset, đã kiểm toán và phân tích trong [ECLIPSE_Ablation_Study_Global_Full_Null.md](ECLIPSE_Ablation_Study_Global_Full_Null.md)); Seed 3 READY.**

## 1. Notebook và phạm vi

| Notebook | Seed | Dataset | Budget | Trạng thái |
| --- | --- | --- | --- | --- |
| [eclipse_global_full_null_seed1.ipynb](../../notebook/P6/eclipse_global_full_null_seed1.ipynb) | 1 | Baby, Sports | 64D, 500epochs/dataset | **COMPLETED** (`logs/ICAART/eclipse_global_full_null_seed1/`) |
| [eclipse_global_full_null_seed2.ipynb](../../notebook/P6/eclipse_global_full_null_seed2.ipynb) | 2 | Baby, Sports | 64D, 500epochs/dataset | **COMPLETED** (`logs/ICAART/eclipse_global_full_null_seed2/`) |
| [eclipse_global_full_null_seed3.ipynb](../../notebook/P6/eclipse_global_full_null_seed3.ipynb) | 3 | Baby, Sports | 64D, 500epochs/dataset | **READY** (Chờ slot thực thi) |

Tổng **6 run** do Chương thực hiện. Không có Electronics trong phạm vi arm này. Mỗi notebook có35cells, chạy hai dataset tuần tự; có thể chia Baby/Sports sang session riêng bằng `SELECTED_DATASETS` trước setup. Không đổi seed/hyperparameters giữa run.

## 2. Câu hỏi khoa học và công thức đúng

Global-full kiểm tra đóng góp của **activity conditioning** khi vẫn dùng toàn bộ người dùng TRAIN, cùng observed counts và excess score. Không phải control chia hai nửa và không phải Count-only.

Với binary TRAIN presence R sau dedup, đặt h_u là số item trong TRAIN history, n_i=Σ_u R_ui và M_active=|{u:h_u>0}|. Chỉ thay:

\[
\mu_{ij}^{\rm activity}=\sum_{b:M_b>0}\frac{n_{ib}n_{jb}}{M_b}
\quad\longrightarrow\quad
\mu_{ij}^{\rm global}=\frac{n_i n_j}{M_{\rm active}}.
\]

Giữ:

\[
c_{ij}=\sum_uR_{ui}R_{uj},\qquad
r_{ij}^{\rm global}=\frac{[c_{ij}-\mu_{ij}^{\rm global}]_+}{c_{ij}+\mu_{ij}^{\rm global}+4},
\]
\[
\widetilde W_{ij}=W_{ij}\exp(\delta r_{ij}^{\rm global}),\quad\delta=1,\qquad
S=(1-\eta)\mathcal N(\widetilde W)+\eta\bar S_{\rm CF},\quad\eta=.1.
\]

Native mode **`pooled_global_excess`**, `evidence_activity_bins=none`, `evidence_halves=0`, tau2, delta1. `null_scope=global_full`, `active_partition=false`, `active_halves=0`, `activity_conditioning=false`. Inactive users không vào mẫu số; không dùng toàn bộ n_users nếu có user không có TRAIN history. Không dùng `unstratified_replication`, vì arm đó còn chia hai nửa và đổi estimator.

WMSG/CSE/LC/MI/FSC/BSC/scorer, embedding dimension và từng dataset gamma/lr/batch/weight decay/workers giữ theo Full. Không tune riêng từ TEST. Đây là reference expectation theo giả định independence, không phải calibrated probability hoặc kiểm định causal.

## 3. Source đã được bổ sung gì?

Source cũ tại commit Full chưa có mode này. Bổ sung tối thiểu ba file:

- [stair_icaart_config.py](../../stair_icaart_config.py): registry mode mới, bins=none, full pooling/halves0, identity global_full. Giữ settings/identity của các modes cũ.
- [stair_icaart_evidence.py](../../models/stair_icaart_evidence.py): candidate-only sparse overlap counter cũ; thêm null scope một pool chứa active users. Cùng integer counts, FP64 marginal products/null/score; không tạo dense item×item.
- [stair_icaart_graph.py](../../models/stair_icaart_graph.py): một global pool, gọi pooled excess cũ, giữ calibration/normalization/blend. Ghi M_active/inactive, one-pool metadata và candidate-wise formula consistency check. Chi phí thêm của check là O(E) CPU theo block, không online training branch hoặc trainable parameters.

`num_activity_strata=0` ở arm này, `num_null_groups=1`, `group_sizes=[M_active]`; không giả ghi log2 strata đang hoạt động. Nếu Full chỉ có một activity stratum, hai null có thể trùng: notebook **không ép** null/graph hashes hoặc metrics phải khác.

## 4. Chạy được trên Kaggle mà không cần push commit mới

Mỗi notebook có **ba file source extension nhúng đầy đủ, khóa SHA256**, trên base commit `5212263e87c5c1cfe6e48c91afa90201a48f5929`.

1. Tạo checkout **riêng cho session Global-full**, không dùng checkout của notebook Full/Count đang chạy.
2. Kiểm base hashes rồi áp đúng ba file extension trước imports/preflight; không thay source tùy ý, không fallback HEAD.
3. Đối chiếu toàn bộ source digest: ba file theo overlay, những file còn lại theo reference Full.
4. Lưu base commit **và overlay ID**, payload/digests, applied-file receipt, `source_diff.patch`, Git status, configs và runtime. Native manifest cũng ghi actual source SHA.

Vì vậy **không cần commit/push mới để dùng ba notebook này**; chỉ tải đúng notebook. Không được mô tả run Global-full là “identical source commit với Full” mà bỏ qua extension. Dùng base commit + overlay ID + source hashes để xác định mã thực thi. `REPO_OVERRIDE` bị khóa nhằm tránh patch nhầm checkout của run khác.

Các notebooks Full/Count cũ vẫn dùng base pin/source checks riêng; không bị notebook mới sửa. References `SEED1_REFERENCE` trong Global notebooks vẫn là Full seed1 protocol/fingerprints, không phải seed đang train. Runtime mong đợi vẫn như các arm trước; nếu lệch, phải ghi/chấp nhận có lý do theo guard, không tùy tiện tắt checks.

## 5. Các bước chạy

1. Upload notebook đúng seed; attach processed Baby/Sports bằng Add Input **trước Save & Run All**. Bật GPU và Internet cho setup.
2. Giữ `RUN_MODE='train'`, SEED đúng1/2/3, `SELECTED_DATASETS=('baby','sports')`, GPU_INDEX đúng GPU của session. Không cần Electronics inputs.
3. Chạy setup, private source/overlay, data checks và native config preflight. Preflight kiểm mode/global bins/halves, cùng training settings và không resume.
4. Chạy Baby và ba plotting cells; Sports và ba plotting cells; session summary/resource profile; packaging cuối notebook.
5. Download ZIP + `.archive_receipt.json` và executed `.ipynb`; có thể cung cấp `NOTEBOOK_SOURCE_PATH` để gói exact submitted file.

Giữ guard directory trống do preflight đã sửa ở Count template. Directory có payload/log/symlink vẫn được bảo toàn; không tự delete/retry/resume. Rerun cell của attempt đã COMPLETED chỉ export. Nếu job fail, giữ logs và đóng gói phần completed; không biến missing thành0.

Hai GPU/session độc lập có thể chạy hai notebook đồng thời; notebook thứ ba chạy khi có slot. Không ngắt run khác và không áp extension lên shared checkout. Train jobs trong một notebook chạy tuần tự, không tự chia hai GPU.

Report-only: `RUN_MODE='report_only'`, mapping `EXISTING_ARTIFACT_DIRS` và `EXISTING_STDOUT_LOGS` đúng dataset/seed/arm. Không clone/pip/GPU/model import/training. Mode/null identity sai bị chặn; khác source/runtime phải được đánh dấu review-required. Không đổi nhãn Full/Count thành Global-full.

## 6. Cần thu những gì mỗi run?

### Kết quả và provenance

- Native selected TEST đủ RECALL@10/20,NDCG@10/20 tại checkpoint VALID NDCG@20 chọn; best_model.pt và checksum;500epoch telemetry, native evaluation JSONL, COMPLETED và stdout.
- Config/resolved CLI, data/feature/candidate fingerprints, source hashes/base+overlay ID/diff/runtime, count/null/score/final-operator hashes.
- Counts/raw semantic/candidates/presence/CF như Full. **Cho phép khác** assignment/groups/null hash, score và calibrated final operator; không ép chúng bằng activity-conditioned Full.
- `global_null_audit.json`: M_active, inactive users, global scope, no halves, one pool, formula-check error và counts/null hashes. Root/graph/checkpoint estimator identities phải nhất quán.
- Preprocessing, `Coach.fit`, sum epoch-train seconds, training allocated/reserved peaks đúng scope; không gọi allocated là whole-process GPU peak hoặc subprocess wall là fit time.

### Hình/bảng

Mỗi dataset/seed xuất PDF vector + PNG300dpi, CSV/JSON nguồn và English caption:

- `eclipse_<dataset>_global_full_null_seed<s>_learning_curves`: BPR/weighted LC/total loss và VALID ranking, marker selected epoch; không TEST selection curve.
- `eclipse_<dataset>_global_full_null_seed<s>_training_memory`: training allocated/reserved và thời gian/epoch.
- `eclipse_<dataset>_global_full_null_seed<s>_calibration_diagnostics`: positive **excess** coverage, operator/poly probe dose và excess-score quantiles. Không gọi coverage này là c>0 hoặc count score.
- `eclipse_global_full_null_seed<s>_resource_profile`: hai dataset, scope timer/memory rõ.

Reference dose là weighted semantic+CSE uncalibrated η=.1, như Full. Polynomial probe không phải actual Adam-direction perturbation. Source overlay và các export đều vào ZIP/inventory/checksum; snapshot là đúng notebook và seed tương ứng.

## 7. Tổng hợp sau khi đủ sáu run

Mỗi notebook chỉ có actual n=1/dataset, không tự tạo SD/p-value. Sau khi kiểm toán receipts, ghép seeds1–3 của Global và Full trên Baby/Sports, tính mean±sample SD, paired Δ, NDCG@20 chính/Recall@20 phụ, đầy đủ bốn metrics. Không gộp epochs hoặc dataset để tăng n.

Nếu Full > Global nhất quán: có bằng chứng incremental cho activity conditioning trong protocol này. Nếu tương đương về point estimates/Global cao hơn: chưa đủ để claim conditioning hữu ích; không giấu số hoặc tune TEST. Không có significant difference không tự chứng minh equivalence. Cùng delta không bảo đảm cùng operator dose, nên log dose/coverage và bàn luận confound cường độ.

## 8. Phạm vi kiểm tra đã làm

[Build receipt](../../reports/eclipse_global_full_null_notebooks/build_receipt.json): 33 kiểm tra tĩnh/notebook, JSON/AST/cú pháp/probe strings/seed/dataset/overlay/snapshot/guards. [CPU validation](../../reports/eclipse_global_full_null_notebooks/validation.json): 48 đối chiếu trên dữ liệu sparse nhỏ tổng hợp và config/manifest guards; M_active khác tổng users, duplicates, counters>127, empty users/items, block invariance, cùng counts khi thay null, cấu hình/identity của modes cũ giữ nguyên.

**Không chạy notebook, không chạy model/GPU/training/network hoặc toàn bộ test suite. Chưa có Kaggle integration smoke run hay real-dataset Global metrics.** READY nghĩa code/notebook được chuẩn bị và kiểm tra trong phạm vi trên; không phải kết quả thực nghiệm COMPLETED.

[Source overlay](../../reports/eclipse_global_full_null_notebooks/source_overlay.json), [builder](../../reports/prepare_eclipse_global_full_null_notebooks.py), [CPU validation script](../../reports/validate_eclipse_global_full_null.py). Full/Count notebooks và raw logs được giữ nguyên.
