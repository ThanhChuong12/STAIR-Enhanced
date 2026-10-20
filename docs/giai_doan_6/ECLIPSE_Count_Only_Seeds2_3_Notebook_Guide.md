# ECL-count_only — notebook seeds 2 và 3

Ngày tạo: 2026-10-20. Hai notebook được dẫn xuất từ bản hiện tại của `eclipse_count_only_seed1.ipynb`; không sửa notebook seed1 hoặc mã nguồn mô hình.

## 1. Phạm vi đúng kế hoạch

| Notebook | Seed mặc định | Dataset chạy | Budget |
| --- | --- | --- | --- |
| [eclipse_count_only_seed2.ipynb](../../notebook/P6/eclipse_count_only_seed2.ipynb) | 2 | Baby → Sports | 500 epochs/dataset, 64D |
| [eclipse_count_only_seed3.ipynb](../../notebook/P6/eclipse_count_only_seed3.ipynb) | 3 | Baby → Sports | 500 epochs/dataset, 64D |

**Tổng còn bốn run Count-only. Electronics chỉ seed1 trong phạm vi hiện tại.** Người dùng xác nhận Count-only seed1 đã hoàn tất cả ba dataset; lần tạo notebook này không audit thêm artifacts seed1 hoặc tự tạo kết quả mới.

## 2. Những gì giữ nguyên từ seed1

- `evidence_mode=pooled_activity_count`, delta1, tau2, halves0. Score c/(c+4), không trừ null; null có thể vẫn được tính offline để diagnostics.
- CSE eta.1, LC lambda.01, WMSG/MI/FSC/BSC, per-dataset gamma/lr/batch/weight-decay/workers và protocol 500epochs giữ như reference Full.
- Pinned runtime source `5212263e87c5c1cfe6e48c91afa90201a48f5929`, input/candidate/count/null/CF fingerprints và runtime checks.
- Chọn checkpoint bằng VALID NDCG@20, báo TEST trong native `selected_metrics.json`; không thay bằng terminal TEST.
- Guard đã sửa lỗi thư mục preflight: cho phép real directory trống khi chưa có log, giữ chặn payload/log/symlink cũ. Không xóa, resume hoặc retry tự động.
- Các engine training, reporting, plotting, packaging giữ nguyên nội dung code từ seed1; thay seed/default datasets, frozen reference subset, headings và snapshot metadata. Bỏ hẳn section Electronics khỏi hai notebook.

`SEED1_REFERENCE` vẫn là **reference protocol/fingerprints Full seed1**, không phải seed đang training. Giữ tên biến này có chủ đích; training seed được ghi bằng `SEED=2/3` trong CLI/config/manifests/checkpoint checks.

## 3. Cách chạy

1. Upload notebook tương ứng, attach processed MMRec Baby/Sports trước Save & Run All, bật GPU/Internet cho setup. Không cần attach Electronics cho hai notebook này.
2. Giữ `RUN_MODE='train'` và seed mặc định. `SELECTED_DATASETS=('baby','sports')`; có thể chọn một dataset trước setup nếu cần tách session.
3. Chạy thứ tự notebook: setup/preflight → Baby/train/plots → Sports/train/plots → summary → ZIP. Mỗi notebook chạy hai child training jobs tuần tự.
4. Hai notebook có thể chạy song song trong hai session/GPU riêng. Không đổi source hoặc seed giữa một run đang chạy. Nếu dùng máy hai GPU, cấu hình `GPU_INDEX` đúng GPU dành cho từng session; không đồng thời sửa cùng thư mục runtime/data bridge bằng hai setup processes.
5. Download ZIP + `.archive_receipt.json`; lưu executed ipynb riêng, hoặc cung cấp `NOTEBOOK_SOURCE_PATH` để gói exact submitted file.

Muốn chỉ hậu xử lý run đã có: dùng `RUN_MODE='report_only'` và điền mapping `EXISTING_ARTIFACT_DIRS`/`EXISTING_STDOUT_LOGS` cho đúng Baby/Sports và đúng seed. Không training/GPU/model import/network ở mode này. Không dùng notebook seed2 để nhận receipts seed1 rồi đổi nhãn.

## 4. Hình và dữ liệu đầu ra

Với s=2 hoặc3, mỗi dataset xuất PDF vector + PNG300dpi, source data và caption:

- `eclipse_<dataset>_count_only_seed<s>_learning_curves`
- `eclipse_<dataset>_count_only_seed<s>_training_memory`
- `eclipse_<dataset>_count_only_seed<s>_calibration_diagnostics`

Session có `eclipse_count_only_seed<s>_resource_profile`, selected TEST overview, resource/diagnostic tables, actual seed status, receipts/checkpoints/telemetry, stdout, configs/commands/runtime/source/input provenance, source template, inventory và checksum.

Mỗi notebook chỉ đo một seed/dataset: không tự tạo SD hoặc p-value ba seed. Sau khi cả Count seeds1–3 được audit, tổng hợp matched Full-vs-Count trên Baby/Sports; Electronics giữ n=1. Operator/probe dose không phải actual Adam-direction perturbation; training allocated/reserved không phải whole-process GPU memory.

## 5. Trạng thái bàn giao

**READY / STATIC_CHECKED_NOT_EXECUTED**, chưa là thực nghiệm COMPLETED. Mỗi notebook có35cells; đã kiểm JSON, AST/cú pháp, child probe strings, seed/dataset defaults, shared code không đổi, không còn Electronics invocation, launch guard và embedded snapshot đúng notebook tương ứng. Không thực thi notebook cells, model, GPU, training, network hoặc test suite.

[Build receipt và hashes](../../reports/eclipse_count_only_seeds2_3_notebooks/build_receipt.json); [script dẫn xuất](../../reports/prepare_eclipse_count_only_followup_notebooks.py). Notebook seed1 được xác nhận byte hash không đổi sau khi tạo hai bản mới.
