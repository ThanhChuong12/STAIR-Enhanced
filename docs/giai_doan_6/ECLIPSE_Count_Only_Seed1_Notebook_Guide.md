# ECL-count_only seed 1 — Notebook handoff

## Material Passport

- Date: 2026-10-19, Asia/Saigon.
- Status: **READY / software checks only**. Không model/GPU/network/training hoặc đo metric Count-only thực tế.
- Notebook: [eclipse_count_only_seed1.ipynb](../../notebook/P6/eclipse_count_only_seed1.ipynb).
- [Validation receipt](../../reports/eclipse_count_only_seed1_notebook/validation.json): **111 kiểm tra đạt (sau bản sửa guard preflight)**. Fixtures chỉ kiểm schema/export, được xóa, không phải số liệu paper. Local harness bỏ riêng IPython display do không có IPython; notebook Kaggle giữ display.

## 1. Cấu hình và giả thuyết

Chạy **Baby → Sports → Electronics, seed1**, fresh64D/500epochs. `pooled_activity_count`, delta1/tau2/halves0; eta.1 và LC.01 giữ như Full. Score là `c/(c+4)`; null vẫn có thể được tính offline nhưng không bị trừ trong score. Mục tiêu là kiểm tra lợi ích của excess correction ngoài việc chỉ boost cạnh có co-occurrence.

Các gamma mặc định được giữ theo Full: Baby0.1, Sports0.2, Electronics0.4; giữ batch/lr/decay/sampling/evaluator và source revision `5212263e87c5c1cfe6e48c91afa90201a48f5929`. Mọi config được preflight từ YAML/CLI thực tế trước training.

Count/null/candidates/raw semantic/CF hashes được đối chiếu Full. Không ép Count score hash hoặc calibrated final graph bằng Full; không ép dose0 hoặc dùng reference semantic-only của w/o CSE. Source builder đã pin và score extrema được kiểm theo công thức; không gọi kiểm tra extrema là chứng minh toàn bộ tensor score. Formula CPU-only của source thực đã được kiểm độc lập trên input nhỏ.

## 2. Cách chạy mới

1. Dùng slot GPU/session riêng, giữ các full/w/o EC jobs đang chạy. Không đổi source giữa run.
2. Attach processed MMRec data bằng Add Input **trước Save & Run All**. Bật GPU; Internet cần cho clone pinned source/dependencies nếu chưa có. AUTO_DOWNLOAD_DATA=False; notebook không attach datasource trong Kaggle batch.
3. Giữ `RUN_MODE='train'`, `SEED=1`, `SELECTED_DATASETS=('baby','sports','electronics')`. Nếu cần tách Electronics sang session riêng, chọn subset trước khi chạy; vẫn giữ64D/500epochs và cùng cấu hình. GPU_INDEX chọn đúng GPU rảnh.
4. Chạy thứ tự notebook. Các child processes tuần tự; checkpoint chọn bằng VALID NDCG@20, báo native selected TEST đủ RECALL@10/20,NDCG@10/20. Terminal TEST không dùng để chọn hoặc thay selected receipt.
5. Download ZIP **và** `.archive_receipt.json`; tải executed Kaggle ipynb riêng hoặc cung cấp NOTEBOOK_SOURCE_PATH để gói exact submitted notebook. Source template cũng được lưu trong gói.

Unique output/attempt, không overwrite/delete artifacts; rerun completed dataset cell chỉ export; incomplete attempt bị giữ và từ chối tự resume/retry. Runtime strict default yêu cầu môi trường khớp; khác biệt phải được ghi/chấp nhận có lý do, không xóa guard tùy tiện. T4 tổng thời gian lịch sử khoảng10GPUh cộng setup/preprocess/export không phải bảo đảm hoàn tất dưới platform limit.

## 3. Các hình/bảng được xuất

Mỗi dataset:

- `eclipse_<dataset>_count_only_seed1_learning_curves`: TRAIN total/BPR/weightedLC + VALID NDCG@20/Recall@20; marker selected epoch.
- `eclipse_<dataset>_count_only_seed1_training_memory`: child training allocated/reservedMiB và seconds/epoch.
- `eclipse_<dataset>_count_only_seed1_calibration_diagnostics`: positive count-score coverage, operator/poly probe dose, count-score quantiles. Reference uncalibrated weighted+CSE eta.1; không phải actual Adam-direction perturbation.
- Selected TEST CSV/JSON, diagnostics/mechanism, comparability, input receipt/checkpoint byte hashes, training/evaluation CSV và resource summary.

Mỗi hình **PDF vector+PNG300dpi**, sourceCSV/JSON, caption tiếng Anh. Session có `eclipse_count_only_seed1_resource_profile`, bảng tổng hợp và trạng thái seed thật. Một seed/dataset chỉ báo điểm đo, **không SD/p-value**; không nhân bản metrics Full hoặc các arms khác để lấp bảng. Không cần architecture figure mới cho mỗi run.

ZIP gồm stdout log ngoài native artifact directory, receipts/checkpoints/telemetry, config/resolvedCLI/source/environment/input provenance, notebook source/template, scripts/tables/figures/captions, file inventory và checksum sidecar. Không được gọi allocated peaks là totalVRAM/whole-run peak, hoặc subprocess duration là Coach.fit.

## 4. Chỉ bổ sung hình cho run đã có

`RUN_MODE='report_only'`, điền `EXISTING_ARTIFACT_DIRS={'baby':'...native...',...}` và `EXISTING_STDOUT_LOGS={'baby':'...stdout.log',...}` cho các dataset đã chọn. Mode này không import model/torch, không clone/pip/download hoặc training. Native originals không đổi, outputs xuất vào root mới. Log thiếu => fit time NOT_MEASURED; mode/config/data/evidence sai bị chặn, source/runtime khác lịch sử được gắn review-required.

Nếu một job fail làm Run All dừng, giữ nguyên log/native attempt và xuất/đóng gói phần completed từ các cells sau; trạng thái dataset chưa có vẫn NOT_MEASURED. Không chạy lại training để có hình đẹp hơn.

## 5. Sau seed1

Chương bàn giao, Hiếu đọc chéo đúng artifacts. Chưa tự xác nhận VERIFIED_ARTIFACT hoặc human review. Count-only tiếp tục Baby/Sports seeds2–3 (bỏ Electronics khỏi SELECTED_DATASETS khi đổi seed); Electronics chỉ seed1 theo phạm vi hiện tại. Tổng Count-only7runs. Không đổi delta/hyperparameters từ TEST. Matched Full-vs-Count so sánh chỉ sau khi có cùng seed set và artifact phù hợp.


## 6. Sửa lỗi FileExistsError trước training (2026-10-19)

**Nguyên nhân đã xác định:** `main_stair_icaart.py` tạo `artifact_dir` ngay trong `post_config()` khi CLI preflight gọi `build_config()`. Guard cũ `artifact.exists()` nhầm thư mục trống đó với run đã bắt đầu; `mkdir(exist_ok=False)` cũng chưa tương thích lifecycle này. Đây là lỗi notebook, không phải lỗi dữ liệu hoặc mô hình Count-only.

Notebook đã sửa: chỉ chấp nhận **real directory trống và chưa có stdout log**; vẫn từ chối symlink, path không phải directory, directory có bất kỳ payload hoặc stdout log đã tồn tại. Không xóa artifacts, không mở tự resume/retry. Tạo directory với `exist_ok=True` sau khi guard xác nhận hợp lệ. Giữ nguyên seed/protocol/hyperparameters.

**Trên phiên Kaggle hiện tại:** copy lại cell ở mục6 chứa `train_dataset` từ notebook đã sửa, chạy cell định nghĩa hàm rồi chạy lại cell Baby. Nếu folder chỉ trống do preflight thì không cần đổi SEED/SESSION_ID, tải lại dataset hoặc chạy lại setup. Nếu guard mới báo payload/log tồn tại, giữ nguyên run đó để audit; chỉ tạo session mới khi thực sự cần fresh attempt, không xóa file để vượt guard.

[Validation111](../../reports/eclipse_count_only_seed1_notebook/validation.json) bổ sung hồi quy còn thiếu ở bộ102 trước: thực thi riêng expression mkdir của native preflight, kiểm controller tiến tới launch bằng Popen interceptor (không launch process), và giữ chặn log/payload cũ. Đã chạy lại toàn bộ fixture export ba dataset; không model/GPU/network/training. [Fix receipt](../../reports/eclipse_count_only_seed1_notebook/empty_directory_fix.json). Template w/o CSE có cùng lỗi guard cũng được đồng bộ sửa; source/model code không đổi.
