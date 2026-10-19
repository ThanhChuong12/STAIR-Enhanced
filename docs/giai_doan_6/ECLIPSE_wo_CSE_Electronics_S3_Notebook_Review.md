# ECLIPSE w/o CSE Electronics seed 3 — Notebook Review for Hiếu

## Material Passport

- Date: 2026-10-19, Asia/Saigon.
- **Verification Status: ANALYZED — static source review only.** Không chạy cells, cài dependencies, gọi network, load checkpoint, test model hoặc training.
- Notebook: [eclipse_wo_cse_electronics_s3.ipynb](../../notebook/P6/eclipse_wo_cse_electronics_s3.ipynb).
- Snapshot SHA256: `02d9ce2e6301aac2ab533e15dd33f6e4f8d249e3f1f881da576755de3a1c4997`.
- Có 11 cells, 5 code cells, không lưu execution outputs; AST parsing thành công không chứng minh notebook chạy đúng trên Kaggle.
- [Numbered source](../../reports/eclipse_wo_cse_electronics_s3_notebook_audit/notebook_source_numbered.txt), [review receipt](../../reports/eclipse_wo_cse_electronics_s3_notebook_audit/review_receipt.json).
- Tham chiếu `Cell N:Lx–Ly` dưới đây dùng **JSON cell index bắt đầu từ 0**, line bắt đầu từ 1 trong source cell. Heading “Cell 1” của notebook chính là JSON Cell 2.
- Không có selected receipts/telemetry w/o CSE được bàn giao cùng notebook này và chưa tìm thấy đường dẫn w/o CSE dưới `logs/ICAART` theo tên. Vì vậy chưa xác nhận run thực tế, metric, hình render hoặc full search mọi vị trí ngoài thư mục đó.

> **Follow-up 2026-10-19:** Notebook đã được sửa trực tiếp theo yêu cầu tiếp theo. Các findings bên dưới mô tả **snapshot gốc** theo SHA256 đã ghi, không phải khẳng định notebook mới vẫn còn tất cả lỗi đó. Xem mục7 về phiên bản đã sửa và cách chạy.

## 1. Verdict

**Notebook có đúng ý định ablation, nhưng chưa đủ để nghiệm thu gói kết quả paper.** CLI đặt eta=0, delta=1 và lambda_nlgcl=0.01 đúng w/o CSE. Chưa có bằng chứng source tĩnh rằng công thức w/o CSE chắc chắn sai. Tuy nhiên có lỗi đọc metric, nguy cơ ghi đè artifacts, source không khóa, thiếu contract checks và thiếu hoàn toàn pipeline vẽ hình/bảng.

Không cần tự động bỏ hoặc train lại một run đã hoàn tất vì notebook thiếu plotting. Trước hết giữ nguyên artifacts, kiểm tra native selected receipts/manifest/telemetry/checkpoint rồi bổ sung hậu xử lý offline. Chỉ quyết định rerun nếu phát hiện lỗi thực sự ảnh hưởng data/config/selection/weights hoặc thiếu bằng chứng thiết yếu không thể khôi phục; không rerun vì metric thấp.

## 2. Điểm làm đúng

1. Cell 6:L35–44,95–101 đặt `eta=0.0`, `delta=1.0`, `lambda_nlgcl=0.01`, `pooled_activity_excess`, halves=0, tau=2. CLI giữ calibration và LC thay vì biến w/o CSE thành w/o EC.
2. Cell 6:L119–139 stream stdout/stderr vào log và raise khi child exit code khác 0; không lấy exit code lỗi làm run thành công.
3. Cell 6:L141–143 đọc native `selected_metrics.json` thay vì parse số làm tròn từ console. Đây là nguồn đúng nếu receipt contract được kiểm tra.
4. Cell 10 đóng artifact directory và tính SHA256 theo chunks, không đọc cả archive vào RAM. Native directory có thể đã chứa manifest/checkpoints/telemetry; không được nói notebook hoàn toàn không lưu provenance.
5. YAML local đang có Electronics 64D, 500 epochs, gamma=0.4, batch=4096, lr=0.001, decay=0.1. Nhưng notebook tải HEAD nên chưa khóa được các giá trị này ở phiên chạy tương lai.

## 3. Findings cần sửa trước các run tiếp theo

| ID / Mức | Vị trí | Phát hiện và tác động | Hiếu cần bổ sung |
|---|---|---|---|
| F01 / HIGH | Cell 6:L147–148 | Key `Recall@10/20` không khớp native `RECALL@10/20`; `.get(m,0.0)` in ra 0 giả. NDCG dùng đúng case. Receipt không bị sửa bởi câu print này, nhưng bảng thủ công lấy từ console sẽ sai. | Dùng bốn canonical keys uppercase, fail khi thiếu/NaN; tuyệt đối không fallback 0. |
| F02 / HIGH | Cell 6:L46–72,119; Cell 10:L16–20 | Output path chỉ có dataset/arm/seed; chạy lại cell sẽ xóa receipts/telemetry/checkpoints cũ rồi log được mở `w`. TAR cũng cùng tên và bị ghi đè. | Session UTC + UUID/attempt_id mới; fail nếu thư mục attempt đã tồn tại. Không tự xóa run cũ. Cache tách khỏi output. |
| F03 / HIGH | Cell 2:L22–42,62–73 | Clone HEAD từ hai repos không pin commit; sau đó sửa validator trực tiếp. Revisions/config thay đổi theo ngày. Native manifest có source digests nhưng Git commit không mô tả đủ dirty patch. | Khóa revision đã nghiệm thu; lưu URL, commit, git status/diff, file hashes, patch trước/sau. Không reset checkout process đang train. |
| F04 / MEDIUM | Cell 2:L62–73,125–127 | Preflight kiểm tra lambda=0 cho **w/o LC**, không kiểm tra eta=0 của w/o CSE. Mở validator toàn cục làm mất guard cho full, dù run này thực tế lambda vẫn 0.01. | w/o CSE không cần bản vá LC. Named-arm allowlist riêng; test đúng effective config của `(dataset,arm,seed)` và lưu diff chỉ eta. Nếu revision cũ buộc patch: lưu diff, không tự coi cùng commit=source giống nhau. |
| F05 / HIGH | Cell 6:L32–42,86–102,141–149 | Chưa so effective config với locked protocol; chưa kiểm tra seed, dataset, budget, runtime/data/candidate hashes, effective_eta hoặc LC còn bật trong manifest. Exit0 không đủ. | Kiểm tra trước/sau run bằng resolved config, `manifest`, `run_status`, telemetry và receipt. w/o CSE được phép graph khác Full; support semantic/count/null/scores phải giữ. |
| F06 / HIGH | Cell 6:L141–149 | Chỉ đọc epoch/metrics, không xác thực mode=test, selected_checkpoint=true, selection=VALID NDCG@20, seed=3 hoặc graph fingerprint đồng nhất. | Dùng native `validate_selected_receipt`, kiểm tra thêm seed/mode/delta/graph/status; đối chiếu selected epoch với VALID history theo tie rule native. Không chọn từ terminal TEST. |
| F07 / HIGH | Toàn notebook; Cell 10:L17–20 | **Không có cell plotting, figure export, CSV table, diagnostics JSON hoặc caption.** TAR chỉ thêm `a_dir`; stdout log ở `WORK_DIR/logs/...` nằm ngoài nên không được gói. YAML/input notebook/command metadata/environment cũng không được chủ động gói. | Thêm hậu xử lý ba figure per-run, source tables/captions; gói log ngoài, YAML + overrides/resolved config, provenance, notebook, export script, receipt/checkpoint. Không chỉ in SHA ra màn hình, lưu sidecar JSON. |
| F08 / MEDIUM | Cell 4:L43–64,75–78 | Tự gọi KaggleHub khi thiếu input có thể thất bại trong noninteractive Save & Run All. Biến `_dl` không được ưu tiên đưa vào search roots. Final check chỉ kiểm tra thư mục tồn tại; bridge bỏ qua file đã có nên có thể giữ stale files. | Attach data trước batch; auto-download mặc định off. Kiểm đủ 5 files, hash đối chiếu reference, resolved source path và writable bridge; mismatch thì fail, không chọn thư mục đầu tiên theo alias tùy tiện. |
| F09 / MEDIUM | Cell 2:L84–100 | Pip fallback FreeRec không pin và `check=False`, PyG không version; import `nvidia-ml-py` dùng tên package thay vì module nên có thể cài lại. Import GPU preflight chỉ print, không chặn CPU hoặc runtime lệch. | Pin matched runtime/dependency lock; fail nếu install/import/runtime không phù hợp. NVML module `pynvml` nếu dùng. Lưu pip freeze + CUDA/driver/GPU + precision; không nâng Torch tùy tiện. |
| F10 / MEDIUM | Cell 2:L2–4; Cell 6:L35; Cell 8:L2–4 | ARM_NAME/arm_name/TARGET_SEED có nhưng runner/call hard-code wo_cse/3; sửa biến trên đầu có thể không đổi seed thật. | Một nguồn ARM/DATASET/SEED; job_id/CLI/config/output dùng cùng nguồn; assert resolved seed sau run. |
| F11 / MEDIUM | Cell 6:L118–129,149; Cell 8:L11 | `dur` bao gồm child startup + preprocessing + fit + selected eval + export, không phải Coach.fit. | Tên `subprocess_wall_seconds`; lấy native fit_seconds từ đúng field/log và train_epoch_seconds_sum riêng. Không dùng dur làm fit time hoặc suy tốc độ tăng do bỏ CSE. |
| F12 / MEDIUM | Cell 0:L8–9 | Claim “an toàn tuyệt đối” dưới 12h không có cơ sở; “Single ... TEST” không đúng native lifecycle hiện có, vốn có terminal TEST và restored-selected TEST. | Ghi timing estimate không bảo đảm; chỉ selected TEST được dùng báo cáo. Không đổi evaluator giữa arm chỉ để khớp câu mô tả. |
| F13 / MEDIUM | Cell 2:L18–20,39–42,54–60 | Tự xóa checkout khi thiếu file; symlinks tồn tại được bỏ qua nên có thể trỏ source cũ; không fail ngay nếu hết các repo đều clone fail. | Checkout riêng attempt, kiểm realpath/import path/commit; raise khi không có đúng source. Không xóa checkout hoặc files của process khác. |
| F14 / MEDIUM | Cell 6:L85 | Tự fallback CPU khi không có GPU; ngân sách ~9h và so sánh efficiency với T4 khi đó không còn đúng. | Fail trước training nếu intended GPU không có; nếu chủ động CPU thì arm vẫn ghi hardware deviation, không gộp timing như T4. |

### Sửa tối thiểu parser/receipt (mẫu cho Hiếu, chưa áp dụng vào notebook)

```python
# Import from the pinned checkout in the reporting process.
from stair_icaart_reporting import validate_selected_receipt

receipt = validate_selected_receipt(json.loads(sel_file.read_text(encoding='utf-8')))
assert receipt['seed'] == SEED
assert receipt['evidence_mode'] == 'pooled_activity_excess'
assert receipt['evidence_delta'] == 1.0
# Also compare receipt graph to manifest final operator fingerprint.
for name in ('RECALL@10', 'RECALL@20', 'NDCG@10', 'NDCG@20'):
    print(f'{name}: {receipt["metrics"][name]:.8f}')
```

Native validator kiểm tra selected/test/selection/finite/range; không thay thế kiểm tra arm/config/runtime/operator. Import từ pinned checkout có source digest đã xác minh. Nếu làm hậu xử lý ở môi trường không có checkout, dùng schema-equivalent validator và lưu script, không reimplement ranking evaluator.

### Không ghi đè attempts (mẫu)

```python
from datetime import datetime, timezone
from uuid import uuid4
attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
run_root = WORK_DIR / 'eclipse_ablation_runs' / f'{ARM}_{DATASET}_seed{SEED}_{attempt_id}'
run_root.mkdir(parents=True, exist_ok=False)
```

## 4. Hình/bảng Hiếu cần bổ sung cho mỗi dataset × arm × seed

Không thể vẽ learning/resource/diagnostic curves thật từ notebook trống outputs. Cần native artifacts; không tạo dữ liệu minh họa thay phép đo.

| Stem | Thiết kế | Dữ liệu / caveat |
|---|---|---|
| `eclipse_electronics_wo_cse_seed3_learning_curves` | Hai panels: TRAIN total/BPR/weighted LC; VALID NDCG@20 và Recall@20, marker epoch selected. | Telemetry + evaluation history + receipt; không TEST curve. Total và raw LC không cùng scale: ưu tiên weighted LC hoặc panel riêng. |
| `eclipse_electronics_wo_cse_seed3_training_memory` | Allocated/reserved MiB; train seconds/epoch, trục/panel riêng. | Child training allocator; không phải tổng VRAM/evaluation/driver. |
| `eclipse_electronics_wo_cse_seed3_calibration_diagnostics` | Positive-score edge coverage/weighted coverage; operator/poly dose; score quantile profile. | Reference **uncalibrated semantic-only eta0** của arm; không lấy weighted+CF full reference và gọi cùng dose. Quantiles không phải histogram. |
| `eclipse_electronics_wo_cse_seed3_selected_test.csv` | Đúng 1 selected TEST row gồm 4 canonical metrics và provenance. | Missing=NOT_MEASURED; không tạo SD/p-value cho một run. |
| `eclipse_electronics_wo_cse_seed3_resource_summary.csv` | preprocessing/cache state, fit, epoch sum, subprocess wall, allocated/reserved scope. | Không trộn timer khác scope. |
| `eclipse_electronics_wo_cse_seed3_comparability.json` | Config diff, support/score hashes, effective eta0, final graph, source/runtime/data checks. | w/o CSE không dùng full final graph fingerprint làm equality target. |

Mỗi hình xuất PDF vector + PNG 300 dpi, source CSV/JSON và caption tiếng Anh. Lưu source/caption là bắt buộc; hình chỉ phục vụ chẩn đoán per-run không bắt buộc chèn hết vào paper. Figure ablation tổng hợp dùng đủ actual seeds của hai arm; seed stability không bịa seed thiếu. Graph diagnostics static qua seeds không có error bars từ ba bản sao giống nhau.

## 5. Checklist w/o CSE riêng

- [ ] Identity thật `(wo_cse,electronics,3)`, fresh 64D run, 500 epochs; không import checkpoint seed khác.
- [ ] Effective eta=0 trong config và graph metadata; PEEC delta=1, LC lambda=.01 vẫn bật.
- [ ] TRAIN/splits/features/candidates và semantic/count/null/score fingerprints tương thích Full cùng protocol; normalized final graph khác là expected, không tự chứng minh lỗi.
- [ ] Không có CF contribution trong final operator; CF metadata/build có thể vẫn tồn tại vì parent/recovery, không suy “không build CF” từ eta0.
- [ ] Reference diagnostics là uncalibrated eta0 operator. Nếu chưa có trực tiếp raw W/operator để xác minh, ghi giới hạn; không suy operator identity chỉ từ tên arm.
- [ ] Telemetry có 500 epochs, finite losses; total=BPR+.01*rawLC trong numerical tolerance phù hợp.
- [ ] Native selected TEST, seed3/status/checkpoint/fingerprints đúng; đủ 4 metrics.
- [ ] Export ba hình + source tables + captions và gói đủ stdout log, notebook/config/command/source diff/environment, checkpoint + receipts.
- [ ] Đọc chéo: Chương kiểm artifacts Hiếu; ghi tên/ngày/phạm vi thực tế, không tự tick independent review.

## 6. Scope và phân công

Trong kế hoạch hiện tại, Electronics w/o CSE chỉ là P2 seed1 tùy ngân sách; **seed3 trong notebook này là extension ngoài phạm vi tối thiểu**, không phải job đã được ledger giao sẵn. Nếu đã chạy thì giữ và báo actual n; không loại vì ngoài kế hoạch hoặc metric xấu. Nếu chưa chạy, ưu tiên Full/w/o EC và core Baby/Sports trước, chỉ bổ sung Electronics khi có budget. Không cần bịa đã có seeds1/2 để ghép với seed3.

Nhận xét này đã được đưa vào [kế hoạch](ECLIPSE_Experiment_Execution_Plan.md), bổ sung gate trước training, danh mục hậu xử lý bắt buộc và quy trình cứu output thiếu hình mà không rerun training. **Không sửa notebook của Hiếu trong lần review này**, tránh thay source của run có thể đang hoạt động.


## 7. Follow-up — notebook đã được sửa

Đã cập nhật [eclipse_wo_cse_electronics_s3.ipynb](../../notebook/P6/eclipse_wo_cse_electronics_s3.ipynb), 27 cells. Backup gốc tại [before_repair.ipynb](../../reports/eclipse_wo_cse_electronics_s3_notebook_audit/eclipse_wo_cse_electronics_s3.before_repair.ipynb).

- Pin source `5212263e87c5c1cfe6e48c91afa90201a48f5929`, checkout riêng; không vá validator LC, không xóa checkout hoặc run cũ. Các source digests được đối chiếu Full reference.
- Cấu hình một nơi: SEED=3, Electronics, eta0/PEEC delta1/LC.01,64D,500 epochs. CLI preflight và post-run manifest/receipt/history guards; final graph không bị ép bằng graph Full.
- Parser dùng bốn native metric keys uppercase, không fallback0; kiểm selected TEST, graph consistency, VALID max và telemetry đầy đủ.
- Unique attempt/export root; incomplete attempts được giữ. Run đã COMPLETED đọc lại để export, không train lại.
- Ba hình bắt buộc và resource profile: PDF vector + PNG300dpi, CSV/JSON nguồn và caption tiếng Anh. Dose reference semantic-only eta0; không suy chi phí CF preprocessing bằng0.
- ZIP gồm cả stdout log ngoài thư mục native artifact, checkpoints/receipts/telemetry, source/config/environment/notebook-template/export code, tables/figures/captions. Có file inventory và sidecar archive bytes/SHA256.
- `RUN_MODE='report_only'` không clone/pip/download/model/CUDA/training. Source/runtime khác lịch sử được gắn review-required; config/data/evidence sai vẫn bị từ chối. Native raw files không bị sửa.

### Cách dùng

**Train mới:** giữ `RUN_MODE='train'`, `SEED=3`, attach đúng processed data trước Kaggle Save & Run All; bật GPU và Internet cho setup. Không bật lại notebook cũ đang chạy bằng bản mới giữa chừng. Strict runtime cần khớp Full; nếu fail, điều tra môi trường thay vì xóa guard.

**Đã train xong, chỉ bổ sung hình:** đặt `RUN_MODE='report_only'`; `EXISTING_ARTIFACT_DIR` là directory có `manifest.json`, `selected_metrics.json`, telemetry và `best_model.pt`; `EXISTING_STDOUT_LOG` là file stdout `.log` tương ứng. Chạy từ đầu notebook: setup training được bỏ qua, outputs nằm ở export root mới. Thiếu log thì fit time là NOT_MEASURED, không bịa0. Nếu lịch sử không đạt input/arm contract, notebook dừng để audit trước; không tự sửa raw manifest.

`NOTEBOOK_SOURCE_PATH` tùy chọn để copy **đúng file ipynb đã submit** vào ZIP. Nếu không cung cấp, gói có source template không tự lồng vô hạn; vẫn tải executed Kaggle notebook riêng khi bàn giao. Human cross-read chưa tự động được xác nhận.

### Kiểm tra đã thực hiện

[Validation receipt](../../reports/eclipse_wo_cse_electronics_s3_notebook_audit/repaired_notebook_validation.json): **57 kiểm tra đạt**, gồm syntax/child-probe dedent, schema negative cases, bảo toàn incomplete attempt, report-only không gọi runner, xuất ba hình/resource profile và archive/sidecar trong thư mục fixture tạm. Local không có IPython nên harness thay riêng display bằng no-op; notebook Kaggle vẫn giữ display bình thường. Fixture metrics/operator/checkpoint chỉ dùng kiểm phần mềm, được xóa; **không tạo kết quả thực nghiệm w/o CSE**. Không model/GPU/network/training, không kiểm chứng native runtime trên Kaggle.


**Follow-up guard launch:** Đã đồng bộ bản sửa preflight-empty-directory từ Count-only: real directory trống/chưa stdout được phép, payload/log/symlink vẫn chặn; không thay eta0/PEEC/LC. Bộ57 fixture checks của notebook này đã chạy lại trên SHA mới, không model/GPU/training. Hồi quy empty-dir→intercepted-launch được kiểm trong bộ111 Count-only cho guard chung. Nếu Kaggle notebook cũ đang hoạt động thì không thay giữa run; chỉ cập nhật trước launch tiếp theo hoặc ở session mới.
