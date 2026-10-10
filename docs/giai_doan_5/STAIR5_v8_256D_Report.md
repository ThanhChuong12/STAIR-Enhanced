# STAIR5-v8-256D — Capacity-Only Implementation Report

**Ngày:** 10/10/2026.

**Experiment ID:** `STAIR5-v8-WMSG-CSE-256D`.
**Trạng thái:** đã triển khai các file mới; kiểm tra cấu hình/analytical memory và syntax đã chạy; kiểm thử PyTorch/FreeRec và training chưa chạy vì máy hiện tại thiếu runtime đó.

## 1. Mục tiêu và phạm vi

Phiên bản này là **V8 WMSG-CSE nguyên kiến trúc, nâng embedding từ 64 lên 256 chiều**. Nó không phải V9-C3V/V8.1-consensus. Câu hỏi thực nghiệm là:

> Với graph, loss, optimizer và protocol V8 giữ nguyên, dimension intervention 64→256 có cải thiện Recall/NDCG không, và với chi phí tài nguyên nào?

Không thêm consensus graph/gate, MFNA, HANS, perturbation, SAP, UCR, RPG, Dirichlet loss hoặc alternate contrastive objective. Không sao chép lịch cosine learning rate/early stopping của CoGate.

Tất cả mã mới được tạo riêng. Không sửa `main.py`, `main_stair5_v8.py`, các model/graph/smoother V8, YAML 64D, `models/__init__.py` hoặc notebook V8 cũ. Đã so SHA-256 trước/sau đối với 10 file V8 tham chiếu: tất cả giữ nguyên. `git diff` tracked files không có thay đổi trong lần triển khai này.

## 2. Source CoGate được tham khảo và những gì không chuyển sang

Đã đọc source trong `CoGate-Consensus-Gated-Graph-Contrastive-Learning-for-Multimodal-Recommendation/`, đặc biệt:

- `models/stair.py`: centered thin SVD, retained left-singular vectors, scale `sqrt(Ni/d)`, user MI từ left-normalized interactions, coordinate FSC schedule.
- `models/cogate.py`: auxiliary graph/gate nằm ngoài đường representation ranking; không được nhập vào phiên bản capacity-only này.
- `main.py`: configuration dimension256 cùng lịch LR warmup/cosine và early stopping; chỉ dimension setup được tham khảo, không chuyển các thay đổi training.
- `configs/Amazon2014Electronics_550_MMRec.yaml`: d256 nhưng batch2048, epochs250, τ=.15 và các CoGate losses. V8-256D **không dùng những override này**; vẫn batch4096, epochs500, τ=.2 như V8 reference.
- `latex.tex:639–654`: dimension-matched tables gợi ý tăng chiều có ích trên Sports/Electronics. Đây là reported evidence, chưa phải kết quả V8-256D hoặc multi-seed verification.

Model V8 đã hỗ trợ variable dimension. Do đó cách triển khai đúng là **reuse toán học V8**, tạo wrapper/config/entry point và checkpoint contract riêng; không viết lại STAIR thành một model gần giống nhưng khác initialization hoặc normalization.

## 3. Các file mới

```text
STAIR-Enhanced/
├── stair5_v8_256d_config.py
├── main_stair5_v8_256d.py
├── models/
│   └── stair5_v8_256d.py
├── configs/
│   ├── Amazon2014Baby_STAIR5_v8_256D.yaml
│   ├── Amazon2014Sports_STAIR5_v8_256D.yaml
│   └── Amazon2014Electronics_STAIR5_v8_256D.yaml
├── tests/
│   ├── test_stair5_v8_256d_contract.py
│   └── test_stair5_v8_256d_pipeline.py
└── docs/giai_doan_5/
    └── STAIR5_v8_256D_Report.md
```

### 3.1 `stair5_v8_256d_config.py`

Module Python standard-library, không import Torch/FreeRec:

- `CAPACITY_CONTRACT`: d256 và các architectural constants của measured WMSG-core.
- `validate_capacity_config`: reject wrong dimension, graph arm, losses/gates, optimizer hoặc selection protocol trước khi construct model. Gamma/lr phải finite-positive; epoch/seed/loss chunk/lr/decay vẫn là runtime options cho smoke và controlled tuning.
- `with_default_config`: mặc định Sports-256D khi caller không chỉ định config; không ghi đè config do user chọn.
- `tensor_payload_ledger`: tính FP32 tensor payload từ số user/item; không đưa ra fabricated peak hoặc guarantee OOM-free.

### 3.2 `models/stair5_v8_256d.py`

Class `STAIR5_v8_256D_Model` kế thừa `STAIR5_v8_Model`. Không override `whitening`, `prepare`, `encode`, `training_objective`, `marked_params`, evaluation/scoring hoặc smoother. Do đó:

- Không có trainable parameters ngoài user/item embeddings.
- Cùng d256 và effective config, numerical behavior phải khớp reference V8(d256).
- Check `beta3` là vector FP32 dài256, đúng coordinate schedule; reject stale vector64 hoặc modified schedule.
- Schedule kiểm tra được tính CPU rồi transfer giống reference parser; tránh yêu cầu bitwise match giữa CPU/CUDA `pow` có rounding khác nhau.
- Extra checkpoint state bổ sung `capacity_experiment={id, contract_version, embedding_dim, consensus:false}` và source hashes của wrapper, launcher, contract module.
- Checkpoint64 hoặc checkpoint khác provenance bị từ chối qua inherited strict pre-copy validation; không pad/copy 64D learned weights để warm-start256 âm thầm.

Import trực tiếp:

```python
from models.stair5_v8_256d import STAIR5_v8_256D_Model
```

Không thêm export vào `models/__init__.py` vì yêu cầu giữ nguyên code cũ.

### 3.3 `main_stair5_v8_256d.py`

- Reuse `main_stair5_v8.build_config`, `load_dataset` và `CoachForSTAIR5_v8`; không monkey-patch original engine globals.
- CLI forwarding tạm thay `sys.argv` trong `try/finally`, restore sau compile. Ghi cả actual/effective arguments trong manifest.
- TF32/AMP policy như reference: TF32 tắt; không thêm AMP.
- Fresh-run artifact directory có manifest/checkpoint/telemetry sẽ bị reject; không reset logs cũ.
- Coach, AdamWSEvo, full ranking, seen-item masking, NDCG computation, validation checkpoint selection và final-selected evaluation giữ nguyên.
- Manifest có `embedding_dim`, effective loss chunks, seed, optimizer options, dataset/source/graph fingerprints, analytic memory ledger, preprocessing memory scope và inherited training telemetry.
- Resume reuse optimizer moments, dataset/loader/global RNG, scheduler/monitor/best states qua original coach.

Không có inner-loop logic mới, `.item()` hoặc catalog-wide dense graph computation mới.

## 4. Mathematical contracts

### 4.1 MI at d256

Với raw modality matrix `Fm ∈ R^(Ni×pm)`:

\[
F_m^c=F_m-\operatorname{mean}_{\mathrm{rows}}(F_m),
\qquad F_m^c=U_m\Sigma_mV_m^\top,
\qquad Z_m=U_m[:,0:256]\sqrt{N_i/256}.
\]

\[
V^0=(5Z_t+Z_v)/6,
\qquad U^0=D_u^{-1}RV^0.
\]

R là reference training interaction operator. Không đổi aggregation/duplicate policy của backbone trong experiment này. Parent validates feature shape/finite values; raw features phải có đủ dimensions/rows. Centered numerical rank và spectrum tails cần được đánh giá trong diagnostics, không được mặc định mọi retained direction đều mang signal hữu ích.

Nếu U retained columns trực chuẩn, `||Zm||F²=Ni`: tăng d phân bổ energy vào nhiều coordinates, không tăng initialization energy bốn lần. Fused MI có cross-terms giữa hai independent SVD bases; basis dependence là limitation giữ lại.

### 4.2 FSC/BSC

\[
b_j=0.1+0.9(j/256)^\gamma,\quad a_j=1-b_j,\quad j=0,\dots,255.
\]

\[
H^{\ell+1}=\mathcal A H^\ell\operatorname{diag}(a),\quad
\bar H_{:,j}=\frac{1-a_j}{1-a_j^4}\sum_{\ell=0}^3 H^\ell_{:,j}.
\]

BSC smooths Adam-preconditioned item directions, không phải riêng raw BPR gradients:

\[
\widetilde D_{:,j}=\frac{1-b_j}{1-b_j^4}
\sum_{\ell=0}^3b_j^\ell S_8^\ell D_{:,j}.
\]

Reuse CSR SpMM và no-grad optimizer recurrence. User optimizer group `smoother=None`, item group original V8 smoother; parameter coverage invariant inherited.

### 4.3 Graph invariance across dimensions

WMSG candidates/cosine lấy từ **raw modality features**, không từ 256D learned/whitened embeddings:

\[
f_2(a)=0.05+0.95[\max(0,a)]^2,\qquad
W=\max(B_t+B_v,(B_t+B_v)^\top),
\]

\[
S_{\mathrm{sem}}=D_W^{-1/2}WD_W^{-1/2},\qquad
S_8=0.9S_{\mathrm{sem}}+0.1\bar S_{CF}.
\]

Identical raw features, candidate cache, train interactions, graph settings/software phải tạo graph fingerprint giống V8-64. Cold cache rebuilds khác device/backend/block signatures có thể đổi FP32 near-tied candidates: explicit frozen supports hoặc same cache/environment được ưu tiên. Không quy đổi p=2 thành inverse-degree exponent α=2.

### 4.4 Objective and prediction

\[
\mathcal L=\mathcal L_{BPR}+0.01\mathcal L_{NLGCL},\quad
\tau=0.2,\ G=1,\ \alpha=0.5,\qquad
s(u,i)=\bar U_u^\top\bar V_i.
\]

Original NLGCL query orientation, diagonal positives/denominators và duplicate treatment giữ nguyên. Decoupled AdamW decay giữ nguyên; không thêm explicit L2 loss hoặc CoGate contrastive terms.

**Interpretation:** đây là dimension intervention bao gồm capacity, MI retained rank và coordinate schedule discretization; không phải mọi effect đều có thể gán riêng cho số parameter.

## 5. Dataset configurations

| Setting | Baby | Sports | Electronics |
|---|---:|---:|---:|
| embedding_dim | 256 | 256 | 256 |
| num_layers | 3 | 3 | 3 |
| gamma | 0.1 | 0.2 | 0.4 |
| lr | 0.001 | 0.001 | 0.001 |
| weight_decay | 0.3 | 0.1 | 0.1 |
| batch_size | 1024 | 1024 | 4096 |
| cl_chunk_size | 1024 | 1024 | 128 |
| epochs / eval_freq | 500 / 5 | 500 / 5 | 500 / 5 |
| λ_NLGCL / τ | .01 / .2 | .01 / .2 | .01 / .2 |
| η / CF top-k / c_min / shrink | .1 / 5 / 2 / 5 | same | same |

All configs: `WMSG-core`, p2, floor.05, semantic_mix1, CPU kNN; alignment/relation/residual/Dirichlet disabled; full ranking; select validation NDCG@20. Electronics chunk128 khớp **effective V8-64 run manifest**, dù YAML64 cũ ghi1024. Đây là memory execution policy, không đổi mathematical loss; original checkpointed chunk implementation được reuse.

Nếu user tuning lr/decay hoặc giảm batch vì memory, phải ghi nhận effective config và rerun same-policy64 controls; không gộp thay đổi đó vào “gain từ dimension” một cách không kiểm soát.

## 6. CPU/GPU performance and memory

### 6.1 Analytical FP32 ledger at d256

| Dataset | Nu / Ni | embedding params | weights+grad+2 Adam moments MiB | one joint activation MiB |
|---|---|---:|---:|---:|
| Baby | 19,445 / 7,050 | 6,782,720 | 103.496 | 25.874 |
| Sports | 35,598 / 18,357 | 13,812,480 | 210.762 | 52.690 |
| Electronics | 192,403 / 63,001 | 65,383,424 | 997.672 | 249.418 |

Formulas: params=`(Nu+Ni)×256`; FP32 table state payload=`16×params` bytes; one joint tensor=`4×params` bytes. Đây không phải runtime peaks. Không gồm saved activations, BSC/Adam direction buffers, sparse graphs, ranking scores, allocator reserve, framework/backend workspace.

Electronics256 **không thể đáp ứng total training VRAM <800 MiB với conventional FP32 Adam state**, vì riêng các table tensors đã khoảng998 MiB. Một item activation `[63001,256]` khoảng61.52 MiB; four retained joint FSC layers khoảng997.67 MiB nếu cùng materialize. Full-ranking score block `[512,63001]` khoảng123.05 MiB.

### 6.2 Kept optimizations

- Graph preprocessing CPU-only, blocked exact kNN và selected-edge cosine budgets inherited; không dense Ni×Ni allocation mới.
- Original CSR graph operators; sparse BSC recurrence no-grad, no dynamic graph snapshots.
- Original activation-checkpointed InfoNCE chunks. Plain chunk loops vẫn có thể giữ B² autograd history; không thay implementation này.
- Không giữ thêm CoGate graph/gate/features vào runtime.
- Graph cache có thể reuse giữa dimensions khi source/features/settings/environment signatures khớp; dimension không phải graph input.

Thin SVD CPU có thể cần nhiều RAM, exact kNN vẫn quadratic compute nếu cold cache. Static review không chứng nhận CPU memory, GPU OOM freedom hoặc speedup. Training telemetry kế thừa không đo evaluation/process GPU peak; cần profile các phase riêng trên T4.

### 6.3 Runtime planning

V8 Electronics64 đã mất khoảng8.78 giờ trên T4. V8-256 có thể vượt một Kaggle session; không hứa runtime bằng bản64. Đo ít nhất5 warm epochs và các lần full evaluation, rồi estimate remaining budget theo throughput thực tế. Nếu phải dừng session, resume từ **completed-epoch checkpoint**; không thay full run bằng một run250epochs như CoGate rồi so trực tiếp bản64-500.

Checkpoint frequency là effective FreeRec option được log trong manifest (`CHECKPOINT_FREQ`); xác nhận tần suất thực tế qua `--help` và artifacts trong môi trường chạy. Không giả định một option chưa kiểm chứng khiến mọi epoch đều đã được persist. Tạo persistent/exported backup trước khi Kaggle session kết thúc.

## 7. Run commands

Chạy từ repository root, trong môi trường Kaggle đã chạy được V8. Không cần dependency mới ngoài reference Torch/FreeRec stack. Các commands sau là hướng dẫn, **chưa được chạy training tại máy hiện tại**.

### 7.1 Runtime tests first

```bash
python -m unittest discover -s tests -p 'test_stair5_v8_256d_*.py' -v
```

Trong environment chuẩn, tất cả16 tests phải được thực thi; “OK (skipped=6)” không đủ để cho phép full training. Nếu runtime đã installed nhưng import lỗi, integration suite fail rõ ràng thay vì che lỗi bằng skip.

### 7.2 Sports smoke — separate disposable attempt

```bash
python -u main_stair5_v8_256d.py \
  --config configs/Amazon2014Sports_STAIR5_v8_256D.yaml \
  --root /kaggle/data --device 0 --epochs 5 --eval-freq 1 \
  --artifact-dir /kaggle/working/stair5_v8_256d_runs/sports_smoke_seed1 \
  --graph-cache-dir /kaggle/working/stair5_v8_shared_graph_cache
```

Smoke khác eval schedule, nên chỉ kiểm tra runtime/shape/finite/memory; không phải fair metric comparison. Full run tạo attempt mới.

### 7.3 Full runs

```bash
python -u main_stair5_v8_256d.py \
  --config configs/Amazon2014Sports_STAIR5_v8_256D.yaml \
  --root /kaggle/data --device 0 --seed 1 \
  --artifact-dir /kaggle/working/stair5_v8_256d_runs/sports_full_seed1 \
  --graph-cache-dir /kaggle/working/stair5_v8_shared_graph_cache

python -u main_stair5_v8_256d.py \
  --config configs/Amazon2014Baby_STAIR5_v8_256D.yaml \
  --root /kaggle/data --device 0 --seed 1 \
  --artifact-dir /kaggle/working/stair5_v8_256d_runs/baby_full_seed1 \
  --graph-cache-dir /kaggle/working/stair5_v8_shared_graph_cache

python -u main_stair5_v8_256d.py \
  --config configs/Amazon2014Electronics_STAIR5_v8_256D.yaml \
  --root /kaggle/data --device 0 --seed 1 \
  --artifact-dir /kaggle/working/stair5_v8_256d_runs/electronics_full_seed1 \
  --graph-cache-dir /kaggle/working/stair5_v8_shared_graph_cache
```

Reference `--v4-support-files TEXT_CACHE.pt,VISUAL_CACHE.pt` vẫn được hỗ trợ khi có authentic cache artifacts; không truyền generated guesses hoặc normalized graph tensor thay candidate payload. Ghi candidate/graph hashes và đối chiếu64 trước khi gán gain cho dimension.

### 7.4 Resume Electronics

```bash
python -u main_stair5_v8_256d.py \
  --config configs/Amazon2014Electronics_STAIR5_v8_256D.yaml \
  --root /kaggle/data --device 0 --seed 1 \
  --artifact-dir /kaggle/working/stair5_v8_256d_runs/electronics_full_seed1 \
  --graph-cache-dir /kaggle/working/stair5_v8_shared_graph_cache \
  --resume-from /kaggle/working/stair5_v8_256d_runs/electronics_full_seed1/training_checkpoint.pt
```

Restored source/data/graph/config/software phải match checkpoint provenance. Không đổi batch/lr/decay/seed/chunk/graph hoặc upgrade source giữa một full run. Khi chuyển session, preserve dataset mappings/caches và checkpoint; strict hash rejection là guard, không phải lỗi cần bypass.

Không dùng `best_model.pt` hoặc `last_model.pt` làm `--resume-from`: chúng là model-only artifacts; continuation cần `training_checkpoint.pt` chứa optimizer/RNG/coach state.

## 8. Verification completed and pending

### 8.1 Completed locally

Đã chạy Python3.12 standard-library unittest:

```text
Ran 25 tests
OK (skipped=6)
```

- **19 tests executed/passed:** 10 capacity-contract tests và 9 notebook/runner tests. Contract coverage: three flat config fixtures và dataset settings; attribute config/no mutation; architectural constants rejection; wrong dimension/additional objective rejection; evaluator/optimizer constraints; invalid scalar guard; explicit/default CLI config routing; analytical Electronics memory arithmetic; invalid shapes.
- Notebook coverage: AST của mọi code cell, không saved outputs; argv đúng launcher256 cho ba datasets; reject model-only resume; selected-test parsing đúng checkpoint; không reuse test từ marker cũ; actual telemetry fields/dedup/corruption; writable FreeRec dataset cache directory; thực thi runner thật với child fixture standard-library cho cả success/error; từ chối ghi đè evidence. Child fixture không phải ML training.
- **6 integration tests skipped:** máy hiện tại không có Torch/FreeRec/NumPy/SciPy runtime đầy đủ. Skip không phải pass.
- **7 source files AST syntax checks passed:** config helper, model wrapper, launcher, notebook utilities và ba test files; code cells của notebook cũng parse được. Đây không xác minh import/backend compatibility.
- **10 old V8 source/config/notebook hashes unchanged.**

Test-only flat-YAML reader không phải production YAML parser; native FreeRec compile thuộc integration test chưa chạy. Không báo cáo training success, GPU speedup, OOM-free hoặc metric improvement.

### 8.2 Integration tests provided for Kaggle/runtime

1. Same256 reference/wrapper loss, gradients và embeddings sau hai optimizer steps, với duplicated IDs.
2. V8-64 và wrapper256 candidate/final graph fingerprints giống nhau; embedding shapes và two parameter groups đúng.
3. Post-step training checkpoint roundtrip, optimizer continuation, reject64 before copying weights.
4. Full/pool scores agreement dùng inherited scorer.
5. Reject stale64/malformed256 BSC schedule.
6. Native CLI compile cả ba256 YAMLs và schedule shape.

Sau đó profile real preprocessing/train/eval peaks và checkpoint save/load trong environment GPU đích. Synthetic CPU tests không thay thế full-dataset memory measurement.

## 9. Fair research protocol and acceptance

- Primary comparisons: V8-64 vs V8-256 cùng source graph/loss/protocol; STAIR64/256 và V4-64/256 là matched-dimension paper controls phù hợp.
- Keep split/feature/ID/candidate/graph hashes, seeds, negative sampling, lr/decay, batch, precision, horizon500, eval5, validation N@20 selection và metric implementation.
- Final results tối thiểu3 seeds, report mean±std và paired differences. Một seed cao hơn không đủ tuyên bố statistically meaningful gain.
- Report R@10/R@20/N@10/N@20 của selected checkpoint; không chọn riêng checkpoint tốt nhất theo test từng metric.
- Không dự báo +5%/+7% trước experiment. Baby có thể ít hưởng lợi hoặc regression; Sports/Electronics có reported dimension evidence mạnh hơn nhưng chưa được V8 xác nhận.
- Với resource regression hoặc metric parity, cân nhắc128D **bằng config/experiment riêng** trong bước sau; entry point này deliberately chỉ256 và reject128 để không nhầm namespace.
- Nếu cần consensus, triển khai namespace V8.1/V9 riêng sau capacity control; không thêm gate vào các file256 này rồi vẫn gọi capacity-only.

**Next step:** chạy integration suite và smoke trên Kaggle, kiểm tra graph fingerprints so64, rồi Sports full matched-seed. Electronics full chỉ bắt đầu sau memory/throughput/resume profiling. Notebook256 riêng đã được bổ sung như mục10; notebook V8 cũ được giữ nguyên.

## 10. Kaggle notebook V8-256D

**File:** `notebook/P5/stair5_v8_256D.ipynb` (22 cells, chưa có execution output). Tham khảo trình tự setup/data/pre-flight/subprocess/plots/export của notebook V8; không sao chép token, mô tả inverse-degree weighting, VRAM ceiling hoặc deletion logic của notebook cũ.

### 10.1 Files added for notebook execution

- `stair5_v8_256d_notebook_utils.py`: standard-library argv builder, writable dataset linking, strict telemetry reader và selected-test log parser. Không import model/Torch trong module này.
- `tests/test_stair5_v8_256d_notebook.py`: kiểm tra notebook/data/runner contracts, including subprocess success/error fixtures. Không kiểm thử GPU thông qua các fixture đó.
- Các model/launcher/config256 ở các mục trước được commit cùng notebook để checkout từ GitHub có đủ dependency nội bộ.

### 10.2 How to run

1. Kaggle bật Internet/GPU, attach processed dataset `rainyle/stair-datasets-mmrec`.
2. Cell settings: `MODE="full"` mặc định500 epochs/eval5; `MODE="smoke"` dùng3/eval1 ở namespace riêng. `SEEDS=(1,)` mặc định; final research study dùng ít nhất3 matched seeds.
3. Checkout riêng `/kaggle/working/STAIR-Enhanced-v8-256D`, fetch `SOURCE_REF` và record resolved SHA. Source thiếu file256 hoặc Git/dependency/CUDA verification lỗi sẽ dừng.
4. Link từng split/feature vào writable dataset directory; không link toàn bộ thư mục vào read-only Kaggle Input. FreeRec có thể tạo cache mới mà không sửa input. Trên Windows không có symlink privilege, helper hỗ trợ file hardlink cùng volume; Kaggle dùng symlink.
5. Copy ba config256 vào session, giữ batch/lr/decay/graph/loss, ghi rõ `CHECKPOINT_FREQ=5` (smoke1). Compile bằng native parser để xác minh option này. Pre-flight chạy suite25tests; bất kỳ fail/skip nào trong Kaggle đều chặn training.
6. Chạy độc lập Sports, Baby hoặc Electronics. `ACTIVE_DATASETS` chỉ điều khiển helper `run_all()`; không làm individual dataset cells bị “Not selected”. Chọn một trong individual cells hoặc `run_all`, không chạy cả hai.
7. Inspect curves và memory sau từng run; export session archive. Không tự giảm batch/horizon hoặc tự chọn chunk lớn theo GPU. Electronics vẫn batch4096/chunk128.

### 10.3 Artifacts and reporting

Session root: `/kaggle/working/stair5_v8_256d_runs/<UTC timestamp>_<suffix>/`.

Mỗi attempt lưu `training.log`, `command.json`, inherited `manifest.json` hoặc `resume_manifest.json`, `training_telemetry.jsonl`, `training_checkpoint.pt`, model-only best/last, `runner_status.json`, `selected_test_metrics.json`, plots và `nvml_memory.jsonl` nếu NVML khả dụng. Source/software/pre-flight/effective YAMLs được lưu tại session root.

- Loss plots đọc `loss`, `bpr_loss`, `weighted_nlgcl_loss`; raw `nlgcl_loss` vẫn có trong JSONL.
- Coach peaks đọc `peak_allocated_gib`/`peak_reserved_gib`, đổi MiB bằng×1024. Scope là training epoch.
- NVML sample scope là whole process lifetime; device-wide memory có thể gồm kernel/process khác. Child-process memory chỉ xuất hiện nếu API hỗ trợ. Không dùng kernel PyTorch allocation làm fallback giả cho subprocess.
- Selected-test parser chỉ đọc TEST sau `Load best model @Epoch`, đúng selected epoch; không dùng test epoch cuối hay best riêng từng test metric. CSV chứa rounded log values và descriptive delta vs historical64 seed1, không phải statistical result.
- Khi nhiều seeds complete, notebook in mean±sample std cho256. Historical64 seed1 không thay thế matched-seed comparator hoặc paired testing.
- Runner interrupt/error giữ artifacts và terminate/wait child; rerun same attempt bị từ chối, không xóa checkpoint. Tạo session mới hoặc explicit resume.

### 10.4 Resume across Kaggle sessions

Export cell tạo một `tar.gz` compression level1, không nhân bản artifact rồi tạo cả zip lẫn tar. Archive chứa session/checkpoints và native FreeRec logs lấy từ manifest; graph cache nằm ngoài archive để tránh duplication, có thể rebuild từ source/data/support không đổi.

Sau khi extract backup vào thư mục riêng trong session mới: pin `SOURCE_REF` về SHA cũ, giữ `MODE="full"`, cùng batch/chunk/seed/lr/decay/eval/horizon; đặt `RESUME_FROM[("electronics",1)]` đến `training_checkpoint.pt`. Notebook tạo attempt mới, copy prior log/telemetry nếu có và append resumed records; inherited Coach trims telemetry về completed checkpoint epoch. Checkpoint provenance guards reject architecture/data/source mismatch. Chưa xác minh GPU continuation của toàn dataset tại máy local.

Không đổi smoke3 sang full500 bằng resume. Không gọi một run incomplete là full benchmark. Chưa có phép đo speed/VRAM/metric256 nào trong deliverable này; mọi outcome phải lấy từ runtime Kaggle thực tế.
