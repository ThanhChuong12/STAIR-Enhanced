# STAIR5-v2 / C-HET implementation and verification

Updated: 2026-10-02. Status: implemented and CPU integration-tested; Amazon ranking gains and Kaggle GPU throughput are not yet measured.

## Files and responsibilities

| File | Responsibility |
|---|---|
| `models/stair5_v2.py` | V1 backbone inheritance; exact MI/FSC/BPR/scorer arithmetic; static BSC graph replacement; audited optional LHC; isolated parameter groups |
| `models/stair5_v2_graph.py` | Binary train interactions, candidate common-user counts, Ochiai × support shrinkage, raw-edge boost, normalization/blend, degree-stratified placebo, fingerprinted atomic cache |
| `models/stair5_v2_geometry.py` | Bounded FP32 Lorentz geodesic squared distance and Euclidean/constant-radius controls |
| `models/stair5_v2_objectives.py` | Uniform multi-positive CE, U0→I1/I0→U1, fixed shared temperature, checkpointed query blocks |
| `models/stair5_v2_utils.py` | Atomic training checkpoints, callback-free optimizer serialization, RNG restoration |
| `optimizers/stair5_v2_smoother.py` | Matrix-free Neumann polynomial on Adam-normalized updates; current model-buffer callback |
| `main_stair5_v2.py` | FreeRec coach, CLI, native ranking/validation checkpoint selection, durable checkpoint/manifest/telemetry export |
| `configs/Amazon2014{Baby,Sports,Electronics}_STAIR5_v2.yaml` | Baseline dataset-specific values with explicit edge-only ET defaults |
| `notebook/P5/stair5_v2.ipynb` | Kaggle dependencies, discovery, preflight, all three datasets, isolated arms/seeds, logs, plots and archives |
| `tests/test_stair5_v2_{graph,objectives,pipeline}.py` | Algebra, source-level parity, full optimizer steps, autograd, caches and checkpoint continuity |

All baseline and v1 sources remain unchanged. V1-L historical training stays in its existing launcher; it is not silently renamed H0-A.

## Arm contracts

| CLI arm | Calibrated BSC | Auxiliary loss |
|---|---|---|
| `B0` | Off | Off |
| `ET` | On | Off; recommended primary |
| `H0-A` | Off | Audited geodesic LHC |
| `E0-A` | Off | Matched Euclidean CE |
| `HC-A` | Off | Constant-radius geometry control |
| `ET-H0` | On | Audited geodesic LHC |
| `ET-placebo` | On, permuted scores within degree strata | Off |

The primary has no learned gate, projector, radial temperature or extra optimizer group. `--lambda-lhc` is ignored for B0/ET/ET-placebo. `--edge-mix 0` or `--edge-strength 0` returns the original S0 object; combined arms can still have an active auxiliary loss and therefore are not baseline controls unless that loss is also disabled.

Raw W0 is captured after baseline modality coalesce(sum)/to_undirected(max), before symmetric normalization. Behavioral counts use only binary-deduplicated train pairs on this candidate support. No dense item-item behavioral matrix is formed. q = n/(n+t) × n/√(di·dj); W+ = W0(1+a q); Sη = (1−η)S0 + η normalize(W+). The user-item FSC adjacency and Euclidean ranking scorer are unchanged.

LHC primary omits the legacy pair-specific self-return correction. A separate exact-weight subtraction utility returns a context-validity mask for future ablation; it is not activated by an undocumented flag. The legacy fallback/degree-minus-one rule is not used. Head scale buffers are initialized at MI and stored in state dicts. Geodesic controls preserve the reference denominator 2R²τ; no hybrid distance is added.

## Large-dataset execution

**Exact kNN in row blocks.** `--knn-chunk-size 1024` bounds the similarity working matrix to chunk × item_count instead of item_count². `--knn-device auto` uses GPU when normalized features fit the memory reserve, otherwise CPU. `cuda` explicitly requested will fail clearly if feature storage does not fit. Feature/k/precision/backend/version/chunk fingerprints cache candidate edges, avoiding repeated all-catalog searches across arms.

This remains exact all-catalog top-k, so arithmetic cost is still quadratic in item count. GPU and CPU GEMM or different block sizes can reorder near ties. Use the same backend/cache for paired arms; the CPU default in the standalone launcher supports comparison with the existing CPU preprocessing. The Kaggle notebook explicitly selects `auto` and records actual backend/chunk/cache settings. Approximate nearest neighbors and TF32 are not silently enabled. MI SVD retains the baseline CPU implementation.

**Sparse online BSC.** The blended graph is materialized once as CSR. Each BSC hop is one sparse matrix × dense update; no S² or dense item-item matrix is constructed. The callback resolves the model's current buffer, including after `.to(cuda)` and checkpoint loads.

**Bounded auxiliary intermediates.** `--lhc-chunk-size 256` computes rows against all candidates with a complete softmax denominator. Non-reentrant activation checkpointing recomputes each block during backward rather than keeping both whole distance matrices. Loss and gradients match the unblocked objective within FP32 tolerance in tests. This reduces memory at the cost of extra backward computation; it does not guarantee faster LHC than the uncheckpointed version. ET disables LHC completely and is the lowest-cost primary.

Positive masks remain O(Bu·Bi) booleans; membership lookup is blocked. For batch 4096, this mask is about 16 MiB when IDs are unique. Graph preprocessing holds sparse edges and sorted train user lists. Intersection cost depends on degrees; high-degree items can still make it slow. Cache hits remove this repeated work.

Each epoch records seconds, samples/s and CUDA peak allocated GiB in `training_telemetry.jsonl`. Compare B0/ET/ET-H0 on the same hardware, settings and cache state; report preprocessing separately from online epochs. The local test environment is CPU-only, so no T4 speedup or peak-VRAM value is asserted here.

## Launching

```powershell
python main_stair5_v2.py --config configs/Amazon2014Sports_STAIR5_v2.yaml --v2-arm ET --knn-device auto --graph-cache-dir runs/stair5_v2/graph_cache --artifact-dir runs/stair5_v2/sports_ET_seed1
```

Use the Baby/Electronics YAML for the corresponding dataset; full training defaults to 500 epochs, with baseline batch/decay/gamma retained. Pilot budgets require an explicit `--epochs 200` override. Prepare tab-separated train/valid/test files and modality pickles in `<root>/Processed/<dataset>`; notebook discovery prepares this bridge without resetting repository state.

FreeRec `--eval-valid` is a store-false flag: do not pass it. Validation remains enabled by default. Test evaluation during tuning is rejected. Ranking and seen-item masking are inherited from FreeRec; native final-test and selected-test records are both retained, and only selected-checkpoint test results should be reported.

The notebook exposes KNN_DEVICE, KNN_CHUNK_SIZE, LHC_CHUNK_SIZE and NUM_WORKERS. Start with workers=0 for reproducible epoch-boundary checks; increase only if telemetry shows input starvation. All three dataset cells can be used. Running all selected datasets sequentially can exceed a Kaggle session; archive/download artifacts after each run.

## Checkpoint and resume

Artifacts include `manifest.json`, `training_telemetry.jsonl`, `best_model.pt`, `last_model.pt` and `training_checkpoint.pt`. The training checkpoint contains model buffers/graph/scale, optimizer moments, schedule epoch, RNG state, FreeRec monitors, best metadata and the prior best model. Smoother callbacks are excluded from serialization and reattached from the newly constructed model when loading. Loads validate data, graph and semantic/optimizer configuration fingerprints.

```powershell
python main_stair5_v2.py --config configs/Amazon2014Sports_STAIR5_v2.yaml --v2-arm ET --resume-from runs/stair5_v2/sports_ET_seed1/training_checkpoint.pt --artifact-dir runs/stair5_v2/sports_ET_resumed
```

Resume at completed-epoch boundaries; keep data, seed, arm, preprocessing and objective settings the same. Increasing the epoch budget is allowed. Exact next optimizer-step continuity was verified on a fixed synthetic batch. Multi-worker datapipe process/prefetch states are not serialized; do not claim bitwise replay of a multi-worker sampler after interruption. FreeRec's validation selection arithmetic is preserved.

## Verification receipt

- 22 tests discovered: 21 passed, 1 CUDA test skipped because no GPU is available locally.
- Source-level parity checks execute the unchanged `main.py` STAIR class: MI embeddings, FSC outputs, BPR, full/pool rank and three real AdamWSEvo steps match on the controlled CPU fixture.
- Graph checks cover duplicate interactions, isolated vertices, symmetry/support, constant-boost normalization, spectral bounds, placebo and cache invalidation.
- Objective checks include both-branch multi-step gradients, origin/cap finiteness, empty/no-positive contexts and checkpointed-block loss/gradient equivalence.
- Checkpoint tests use the actual callback-safe save/load helpers, restore frozen scales and compare the next optimizer update.
- Real FreeRec synthetic-data CLI smoke runs completed ET and ET-H0 train/validation/test; ET-H0 resume from completed epoch 2 ran epoch 3 successfully. These are engineering checks, not Amazon performance results.
- Notebook code cells compile; CLI help and config parsing succeed. Kaggle execution and large-dataset GPU performance remain unverified.

Local receipts are under `scratch/g5v2_review/`, including `implementation_tests.log` and smoke logs. TorchData/CSR/deprecation warnings from the existing runtime are informational; they were not suppressed to imply a warning-free environment.
