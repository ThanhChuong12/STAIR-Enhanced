# STAIR6-v2 — Implementation and Verification

## Material Passport

- Specification: `docs/giai_doan_6/STAIR6_v2_Report.md`.
- Parent: the existing `STAIR5_v8_Model`, native V8 coach, evaluator, NLGCL and AdamWSEvo.
- Implementation: AKP-WMSG-CSE, frozen text5/visual1, mean auxiliary text support4, static interpolation0.25.
- Verification: numerical/graph tests, native synthetic training and resume, and real Baby preprocessing.
- Scope: implementation verified on CPU; real-dataset ranking improvement and CUDA performance remain unmeasured. No production Baby/Sports/Electronics training was run for this delivery.

## Files and responsibilities

| File | Responsibility |
|---|---|
| `stair6_v2_config.py` | Pure validation, defaults, canonical-parent constraints and bypass detection |
| `models/stair6_v2_can.py` | Exact FP64 simplex solution, KKT checks, content-only budget calibration, fixed/shuffled support controls |
| `models/stair6_v2_graph.py` | Bounded selected-edge cosine, native magnitudes, sum-before-max symmetry, own-degree normalization, static interpolation, cache/provenance and offline diagnostics |
| `models/stair6_v2.py` | V8 inheritance, one prepare call, capture/release CPU inputs, final CSR replacement and strict checkpoint identity |
| `main_stair6_v2.py` | CLI, manifests, preprocess-only mode and inherited native training/resume lifecycle |
| `configs/Amazon2014{Baby,Sports,Electronics}_STAIR6_v2.yaml` | Matching dataset hyperparameters and CAN defaults |
| `tests/test_stair6_v2_can.py` | Independent optimization reference, ties, budgets, invalid configuration and Electronics-scale allocation |
| `tests/test_stair6_v2_graph.py` | Algebra, native recovery, contraction fixtures, cache corruption and control arms |
| `tests/test_stair6_v2_pipeline.py` | Exact V8 steps, autograd, ranking, checkpoint continuation, YAML/CLI and real FreeRec synthetic training |

Existing source files are unchanged. No new loss, smoother, auxiliary optimizer group or package-level eager export is required. Import the model with `from models.stair6_v2 import STAIR6_v2_Model`.

## Exact data flow

1. Validate the raw configuration before the V8 arm resolver can silently change an incompatible arm.
2. Run parent prepare once. Observe its original CPU features and cached candidates through the inherited kNN method; do not rerun search or SVD.
3. Compute the native selected-edge cosine and FP64 signed text costs. Stable row grouping supports arbitrary valid edge order.
4. Solve/calibrate CAN. Infeasible support budgets raise `CANBudgetError` with a NO-GO reason. Equal costs retain full support; no invented sentinel or uniform full-support fallback.
5. Keep original WMSG magnitudes on the selected text edges and all visual edges; sum directed modal occurrences before maximum symmetrization.
6. Normalize the native and auxiliary semantic graphs separately. Load the native raw CF cache written by parent prepare, using the existing shared helper in `stair6_v1_graph.py`; do not repeat co-occurrence discovery.
7. Reconstruct the native S8 and require its exact fingerprint to match the current parent. This catches CF normalization/precision drift before attributing a change to CAN.
8. Form one final CSR. The inherited item smoother reads it with sparse SpMM; forward representations, loss and evaluation stay native.
9. Delete captured raw features and candidate arrays. Masks/probabilities and auxiliary graph views are not retained in model buffers or transferred to GPU.

The read-only dataset overlay loader is reused from `main_stair6_v1.py`; both reused V1 source files are included in model source provenance. This dependency adds no category/latent-category operation to v2.

At interpolation strictly between0 and1, final support remains the native support. Adaptive K is a property of the auxiliary view. Interpolation1 is the hard-pruning ablation. `off` and interpolation0 return the exact parent tensor before any CAN work.

## Running preprocessing and training

Run commands from the repository root. Use a new artifact directory for each attempt and a different directory for preprocessing and training. This prevents overwriting an existing attempt.

```bash
python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml --device cpu --artifact-dir runs/stair6_v2/baby_preflight --graph-cache-dir .cache/stair6_v2 --preprocess-only

python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml --device cuda:0 --artifact-dir runs/stair6_v2/baby_seed1 --graph-cache-dir .cache/stair6_v2
python main_stair6_v2.py --config configs/Amazon2014Sports_STAIR6_v2.yaml --device cuda:0 --artifact-dir runs/stair6_v2/sports_seed1 --graph-cache-dir .cache/stair6_v2
python main_stair6_v2.py --config configs/Amazon2014Electronics_STAIR6_v2.yaml --device cuda:0 --artifact-dir runs/stair6_v2/electronics_seed1 --graph-cache-dir .cache/stair6_v2
```

Dataset directories must contain the matching processed splits and modality PKLs in item-ID order. No category metadata or ASIN mapping is required. Sports/Electronics were not locally available for a full dataset preflight; their CLI/configuration contracts are tested. Full training retains the500-epoch budget, validation selection by NDCG@20, full ranking and native seen-item masking. Electronics uses CL chunk128 in **both** the new V2 run and its new V8 control.

To freeze previously cached native candidate IDs across runtime versions, provide the existing text and visual support files in that order:

```bash
python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml --artifact-dir runs/stair6_v2/baby_frozen --v4-support-files path/to/text_support.pt,path/to/visual_support.pt
```

## Controlled arms

All commands below use the same model/loss/evaluator and can share the content-addressed graph cache. Change the artifact directory for every arm/seed.

| Experiment | Overrides |
|---|---|
| Exact V8 control | `--can-mode off` |
| Exact zero-dose control | `--can-interpolation 0` |
| Primary adaptive view | `--can-mode prune --can-interpolation 0.25` |
| Fixed-K control | `--can-mode fixed --can-fixed-text-k 4 --can-interpolation 0.25` |
| Distribution-matched placebo | `--can-mode shuffled --can-placebo-seed 2718 --can-interpolation 0.25` |
| Hard pruning | `--can-mode prune --can-interpolation 1` |

Use the same dose grid0/0.25/0.5/1 for adaptive, fixed and shuffled controls. Fixed3/4/5 are supported. Shuffling preserves the measured adaptive K distribution within binary TRAIN-degree log2 strata; its graph dose is logged because it is not a perfect causal placebo. No validation/test interactions participate in CAN calibration or degree strata.

Candidate expansion, density-dependent regularization, probability edge weighting, learned gates and category/consensus combinations are intentionally excluded from this primary implementation. They require separate experiments rather than hidden CLI changes.

## Checkpoint and cache contracts

```bash
python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml --device cuda:0 --artifact-dir runs/stair6_v2/baby_seed1 --graph-cache-dir .cache/stair6_v2 --resume-from runs/stair6_v2/baby_seed1/training_checkpoint.pt
```

Resume reconstructs the static graph before loading, then verifies parameters, settings, candidate/cost/mask/operator fingerprints, features, splits, source hashes and runtime identity. Changed dose/mask/source/data is rejected **before** model weights are mutated. Cache-hit flags and elapsed time are telemetry and do not invalidate resume. A V8 checkpoint is not silently accepted as an active V2 checkpoint.

The CAN cache is written atomically, loaded with `weights_only=True`, validated by shape/symmetry/nonnegativity and checksum, and rebuilt with a warning if corrupt. The cache does not authorize a different graph or forced support fallback.

## Verification results

Executed:

```bash
python -m pytest tests/test_stair6_v2_can.py tests/test_stair6_v2_graph.py tests/test_stair6_v2_pipeline.py tests/test_stair5_v8_graph.py tests/test_stair5_v8_pipeline.py -q --disable-warnings --junitxml=reports/stair6_v2/verification.xml
```

**62 passed, 4 warnings**,38.04 seconds on the local CPU test environment. This covers44 new V2 test cases and18 inherited V8 regression cases, including an actual two-epoch FreeRec synthetic training run followed by resume to epoch3. Checkpoint/model/optimizer continuation is tested independently. The63001-item test exercises CAN arrays of shape[63001,5]; it is not an Electronics training or GPU-memory benchmark.

Environment: Python3.14.0, PyTorch2.12.0+cpu, NumPy2.4.6, SciPy1.18.1, FreeRec0.9.7; OMP/MKL threads1 for verification. CUDA hardware was not available for these tests. The warnings originate from native sparse API/library paths; no warning was hidden by an algorithmic fallback.

### Real Baby preprocessing

Receipt: `reports/stair6_v2/baby_can_preflight/manifest.json`. Existing native text5/image1 support files were frozen, so no new nearest-neighbor search occurred. No optimizer or ranking evaluation ran.

| Measurement | Observed value |
|---|---:|
| Items/users | 7050 / 19445 |
| Calibrated lambda | 0.10227920114994049 |
| Mean auxiliary text K | 4.0 |
| K1/2/3/4/5 counts | 352 / 821 / 1047 / 1085 / 3745 |
| KKT residual | 4.44e-16 |
| Native/auxiliary semantic nnz | 59852 / 50944 |
| Final native/V2 nnz | 84138 / 84138 |
| Removed raw semantic mass | 12.6137% |
| Relative final operator delta | 9.5740% |
| Eight-coordinate BSC probe delta | 2.1546% |
| Extra graph-stage elapsed time | approximately0.403 seconds |
| Total preprocessing elapsed time | approximately29.946 seconds |
| Native raw CF cache reused | Yes |

Timing scope matters: total preprocessing includes native MI/SVD and dataset preparation; the extra graph-stage timer does not. The BSC probe uses the first eight native coordinates with local seed2718 and L3. It is not actual training-update utility or ranking performance.

Current native/V2 fingerprints are respectively `1fc05d0db465652a554e197fb437986ee81394bd46019d59b99f2b4a12c54c56` and `dcfe881d5b68e8b927003a8fd00869f8ef3d155613df743c503f1921d206df0b`. They differ from historical runtime fingerprints; the implementation requires exact recovery against its **current native parent**. Do not infer cross-runtime bitwise identity from rounded diagnostic agreement. Rerun matched controls in the target Kaggle environment.

## Resource and scientific limits

- CAN workspaces scale with items×5; production graph construction remains sparse/chunked and CPU-only. No dense catalog-square matrix is created by the new code.
- Only one final item CSR enters training. Primary final support and number of SpMM calls match V8, but actual GPU peak memory and speed require a CUDA run.
- Workspace budgets do not cover the entire parent raw-feature/SVD process or external allocator overhead. An absolute OOM-free guarantee is not supported.
- Norm bounds concern the static graph/polynomial operator, not absolute Adam-direction magnitudes or training convergence.
- No production Recall/NDCG gain has been observed. Follow the report's fixed-K/placebo, common-dose and multi-seed plan before making a performance or publication claim.
