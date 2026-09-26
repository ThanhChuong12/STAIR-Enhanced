# STAIR-RAM v5 implementation

## Material Passport

- Origin: academic-research-suite / experiment-agent.
- Scope: implement the architecture in STAIR4_v5_Report.md, with synthetic and integration verification.
- Date: 2026-09-26.
- Verification status: software and synthetic integration checks passed (37 tests); no full benchmark experiment performed.
- Baseline policy: main.py, existing model/optimizer modules and baseline YAML files are unchanged.
- Scientific status: no claim of a measured accuracy gain or attainment of the >6% target.

## Implemented components

| File | Responsibility |
|---|---|
| [stair4_v5.py](../../models/stair4_v5.py) | FreeRec model adapter, frozen final FSC vectors, exact disabled-residual scorer, fresh candidate projections and full/pool ranking |
| [stair4_v5_features.py](../../models/stair4_v5_features.py) | Separate PCA, missing-feature handling, authenticated caches, binary train histories and row-specific LOO |
| [stair4_v5_heads.py](../../models/stair4_v5_heads.py) | Bounded residual, nonzero amplitude initialization, user-only gates, normalized projections and optimizer groups |
| [stair4_v5_sampling.py](../../models/stair4_v5_sampling.py) | Uniform distinct train-unseen candidates, finite-catalog masks and private resumable RNG |
| [stair4_v5_objectives.py](../../models/stair4_v5_objectives.py) | Masked sampled candidate CE; K=1/T=1 BPR identity |
| [stair4_v5_utils.py](../../models/stair4_v5_utils.py) | Teacher/data/source identities, train-only score calibration and fixed ID-only control |
| [main_stair4_v5.py](../../main_stair4_v5.py) | Stage A/B orchestration, validation selection, selected teacher export, atomic checkpoints and epoch-boundary resume |
| [analyze_stair4_v5.py](../../scripts/analyze_stair4_v5.py) | Selected-test-only paired summaries, missing-run audit and descriptive uncertainty |
| [stair4_v5.ipynb](../../notebook/P4/stair4_v5.ipynb) | Kaggle dependencies, dataset staging, preflight, teacher reuse, controls, plots and paired reporting |

There are three inherited dataset configurations under configs/dataset_stair4_v5_*.yaml. Stage A retains each dataset's baseline batch size, weight decay, gamma and 500-epoch budget. Stage B defaults to 100 epochs, K=32, PCA width 128, residual width 32 per modality, c_eta=.5 and an effective batch equal to the corresponding baseline.

Stage A reuses the audited v4 **B1/alpha=0** path, including blocked exact kNN, the baseline MI/FSC arithmetic and AdamWSEvo/BSC. It does not enable CSGC. FreeRec still handles seen-item masking, full/pool evaluation and Recall/NDCG. The v5 coach preserves baseline training and validation order while exporting structured artifacts; plotting is not part of the training completion path.

## Running

From the repository root, after installing the dependencies appropriate to the host CUDA/PyTorch environment:

~~~shell
python main_stair4_v5.py --config configs/dataset_stair4_v5_baby.yaml --stage all --device 0 --run-dir runs/baby_seed1 --no-final-test
~~~

This is a validation-only run with 500 Stage A epochs and up to 100 Stage B epochs. A shorter head pilot is explicit:

~~~shell
python main_stair4_v5.py --config configs/dataset_stair4_v5_sports.yaml --stage all --head-epochs 50 --device 0 --run-dir runs/sports_pilot_seed1 --no-final-test
~~~

Changing head-epochs does **not** shorten Stage A. For confirmation after configuration is locked, omit no-final-test and use a new run directory. The test split is evaluated only at the validation-selected checkpoint, including Stage B epoch zero if it wins.

Reuse an exported teacher for another head/control:

~~~shell
python main_stair4_v5.py --config configs/dataset_stair4_v5_baby.yaml --stage b --teacher-dir runs/baby_seed1/stage_a --run-dir runs/baby_id_control_seed1 --arm id-ss --seed 1 --device 0 --no-final-test
~~~

Supported arms:

- mm-ss: primary multimodal residual with K sampled negatives.
- mm-bpr: identical residual with one negative and temperature 1.
- id-ss: the same residual capacity using fixed lifts of frozen item ID embeddings instead of content features.
- baseline-ss: trainable STAIR continuation from selected model and Adam moments, using sampled CE.
- baseline-bpr: the corresponding one-negative restart control.

The continuation controls use the v5 explicit train-pair order/sampler. They preserve the STAIR optimizer and score but are not claims of a bitwise continuation of the original FreeRec data stream. A continuous 600-epoch baseline can be produced with Stage A and epochs=600; that is a separate budget control, not automatically pooled with B0=500 by the strict paired analyzer.

Global and fixed modality gates are available through gate-mode. Semi-hard mining, feature-permutation experiments and all optional report ablations are not silently enabled by the primary implementation.

## Artifacts and provenance

Each run contains stage_a and/or stage_b directories with:

- run_manifest.json: dataset, seed, baseline/source/evaluator identities, data hashes, selected epoch, timing and completion state.
- resolved_config.json: effective configuration for that stage.
- evaluation.jsonl: validation records and, when enabled, one selected-checkpoint test.
- epochs.jsonl: training loss, duration and memory; Stage B also records gradient norms, amplitude/gate diagnostics and candidate-margin statistics.
- best.pt: validation-selected model/head state.
- model.pt: last model/head state before selected evaluation.
- checkpoints: epoch-boundary state for interruption recovery.

Stage A additionally exports teacher.pt containing **selected final FSC vectors**, the selected baseline model, selected optimizer state and provenance. Stage B authenticates this artifact against its checksum, dataset, seed, baseline configuration and source/evaluator identity. Arbitrary legacy model.pt files are not accepted as teachers without the required provenance.

Head checkpoints deliberately omit duplicate static teacher/feature tensors. Reconstruct them from the verified teacher and PCA cache. Do not edit, replace or delete the teacher while its head runs depend on it.

PCA caches authenticate the fitted tensors and preprocessing options. Fingerprints establish that the same input files and internal IDs were used; they cannot prove that an externally prepared feature row was semantically assigned to the correct item. Dataset preprocessing must already follow the baseline mapping.

## Resume

For a failed/interrupted Stage B run, repeat the original command with the same directory and add resume:

~~~shell
python main_stair4_v5.py --config configs/dataset_stair4_v5_baby.yaml --stage b --teacher-dir runs/baby_seed1/stage_a --run-dir runs/baby_id_control_seed1 --arm id-ss --seed 1 --device 0 --no-final-test --resume
~~~

Resume is supported at epoch boundaries with num_workers=0. Model, optimizer, best checkpoint, selection counters, private pair-order/negative RNG and evaluation sampler states are restored. Changing the objective, teacher, source, optimizer settings, device or epoch budget is rejected. Stage A also stores the selected optimizer snapshot inside its resume checkpoint.

Completed runs cannot be resumed into the same directory. A run that already disclosed test metrics cannot resume optimization. Use a distinct registered experiment for a new budget or configuration. For stage=all interruption, resume the affected individual stage explicitly rather than restarting both.

## Kaggle notebook

The notebook defaults to pilot mode: all three datasets are selectable, Stage A remains 500 epochs, Stage B uses 50 and test is not disclosed. GPU training itself has not been executed in this implementation session.

Set DATASET_SOURCES when multiple input folders match. The notebook stages the five processed files and mapping files into a writable FreeRec layout; it does not split/reindex data. Existing Git checkouts are preserved; upload or push the new files yourself before running a notebook that clones the repository.

An explicit call to train_dataset('sports') or train_dataset('electronics') runs that dataset even if the Run All selection contains only Baby. Teacher training runs once per dataset/seed and is reused across head arms. Subprocess errors preserve combined stdout/stderr and stop the arm; a plotting failure does not convert a successful training process into a failed training artifact.

Install requirements-stair4-v5.txt with a constraint for the existing Torch wheel as shown in the notebook. The inherited TorchData deprecation warning can still appear; no warning is treated as evidence that an import or training command succeeded.

## Verification and interpretation

The test suite covers:

- Actual baseline disabled-branch full/pool score equality.
- Bound/separability algebra and K1/T1 value/gradient equivalence.
- Nonzero multi-step gradients, frozen teacher tensors, disjoint parameter groups and duplicate candidates.
- Constant/missing-feature PCA, canonical signs, deterministic cache validation and no global RNG consumption.
- Binary histories, singleton/empty users, per-row LOO, excluded train positives and finite candidate masks.
- Real CLI Stage A+B, standalone Stage B, ID and objective-only controls.
- An intentional Stage B interruption followed by resume in a new interpreter; compare final head, optimizer and negative RNG to an uninterrupted run.
- An intentional Stage A interruption followed by resume in a new interpreter; compare selected teacher vectors and selected epoch to an uninterrupted run.
- The existing production v4 Gate-0 multi-step optimizer/ranking regression, verifying the Stage A baseline path.
- Selected-checkpoint-only analysis, strict per-dataset growth criteria and notebook cell syntax/manual dataset selection.

Verification completed on 2026-09-26: **37 passed, 12 warnings in 73.37 seconds** on the local CPU environment. This includes 36 v5 tests and the existing production baseline regression. The warnings concern inherited dependency deprecations and sparse tensor support; they are not benchmark results.

Reproduce the verification from the repository root:

~~~shell
python -m pytest tests/test_stair4_v5.py tests/test_stair4_v5_features.py tests/test_stair4_v5_heads.py tests/test_stair4_v5_analysis.py tests/test_stair4_v4.py::test_production_gate0_multistep_and_ranking -q -p no:cacheprovider --basetemp=scratch/v5_tests_final
~~~

These are software and synthetic-data checks. They do not estimate Baby/Sports/Electronics accuracy, CUDA determinism, real-data throughput or VRAM peaks. The report's >6% objective remains an empirical hypothesis.
