# STAIR-MHD v3: implementation and verification

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21
- Verification Status: VERIFIED for the synthetic CPU checks below; recommendation improvements UNVERIFIED
- Version Label: mhd_v3_implementation_v1

## Implementation

The implementation follows `v3.md` and the explicit implementation constraints in the user request. Baseline `main.py` and existing source modules are unchanged.

| New file | Responsibility |
|---|---|
| `models/stair_mhd_v3.py` | GenRecArch model, baseline MI/FSC/scorer, gates, shared projector, unique-positive InfoNCE, budget loss, diagnostics and parameter partitioning |
| `models/stair_mhd_v3_utils.py` | Blocked exact KNN, binary incidence, candidate-only train interaction statistics, differentiable matrix-free hypergraph operator |
| `optimizers/mhd_smoother.py` | Parameter-free Neumann callback using a detached pre-step snapshot |
| `main_stair_mhd_v3.py` | Baseline AdamWSEvo updates, isolated groups, snapshot cleanup, checkpoint/RNG management, inherited FreeRec evaluation |
| `configs/dataset_mhd_v3.yaml` | Baby baseline config inheritance and explicit v3 pilot settings |
| `tests/test_stair_mhd_v3.py` | Algebra, autograd, baseline recovery, lifecycle, checkpoint and real CLI integration tests |

Static preparation retains no active autograd graph. Gate weights and weighted degrees are rebuilt each step. The smoother receives cloned detached values before any parameter update; a `finally` block clears its snapshot after the step, including failed steps. Optimizer checkpoints exclude bound smoother callbacks and reconnect live callbacks when loaded.

The three groups use baseline lr/decay for users and items, and 0.1 times baseline lr with zero decay for auxiliary parameters. Only the item group has a smoother. Parameter coverage and uniqueness are enforced.

Evaluation, seen masking, best-checkpoint selection and metric calculation remain inherited from FreeRec. Full/pool model scoring is checked against the actual baseline class. FreeRec supplies one-based training epochs; warmup therefore ends at epoch 10 and the first ramp increment occurs at epoch 11.

For deterministic epoch-boundary training resume, the new coach seeds each epoch and canonicalizes FreeRec Launcher's index list before it is shuffled. This is necessary because Launcher shuffles that list in place. Global RNGs are seeded after loader initialization, covering the Python random stream used for negatives. This controls sampling order without changing the sampling distribution. Baseline recovery tests use identical batches; an entire training trajectory is not claimed identical to the original entry point's RNG schedule.

## Executed verification

Working directory: `D:\4thY_HCMUS\KLTN\STAIR-Enhanced`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv-mhd-v3\Scripts\python.exe -B -m pytest tests/test_stair_mhd_v3.py -q -p no:cacheprovider --basetemp=.venv-mhd-v3/pytest-run7
```

Result: **16 passed, 2 warnings, 31.42 seconds, exit code 0**.

- Matrix-free operator agrees with dense reference; symmetry, PSD, spectral bounds and weight-scale invariance pass on fixtures.
- Double-precision gradcheck includes the dependence of weighted degrees on gate weights.
- KNN agrees with FreeRec; duplicate incidence entries and duplicate training interactions are handled.
- Contrastive loss deduplicates positive item IDs and handles singleton batches.
- Across four training steps, output gate gradients are nonzero and initially zero hidden-layer gradients become nonzero.
- With auxiliary objectives and mixing disabled, baseline initialization, BPR loss, embedding gradients, optimizer update, full scores and pool scores match exactly on the same synthetic batch.
- Checkpoint roundtrip preserves model/optimizer state, RNG draws, scores and the next update; inconsistent data hashes are rejected.
- A subprocess runs actual FreeRec Dataset/DataLoader2/Coach training and evaluation for two epochs, then resumes from the real Coach checkpoint. All epoch-2 scalar diagnostics, including losses and gate gradients, match uninterrupted training exactly (timing excluded).

The CLI fixture has four users and 24 items. Tests use real Torch and FreeRec, with no mock replacement for their model, optimizer, graph or evaluation APIs.

## Environment and usage

Validated on Windows with Python 3.12.14, Torch 2.5.1+cpu, FreeRec 0.9.7, torchdata 0.8.0, torch-geometric 2.6.1, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1 and PyYAML 6.0.3. The local environment is `.venv-mhd-v3` and is ignored by Git.

The baseline declares FreeRec 1.0.1, while the available PyPI runtime used here is 0.9.7. That declaration is retained and actual versions are logged. Compatibility with a different FreeRec build must be validated separately. The two test warnings concern deprecated torchdata DataPipes and beta sparse CSR support.

```powershell
.\.venv-mhd-v3\Scripts\python.exe -B main_stair_mhd_v3.py --config configs/dataset_mhd_v3.yaml
```

The dataset must already be prepared under `<root>/Processed/<dataset>` with the baseline modality files. The supplied config inherits Baby's hyperparameters. For Sports or Electronics, create a separate v3 config inheriting that dataset's baseline YAML; changing only `--dataset` retains Baby's settings.

## Limits and research follow-up

No real-dataset benchmark or GPU training was executed. These checks establish implementation behavior on synthetic inputs, not improved NDCG, convergence or generalization. Exact resume is verified only for CPU, `num_workers=0`, soft gates and epoch boundaries. GPU sparse nondeterminism, multiworker continuation, pool-evaluation RNG continuation, full-catalog memory and throughput require separate measurement.

The mathematical properties of each normalized operator do not prove that the adaptive optimizer will improve ranking. Gates receive auxiliary-loss gradients, not a differentiable path through optimizer updates from BPR; the test suite explicitly checks this distinction. Run baseline recovery, fixed-gate and component ablations on real data with matched budgets and multiple seeds before making thesis claims about effectiveness.
