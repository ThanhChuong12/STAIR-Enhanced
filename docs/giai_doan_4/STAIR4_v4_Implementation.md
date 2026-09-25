# STAIR4-CSGC v4 — implementation and execution guide

The implementation follows the static CSGC core in `STAIR4_v4_Report.md`. It does not add contrastive learning, a learned gate, extra projectors or graph refresh. Baseline `main.py`, existing models and optimizers are unchanged.

## Files

| File | Responsibility |
|---|---|
| `models/stair4_v4_utils.py` | Binary train CSR, compiled candidate intersections, confidence statistics, tied CDF, strata, hashes, atomic writes and RNG/sampler states |
| `models/stair4_v4_graph.py` | Options validation, blocked exact cosine top-k, baseline raw graph, cache, normalization, CSGC operator and graph audit |
| `models/stair4_v4.py` | FreeRec model, original MI/FSC/BPR/scoring, two isolated parameter groups and checkpoint identity validation |
| `optimizers/stair4_v4_smoother.py` | Detached static Neumann BSC applied to Adam directions; explicit arm/clear lifecycle |
| `main_stair4_v4.py` | YAML inheritance, CLI, baseline FreeRec evaluation, checkpoint/resume, epoch telemetry and selected-checkpoint metrics |
| `configs/dataset_stair4_v4_*.yaml` | Inherit the respective baseline; add only calibration/runtime fields |
| `tests/test_stair4_v4.py` | Algebraic oracle, sparse invariants, actual baseline comparison, checkpoint and real CLI smoke tests |
| `notebook/P4/stair4_v4.ipynb` | Fresh-process Kaggle preflight, writable data bridge, graph audit, matched benchmark, full/pilot training, resume and measured-result plots |
| `tests/test_stair4_v4_notebook.py` | Clean-namespace plotting, epoch budgets, selected metrics, matched splits, runner failures, completed-run reuse, resume and report exports |
| `scripts/audit_stair4_v4_graph.py` | Run production preparation without optimization |
| `scripts/summarize_stair4_runs.py` | Read metrics at the checkpoint selected by validation NDCG@20 |

The literature/code review is recorded separately in `STAIR4_v4_Source_Audit.md`.

## Core contracts

1. Train interactions are deduplicated only for behavioral statistics. Baseline MI and forward adjacency continue to use FreeRec's original train graph contract.
2. Candidate pairs are canonical existing off-diagonal semantic edges. Numba compiles exact sorted-history intersections; production never materializes a full co-occurrence matrix.
3. Both directed copies receive the same bounded multiplier. Diagonal weights remain unchanged. The calibrated graph is normalized before mixing with S0; the mixture is not normalized again.
4. `alpha=0` returns the exact supplied FreeRec S0 object without statistics, ranking or shuffling. Graph preparation preserves global RNG. Blocked top-k has the same rule as the baseline but may differ on numerical ties across backends; compare recorded neighbor hashes.
5. The FSC final multiplier uses the baseline's floating-point expression `1 - (1 - b)`, not an algebraically equivalent rewrite. This matters for bitwise recovery.
6. Only user and item embeddings are trainable. Both use the original AdamWSEvo lr/decay; the item group alone has BSC. One sparse matrix multiplication per BSC hop; no additional forward propagation.
7. Large static graph buffers are reconstructed and identity-checked on load, rather than serialized in each best model. Source/data/config mismatches fail before model parameters are overwritten.

## Installation and dataset layout

Use a supported Python 3.12 environment with a suitable CPU or CUDA PyTorch build. On Kaggle keep its installed CUDA build:

```bash
python -m pip install -r requirements-stair4-v4.txt
python -m pytest tests/test_stair4_v4.py -q
```

Expected layout:

```text
data/Processed/Amazon2014Baby_550_MMRec/
    train.txt
    valid.txt
    test.txt
    textual_modality.pkl
    visual_modality.pkl
```

FreeRec uses tab-separated `USER`, `ITEM`, and optionally `TIMESTAMP` columns. Preserve original feature-row/ID alignment. The notebook creates individual links into a writable dataset directory, allowing FreeRec to write its schema cache without modifying Kaggle input mounts.

## Launch sequence

```bash
# Graph signal audit (still includes baseline MI preparation).
python scripts/audit_stair4_v4_graph.py --config configs/dataset_stair4_v4_baby.yaml --device cpu --id baby_audit --run-dir logs/v4_baby_audit

# Exact baseline path through the new entry point.
python main_stair4_v4.py --config configs/dataset_stair4_v4_baby.yaml --ablation-id V4-B1 --alpha 0 --epochs 50 --id baby_b1_s1 --run-dir logs/v4_baby_b1_s1

# Matched core pilot. Use the same device/seed/data/evaluation schedule.
python main_stair4_v4.py --config configs/dataset_stair4_v4_baby.yaml --ablation-id V4-C --alpha 0.25 --epochs 50 --id baby_c_s1 --run-dir logs/v4_baby_c_s1

python scripts/summarize_stair4_runs.py logs/v4_baby_b1_s1 logs/v4_baby_c_s1 --output logs/v4_pilot_selected.csv
```

FreeRec owns full/pool ranking, seen-item masking, NDCG computation, validation frequency and best-model selection. The v4 coach only records the resulting meter values. Its JSONL explicitly distinguishes final-epoch evaluation from the re-evaluation of the validation-selected model. Do not substitute per-metric maxima from FreeRec's summary for selected-checkpoint results.

## Ablations

| ID | Overrides relative to core |
|---|---|
| V4-B0 | Original `main.py`; same data, runtime and sampler protocol for comparison |
| V4-B1 | `--ablation-id V4-B1 --alpha 0` |
| V4-C | Defaults, alpha .25 |
| V4-NS | `--ablation-id V4-NS`; reliability 1 when overlap is positive, 0 otherwise |
| V4-ND | `--ablation-id V4-ND`; global tied CDF |
| V4-NA | `--ablation-id V4-NA --activity-weighting uniform` |
| V4-SH | `--ablation-id V4-SH --shuffle-seed 1`; shuffle h within strata with local RNG |
| V4-SM | `--ablation-id V4-SM --alpha 0 --identity-mix 0.1`; mix the BSC **output** with the original direction |

The SH control deliberately breaks the correspondence between evidence and weights; unlike the core, an originally zero-evidence edge can receive nonzero shuffled h. It tests the information content of calibration, not the core's zero-evidence guarantee.

## Checkpoint and cache behavior

- Checkpoints default to the run-specific FreeRec log directory's `checkpoints/`, avoiding collisions between ablations. The exact path is in the run manifest.
- Resume with `--resume --checkpoint-dir <recorded path>` and identical source, data, optimizer/sampling/evaluation settings. Epoch count may be extended. Use `num_workers=0` for the verified resume contract.
- FreeRec checkpoints at the **beginning** of the configured epoch interval. Work since that checkpoint is repeated. Mid-batch recovery and multi-worker exact recovery are not supported.
- Preserve model, Adam moments, best-selection metadata, best model, monitor histories, Python/NumPy/Torch/CUDA RNG, DataLoader2 seed generator and FreeRec's in-place shuffled source order.
- Reconstruct serialized Field objects in the new process to avoid process-dependent cached Python hashes. Restoring only the RNG, without the source permutation, does not reproduce FreeRec's next epoch.
- Raw neighbors/W0 and calibrated CSR values use content-addressed caches. Identity includes source version, feature and neighbor hashes, train CSR and shapes, explicit mapping files when present, architecture settings and backend. Cache tensors have checksums; stale identities do not match.
- MI whitening is recomputed. It is not replaced with approximate/randomized SVD. Preprocessing can therefore remain expensive even on cache hits; total runtime is recorded.

## Validation status and limits

Executed locally on Python 3.12 / PyTorch 2.14 CPU / FreeRec 0.9.7:

Model/engine test suite: **17 passed, 1 skipped** (CUDA unavailable). Notebook cells are syntax-checked and its reporting helpers are executed in a fresh namespace by tests/test_stair4_v4_notebook.py. The full updated notebook was not executed on Kaggle locally.

- Dense candidate-statistics oracle, duplicate interactions, zero overlap, tied ranks, isolated nodes and empty support.
- Randomized sparse symmetry/support/spectral bounds and cache identity/corruption checks.
- Actual unchanged STAIR class extracted from local `main.py`: four optimization steps give bitwise-equal embeddings, Adam moments and full/pool scores on a 24-item dataset at alpha zero.
- Real FreeRec CLI training, evaluation and two-process epoch-boundary resume on a toy dataset. The resumed epoch's BPR equals the uninterrupted epoch.

### Local Baby graph audit

The production `--audit-only` path was also executed on the local Baby split (19,445 users; 7,050 items; 118,551 training interactions). With the registered defaults:

| Diagnostic | Observed value |
|---|---:|
| Undirected candidate pairs | 29,926 |
| Directed sparse nonzeros | 59,852 |
| Candidate pairs with positive overlap | 10.4524% |
| Maximum effective support | 20.1636 |
| Maximum reliability | 0.741339 |
| Raw multiplier range | [1.0, 1.366291] |
| Relative Frobenius change of normalized operator | 0.395357% |
| Relative action change on a fixed eight-column random probe | 0.400191% |

The core only **increased** evidenced raw weights in this audit; it did not downweight raw edges. With most scores zero, supported candidates rank high in their strata. Degree renormalization still changes neighboring normalized weights. This is a consequence of the registered formula, not a coding failure; do not describe this run as observed bidirectional denoising. The perturbation is small, so a large ranking gain cannot be inferred. The probe is a diagnostic, not the spectral operator norm. Full-dataset training was not launched.

The compact graph diagnostics and source/data identity are preserved in `artifacts/stair4_v4_baby_graph_audit.json`.

These checks establish implementation contracts on tested inputs. They do not establish GPU bitwise parity, Kaggle throughput, full-dataset accuracy or an improvement over STAIR. The notebook is a runnable pipeline whose full Kaggle training remains to be executed. TorchData deprecation and PyTorch CSR notices remain visible; they are not silently suppressed.

Use the registered Baby pilot, paired seeds and validation-only tuning before Sports/Electronics. A null or negative result remains a valid outcome.

## Kaggle notebook correction — 2026-09-25

Local verification: **11 notebook tests passed**, including actual PNG rendering and CSV/LaTeX export from fixture JSONL. The model/engine suite was rerun: **17 passed, 1 skipped** (CUDA unavailable), including real toy CLI training/evaluation/resume. Fixture metrics are test data, not research results. No 500-epoch Kaggle run was performed during this correction.

The supplied Kaggle log shows both Baby V4-B1 and V4-C completing 50 training epochs, loading the validation-selected model at epoch 50 and writing their artifacts. The subsequent NameError occurred in notebook plotting, after training had finished. The local fixes do not change the model, optimizer, baseline evaluation or checkpoint selection.

### Findings and corrections

- The learning-curve function called np.argmax without importing NumPy. Plotting now marks the epoch recorded in the completed run manifest, using the same validation-NDCG selection for both Recall and NDCG. Plot helpers have local imports and are tested without hidden kernel variables.
- FreeRec JSONL uses RECALL@20, whereas the old per-dataset plot requested Recall@20 and substituted zero when missing. Metric spelling is now normalized, and missing values are omitted rather than plotted as zero.
- Hard-coded historical baseline values were overlaid on validation curves without establishing split/protocol compatibility. These overlays were removed; measured B1/C curves provide the control comparison.
- Training invoked plotting automatically and then the following cells repeated the plots. Training now returns after saving its runs. The dedicated plot cells can regenerate figures without launching training.
- Plotting reads one explicit stage, with stage-specific output filenames. It no longer falls back from full to benchmark silently. Resumed duplicate epochs use the last record.
- CUDA telemetry is labelled cumulative PyTorch peak allocated memory sampled at epoch boundaries, not instantaneous or whole-device VRAM. The counter resets after preparation, so this is not a measurement of preparation's peak.
- Report deltas also require matching whole-dataset fingerprints, including held-out files. Seed summaries separate epoch budgets, preventing a resumed 500-epoch pilot from being averaged with a short pilot.
- Resume setup keeps the original experiment's artifact root for reports/export. Repeated resume uses the latest requested epoch budget, rather than reverting to the original 50-epoch command.

### Epoch budget and use

The previous notebook explicitly passed --epochs 50 because RUN_MODE was pilot and PILOT_EPOCHS was 50. This overrode the inherited YAML value of 500; it was not early stopping or a model failure.

The corrected notebook defaults to RUN_MODE='full': each selected B1/C run receives the inherited **500 epochs**. Optional pilot mode now defaults to **100 epochs**, matching Stage B of the architecture report. The resolved budget is printed before training. Baby/seed 1 remains the default dataset/seed; a single seed is exploratory, not the report's multi-seed confirmation protocol.

To repair figures for the existing Kaggle session, retain the original artifact root, execute the updated train-helper definition cell, then run:

~~~python
plot_single_dataset_learning_curves('baby', stage='pilot')
plot_vram_profile('baby', stage='pilot')
~~~

Do not call train_dataset merely to regenerate figures. For a fresh full comparison, set RUN_MODE='full' in the control panel and execute the setup/training cells; /full/ output remains separate from the old /pilot/ output. Existing repository checkouts are preserved, so explicitly upload/sync the corrected notebook and required code rather than assuming clone refreshes an existing checkout.

Rounded selected-test values in the supplied log were B1 Recall@20=0.0960 / NDCG@20=0.0418 and C Recall@20=0.0961 / NDCG@20=0.0419. These 50-epoch, single-seed observations do not establish improvement at convergence or statistical significance. Read exact values from evaluation.jsonl; do not derive a precise gain from the rounded console output.
