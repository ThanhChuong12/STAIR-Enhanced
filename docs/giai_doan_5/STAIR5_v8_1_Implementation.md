# STAIR5-v8.1 BSCV — Implementation and Training Guide

Implementation review: 2026-10-15. Architecture specification: [STAIR5_v8_1_Report.md](STAIR5_v8_1_Report.md).

## Scope and architecture

V8.1 subclasses `STAIR5_v8_Model`. Existing V8 source, MI, FSC, item BSC smoother, BPR, full/pool ranking, seen-item masking and validation-NDCG@20 checkpoint selection remain unchanged. The added route affects only the item-query branch of NLGCL during training. There is no new inference route, optimizer smoother, negative sampler or explicit regularization loss.

The primary query is

\[
M=CV^0,\quad
\kappa_i=\operatorname{stopgrad}\min\left(1,\frac{\|I_i^1\|}{\|M_i\|+10^{-8}}\right),\quad
J_i=I_i^1+\rho_e g_i\kappa_iM_i.
\]

The scalar gate uses detached cosine, clipped log norm ratio and graph coverage as its three inputs. Its 3→8→1 SiLU network has 41 trainable parameters. Gate initialization preserves the caller's RNG. The final layer has small nonzero initial weights so that the first layer can receive gradients immediately after activation. Messages and FSC inputs retain autograd; the context and cap do not.

`C` is normalized independently from the BSC graph, uses existing raw CSE support only and has **zero isolated rows**. CPU construction uses selected-edge cosine chunks, sparse normalization and dimension-independent caches. Raw features are observed during the inherited preprocessing call and released afterwards; they are not loaded a second time or retained on GPU.

## Files

| File | Responsibility |
|---|---|
| `models/stair5_v8_1.py` | V8 subclass, static consensus buffers, fresh live messages, item-query override, parameter groups, diagnostics and checkpoint provenance |
| `models/stair5_v8_1_consensus.py` | Scalar gate, detached norm cap/context, residual bound and one-based dose schedule |
| `models/stair5_v8_1_graph.py` | Existing raw CF cache reuse, selected-pair semantic weights, normalization, support validation and corrupt-cache recovery |
| `models/stair5_v8_1_objectives.py` | Override only I1→U0 queries, preserving live keys, duplicate batch IDs and the original denominator/chunking |
| `stair5_v8_1_config.py` | Dependency-light validation and analytical tensor-payload estimates |
| `main_stair5_v8_1.py` | Own CLI parser and V8-derived coach; preprocessing, backend probe, manifests, training, checkpoint/resume and epoch logging |
| `configs/Amazon2014{Baby,Sports,Electronics}_STAIR5_v8_1.yaml` | Three primary 64D configurations |
| `configs/Amazon2014{Baby,Sports,Electronics}_STAIR5_v8_1_256D.yaml` | Three matched 256D configurations |
| `tests/test_stair5_v8_1_{consensus,graph,pipeline}.py` | Algebraic contracts, graph invariants and real FreeRec entrypoint integration |

The runner reuses the V8 coach and the existing read-only-mount-aware dataset loader from `main_stair6_v1.py`. That loader does not activate the Stage 6 model or category algorithms. Its source hash is included in checkpoint provenance. Imports use the explicit model module; modifying `models/__init__.py` is unnecessary.

## Corrections from the initial core

1. Corrupt weights-only cache reads can raise `IndexError` on the installed PyTorch version. The consensus cache now treats this as corruption and rebuilds from validated train inputs.
2. The model validates that `beta3` exactly matches the native FP32 schedule for the requested dimension. A stale 64D schedule cannot be reused for 256D.
3. Active models automatically create a graph cache under the artifact directory when no cache path is supplied. The parent and consensus builders can then reuse the same raw CF cache instead of building it twice.
4. `off` and permanent zero-dose controls bypass auxiliary feature capture and graph construction. Warmup bypasses message/gate computation while retaining the static graph for subsequent active epochs.
5. `auto` layout now verifies sparse-dense backward on the selected device before optimizer creation/resume. Unsupported CSR kernels trigger an explicit, recorded COO fallback. OOM and unrelated failures propagate. Explicit `csr` does not silently fall back.
6. Epoch diagnostics pool means by sampled unique-row counts, reconstruct gate standard deviation from pooled first/second moments, and retain the global sampled residual maximum. Averaging batch maxima or standard deviations is avoided.
7. The checkpoint contract includes consensus settings, graph fingerprint, resolved sparse layout and source hashes. Epoch state is restored with the gate and optimizer. Changed-source/configuration checkpoints are rejected before model weights are copied.

## Optimizer and schedule

| Group | LR | Weight decay | Smoother |
|---|---:|---:|---|
| User embeddings | 0.001 | Dataset baseline | None |
| Item embeddings | 0.001 | Dataset baseline | Native V8 BSC |
| Learned gate | 0.0001 | 0 | None |

Every trainable parameter occurs in exactly one group. The fixed gate has no auxiliary parameters/group. The engine calls `zero_grad(set_to_none=True)`; zero-dose gate gradients stay absent and Adam does not advance their state.

The epoch convention is one-based: epochs 1–30 have `rho=0`; epoch 31 has `rho=0.2/30`; epoch 60 reaches `rho=0.2`. The original NLGCL coefficient remains 0.01 throughout. A smoke test shorter than 31 epochs does **not** exercise the consensus branch unless warmup is explicitly disabled for that smoke test.

## Training commands

Run these commands from the repository root in the configured FreeRec environment. Select the correct dataset root/device. Each independent attempt needs a fresh artifact directory; resume uses its original directory.

Preprocessing only, without training or optimizer construction:

```bash
python main_stair5_v8_1.py --config configs/Amazon2014Baby_STAIR5_v8_1.yaml --device cpu --preprocess-only --artifact-dir runs/v81/baby/preprocess
```

Short active-branch smoke test; its zero warmup is intentionally different from the primary experiment:

```bash
python main_stair5_v8_1.py --config configs/Amazon2014Baby_STAIR5_v8_1.yaml --device cuda:0 --epochs 3 --consensus-warmup 0 --consensus-ramp 0 --artifact-dir runs/v81/baby/smoke
```

Primary 64D run with the specified 30+30 schedule:

```bash
python main_stair5_v8_1.py --config configs/Amazon2014Baby_STAIR5_v8_1.yaml --device cuda:0 --artifact-dir runs/v81/baby/64d/seed1
```

Matched 256D run:

```bash
python main_stair5_v8_1.py --config configs/Amazon2014Baby_STAIR5_v8_1_256D.yaml --device cuda:0 --artifact-dir runs/v81/baby/256d/seed1
```

Resume the same configuration, source and dataset after a completed epoch:

```bash
python main_stair5_v8_1.py --config configs/Amazon2014Baby_STAIR5_v8_1.yaml --device cuda:0 --artifact-dir runs/v81/baby/64d/seed1 --resume-from runs/v81/baby/64d/seed1/training_checkpoint.pt
```

Replace `Baby` with `Sports` or `Electronics` to use the other configurations. Control flags are `--consensus-mode off`, `--consensus-gate fixed`, `--consensus-mode cf-only` and `--consensus-mode placebo`. Use separate attempt directories for each. A V8 checkpoint is not a full-resume V8.1 checkpoint.

## Fair comparisons and performance

Run the four cells of consensus off/on × 64D/256D with matched seeds, batch sizes, chunking, epoch budgets and evaluation. Compare consensus against its **same-dimension control**. Do not attribute a 64→256D gain to consensus. Three seeds minimum are needed before claiming a reliable incremental improvement.

Electronics retains batch size 4096 and uses checkpointed NLGCL anchor chunks of 128 in all six-config experiments, including controls. Baby/Sports use batch 1024 and chunks 1024. Evaluation remains full ranking, with validation NDCG@20 selection. Use the native final **selected** test record for comparisons, rather than mixing it with the final-epoch test record.

Each active step adds one full sparse forward message and its backward path. Unique positive IDs save repeated gate work; inverse expansion preserves all sampled duplicate keys and queries. No catalog-sized dense similarity matrix or edge×embedding feature gather is added. No full message is cached between steps.

Sparse preprocessing does not guarantee that training/evaluation cannot OOM. At 256D, the native embedding/Adam tables, message activations, chunked loss and full-ranking workspace still require device memory. Manifest payload estimates are analytical tensor sizes, not measured total VRAM. CUDA throughput and peak memory must be measured on the target Kaggle GPU.

## Artifacts and diagnostics

`manifest.json` records the initial attempt; `resume_manifest.json` records a resume invocation. Both include source/config/data contracts, effective graph cache, graph coverage and weights audit, dimension, resolved sparse layout, runtime, precision policy and preprocessing time.

`training_telemetry.jsonl` is inherited from V8 and records epoch losses, throughput and training allocated/reserved peaks. `consensus_telemetry.jsonl` adds dose, sampled gate moments, cap activation, residual magnitude, angular shift, orthogonal-message fraction and zero-message fraction. Its gradient norm refers explicitly to the **last batch**, not a mean across the epoch. Warmup reports no gate gradient.

Sampling occurs every 25 active batches by default, beginning with the first batch. Counts refer to unique items within each sampled batch; an item can contribute again in another sampled batch. Detached device scalars are transferred at epoch boundaries. Resume removes telemetry beyond the saved epoch to avoid duplicate records after rollback.

`training_checkpoint.pt` retains model/gate, optimizer including the auxiliary group, completed epoch, scheduler, RNG/sampler/loader states, selection monitors and the saved best model. `best_model.pt` and `last_model.pt` remain separate. Strict provenance means editing implementation files between save and resume intentionally invalidates a checkpoint.

## Verification scope

The automated suite covers exact two-step V8 recovery during warmup/off/zero-dose, independent group membership, live gate gradients over three active steps, duplicate-ID expansion, unchanged full/pool scoring, checkpoint continuation and rejection, CSR/COO backward parity, recorded fallback, pooled diagnostics and active 256D backward. A subprocess integration test invokes the real CLI for preprocessing, two active training epochs and a third resumed epoch on synthetic data.

These checks establish implementation contracts. They do not establish Recall/NDCG gains on Amazon datasets, large-dataset throughput, CUDA determinism or measured CUDA memory bounds. Those remain experimental work.

### Observed validation results — 2026-10-15

Environment: Python 3.14.0, PyTorch 2.12.0+cpu, FreeRec 0.9.7; OMP/MKL thread counts 1. CUDA was unavailable for this validation.

```bash
python -m pytest tests/test_stair5_v8_1_consensus.py tests/test_stair5_v8_1_graph.py tests/test_stair5_v8_1_pipeline.py tests/test_stair5_v8_graph.py tests/test_stair5_v8_pipeline.py -q --disable-warnings --maxfail=1
```

Observed: **58 passed, 10 subtests passed, 4 warnings, 62.00 seconds**. This comprises 25 existing core tests, 15 pipeline cases including parametrizations, and 18 V8 regression tests. The original core initially had one failing corrupt-cache recovery test; the final result includes its fix. The subprocess test verifies preprocessing, two active training epochs and resume to epoch 3 with nonzero gate gradients. Both explicit sparse layouts pass CPU backward comparison; the unsupported-CSR fallback is tested by an injected kernel error, not by claiming that the current CPU backend lacks CSR.

The real Amazon Baby **64D preprocessing-only** invocation completed in 43.219 seconds inside the runner (dataset loading plus inherited/consensus preparation; excludes interpreter startup). Its consensus portion took 0.885 seconds, with the raw CF cache reused and a fresh consensus cache built. This is a single CPU preflight measurement, not a GPU throughput benchmark.

| Quantity | Observed value |
|---|---:|
| Users / items | 19,445 / 7,050 |
| Raw CF nnz / consensus nnz | 23,054 / 23,054 |
| Covered item rows | 4,602 |
| Consensus operator fingerprint | `4f9ec3748b925e2a781bc294dc5690456626727daef83c6c3def76b7d804acf4` |
| Analytical CSR payload | 333,056 bytes |
| Analytical one-message 64D payload | 1,804,800 bytes |
| Training checkpoint generated | No |

The local preflight manifest is `scratch/v81_baby_preflight_20261015/manifest.json`. A compact validation receipt is saved at `reports/stair5_v8_1/implementation_validation_20261015.json`. Scratch outputs/caches are local artifacts, not additional source modules. No production Amazon training, CUDA benchmark or new recommendation metric was generated during this implementation review.
