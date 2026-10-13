# STAIR6-v1 implementation and validation

Date: 2026-10-13. Canonical specification: [STAIR6_v1_Report.md](STAIR6_v1_Report.md).

Primary implementation: **LCAE-WMSG-CSE**, static latent cluster association
enrichment of the native V8 item smoother. The primary now works from native
feature rows without external category labels or ASIN mapping. Verified Amazon
metadata remains a separate `category_source=metadata` experiment.

## Material Passport and review boundary

- Material: revised architecture, isolated STAIR6 implementation, methodology review.
- Evidence: CPU numerical/integration tests, synthetic actual FreeRec training/resume,
  real Baby feature clustering and complete graph-only preprocessing.
- Not measured: LCAE Amazon ranking gains, target GPU memory/time, real Sports/Electronics clustering.
- Reviewer provenance: separately invoked methodology reviewer plus primary-agent
  synthesis, same model family, NOT_CALIBRATED. No independent human certification.
- Baseline V8/V7 implementation files remain unchanged.

## Source layout

```text
main_stair6_v1.py
stair6_v1_config.py
models/
    stair6_v1.py
    stair6_v1_categories.py
    stair6_v1_latent_categories.py
    stair6_v1_graph.py
configs/
    Amazon2014Baby_STAIR6_v1.yaml
    Amazon2014Sports_STAIR6_v1.yaml
    Amazon2014Electronics_STAIR6_v1.yaml
tools/
    audit_stair6_v1_latent.py
    audit_stair6_v1_metadata.py
tests/
    test_stair6_v1_latent_categories.py
    test_stair6_v1_categories.py
    test_stair6_v1_graph.py
    test_stair6_v1_pipeline.py
requirements-stair6-v1.txt
```

The model inherits V8 encode/evaluation, training_objective, fit, marked_params,
sampling and scorers. It observes native CPU feature/kNN inputs during prepare,
generates latent memberships or validates metadata, enriches raw semantic
weights, then replaces mAdj with one final CSR operator. Transient raw features,
centers/memberships and baseline graph are not retained on GPU.

No additional optimizer, loss or smoother module is needed. The native V8
AdamWSEvo groups and BSC recurrence are reused. The launcher reuses the native
coach/evaluator and creates a writable FreeRec Processed bridge for a directly
located dataset. It also provides --preprocess-only without optimizer steps.

## Source selection and exact recovery

| Setting | Meaning |
|---|---|
| category_source=latent | Generate unlabeled groups from existing text/image rows |
| category_source=metadata | Require verified mapping, taxonomy and manifest |
| category_mode=off | Exact native V8 path; no grouping/metadata work |
| category_strength=0 | Exact native V8 path, irrespective of source |

There is no auto source detection or silent metadata-to-latent fallback. Unknown
source/invalid parameters fail. The default three YAMLs select latent at64D.
Electronics preserves batch4096 and effective historical CL chunk128. Native
500epoch horizon, validation NDCG@20 selection, full-ranking masks, MI/FSC/BPR/
NLGCL settings and dataset LR/decay remain unchanged.

## Implemented grouping/statistics contracts

1. Read CPU FP32 contiguous feature rows in native processed item order.
2. Normalize each raw row safely, use separate seeded Gaussian projections to
   r64 per modality, normalize projected views and concatenate sqrt(5/6) text
   and sqrt(1/6) image blocks. No coordinate-wise modal alignment is assumed.
3. Spherical k-means++ initialization and up to20 full blocked spherical Lloyd
   passes, C50, local seed2718. No Euclidean MiniBatchKMeans approximation is
   called spherical. No training RNG mutation.
4. Hard sparse one-hot H. Double-invalid content rows have no assignment;
   single-view rows use the valid block. Nonfinite inputs/collapsed partitions
   fail. All group IDs are anonymous and item_asins remains empty.
5. Binary user-group OR uses unique TRAIN pairs only. Dense positive GMM/softmax
   membership is intentionally excluded: it would make every group present for
   every active user and destroy lift.
6. Requested off-diagonal pairs only, with minimum count2 and support attenuation
   n/(n+5) times tanh(positive log-lift). This is not a calibrated Bayesian model.
   Integer comparison handles exact independence before evaluating logs.
7. h_aa0: within-group semantic edges are neutral. Existing edge scores enrich
   raw W by1+delta*h, delta0.5; support is unchanged. Reverse reuse ensures exact
   symmetry. Recompute current degrees and retain original CF normalization and
   empty-CF behavior before eta0.1 blending.
8. All-zero active evidence raises InactiveCategorySignal. A small raw change
   can disappear in normalization, so operator and BSC probe doses are logged.

## Cache, checkpoints and reproducibility

Latent cache includes exact raw-array hashes as well as ordered feature-file
hashes, dataset/splits, algorithm/source version, rank/C/iterations/seed, NumPy/
Torch versions, recorded thread setting and effective block. It stores CPU
assignments/centers, content hashes and deterministic diagnostics. Corrupt caches
warn and recompute. Group centers are offline artifacts, not model parameters.

Checkpoint identity includes source revision, source kind, group provenance,
configuration and graph/data fingerprints. Runtime seconds/cache-hit flags are
not used as deterministic model identity. Changing source/rank/C/seed/data or
code invalidates an old run. Do not resume a metadata-only checkpoint under the
new latent model or silently cross this source revision.

Within-environment repeatability is tested. Different BLAS/hardware/near-tie
rounding can change hard assignments; record fingerprints and use a fixed
clustering realization across training seeds. Full-catalog content fitting is
transductive, matching V8 feature availability; held-out interaction values are
excluded from all fitted statistics. Hashing valid/test files binds the protocol,
not their use in model fitting.

## Commands

Install the runtime consistent with V8; preserve Kaggle's CUDA-enabled Torch.
requirements-stair6-v1.txt records the dependencies used for CPU verification.

Feature-only grouping audit (does not build the ranking model or graph):

```bash
python tools/audit_stair6_v1_latent.py \
  --dataset-dir data/Amazon2014Baby_550_MMRec \
  --cache-dir .cache/stair6_v1 \
  --output reports/stair6_v1/baby_latent_clustering_audit.json
```

Complete graph/dose preflight, with native MI and graph construction:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --device cpu --preprocess-only \
  --artifact-dir runs/stair6_v1/baby_graph_preflight
```

The preflight writes a manifest but runs no training/evaluation. Use a separate
fresh artifact directory for training, so a preflight cannot masquerade as or
overwrite a training attempt.

Matched V8 control:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --category-mode off \
  --artifact-dir runs/stair6_v1/baby_control_seed1
```

Latent pilot, then decide whether to run the matched full budget:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --category-source latent --epochs 100 \
  --artifact-dir runs/stair6_v1/baby_latent_pilot_seed1
```

Use Sports/Electronics YAMLs similarly after making the native feature/split
bundles available. Clustering requires no outside category files, but missing
base datasets/features still fail. A positive short-pilot trend is not final
TEST superiority. Do not select C/r/delta or clustering seeds on TEST.

Resume only the same source/config/data attempt:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --artifact-dir runs/stair6_v1/baby_latent_seed1 \
  --resume-from runs/stair6_v1/baby_latent_seed1/training_checkpoint.pt
```

For verified metadata CAE, explicitly pass --category-source metadata and the
three original metadata paths. The schema and provenance boundaries are in
STAIR6_v1_Metadata_Design.md and the loader. The current Baby metadata preflight
still returns NO-GO; this does not block the separately declared latent source.
No helper invents ASIN mappings, text category names or verified attestations.

## Real Baby preprocessing receipts

Executed on the existing7050item/19445user Baby bundle,118551 TRAIN interactions,
CPU, with C50/r64/T20/seed2718. No optimizer training or ranking evaluation was
performed in this real-data preflight.

| Observation | Executed value |
|---|---:|
| Occupied groups | 50/50 |
| Informative feature coverage | 7050/7050 in this bundle |
| Largest group fraction | 4.0709% |
| Normalized group entropy | 0.98060 |
| Standalone uncached grouping stage | 0.8713s |
| Full native model/graph preprocessing | 41.41s |
| Within-group semantic weight mass | 61.3497% |
| Cross-group semantic weight mass | 38.6503% |
| Active positive-association semantic mass | 37.9745% |
| Requested/positive group pairs | 1094/1034 |
| Relative normalized-operator Frobenius delta vsV8 | 5.2710% |
| Relative BSC deterministic probe delta vsV8 | 1.3396% |
| Semantic/final nonzeros | 59852/84138, native support retained |

The0.8713s grouping-stage clock excludes caller PKL loading/file hashing and the
parent native kNN/SVD/CF work. The full preflight reused that grouping cache and
completed in41.41s. Neither observation supports a universal two-second GPU
claim. Measured coverage100% is specific to this Baby bundle, not a theoretical
guarantee for other features. A20pass run need not be at the global clustering
optimum; assignment/objective history and margins are recorded.

Receipts:

- reports/stair6_v1/baby_latent_clustering_audit.json
- reports/stair6_v1/baby_latent_graph_preflight/manifest.json

These show usable clustering and non-negligible intervention on Baby, not
ranking improvement, semantic-label accuracy, statistical significance or
measured GPU/process memory. They support progression to a matched pilot.

## Performance and validation scope

CPU blocks bound projection and item-to-centroid logits; no item²/category²
matrix, full cdist or dense HhH-transpose is used. Fused features are128D and
membership has at most one nonzero per item. Native raw-CF cache is shared,
avoiding duplicate catalog-wide co-occurrence construction. Native CL chunking
and no-grad BSC recurrence are inherited.

Workspace budgets constrain additional grouping/query/probe allocations, not
total RSS, raw feature arrays, baseline full SVD/kNN/CF work or libraries. Final
CSR replaces S8, with no extra train-time GPU branch. Actual Electronics/T4
memory and wall time still need measurement; no universal OOM guarantee.

Executed test environment: Python3.12.10, Torch2.10.0+cpu, FreeRec0.9.7,
NumPy2.5.4, SciPy1.18.1, PyYAML6.0.3, Polars2.0.0, torch-geometric2.8.1,
pytest9.1.1. Tests include hard membership, normalized centers, RNG isolation,
missing/degenerate content, corrupt-cache recovery, feature binding, global
cluster-ID permutation invariance, dense-soft-incidence counterexample, graph
bounds/support and exact off/zero forward/gradient/optimizer recovery.

Final regression receipt: **87 passed, 3 upstream warnings in58.33s** on
2026-10-13. CPU Torch/OMP/MKL thread settings were set to one for this run.

Pipeline smoke tests use the actual FreeRec dataset/datapipe/coach, train two
synthetic epochs and resume to epoch three for **both metadata and latent**.
The latent smoke has no external metadata files; its synthetic TRAIN co-interest
is deliberately constructed to test integration, not presented as natural
Amazon behavior or a ranking experiment. Checkpoint continuation on fixed
batches and full/pool scorer parity are separately covered.

```bash
python -m pytest -q \
  tests/test_stair6_v1_latent_categories.py \
  tests/test_stair6_v1_categories.py \
  tests/test_stair6_v1_graph.py \
  tests/test_stair6_v1_pipeline.py
```

Keep upstream torch.jit.script deprecation and sparse CSR warnings visible.
The research report defines the matched-seed, placebo and GO/NO-GO protocol.
Do not convert preprocessing readiness or test success into an improvement claim.
