# STAIR6-v1 implementation and validation

Date: 2026-10-13. Specification: `STAIR6_v1_Report.md`.

Implemented architecture: **CAE-WMSG-CSE**. This implementation enriches the
existing semantic graph with shrunken positive category associations. It does
not classify substitutes/complements, introduce signed edges, add topology or
change the V8 representation/loss/evaluation methods.

## Material Passport

- Material: isolated STAIR6-v1 source implementation and synthetic validation.
- Scope: CPU numerical/integration tests, CLI training/resume smoke and metadata preflight.
- Evidence status: executed checks; no Amazon CAE benchmark or GPU performance result.
- Data prerequisite: verified ASIN mapping and informative category metadata are missing from the current Baby bundle.
- Baseline: original STAIR5-v8 source retained unchanged.

## File layout

```text
main_stair6_v1.py
stair6_v1_config.py
models/
    stair6_v1.py
    stair6_v1_categories.py
    stair6_v1_graph.py
tools/
    audit_stair6_v1_metadata.py
configs/
    Amazon2014Baby_STAIR6_v1.yaml
    Amazon2014Sports_STAIR6_v1.yaml
    Amazon2014Electronics_STAIR6_v1.yaml
tests/
    test_stair6_v1_categories.py
    test_stair6_v1_graph.py
    test_stair6_v1_pipeline.py
```

The model inherits V8 `encode`, `encode_for_eval`, `training_objective`,
`fit`, `marked_params`, sampling, ranking buffers, full scoring and pool scoring.
Its preparation observes the original CPU kNN inputs, runs native V8 MI/graph
construction, then replaces the item operator with the CAE graph. Transient
raw features, category memberships and baseline operators are not kept on GPU.

No new optimizer or loss module is needed. The existing V8 smoother is reused,
with the final operator exposed by `self.mAdj`. User parameters remain
unsmoothed; item parameters use the same BSC polynomial. The same V8 coach
provides model selection, telemetry and checkpoint lifecycle.

## Mathematical contracts

For canonical informative terminal category memberships, rows of H sum to
one; missing/root-only rows are zero. Train user-category incidence is binary
and built from distinct train pairs. Its population includes users with at
least one informative category.

For requested off-diagonal category pairs only:

\[
h_{ab}=\frac{n_{ab}}{n_{ab}+5}
\tanh\left(\max\left(0,\log\frac{n_{ab}N_*}{n_an_b}\right)\right)
\]

when shared-user count is at least two; otherwise h is zero, including h_aa.
Item-edge scores pool this evidence using the normalized memberships. The
enrichment multiplier is `1 + category_strength * h_ij`, with strength in [0,1].
The enriched graph is normalized with current degrees and blended with the
original CF graph using the native globally-empty-CF fallback.

Support stays fixed. Reverse-edge scores are evaluated once and reused for
exact FP32 symmetry. Missing labels preserve the raw edge; normalization can
still change that edge's normalized magnitude through changed node degrees.

`--category-mode off` or `--category-strength 0` bypasses all category inputs
and returns the exact original V8 numerical path. Enabled but empty/generic
metadata or all-zero positive association raises `InactiveCategorySignal`.
It must not be reported as a successful category-enabled experiment.

## Metadata inputs

Each dataset configuration declares paths to three required files:

1. `item_id_to_asin.json`: exactly N_i original ASINs in processed item order,
   or records with unique contiguous `item_id` and `asin` fields.
2. `item_categories.jsonl`: records with `item_id`, matching `asin`, and
   `categories` as lists of full hierarchy paths. Missing item records or
   empty category lists are neutral; duplicate records and malformed explicit
   category paths are rejected.
3. `category_metadata_manifest.json`: source attestation and hash binding.

The manifest uses this schema; angle-bracket values below are placeholders,
not actual verified inputs:

```json
{
  "schema_version": 1,
  "dataset": "Amazon2014Baby_550_MMRec",
  "n_items": 7050,
  "mapping_verified": true,
  "mapping_source": "<the actual provider/preprocessing artifact establishing item order>",
  "metadata_source": {
    "source": "<the actual metadata source>",
    "snapshot": "<the documented snapshot>"
  },
  "files": {
    "mapping_sha256": "<SHA256 of the mapping file>",
    "metadata_sha256": "<SHA256 of the JSONL file>"
  },
  "baseline": {
    "features": {
      "textual_modality.pkl": "<SHA256>",
      "visual_modality.pkl": "<SHA256>"
    },
    "splits": {
      "train.txt": "<SHA256>",
      "valid.txt": "<SHA256>",
      "test.txt": "<SHA256>"
    }
  }
}
```

Only set the mapping attestation after obtaining and validating the provider's
actual item order. A one-to-one array plus matching hashes cannot prove that an
invented ASIN mapping is semantically correct. Do not infer order from metadata
file position, sorted ASINs or approximate feature matching. External
`also_bought`/`also_viewed` links are not used.

The model checks dataset identity before MI, rejects entirely uninformative
taxonomy before SVD, and checks the actual split/feature hashes after native
preparation. Root-only paths and generic terminals are excluded; full parent
paths distinguish identical leaf names under different parents.

## Run the preflight

```bash
python tools/audit_stair6_v1_metadata.py \
  --dataset-dir data/Amazon2014Baby_550_MMRec \
  --output reports/stair6_v1/baby_metadata_audit.json
```

Exit 0 means `READY_FOR_GRAPH_DIAGNOSTICS`, not guaranteed ranking usefulness.
Exit 2 means `NO-GO`. Missing metadata is detected without importing the ML
runtime. Full validation uses the same loader/provenance rules as the model.
The audit does not modify mappings or baseline dataset inputs.

The current Baby bundle returns NO-GO with all three metadata inputs missing.
This is expected and is not a model import/training failure.

## Training commands

Explicit V8 control, which works without category metadata:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --category-mode off \
  --artifact-dir runs/stair6_v1/baby_control_seed1
```

Category-enabled run, after valid metadata and graph diagnostics:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --artifact-dir runs/stair6_v1/baby_cae_seed1
```

Use the Sports/Electronics YAMLs similarly. Their embedding dimension is 64.
Electronics preserves batch 4096 and CL chunk 128, matching the effective
historical V8 run. All three preserve validation NDCG@20 model selection,
full-ranking masks, 500 epochs and the native sampler/optimizer settings.

Resume an existing attempt:

```bash
python main_stair6_v1.py \
  --config configs/Amazon2014Baby_STAIR6_v1.yaml \
  --artifact-dir runs/stair6_v1/baby_cae_seed1 \
  --resume-from runs/stair6_v1/baby_cae_seed1/training_checkpoint.pt
```

Checkpoint provenance rejects changed metadata, splits, features, graph,
strength, dimensions or source contracts. It is not a V8-to-CAE warm-start
utility. A fresh attempt cannot silently overwrite existing training artifacts.

## Performance implementation

- Graph/statistics processing is CPU-only. Category pair counts are queried
  from sorted CSC user postings; the smaller posting list uses vectorized
  search in the larger list. No dense N_i²/C² matrix or full HhH-transpose is built.
- Requested-pair generation avoids duplicate reverse edges on symmetric input.
  Pooling uses dictionary lookup and reverse-edge reuse, avoiding a scalar
  binary search for every label combination.
- Pair-set and probe workspaces have explicit budgets; pathological multi-label
  inputs fail rather than silently truncate the model. Budgets do not bound
  total RSS, which also includes raw features and caller graphs.
- Enabled models share the validated native raw-CF cache, avoiding a second
  catalog-wide co-occurrence multiplication. CAE scores have their own
  content-addressed cache binding taxonomy, train keys, raw weights and sources.
- Final S6,1 replaces S8 with the same support/CSR dimensions. No additional
  parameters or per-step graph branch are introduced. Original checkpointed
  NLGCL chunking and no-grad optimizer recurrence are retained.
- Manifests include raw/normalized graph fingerprints, coverage, active edge
  mass, score quantiles, operator perturbation and deterministic BSC probe dose.
  Probe dose is not a measurement of actual Adam-direction or ranking benefit.

A synthetic CPU profile with 63,001 items, 504,008 semantic edges and 100
single-label categories measured approximately 0.778 s for requested-pair
generation and 2.091 s for pooling, with a 1.923 MiB edge-score tensor and exact
symmetry. This is a synthetic component benchmark, not real Electronics
preprocessing, GPU training throughput or an OOM guarantee.

## Verification and limits

Executed in an isolated Python 3.12.10 environment with Torch 2.10.0+cpu,
FreeRec 0.9.7, NumPy 2.5.4, SciPy 1.18.1, PyYAML 6.0.3, Polars 2.0.0,
torch-geometric 2.8.1 and pytest 9.1.1.

Final validation receipt: **62 passed, 3 upstream warnings in 30.93 s** on
2026-10-13, with Torch CPU threads set to one. The exact-independence regression
also verifies that integer count equality yields zero category evidence rather
than a spurious positive score from floating-point log cancellation.

```bash
python -m pytest -q \
  tests/test_stair6_v1_categories.py \
  tests/test_stair6_v1_graph.py \
  tests/test_stair6_v1_pipeline.py
```

The suites cover taxonomy/mapping/hash integrity, duplicate-aware counts,
multi-label pooling, graph bounds/support/cache recovery, zero-intervention
forward/RNG/gradient/two-step optimizer parity, live autograd, exact checkpoint
continuation on the same batches and full/pool scoring parity.

A subprocess smoke additionally executes the actual FreeRec dataset, datapipe,
coach, validation, selected-checkpoint evaluation and artifact writer for two
synthetic epochs, then resumes the same attempt to epoch three. This confirms
the production launcher path; it does not replace the Amazon benchmark.

Upstream warnings about deprecated `torch.jit.script` and beta CSR support are
left visible. They were not failures in the CPU checks. GPU kernels, process
VRAM, evaluation peaks, real metadata coverage and ranking gains still require
measurement on the target environment.

Use the research report's matched controls and GO/NO-GO protocol before making
improvement claims. No real dataset metrics or guaranteed percentage gains
were produced by this implementation request.
