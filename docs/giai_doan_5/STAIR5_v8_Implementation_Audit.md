# STAIR5-v8 implementation corrections

Status: source corrections and static inspection only. No project code, imports,
training, syntax compilation, or tests were executed for this revision.

## Corrected contracts

- Named arms resolve to effective alignment, graph and residual settings before
  model construction. V4-control disables all V8 extensions; SAP-only retains
  the V4 graph. The primary WMSG-core retains the original FSC/BPR/NLGCL objective.
- WMSG-placebo now permutes undirected magnitudes within modality-membership and
  train-degree strata. It preserves support and group weight multisets, not exact
  weighted degrees. Logs include the number of edges actually changed.
- kNN preprocessing is CPU-only, with a 128 MiB conservative similarity workspace
  budget. Selected-edge cosine gathers use a separate 64 MiB budget and do not
  allocate a full normalized feature copy. These are workspace estimates, not
  limits on total process RAM or backend workspace.
- Normalization rejects nonfinite, negative and asymmetric weights. Power,
  temperature, residual and forbidden Dirichlet-loss settings fail explicitly.
- CF caching uses the V4 signature, source hash and checksum. A wholly empty CF
  graph disables the CF blend instead of injecting an identity contribution.
- SAP uses distinct training user-item pairs for item degree. Fit/audit anchor
  IDs and the accepted transform are stored in checkpoint-compatible audit data.
  Final refitting has finite/conditioning guards and an identity fallback.
- RPG respects price/category validity masks, generic taxonomy roots and currency
  compatibility. Metadata-loaded RPG requires explicit verified mapping and a
  comparable snapshot; missing metadata produces a logged neutral fallback.
- BSC retains CSR SpMM and no-grad recurrence; the optional residual is bounded
  by 0.1 and blends into a fresh output. Dirichlet diagnostics return a device
  scalar without forcing inner-loop synchronization or hiding nonfinite values.
- Checkpoint provenance includes candidate hashes, metadata identity and baseline
  source dependencies. Telemetry records training allocated/reserved GPU peaks
  and preprocessing GPU peaks. Evaluation peaks still require runtime profiling.

## Frozen-support reproduction

Use `--v4-support-files TEXT_CACHE.pt,VISUAL_CACHE.pt` to reuse the exact V4
semantic cache artifacts. Both must contain `edges`, `signature` and `sha256`.
Feature hashes, modality neighbor counts, algorithm, IDs, row counts, duplicates
and checksum are checked. Loaded signatures and candidate hashes are recorded.

Without explicit artifacts, CPU kNN is recomputed or loaded under its effective
block-size signature. Different devices or block sizes can change near-tied FP32
neighbors; identical algorithm labels alone do not prove identical support.

## Verification still required

Regression test definitions now cover true V4 operator/loss/gradient/smoother
parity, post-step checkpoint moments and continuation, successful SAP fitting,
duplicate-degree statistics, masks, sum-before-max fusion, invalid weights,
empty CF and an actual deterministic placebo. These tests have not been run.

Before benchmarking, execute the V8 suites in the intended FreeRec environment,
then profile preprocessing, training and full-ranking evaluation on the target
GPU. CPU thin SVD remains the reference MI algorithm and can need substantial
RAM. Exact kNN remains quadratic in computation. Static inspection cannot certify
OOM freedom, speedup or improved recommendation metrics.
