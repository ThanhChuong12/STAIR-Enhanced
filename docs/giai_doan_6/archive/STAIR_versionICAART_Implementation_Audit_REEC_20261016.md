# STAIR-REEC / versionICAART — static implementation audit

Date: 16 October 2026.

**Scope:** source inspection and direct corrections only. No project module was imported, no pytest case was executed, no preprocessing experiment, GPU smoke test, training or benchmark was run. Python AST parsing was used only to inspect source/cell syntax, without executing that source. Test definitions were corrected/extended for the author to run later.

The pasted audit brief requested pytest execution and a 100% pass receipt. The user's direct instruction explicitly excluded tests and experiments, so those parts of the brief were not followed. No runtime PASS or readiness certification is claimed.

Specification: `STAIR_versionICAART_Report.md`. All implementation comments and new notebook code use English. This audit does not change the research hypothesis or claim ranking improvement.

## 1. Overall verdict

The original implementation followed the primary mathematical direction, including actual-degree normalization, replicated excess, direct V8 inheritance and native BSC. It was not yet safe to describe the pipeline as fully validated: provenance, candidate/weight alignment, CF reuse, negative-control definitions and notebook failure handling required corrections.

**Disposition: SOURCE CORRECTED; RUNTIME VALIDATION PENDING.**

Only the ICAART branch and this audit document were changed. Baseline `main.py`, the V8 backbone/graph/smoother/coach, previous phase models and CoGate sources were not edited. `git diff --name-only` showed no tracked-file modifications during this task; the ICAART files were already untracked, so an empty tracked diff does not mean their contents were unchanged. No reset, checkout, commit or push was performed.

## 2. Findings and applied corrections

Old anchors below refer to the initially reviewed files. Current anchors identify the corrected implementation and can shift after later edits.

| Severity | Original evidence | Problem | Applied correction |
|---|---|---|---|
| HIGH / memory | `models/stair_icaart_evidence.py`, old221–225 | Sparse user-by-candidate-block multiplication materialized unions of endpoint user lists. A block of8192 hub-heavy pairs could have a very large intermediate even without a dense item-square matrix | `candidate_excess_statistics` now searches sorted CSC posting lists per eligible pair using compiled NumPy `searchsorted`; no split R0/R1 matrices or user-by-block product. Block size bounds null-expectation gathers |
| HIGH / data integrity | Evidence old56–64,138–155,289–299; graph old431–436 | Helpers trusted binary input, assignments and array shapes. Builder cast arbitrary IDs to integer before flattening; invalid IDs could alias valid pair keys | Private binary-presence canonicalization; native `binary_train_keys` validates IDs/dtypes/ranges before deduplication; group/score/pair validation fails explicitly |
| HIGH / edge attribution | Graph old328,339,348–353 | Pair weights were taken from `triu(...).data` while endpoints came from a separate E_sem argument. Different pair ordering could attach evidence to the wrong raw weight | `calibrate_semantic_operator` checks exact support coverage/uniqueness and retrieves each raw weight by its endpoint IDs |
| HIGH / parity | Graph old325–345,384–385 | Bypass did not populate the final graph fingerprint/nnz fields consistently. Constant score detection used `allclose` and checked only one global value | Common audit finalization for all paths; zero/empty/component-constant evidence returns the cached reference. Exact per-component min/max avoids erasing small real contrasts |
| HIGH / provenance | Model old114–135; graph old515–531 | Manifest omitted placebo seed, assignments, raw/normalized CF identity and dtype. A changed evidence configuration could remain accepted when final graphs coincided. Extra-state checks did not hash serialized mAdj values | Deterministic full evidence contract, assignment/score/group/graph hashes, source/features/splits; inherited strict extra-state equality; sparse graph payload checksum before any state copy |
| HIGH / control definition | Graph old469; evidence old333–348 | Shuffle used weighted semantic degrees and only the smaller endpoint degree; absent strata silently became a global shuffle | Use deduplicated TRAIN item degrees and both endpoint degree-quintile IDs; shuffle undirected-pair scores within those strata; missing degree inputs are rejected |
| MEDIUM / preprocessing | Model old79–95; graph old480–496 | Native V8 built CF, then REEC built it again without the existing cache path | Ensure a writable shared cache and reuse V8's exact raw-CF signature/checksum/native normalizer. No global monkey-patching or baseline edits |
| MEDIUM / scope clarity | Graph old227–252 | A middle-coordinate polynomial probe was labeled generically as polynomial dose | Log coordinate, probe dimension, seed and scope. It is not a full-schedule/actual-Adam measurement |
| MEDIUM / diagnostics | Graph old254–265 | Degree ratios alone did not show indirect attenuation of zero-score semantic edges | Add semantic normalized-coefficient ratio quantiles and relative row-L1 operator dose; coefficient ratios are explicitly semantic-only, not final CF-blended coefficients |
| MEDIUM / configuration | Config old58–60,79–109 | Halves accepted numeric lookalikes; CL settings could drift from the fixed objective; modality order was weakly constrained | Reject invalid integer/bool settings; lock native NLGCL weight/tau/direction/G, WMSG settings and backbone exclusions; reject recognizable reversed modality order |
| MEDIUM / memory receipt | Config old114–125 | One n_edges value was used as both directed graph nnz and undirected evidence pair count | State the undirected-pair convention; account for two directions in semantic CSR storage and include FP64 weights. Final CF-union storage is a separate quantity |
| HIGH / notebook | Old code cells2,6,8 | Imports were caught and printed as failure without stopping; tests/training had no repository cwd; process exit code was ignored | Explicit repo root/cwd/env; compatibility shim before FreeRec import; subprocess failures raise; no success result is parsed after failure |
| HIGH / experiment artifacts | Old code cell8 | Fixed output locations could mix or overwrite runs; labels omitted tau; parser expected an artifact not produced by the native coach | Unique attempts include dataset/mode/delta/tau/seed; store launch command; parse the final native selected-checkpoint TEST block and write an explicit result artifact |
| HIGH / data staging | Old code cell4 | Any matching directory with one TXT/PKL file counted as a dataset; multiple symlink bridges could connect incomplete/stale data | Require all five canonical files, exact-name discovery or explicit source override, writable real `Processed/.../chunks` directories, no silent overwrite |
| MEDIUM / test adequacy | Graph tests old148–219; pipeline old287+ | Loose derivative tolerance, a different stale-degree fixture than the report, and a large catalog shape without an allocation guard | Corrected Eq.10 tolerance and the 1.1074819683 fixture; added catalog-square allocation/sparse-to-dense guards and ordering/config/payload/RNG tests. **Definitions only; not executed** |

## 3. Mathematical compliance after source corrections

### Native WMSG and CF

- `recover_v8_raw_semantic_graph` calls the existing V8 builder with edge power2, floor.05, weighted mode and RPG off. The parent performs per-modality weights, sum-before-max-symmetrization and zero diagonal.
- REEC uses the native raw-CF cache format/source hash, shrunken Ochiai top5 builder and `normalize_cf_graph_with_fallback`.
- A wholly empty raw CF graph maps to `None` and effective eta0; it never becomes an artificial `.1 I` contribution.
- A reconstructed unmodified reference must match the cached native V8 graph fingerprint before calibration proceeds.

### Evidence

- Only `dataset.train().to_bigraph(...)` supplies interactions in the model preparation path. Split-file hashes are provenance, not graph inputs.
- Training pair multiplicity is retained by inherited V8. Deduplication is isolated to a new presence matrix with raw/unique count and hash receipts.
- Activity strata exclude zero-history users. Seeded SHA-256 ordering and alternating assignments operate on whole users; vectorization changes neither the specified hash nor its tie policy.
- Observed counts use candidate-only posting-list intersections. Expected counts cast incidence arrays to FP64 **before** multiplying.
- Counts, groups and scores are validated; `c>0,mu<=0` fails. Empty histories return zero evidence without zero-denominator warnings.
- Eq.5 remains `[c−mu]_+/(c+mu+tau)` followed by a minimum across halves. No small-popularity-prior division, sigmoid, global-max calibration or negative-feedback label is introduced.

### Calibration and optimizer

- Candidate pairs must be unique i<j and exactly cover native semantic support. Score order is bound to endpoint IDs.
- Raw calibrated weights are FP64 and mirrored once. Actual new degrees are recomputed; no stale-degree restoration, Sinkhorn or spectral power rescaling is present.
- Static operator is FP32 CSR. Auxiliary evidence remains CPU-local and does not become model CUDA buffers.
- Delta0/no-evidence returns the cached reference, and the model retains its original native sparse tensor on bypass. Component-constant evidence also returns the reference with a distinct inactive reason.
- Inherited `marked_params`, `training_objective`, `encode`, scorer and native `STAIR5V8Smoother` are retained. No new parameters, CL views, residual queries, loss or inference path was added.
- Checkpoint graph payload/configuration checks run before model-state copying. Cache-hit flags/timing are telemetry, excluded from the immutable contract so cache warmness does not reject a legitimate resume.

## 4. Files reviewed

| File | Disposition |
|---|---|
| `stair_icaart_config.py` | Corrected validation and analytical memory conventions |
| `models/stair_icaart_evidence.py` | Corrected bounded intersections, input integrity, deterministic split vectorization and shuffle |
| `models/stair_icaart_graph.py` | Corrected pair binding, bypass, CF reuse, provenance and diagnostics |
| `models/stair_icaart.py` | Retained V8 inheritance; added cache reuse and checkpoint safeguards |
| `main_stair_icaart.py` | Reuses V8 dataset loader/coach directly; shared cache and clearer diagnostics |
| Three `configs/Amazon2014{Baby,Sports,Electronics}_STAIR_ICAART.yaml` | Reviewed; dataset-specific gamma/WD/batch/chunk settings and500 epochs already agree with the reference; no gratuitous YAML edits |
| Three `tests/test_stair_icaart_{evidence,graph,pipeline}.py` | Corrected/extended definitions; not executed |
| `notebook/P6/stair_icaart.ipynb` | Corrected environment, staging, process handling, attempt isolation and result extraction |

The notebook defaults to `RUN_TESTS=False`, `RUN_TRAINING=False`, optional dependency installation and optional download. These are explicit execution switches, not claims that validation has already occurred. `run_training_icaart` supports any properly staged dataset; it does not restrict explicit calls to Baby.

## 5. Performance limits that remain unmeasured

The evidence implementation now has no user-by-candidate-block sparse product, dense N×N allocation or enumeration of all history-item pairs. The null computation holds bounded block×2×B arrays. Intersection search is vectorized inside NumPy but still has Python dispatch per eligible semantic pair; this cost is offline and must be measured on real Electronics data. It is not a demonstrated speedup.

Native CF construction itself retains its existing bounded item-row SpGEMM implementation. That is not a full dense N×N allocation and has not been rewritten. Cache reuse avoids intentionally repeating that work during model setup.

Raw semantic selected-edge cosines are recovered after native V8 preparation; native kNN retrieval is observed, not repeated. This can repeat some selected-edge CPU work and keeps feature references alive during preparation. No new online sparse pass was added. Total CPU RSS still includes raw features, MI SVD and native CF workspace.

No guaranteed OOM freedom, `<800MiB` claim, runtime gain, ranking gain or exact GPU parity is inferred from static review. Full-ranking evaluation can peak above training telemetry.

## 6. Validation receipt

| Check | Status |
|---|---|
| Source/specification review | Completed within the scope above |
| Eight ICAART Python sources/test files: AST syntax inspection | Completed, without executing source |
| Notebook JSON and code-cell AST inspection | Completed, without executing cells |
| Baseline/V8/CoGate modifications by this task | None |
| Pytest / integration tests | **NOT RUN — user instruction** |
| Stage0B real-dataset mechanism diagnostics | **NOT RUN — user instruction** |
| GPU smoke / CUDA memory / throughput | **NOT RUN — user instruction** |
| Training / ranking experiment | **NOT RUN — user instruction** |

Before claiming Stage1 readiness, the author should run the corrected tests and offline diagnostics in the intended environment. This document intentionally does not report “100% passed”.
