# STAIR-PEEC — Implementation Audit, versionICAART R2

Date: 16 October 2026. Scope: source review and corrections against `STAIR_versionICAART_Report.md`.

**Verdict: IMPLEMENTED AS A RESEARCH CANDIDATE; RUNTIME VALIDATION PENDING.**

No project test, training, preprocessing experiment, CUDA execution or benchmark was run, following the author's instruction. AST/JSON parsing and text/diff inspections do not establish mathematical runtime correctness, parity, memory usage or ranking gains. The earlier REEC audit is retained in `archive/STAIR_versionICAART_Implementation_Audit_REEC_20261016.md`.

## 1. Blocking discrepancies corrected

| Severity | Finding before this source revision | Correction | Evidence still needed |
|---|---|---|---|
| CRITICAL | Only REEC and pooled-half existed. The primary full-activity estimator was absent. | Added `build_activity_strata`, `candidate_pooled_statistics`, `pooled_activity_score`; explicit full-mode routing. | Enumerated fixtures and real-dataset diagnostic runs. |
| CRITICAL | Pooled-half and pooled-full could be confused despite different null expectations. | Preserve historical `pooled_excess`; introduce `pooled_activity_excess`. Record null scope and contract version2. | Run equation and cross-mode checkpoint tests. |
| HIGH | Halves were unconditionally required to equal2; PEEC could not be configured. | Full modes require0, legacy modes2. Full-mode partition seed is inactive and canonicalized to `None`. | CLI/config checks in intended environment. |
| HIGH | Hardcoded REEC identity did not describe the new estimand. | Versioned identity in graph, model extra state and run manifest, with counts/null/score hashes and shapes. | Checkpoint roundtrip and mismatch rejection. |
| HIGH | Two-half disagreement is inapplicable for PEEC; numeric formatting of `None` would crash. | Emit `None`, print N/A, use generic positive-evidence coverage keys; do not call pooled evidence replicated. | Preprocess-only smoke. |
| CRITICAL / evidence integrity | Old notebook generated Gaussian pseudo-samples around aggregate metrics and ran statistical tests. Missing results could become V8 values. | Removed the entire synthetic statistical layer and missing-result substitution. Future summaries read native unrounded selected-checkpoint receipts. | Inspect future receipts against actual evaluator output. |
| HIGH | Exporting a final or rounded TEST line can misidentify the selected checkpoint or hide tiny differences. | Observe FreeRec AverageMeters before reset; mark only native `eval_at_best` TEST receipts as selected. Selection remains validation NDCG@20. | Compare native best epoch/metric histories with receipts. |
| HIGH | Failed/resumed attempts could expose stale result files. | Unique notebook attempts, checked process return codes, explicit status receipts, stale selected-receipt removal on resume. | Interruption/resume smoke in target environment. |
| MEDIUM | Model edge fallback could choose an arbitrary heterogeneous relation. | Accept only a direct tensor or explicit user-to-item relation keys. Fail closed if unresolved. | Actual FreeRec dataset graph integration. |
| MEDIUM / memory | Endpoint diagnostics assembled extra sparse score matrices; one-coordinate probes did not describe the filter schedule. | Edge-array `bincount` accumulation; five coordinate probes reusing three sparse powers per operator. | Measure preprocessing RSS and time. |

## 2. Estimator mathematics and boundaries

Let binary TRAIN presence be R[M,N]. Deduplication affects evidence only; inherited V8 training retains its original multiplicity. For active user u, b(u)=floor(log2(h_u)); inactive users receive -1 and do not enter any denominator.

For each frozen undirected semantic candidate e={i,j}, i<j:

\[
c_e=\sum_uR_{ui}R_{uj},\qquad
\mu^F_e=\sum_{b:M_b>0}\frac{n_{ib}n_{jb}}{M_b},\qquad
r^F_e=\frac{[c_e-\mu^F_e]_+}{c_e+\mu^F_e+2\tau}.
\]

At tau2, pooled kappa=4. FP64 conversion precedes multiplication of item/bin integer counts. Candidate counts are int64. Empty groups contribute zero. Positive observed overlap with nonpositive expected overlap is an error, not silently repaired. Finite positive regularizers and scores in [0,1) are required.

The full estimator never calls user hashing or half assignment. Fixed local randomness used by diagnostic probes does not enter graph weights. The P1 placebo has its own RNG and shuffles P1 scores within unordered endpoint TRAIN-degree quintile pairs.

| Mode | Estimator | Active halves | Preserved purpose |
|---|---|---:|---|
| `replicated_excess` | Minimum of two separately regularized half-excess scores | 2 | Original REEC |
| `pooled_excess` | Sum counts and sum half-conditioned nulls, pooled regularizer2tau | 2 | Historical bridge; NOT full null |
| `pooled_activity_excess` | Full-stratum null and pooled excess | 0 | Primary PEEC |
| `pooled_activity_count` | c/(c+2tau), same candidate support | 0 | Count-only control |
| `pooled_activity_shuffle` | P1 scores permuted within TRAIN-degree pair strata | 0 | Primary placebo |
| `degree_stratified_shuffle` | Original replicated scores permuted | 2 | Historical placebo; not a P1 substitute |
| `v8_exact` | Zero evidence / exact native operator | 2 | V8 graph control on identical inherited training path |

PEEC does not identify causality, negative feedback, substitute/complement relations or update compatibility. Greater score coverage is not proof of ranking utility. The independence-style activity null is a model approximation, not a calibrated significance test.

## 3. Graph, optimizer and memory contracts

- Recover W8 through the native WMSG builder on cached text5/vision1 candidates: floor.05, power2, sum modalities before max-symmetrization, zero raw diagonal.
- Apply `w_delta = w8 * exp(delta*r)` only to those undirected edges; mirror exactly once. Delta is constrained to [0,2]. No support expansion or semantic-CF graph redefinition is introduced.
- Recompute actual degrees, normalize symmetrically, and give isolated semantic nodes identity blocks. Blend the exact native CF operator at the configured eta. Globally empty CF implies effective eta0.
- When supplied, reconstructed V8 must match the cached V8 fingerprint. Delta0, zero evidence and component-constant evidence retain that reference exactly; the model retains its original native tensor on bypass.
- New evidence/calibration work stays on CPU and returns FP32 CSR. Auxiliary arrays are not retained as model CUDA buffers. The native coach transfers the finished model.
- No dense catalog-square array, full dense similarity, full `R.T @ R`, history-pair enumeration, new trainable parameter or online evidence recomputation is introduced by the new estimator.
- Parent MI/FSC, BPR/NLGCL, user/item parameter groups, AdamWSEvo, native `STAIR5V8Smoother`, ranking, seen masking and checkpoint selection are inherited.
- A symmetric nonnegative actual-degree-normalized graph has spectral norm at most1 in exact arithmetic, including isolate identity blocks. The unchanged convex blend preserves this bound. This is not a proof of bounded Adam trajectories, fewer epochs or higher Recall/NDCG. FP32 validation needs numerical tolerances.

CPU storage includes N-by-B item counts, candidate arrays and bounded block-by-B expectation workspace; B is the number of log2 activity bins. Per-edge NumPy posting searches still incur Python dispatch and are offline. Sparse normalization/CF union and simultaneous reference/calibrated graphs also consume CPU memory. Native feature loading, SVD, CF construction and full-ranking evaluation remain material costs.

Five diagnostic probe coordinates cover {0,d/4,d/2,3d/4,d-1}, with integer division. Reusing sparse powers removes repeated SpMM work. Report these as fixed Rademacher probes, not observed Adam update dose. Additional stored diagnostics include candidate count/null/score quantiles, row-L1 dose and zero-score semantic-coefficient attenuation.

No measured speedup, guaranteed OOM freedom or sub800MiB claim is made. The conditional reference-degree KL solver is deliberately **NOT implemented or enabled**; it tests a separate hypothesis and must not be smuggled into the primary method.

## 4. Checkpoints and reproducibility

The immutable manifest binds architecture/contract, estimator/null scope, active partition, score family, actual-degree normalization, assignments, counts/null/score hashes, native CF identity, final graph, candidates, features, split hashes and training multiplicity. Source hashes and strict inherited provenance checks remain in force. Sparse graph payload checksum is verified before state copying.

Two different estimators remain different contracts even when delta0 produces the same final graph. Full PEEC canonicalizes only its inactive partition seed. Historical R1 source-revision checkpoints are **not silently migrated**: changing audited source hashes can intentionally reject them. Preserve historical runs and start a fresh R2 attempt; resume only under the same validated code/config/data contract.

Cache warmness/build time are telemetry, not immutable identity. Native content-keyed semantic/CF caches are reused; prior raw experimental artifacts are not overwritten. New PEEC configs are separate from original REEC YAMLs.

## 5. Native metrics and notebook

`CoachForSTAIRICAART` inherits the V8 training engine. Its evaluation override calls the native evaluator once and observes active AverageMeters before `step()` resets them. It does not alter score calculation, metric formulas, evaluation frequency or best selection. Only native `eval_at_best()` activates the selected-checkpoint marker. All four primary metrics must exist and be finite; missing metrics are errors, never substitutes.

`evaluation_metrics.jsonl` stores unrounded native averages. `selected_metrics.json` stores TEST at the validation-selected checkpoint. `run_status.json` distinguishes RUNNING, FAILED, PREPROCESSED and COMPLETED. JSON replacement is atomic and rejects NaN/Inf. Metric export occurs at evaluation boundaries, not inside the minibatch optimizer loop.

The observer was designed from inspected **FreeRec0.9.7** source: `AverageMeter` (`avg`, `count`, `active`, `step`) and `Coach.evaluate/valid/test/eval_at_best`. Wheel source was read without importing or executing it. PyPI release metadata: https://pypi.org/project/freerec/0.9.7/ . Inspected wheel SHA256: `dc946d048e7f38261fb3e5a1c76196f56d387bd4964accbe3cd98ecbb34a110c`. Other versions are rejected until audited.

The rewritten canonical notebook:

1. Uses an explicit repository root and subprocess cwd/env; optionally pulls fast-forward only.
2. Checks environment imports in a child process before training; does not upgrade PyTorch/CUDA.
3. Requires all five dataset files, hashes staged content, refuses different existing files and provides writable FreeRec chunk directories.
4. Provides explicit Baby/Sports/Electronics calls, all with native dataset-specific YAML hyperparameters and500 epochs by default.
5. Separates historical REEC, pooled-half, pooled-full, count, shuffle and exact-V8 arms and unique attempt directories.
6. Fails on nonzero process exit and exports only measured selected-checkpoint receipts from completed attempts.
7. Plots native validation curves and training-epoch VRAM, labeled to exclude evaluation-only peaks.
8. Reports actual matched-seed differences and mean/std only; duplicate seed attempts require explicit resolution. No Gaussian pseudo-observations, fabricated p-values, baseline substitution or unmeasured Recall@1.
9. Disables tests and training until the author explicitly enables/calls them. Archives only the current session.

Uploaded historical `logs/GD6/.../icaarts.ipynb` is preserved. Its synthetic statistical tables remain invalid empirical evidence; rewriting future reporting does not validate or retroactively replace them.

## 6. Source and validation inventory

| Files | Change |
|---|---|
| `models/stair_icaart_evidence.py` | Split-free activity statistics, candidate pooled null/counts, full scores and placebo; historical equations preserved |
| `models/stair_icaart_graph.py` | Explicit routing, CPU final CSR, versioned manifests, correct pooled diagnostics, efficient probe/scatter computations |
| `models/stair_icaart.py` | Strict user-to-item relation extraction; estimator-aware checkpoint identity; unchanged backbone |
| `stair_icaart_config.py` | Conditional halves, inactive seed normalization, explicit contracts and pooled payload accounting |
| `main_stair_icaart.py` | Native evaluator observer, accurate diagnostics, unrounded selected receipts and attempt statuses |
| `stair_icaart_reporting.py` | Dependency-free atomic JSON and measured-metric validation |
| Three `configs/Amazon2014{Baby,Sports,Electronics}_STAIR_PEEC.yaml` | New R2 configs; original ICAART YAMLs untouched |
| `tests/test_stair_peec_evidence.py` | Exact full-vs-half null counterexample, duplicates, empty/singleton, ordering, score/control/config contracts |
| `tests/test_stair_icaart_{graph,pipeline,reporting}.py` | Five-coordinate/no-half diagnostics, inherited parity, no partition, resume identity/roundtrip and result integrity definitions |
| `notebook/P6/stair_icaart.ipynb` | Replaced future run/report pipeline; all stored outputs cleared |

| Validation | Status |
|---|---|
| Source/specification/API review | Completed within the disclosed scope |
| Python AST / notebook JSON and cell AST / whitespace diff | Completed: 11 Python files, 17 notebook cells, clean diff whitespace; no source/cell execution |
| Three PEEC YAMLs | Text-level comparison completed: native non-evidence fields match original dataset configs; runtime parser not exercised |
| Unit/integration/CLI tests | **NOT RUN — user instruction** |
| Native exact-parity runtime check | **NOT RUN** |
| Real graph diagnostics / throughput / memory | **NOT RUN** |
| Baby/Sports/Electronics PEEC training / multi-seed ranking | **NOT RUN** |
| Demonstrated PEEC metric improvement | **NOT ESTABLISHED** |

Static inspection receipt: `reports/peec_static_audit_20261016/source_receipt.json`. The adjacent FreeRec source extracts are reference material only; they are not imported by the application.

## 7. Author execution gates

Use the notebook or the following explicit primary command in the intended environment:

```text
python main_stair_icaart.py --config configs/Amazon2014Sports_STAIR_PEEC.yaml --artifact-dir runs/peec_sports_seed1_d1 --preprocess-only
```

Use a **new** artifact directory for training after inspecting the preprocessing receipt; remove `--preprocess-only`. Do not overwrite an earlier attempt. The standalone CLI preserves legacy default mode for compatibility: select PEEC through its new YAML or pass both `--evidence-mode pooled_activity_excess --evidence-halves 0` explicitly.

Before a full-scale claim, the author should execute the defined tests, exact-V8 parity, a2–5 epoch smoke, offline coverage/contrast/dose diagnostics, matched500-epoch V8/REEC/pooled-half/PEEC controls and then3–5 real paired seeds. Delta must be chosen on validation. Include count and P1-shuffle controls. Increased association or graph dose alone is not success; reproducible ranking/margin evidence and subgroup checks determine GO/NO-GO.
