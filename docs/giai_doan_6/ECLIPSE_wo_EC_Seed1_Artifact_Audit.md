# ECLIPSE w/o EC — Seed 1 Artifact Audit and Execution Handoff

## Material Passport

- Date: 2026-10-18.
- Scope: existing local training artifacts, recorded source revisions, notebook orchestration and selected-checkpoint reporting.
- Verification status: **ANALYZED** by static source inspection, JSON/log parsing and byte-level SHA256 checks. No model execution, training, checkpoint deserialization or unit tests were performed.
- Research arm: **ECLIPSE w/o EC**, meaning the complete method with only `evidence_delta=0`. LC, continuous semantic weights, CF support expansion, MI/FSC/BPR/BSC and dataset hyperparameters remain enabled.
- Initial acceptance state before user recovery: **REUSED_TRAINING_PENDING_SELECTED_REEVALUATION**. Superseded by the acceptance update below.

### Acceptance update after user-executed recovery

Session `eclipse_wo_ec_seed1_reuse_20261018T152158Z_69603726` supplies successful native TEST receipts at Baby 430, Sports/Electronics 500. File-hash and receipt audit passes for all three: **REUSED_VERIFIED**. See [acceptance JSON](../../reports/eclipse_wo_ec_seed1_audit/recovered_seed1_acceptance.json) and [corrected ablation report](ECLIPSE_Ablation_Study_wo_EC.md). This review parsed outputs and hashed checkpoint bytes; it did not rerun models locally. Statistical significance remains unestablished at one training seed. The seed-2 notebook now embeds the separately verified measured w/o EC seed-1 receipts for aggregation; full-model reference metrics remain excluded. Statements below describing pending recovery document the earlier pre-recovery handoff.

## 1. Decision

**Do not repeat the three 500-epoch seed-1 training runs.** Existing native uncalibrated runs have the required data, candidates, backbone configuration and exact item-operator fingerprints. Recover selected TEST metrics directly from their existing best checkpoints.

The remaining work is checkpoint evaluation and receipt export, not training. Do not describe this recovery as a second training seed. The historical implementation names below are provenance details; the paper arm is simply **w/o EC**.

| Dataset | Existing artifact directory, relative to repository | Selected epoch | Historical Coach.fit wall time | Status |
|---|---|---:|---:|---|
| Baby | `logs/GD5/stair5_v8_artifacts/stair5_v8/baby_WMSG-core_seed1` | **430** | 1,158.378381 s | Reuse training; recover selected TEST receipt |
| Sports | `logs/GD5/stair5_v8_artifacts/stair5_v8/sports_WMSG-core_seed1` | **500** | 2,499.692581 s | Reuse training; recover selected TEST receipt |
| Electronics | `logs/GD5/stair5_v8_artifacts/electronics_WMSG/stair5_v8/electronics_WMSG-core_seed1` | **500** | 31,602.465461 s | Reuse training; recover selected TEST receipt |

Reusing training avoids approximately **9.79 GPU-hours** of historical fit time. This is a saved-work estimate from recorded runs, not a measured speedup of the method. Graph reconstruction and native TEST evaluation still consume time.

## 2. Evidence supporting reuse

The machine-readable audit is [reuse_audit.json](../../reports/eclipse_wo_ec_seed1_audit/reuse_audit.json). For each dataset it includes the complete original manifest, checkpoint/log/telemetry digests, source digests, reference operator and acceptance checks.

The following checks passed for **all three datasets**:

1. TRAIN/VALID/TEST file digests and both modality feature digests match full ECLIPSE seed 1.
2. Text/vision candidate hashes match, in the same modality order.
3. The TRAIN graph/data fingerprint matches, retaining the original training-pair multiplicity.
4. Every shared recorded backbone checkpoint setting matches: dimension 64, layers, gamma, optimizer hyperparameters, batch, seed, evaluation frequency, ranking, worker count, LC settings, CSE settings and weighted semantic settings.
5. Every full-model `v8_config` field matches the corresponding historical native setting. Additional inactive native settings are checked through the shared checkpoint configuration: alignment off, relation gate off, BSC residual zero and Dirichlet loss zero.
6. Historical graph fingerprint equals the full-model manifest's **`baseline_v8_fingerprint`**, not its calibrated graph fingerprint.
7. All 11 recorded source files match their recorded Git commit byte-for-byte. Baby/Sports use `014886e7e7cd03096c724ba67815907e87e05d7c`; Electronics uses `d401a27511a3f77b13c082d68c7f2359b7e4e61c`.
8. Telemetry contains exactly epochs 1–500, and the log records normal completion of native `Coach.fit` and loading the best checkpoint selected by VALID NDCG@20.
9. Python 3.13.15, PyTorch 2.11.0+cu128, FreeRec 0.9.7, Tesla T4, disabled TF32 and no AMP are recorded in the historical manifests.

### Operator equivalence

The PEEC graph builder explicitly returns the cached native operator when `delta == 0`. The model also reuses the original native `mAdj` on that path. The audit compares each historical item operator against precisely that uncalibrated operator:

| Dataset | Exact uncalibrated operator SHA256 |
|---|---|
| Baby | `eab753510c9d830425ba0f1f44f6d4964aa0143c574d4707fb2e7c9d93532041` |
| Sports | `cc0ff8deb6bbdac1c9295e27781dbb96de422731779794d79b4e1fd355ee75ea` |
| Electronics | `3de143b8c1d3b5fe5c8107964b1131be71d9b49b2eec36a37d6c160d08bcc22b` |

Static inspection of the old source against the full seed-1 source shows changes in robust TRAIN edge-store extraction and FreeRec/PyG compatibility routing. The objective, FSC, scorer, smoother, AdamW update and native evaluator paths remain unchanged. Equal actual data/operator fingerprints support equivalence for these recorded datasets. This does **not** assert bitwise-identical stochastic training trajectories across separate implementations or runtime revisions.

## 3. Reporting issue that must be fixed before paper aggregation

The historical native runs did not export an unrounded `selected_metrics.json`. The selected TEST lines have four-decimal precision.

**Baby is the important counterexample:** native `Coach.fit` evaluated epoch 500 before printing its six-decimal summary; it then loaded epoch **430** and evaluated the selected model. The summary-at-500 is not selected-at-430. It cannot be used to invent six-decimal metrics for the selected checkpoint.

Sports and Electronics selected epoch 500, so their six-decimal summaries describe the selected epoch, but remain rounded. Reevaluate all three original best checkpoints to obtain consistent native unrounded receipts.

The historical selected rows below are **rounded log evidence only**, not newly measured or final paper-table values:

| Dataset | R@10 | R@20 | N@10 | N@20 | Precision |
|---|---:|---:|---:|---:|---|
| Baby | 0.0675 | 0.1054 | 0.0365 | 0.0462 | Four-decimal selected log |
| Sports | 0.0762 | 0.1142 | 0.0420 | 0.0518 | Four-decimal selected log |
| Electronics | 0.0460 | 0.0680 | 0.0260 | 0.0317 | Four-decimal selected log |

No p-value, multi-seed SD, new TEST measurement or successful checkpoint load is claimed by this audit.

## 4. Prepared notebooks and input bundle

### 4.1 Seed 1 — recover existing training, without retraining

Notebook: [eclipse_wo_ec_seed1.ipynb](../../notebook/P6/eclipse_wo_ec_seed1.ipynb).

Input ZIP: [eclipse_wo_ec_seed1_reuse_bundle.zip](../../reports/eclipse_wo_ec_seed1_reuse_bundle.zip), approximately 113 MiB. Its current byte count and SHA256 are in [bundle_receipt.json](../../reports/eclipse_wo_ec_seed1_audit/bundle_receipt.json).

The ZIP contains only the three best checkpoints, original manifests, training telemetry, native logs and audit. It excludes large optimizer checkpoints and last-model duplicates.

Steps:

1. Attach this ZIP as a private Kaggle dataset/input. Kaggle may expose its extracted contents directly; both an extracted folder and the original ZIP are supported.
2. Attach the same processed MMRec data, or retain the notebook's existing dataset download/discovery setting. Data are checked against original split/feature fingerprints before model construction.
3. Enable GPU and Internet for detached historical source checkouts. Keep runtime versions matched; changing `STRICT_RUNTIME=False` records a deviation and prevents silent matched-runtime aggregation later.
4. Run setup cells, then Sports, Baby and Electronics recovery cells. Each uses a separate source directory pinned to its original commit. No checkout of a running full-model repository is reset.
5. The helper reproduces the complete original checkpoint contract, loads the original best state dict with `weights_only=True` and `strict=True`, then calls native `Coach.test(selected_epoch)`. It observes native AverageMeters before their reset; it does not reimplement ranking metrics or seen-item masking.
6. Download the new output ZIP. For each dataset require `selected_metrics.json`, `run_status.json`, original `manifest.json`, original `best_model.pt`, evaluation log and provenance plans. The new receipt marks `training_reused=true`, seed 1 and `arm=wo_ec` explicitly.
7. After strict recovery succeeds and Hiếu reads the receipts/provenance, change the corresponding plan status to `REUSED_VERIFIED`. Until then keep the pending status.

On missing input, hash/config/runtime disagreement, checkpoint mismatch or disagreement with the selected log's rounding interval, the notebook stops. It does not change metadata, relax checkpoint loading or launch a replacement 500-epoch run automatically.

### 4.2 Continue with seed 2, then seed 3 (updated 2026-10-19)

Notebook: [eclipse_wo_ec_seed2.ipynb](../../notebook/P6/eclipse_wo_ec_seed2.ipynb).

- Default `SEED=2`, all three datasets, 500 epochs, fresh initialization.
- The only algorithmic change from the full-model protocol is `evidence_delta=0.0`.
- The guard expects the native **uncalibrated** graph fingerprint, `exact_v8_bypass=true`, `inactive_reason=delta_zero` and exactly zero operator/polynomial calibration dose. It does not incorrectly demand the full calibrated graph hash.
- Use the dedicated [eclipse_wo_ec_seed3.ipynb](../../notebook/P6/eclipse_wo_ec_seed3.ipynb) in a separate session for the remaining independent training seed. Change `GPU_INDEX` when using the other physical GPU; use separate sessions/output directories. Attach processed data through Add Input before Kaggle Save & Run All; the seed-3 notebook disables automatic download/attachment by default.
- Attach the seed-1 recovery output and other w/o EC output directories through `ADDITIONAL_ARTIFACT_ROOTS`. The aggregator verifies native-reuse provenance separately from modern-run provenance and never inserts embedded full-model seed-1 metrics into this arm.
- A seed-1 reevaluation with runtime deviations is not silently pooled as matched-runtime evidence.

**2026-10-19 acceptance:** all three seed-2 runs have completed and passed [151 artifact checks](../../reports/eclipse_wo_ec_seed2_audit/acceptance.json). The corrected [ablation report](ECLIPSE_Ablation_Study_wo_EC.md) reports w/o EC n=2, but only matched Full–w/o EC seed 1 from locally available receipts. This does not establish an EC gain across two matched seeds.

The dedicated seed-3 notebook is **READY** after [46 syntax/data-only checks](../../reports/eclipse_wo_ec_seed2_audit/seed3_notebook_validation.json). It embeds verified w/o EC seed-1 and seed-2 measurements for aggregation only; uploading their checkpoints is unnecessary for that purpose. It never uses those measurements as training initialization or checkpoint-selection evidence. No seed-3 result is currently claimed. Human cross-read by Hiếu remains separately attested.

This continues the existing plan without another architecture modification. Full ECLIPSE notebooks and model/runner code were not edited.

## 5. Figure and data outputs

Both notebooks export PDF vector + PNG 300 dpi, source CSV and figure captions. No synthetic measurements or epoch-based p-values are generated.

| Stem | Purpose and limits |
|---|---|
| `eclipse_wo_ec_architecture` | Explicit EC bypass; LC, WMSG, CF expansion and item-BSC remain active. Supplementary ablation schematic. |
| `eclipse_wo_ec_<dataset>_seed1_learning_curves` | Historical training losses and VALID curves. VALID values come from four-decimal logs; selected epoch comes from the original best-checkpoint record, not argmax of rounded curves. |
| `eclipse_wo_ec_<dataset>_seed1_training_memory` | Original per-training-epoch CUDA allocator peaks and epoch time; excludes evaluation/driver/other-process memory. |
| `eclipse_wo_ec_<dataset>_seed1_calibration_diagnostics` | Zero calibration perturbation follows analytically from operator identity. Legacy evidence coverage/quantiles are `NOT_MEASURED`, not zero. |
| `eclipse_wo_ec_seed1_resource_profile` | Separates historical fit resources from newly measured checkpoint reevaluation time. |
| `eclipse_ec_seed1_paired_descriptive` | Full minus w/o EC on the same training seed at each arm's own VALID-selected checkpoint. Descriptive only; no error bars or significance claim. |
| Corresponding `...seed2...` stems | Fresh measured losses, native unrounded VALID receipts, selected TEST and telemetry from seed-2 training. |
| `eclipse_wo_ec_seed_stability_ndcg20`, `...recall20` | Actual available independent w/o EC seeds; sample SD only when n ≥ 2, missing seeds left unmeasured. |

Use multi-seed arm-level ablation figures for the main paper after all required seeds are available. Seed-1 diagnostics and resource panels can support the supplement; they do not replace multi-seed results.

## 6. Completion checklist

- [x] Existing equivalent seed-1 training identified for all three datasets.
- [x] Original file digests, source commits, source hashes, configuration/data/operator equivalence audited statically.
- [x] Original best checkpoints bundled for portable Kaggle recovery.
- [x] Recovery-only seed-1 notebook prepared.
- [x] Fresh w/o EC seed-2 notebook prepared; dedicated seed-3 notebook created with unchanged training configuration.
- [x] Notebook JSON and Python cell/helper syntax parsed without executing cells.
- [x] User runs native selected-checkpoint recovery and verifies receipts (completed in run `eclipse_wo_ec_seed1_reuse_20261018T152158Z_69603726`).
- [ ] Independent human cross-read by Hiếu remains to be explicitly attested; automated file/hash review does not establish who performed a human review.
- [x] Mark three seed-1 jobs `REUSED_VERIFIED`.
- [x] Complete independent w/o EC seed 2 on all three datasets (completed in run `eclipse_wo_ec_seed2`).
- [x] Prepare dedicated seed-3 notebook with safe Kaggle data preflight and verified seed-1/2 aggregation snapshots.
- [ ] Complete independent w/o EC seed 3 on all three datasets.
- [ ] Aggregate actual seeds, then analyze matched differences against full ECLIPSE without inventing missing measurements.
