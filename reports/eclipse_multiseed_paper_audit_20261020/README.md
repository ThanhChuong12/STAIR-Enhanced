# ECLIPSE A0/A1 — derived evidence package, 2026-10-20

Status: **ANALYZED_ARTIFACTS_ONLY**. No training, unit/model tests, checkpoint deserialization, GPU work, or independent reevaluation was performed. Raw artifacts were not modified. This package derives tables from existing native receipts, not new recommendation experiments.

## Contents

- `selected_test_runs.csv`: 18 runs (A0/A1 × three datasets × seeds1–3), four selected TEST metrics, epochs, scoped resources, learning-dynamics summaries, source revisions, fingerprints, file checksums and checkpoint availability/size.
- `mean_std.csv`: full-precision means and sample SD, ddof1.
- `paired_differences.csv`: 36 metric-wise differences. Metrics are correlated, not36 independent experiments.
- `paired_statistics.csv`: exploratory paired t-tests, df2; unadjusted95% CIs; Holm3 primary NDCG@20 and Holm12 exploratory adjustment; exact two-sided sign-flip probabilities. Families are retrospective, not preregistered.
- `mechanism_diagnostics.csv`: static graph diagnostics recorded per run. Repetition over training seeds is not independent graph replication.
- `artifact_checks.csv`, `audit_acceptance.json`: 676 data/provenance comparisons; matching hashes are not independent metric reproduction or proof of no leakage.
- `a0_a1_metrics_table.tex`: data-only LaTeX tabular snippet, needs booktabs; no significance stars or Experiments prose. Not compiled in this audit.
- `before_revision/`: copies of the four reports/plan before this audit. Historical claims retained there are superseded, not additional evidence.

## Recompute numeric outputs

Run `python reports/audit_eclipse_full_multiseed.py` from the repository root with Python and scipy. It reads existing artifacts/Git objects, hashes files and writes derived CSV/JSON outputs here. It does not rewrite narrative reports, raw artifacts, notebooks or model code. Record the Python/scipy environment when reusing it; historical Git commits must remain available.

Source digests are checked against recorded Git objects, and data/candidate fingerprints against other manifests. The script does not rehash original Kaggle feature/split bytes. Checkpoints are hashed as bytes, never loaded. All three Full seed1 checkpoints are absent locally; their selected receipts/logs remain available. A1 seed1 reuses historical uncalibrated checkpoints under the prior recovery audit, with different source revisions; it is not an identical-commit rerun.

`Coach.fit` and summed epoch-train seconds are distinct timers. Allocated/reserved GPU peaks are training telemetry, not process-wide/evaluation peaks. Terminal VALID selects epoch500 before checkpoint restore, not the final reevaluation event. TEST comes from `selected_metrics.json`.

## Paper guidance

See [Full audit](../../docs/giai_doan_6/ECLIPSE_Full_Model_Multi_Seed_Analysis.md), [w/o EC audit](../../docs/giai_doan_6/ECLIPSE_Ablation_Study_wo_EC.md), and [paper readiness/presentation plan](../../docs/giai_doan_6/ECLIPSE_Paper_Experimental_Readiness_and_Presentation_Plan.md).

A2/A3 are human-reported complete at Hiếu's computer and were unavailable locally; no metrics or coverage were inferred. No new figures were generated: requested figure/table and ablation guidance is in the plan. Visual QA of historical figure exports and LaTeX compilation were outside the executed checks.
