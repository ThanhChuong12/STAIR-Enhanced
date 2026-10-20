# ECL-global_full_null notebooks — 2026-10-20

Three notebooks, seeds1/2/3, Baby/Sports only: six planned 64D/500-epoch runs.

- `source_overlay.json`: exactly three estimator/config/graph files extending Full base commit5212263. Includes original and extension SHA256, plus UTF-8 source payload. This is embedded in every notebook; no GitHub push is required.
- `build_receipt.json`: notebook hashes, base commit, overlay ID and33 static comparisons per notebook. Snapshot points to its own seed/arm; no recursive self-embedding.
- `validation.json`:48 CPU checks on small synthetic sparse TRAIN matrices, config compatibility and standalone manifest guards. No model, GPU, training, notebook execution, network or full test suite.

Global mode is `pooled_global_excess`, bins=`none`, halves0. Null is n_i*n_j/M_active, excluding zero-history TRAIN users. Score remains positive excess/(c+mu+4), delta1. No halves, no raw-count substitution. Core candidate/count/CF/training contracts stay fixed; null/assignment/score/final graph may differ.

Each notebook clones the recorded base into its own session checkout, verifies base file hashes and applies only the three declared files before imports. It saves base+overlay identity, payload/diff, applied-file receipt and native source hashes. Existing Full/Count runtime checkouts are never modified. Native root/graph/checkpoint identities must agree on global scope.

Generation: `python reports/prepare_eclipse_global_full_null_notebooks.py`. It refuses to overwrite a notebook modified since the recorded build. CPU validation: `python reports/validate_eclipse_global_full_null.py`. Raw experiment artifacts are not touched by either script. The scripts require historical Git objects to remain available. This package contains **no real Global-full training results**.

See [run guide](../../docs/giai_doan_6/ECLIPSE_Global_Full_Null_Notebook_Guide.md) and [execution plan](../../docs/giai_doan_6/ECLIPSE_Experiment_Execution_Plan.md), section21. Status READY refers to prepared code/notebooks, not completed Kaggle integration or experiments.
