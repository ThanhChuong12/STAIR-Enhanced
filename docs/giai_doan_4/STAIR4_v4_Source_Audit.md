# STAIR4-CSGC — author-code audit

Access date: 2026-09-23. This is an implementation/provenance audit, not a reproduction of external papers. Repository branches can change; the links below identify the inspected paths. External implementations were read for comparison, not vendored into CSGC. CSGC implements the equations in `STAIR4_v4_Report.md`; STAIR's existing local optimizer is reused unchanged.

## Inspected implementation paths

| Work | Author source inspected | Relevant observation and implementation decision |
|---|---|---|
| STAIR | [main.py](https://github.com/yhhe2004/STAIR/blob/main/main.py), local `main.py`, `optimizers/AdamW.py`, `optimizers/utils.py` | Preserve MI, opposite FSC/BSC coefficients, dot scoring and the application of BSC **after Adam moment normalization**. Test against the actual local baseline class, rather than a rewritten reference formula. |
| FREEDOM | [src/models/freedom.py](https://github.com/enoche/FREEDOM/blob/master/src/models/freedom.py), `get_knn_adj_mat`, `pre_epoch_processing`, `forward` | Static modality graphs and degree-sensitive interaction pruning are separate components. CSGC keeps STAIR's interaction graph unchanged. FREEDOM's dense similarity construction is not copied. |
| SMORE | [src/models/smore.py](https://github.com/kennethorq/SMORE/blob/main/src/models/smore.py), `spectrum_convolution`, `forward` | Uses feature-coordinate FFT, learned complex filters and preference gates. This does not justify a unitary graph-gradient filter. None of these extra objectives/parameters enter CSGC. |
| DGMRec | [SIGIR branch src/models/dgmrec.py](https://github.com/ptkjw1997/DGMRec/blob/sigir25/src/models/dgmrec.py), `mge`, `calculate_loss`, graph refresh | Contains modality-specific/shared representations, generation and multiple auxiliary losses; refresh paths densify modality graphs. It addresses a different missing-modality setting. CSGC does not inherit these costs. The `sigir25` branch is distinguished from the extended main branch. |
| GUIDER | [src/models/guider.py](https://github.com/Neon-Jing/Guider/blob/main/src/models/guider.py), `calculate_loss`; [author README](https://github.com/Neon-Jing/Guider) | Current code includes Sinkhorn-based selection and hashing-based BPR weights; README describes teacher/student invocation. Source availability does not establish full reproduction of the paper's distillation protocol. Do not import this weighting into baseline BPR. |
| Structured Spectral Reasoning | [src/models/ssr.py](https://github.com/llm-ml/SSR/blob/main/src/models/ssr.py), `_wavelet_band`, `_cheb_apply`, `calculate_loss` | Switches between small-graph eigenspace and polynomial paths; uses spectral consistency/contrastive terms. These are materially different from a single static BSC operator. No eigendecomposition is introduced in CSGC production. |

## Searches without a verified author implementation

“Not verified” means the audit did not establish an author-code link; it does **not** mean the code does not exist.

| Work from the v4 report | Verified material / limit |
|---|---|
| EVEN | [AAAI full paper](https://ojs.aaai.org/index.php/AAAI/article/download/33358/35513) was checked for a repository link; no verified author implementation was found. The conceptual semantic/behavior graph precedent remains relevant. |
| DAMGO | [Publisher DOI](https://doi.org/10.1016/j.eswa.2025.130899); no verified author repository. A third-party page mentioning a generic repository name is not sufficient provenance. |
| MM-GF | [Author manuscript](https://arxiv.org/html/2503.04406v1) points to the MMRec framework. That framework link is not evidence of a released MM-GF implementation. |
| FastMMRec | [Author manuscript](https://arxiv.org/html/2507.18489v1); no author implementation verified in this audit. Do not claim code-level reproduction. |
| GRE-MC | [Author manuscript](https://arxiv.org/html/2605.00670); search found an explicitly third-party reproduction, not a verified author release. No code imported. |
| Critical analysis of multimodality | [Author manuscript](https://arxiv.org/html/2508.05377) lists comparator repositories; no independent author experiment package verified. Comparator links are not a release of the analysis pipeline itself. |

## Additional close work: BRIDGE

The author [repository](https://github.com/LIZESHENG13/bridge) and [src/models/bridge.py](https://github.com/LIZESHENG13/bridge/blob/main/src/models/bridge.py) were inspected after the original twelve-work report. Relevant paths include `_build_sparse_behavior_evidence`, `_candidate_behavior_scores`, `_combine_score_components`, and `_sampled_candidate_masks`.

The inspected code constructs train co-user similarity, optionally prunes it, and applies behavior-derived corrections within score candidates. Its sparse evidence path computes `train_mat.T @ train_mat`. CSGC instead intersects histories only on existing semantic edges, preserves their support and modifies a static optimizer-direction operator. This is an architectural distinction, not evidence of superior accuracy or sufficient novelty. BRIDGE strengthens the need to qualify any claim that behavior-guided calibration itself is new. The repository currently labels its work CIKM 2026; that label is reported as an author claim, not independently certified here.

## Traceability to the implementation

- `models/stair4_v4_utils.py`: independently implemented weighted intersections, effective support, tied midranks and degree strata from the local report.
- `models/stair4_v4_graph.py`: original STAIR raw graph aggregation rule; bounded recalibration and normalized mixture from the local report.
- `models/stair4_v4.py`: baseline backbone; no copied external model modules.
- `optimizers/stair4_v4_smoother.py`: baseline Neumann arithmetic with explicit lifecycle; reuses the local unchanged `AdamWSEvo` application point.
- No claim that the external papers prove CSGC's shrinkage or ranking improvement. Only matched experiments can test that hypothesis.
