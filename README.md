# DXI-TWNCR (IX-WNCROD)

Interpretable **incremental** anomaly detection via **three-way neighborhood characteristic regions** (NEB / NPB / Lower).

DXI-TWNCR extends the batch WNCROD outlier detector (Zhang, Yuan & Miao, *IEEE TKDE* 2024) with:

- **Bounded-memory incremental updates** — per-point or per-batch state extension in `O(m·n)` with a mobile forgetting / reset mechanism (Phase C) that keeps the reference set within a fixed budget.
- **Intrinsic additive attribution** — the anomaly score is *exactly* decomposed into per-feature contributions, plus region-level contributions (NEB / NPB / Lower), without any post-hoc surrogate model.
- **Theoretical guarantee** — a core invariant that update variants produce identical scores and attributions for any point.

---

## Repository structure

```
.
├── src/
│   ├── base/                  # BaseDetector / BaseExplainer / BaseIncrementalStrategy
│   ├── detectors/
│   │   ├── ix_wncrod.py       # IXWNCROD — proposed incremental + explainable detector
│   │   ├── wncrod_base.py     # WNCRODBase — batch 3WNCROD reimplementation
│   │   └── baselines/         # 10 streaming baselines (IncLOF, HST, RRCF, xStream, ...)
│   ├── explainers/            # three-way region + NOM-path attribution
│   ├── incremental/           # sliding-window update strategy
│   ├── data/                  # synthetic streaming data generator
│   └── utils/                 # metrics + visualization
├── experiments/               # reproducible experiment scripts (see below)
├── DATASETS/{Norm,High}/      # 14 benchmark .mat datasets
├── parameters.txt             # authoritative record of ALL experiment parameters
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `openpyxl` (Excel export). The optional `rrcf` package is used by the RRCF baseline and falls back to a built-in path when absent.

---

## Quick start

```python
from src.detectors import IXWNCROD

det = IXWNCROD(lammda=1.0, n_max=2000, fast_update=True, batch_size=1)
det.fit(U)                                   # fit on the initialization base U = P1 + X

scores, attr = det.predict_with_attribution(X_stream)   # anomaly scores + per-feature attribution
sub = det.get_attribution_subregions(X_stream)          # {"total", "NEB", "NPB", "Lower"}
```

`predict_with_attribution` returns the anomaly score `f(x)` together with the additive decomposition
`f(x) = Σ_j φ_j` where `φ_j` is the contribution of feature `j`; region-level contributions decompose
the same score over the positive (NEB), boundary (NPB) and negative (Lower) regions.

---

## Data partition

Following the unified scheme used across all experiments (`SEED = 42`):

```
D      = P0 + P4                 # full dataset
P0     = P1 + X + P3             # stratified sample of D
U      = P1 + X                  # initialization base, fit(U)
Stream = P3 + P4                 # per-point / per-batch ingestion
```

Datasets are stratified into five traffic tiers `T1–T5` with per-tier sampling ratios (`alpha`)
and `P0` caps — see [parameters.txt](parameters.txt) for the exact values.

---

## Experiments

Each script is self-contained and checkpointed (results are written per-dataset to CSV so runs can be resumed after interruption).

| Script | Purpose |
| --- | --- |
| `incr_core.py` | Core benchmark: 4 update variants (Rebuild1 / RebuildK / Incr1 / IncrK) on 14 datasets |
| `baselines.py`, `baselines_fixed.py` | 10 streaming baselines under the same split |
| `drift.py`, `run_drift.py` | Multi-level concept drift (L0–L3) + Phase-C ablation |
| `ablation_beta.py`, `ablation_gamma.py`, `ablation_lambda.py`, `ablation_lambda_supplement.py` | Sensitivity ablations (β background ratio, γ forgetting strength, λ radius scale + K / P1-X separation) |
| `fidelity.py`, `fidelity_full.py` | Explanation fidelity (perturbation + imputation) |
| `f1_compute.py`, `f1_incrk_compute.py` | F1 score computation paths |
| `significance.py`, `significance_fixed.py`, `gen_cd.py` | Friedman + Nemenyi significance and CD diagram |

Run from the repository root, e.g.:

```bash
python experiments/incr_core.py
python experiments/run_drift.py
```

All parameter grids, data-split formulas and metric definitions are documented in [parameters.txt](parameters.txt).

---

## Evaluation metrics

- **Detection accuracy**: AUC-ROC, AUC-PR, F1.
- **Update efficiency**: `avg_update_ms` (per-batch `partial_fit` wall time).
- **Detection latency**: `avg_detect_ms` (per-point `predict_with_attribution` wall time).
- **Explanation quality**: `completeness_gap = | f(x) − f(x_base) − Σ_j φ_j |`.
- **Storage**: `n_ref_actual` (bounded-memory reference count).
- **Statistical tests**: Friedman (chi-square approx., `df = k−1`) + Nemenyi post-hoc (`CD = 4.454`, `α = 0.05`, `k = 12`, `N = 14`).

---

## License

[MIT](LICENSE)