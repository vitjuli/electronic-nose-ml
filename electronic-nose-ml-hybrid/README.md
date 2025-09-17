# 🫧 Electronic Nose — Gas Sensor ML (Hybrid)

This repository implements **machine learning for gas sensor time-series** (SnO₂-based MOS sensors) to predict **dangerous gas concentrations** (Hydrogen, Propane).  
It ships a **small demo dataset** (lite) to keep repo size low, **but** contains the **full codebase** (ensembles, CV, uncertainty, drift, features, baselines, tests, CI) and an **extended README** with methods and literature.

> Hybrid = *Lite data* + *Full documentation & methods*.

---

## 1) Problem & Data

- **Task:** Multitarget regression → \(f: \mathbb{R}^{T} \to \mathbb{R}^2\) mapping a single temperature-modulated sensor cycle (length \(T\)) to concentrations `[H2_ppm, Propane_ppm]`.
- **Sensors:** SnO₂-based MOS sensors.
- **Signals:** resistance traces per cycle (heating/cooling phases).
- **Demo data (lite):** `data/processed/dataset.csv` with columns:
  - `x0..x255` (256 points), `H2_ppm`, `Propane_ppm`, `group`, `split`.

Replace the CSV with your real processed data keeping same column names. Handcrafted feature builder is available in `src/features.py` and `notebooks/00_build_features.ipynb`.

---

## 2) Methods (as in the thesis)

### Modeling
- **MLP (PyTorch)** — default: 1 hidden layer (32), Sigmoid, linear head for 2 targets.
- **Baselines:** compact **1D-CNN** and **GRU** for time-series benchmarks.

### Validation
- **Random split** and **Seasonal/Grouped CV**: `GroupKFold` by `group` (e.g., month/day) to avoid temporal leakage.  
- Scenario “Train on March → Test on April” is recommended to quantify non-stationarity.

### Ensembling
- Train **N independent models** (`--n_models 5`) with different seeds and **average** predictions.  
- Manifest `ensemble.json` is written automatically; `evaluate.py` will detect multiple checkpoints.

### Uncertainty & Intervals
- **Quantile regression** via **pinball loss** (q = 0.05 / 0.5 / 0.95).  
- **Conformal prediction** via **residual calibration** on validation; report coverage & widths.

### Drift checks
- **PSI** (Population Stability Index) & **KS** stats between Train vs Test (or Month→Month).

### Interpretability & Feature Selection
- **Weight/Connection Analysis** (aggregate absolute incoming weights × outgoing head magnitudes).  
- **Permutation Importance** (shuffle feature columns; measure metric drop).  
- **Feature Fixing** (set feature to constant; measure metric drop).  
- **Deep Taylor Decomposition (DTD)** (Montavon et al., 2017) — conceptual guidance; see interpretability notebook for weight- and permutation-based proxies and ablation checks.

---

## 3) Repository Structure

```
README.md
requirements.txt
requirements-dev.txt
src/
  dataset.py        # CSV → matrices, split, scaler save/load
  model.py          # MLP
  baselines.py      # Simple1DCNN, SimpleGRU
  train.py          # training (+ensembles), early stopping, checkpoints
  evaluate.py       # metrics + ensemble averaging + plots
  train_quantile.py # pinball loss quantile model
  losses.py         # pinball loss
  features.py       # handcrafted signal descriptors
  drift.py          # PSI & KS
  calibration.py    # residual-based conformal intervals
  hparam_search.py  # Optuna search (hidden, dropout, lr, batch)
  logging_utils.py  # CSV logger + save_config
  utils.py          # set_all_seeds
data/
  processed/dataset.csv   # small demo (200 cycles × 256 points)
notebooks/
  00_build_features.ipynb
  01_preprocessing.ipynb
  02_training.ipynb
  03_evaluation.ipynb
  04_cross_validation.ipynb
  05_interpretability.ipynb
  06_drift.ipynb
  07_intervals.ipynb
tests/
  test_dataset_scaler.py
  test_model_forward.py
.github/workflows/ci.yml
```

---

## 4) Install & Quickstart

```bash
pip install -r requirements.txt -r requirements-dev.txt

# Train 5 models (ensemble) on the included demo CSV
python -m src.train --data data/processed/dataset.csv --outdir models   --targets H2_ppm Propane_ppm --epochs 100 --patience 20 --n_models 5 --seed 123

# Evaluate (auto-ensemble if you pass the folder)
python -m src.evaluate --data data/processed/dataset.csv --ckpt models   --scaler models/scaler.joblib --targets H2_ppm Propane_ppm
```

Optional:
```bash
# Hyperparameter search (Optuna)
python -m src.hparam_search --data data/processed/dataset.csv --outdir models --trials 20

# Quantile model (0.05/0.5/0.95)
python -m src.train_quantile --data data/processed/dataset.csv --outdir models   --targets H2_ppm Propane_ppm --quantiles 0.05 0.5 0.95
```

---

## 5) Extended Literature (selected, thematically grouped)

### Interpretability / Relevance Propagation / DTD
- **Montavon, G., Lapuschkin, S., Binder, A., Samek, W., Müller, K.-R.** (2017). *Explaining nonlinear classification decisions with deep Taylor decomposition.* **Pattern Recognition, 65**, 211–222.  
- **Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W.** (2015). *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.* **PLOS ONE, 10(7)**.

### Feature Importance / Connection Weights / Permutation
- **Sarle, W.S.** *How to measure importance of inputs?* SAS Institute (tech note).  
- **Breiman, L.** (2001). *Random forests.* **Machine Learning, 45(1)** — permutation importance popularized; regression adaptation follows same idea.  
- **Kohavi, R., John, G.H.** (1997). *Wrappers for feature subset selection.* **Artificial Intelligence, 97(1–2)**.  
- **Guyon, I., Elisseeff, A.** (2003). *An introduction to variable and feature selection.* **Journal of Machine Learning Research**.

### Foundations of Neural Nets & Optimization
- **Cybenko, G.** (1989). *Approximation by superpositions of a sigmoidal function.* **Math. of Control, Signals and Systems**.  
- **Hornik, K., Stinchcombe, M., White, H.** (1989). *Multilayer feedforward networks are universal approximators.* **Neural Networks**.  
- **Rumelhart, D.E., Hinton, G.E., Williams, R.J.** (1986). *Learning representations by back-propagating errors.* **Nature**.  
- **Robbins, H., Monro, S.** (1951). *A stochastic approximation method.* **Annals of Mathematical Statistics**.

### Uncertainty, Quantiles, Conformal
- **Koenker, R., Bassett, G.** (1978). *Regression quantiles.* **Econometrica**.  
- **Angelopoulos, A.N., Bates, S.** (2023). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* **Foundations and Trends in ML**.

### Drift / Monitoring
- **PSI** (Population Stability Index) — standard risk monitoring metric; see model risk documentation.  
- **Massey, F.J.** (1951). *The Kolmogorov–Smirnov Test for Goodness of Fit.* **JASA**.

### Electronic Nose / Gas Sensors (general refs)
- **Gardner, J.W., Bartlett, P.N.** (1994). *A brief history of electronic noses.* **Sensors and Actuators B**.  
- **Bermak, A., et al.** — works on machine olfaction with sensor arrays and gas identification.

> Extend with institution-specific citations from your full thesis when publishing.

---

## 6) Notes & Reproducibility
- Fix seeds with `--seed` and `src/utils.py`. Use ensembling to reduce variance.  
- For non-stationarity, always report grouped CV and time-separated train/test.  
- Keep sensitive data out of version control; consider DVC/LFS for large datasets.

---

## 7) License & Citation
- License: MIT (see `LICENSE`).  
- Citation: see `CITATION.cff` (release date: 2025-09-17).
