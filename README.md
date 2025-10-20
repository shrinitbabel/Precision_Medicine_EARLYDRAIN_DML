# EarlyDrain Dynamic Machine Learning

This repository contains code and data to study prophylactic lumbar drainage (LD) in aneurysmal subarachnoid hemorrhage.  It uses causal machine learning to estimate treatment effects, perform clustering of heterogeneous responses and provides an interactive Streamlit application.

The work associated with this repository is currently under review at a journal.

## Repository Structure

- `ed.csv` – main dataset with baseline variables and outcomes
- `ed_daily.csv` – daily ICU measurements aggregated with `build_dynamic_covariates.py`
- `modules/` – reusable Python modules
  - `preprocess.py` – load and clean data
  - `dml.py` – train CausalForestDML models and export CATEs
  - `analysis.py` – utilities for CATE analysis and DoubleML sensitivity
  - `cluster.py` – cluster patients by CATEs using UMAP/KMeans
  - `synergy.py` – evaluate synergy between binary treatments
- `models/` – pretrained causal forest models for several outcomes
- `cate_results/` – exported conditional treatment effects
- `streamlit_app.py` – web app to estimate individual effects
- Jupyter notebooks (`*.ipynb`) demonstrate analyses for outcomes like modified Rankin Score, vasospasm and others.
- `ATE_stability_analysis.ipynb` helps validate whether the heterogeneous treatment effect estimates (like CATEs or ATEs) are stable across different training strategies. 

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   The app lets you enter patient characteristics and shows predicted effects of prophylactic LD on selected outcomes.

Model files are provided in `models/`.  If you retrain models using the notebooks or `modules/dml.py`, save them in this directory so the app can load them.

## License

This project is released under the MIT License (see `LICENSE`).
