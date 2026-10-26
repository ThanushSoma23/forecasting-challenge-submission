# Cloud Workload Forecasting Challenge

An end-to-end cloud workload forecasting solution built from the curated **Microsoft Azure Public Dataset V2** VM trace (`data-v1` release).

This project implements a leakage-safe time-series forecasting pipeline predicting 2-hour near-term aggregate cloud demand across 10,000 virtual machines.

---

## 1. Project Overview & Results Summary

We forecast two operational targets over a 24-step (2-hour at 5-minute cadence) holdout horizon:
1. **`active_vm_count`**: Number of active VMs observed in the telemetry pool at timestamp $t$ (constant at 10,000 in this trace; forecasted via persistence).
2. **`aggregate_cpu_utilization`**: Mean CPU utilization percentage across all active VMs at timestamp $t$.

### Key Quantitative Results
- **Expanding-Window Cross-Validation (Training Set Only, $N=110$):**
  - **Selected Method:** **Mean-Reversion Rule with Fixed Decay Factor** ($\phi = 0.90$) achieved the lowest validation error: **MAE = 0.1876%** ($\pm 0.0646$), **RMSE = 0.2166%**, **sMAPE = 1.985%**, providing an **8.74% relative improvement** over the persistence baseline (MAE = 0.2056%).
  - **Moving Average ($K=3$, 15m):** MAE = 0.2353% (performed worse due to lag during transitions).
- **Retrospective Holdout Evaluation (Final 24 Observations):**
  - **Mean-Reversion Rule Holdout MAE:** **0.1055%** (vs. Persistence Baseline: 0.1468% — **28.13% error reduction**).
  - **Holdout RMSE:** **0.1290%**.
  - **Holdout sMAPE:** **1.138%**.
  - **Peak Load MAE ($> P75$):** **0.0885%** (remains accurate during higher-demand intervals).

---

## 2. Environment Setup

Python **3.11 or newer** is required.

Dependencies: `numpy>=1.26`, `pandas>=2.0`, `matplotlib>=3.8`, and `pytest>=8.0`.

### Windows PowerShell

```powershell
# Create virtual environment
py -3.11 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install package in editable mode with development dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Linux / macOS

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install package
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 3. Dataset Download & Verification

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

### Download and Extract

```powershell
# Windows PowerShell / Windows Command Prompt
curl.exe -L -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
tar -xzf data-v1.tar.gz -C challenge_data
```

### Verify Checksums

#### Windows PowerShell
```powershell
Get-FileHash challenge_data/vm_cpu_readings.csv.gz, challenge_data/vm_metadata.csv.gz, challenge_data/DATA_CARD.md -Algorithm SHA256 | Select-Object Hash, Path
```

#### Linux / macOS
```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

Expected SHA256 checksums:
```text
f961964e51f2250534a1273ce8b5aec76816b1ef9628d6356daad0f5af345440  vm_cpu_readings.csv.gz
acc07b52b3757d0c62628fea9cc988933aac93c48d40a8e4a527a15c10e1334b  vm_metadata.csv.gz
3ae8a73f19243ee57e49b8bb94ca4baa4687a7cb4b12ae8326026bb84f547e43  DATA_CARD.md
```

---

## 4. Run Tests

Execute the automated test suite (19 unit and integration tests covering data aggregation, bounds enforcement, temporal split integrity, leakage safety, schema validation, and synthetic smoke tests):

```powershell
pytest -v
```

---

## 5. Generate Predictions

Generate the official submission file `predictions.csv` with exactly 24 rows conforming to the submission schema:

```powershell
python -m cwfc.predict --data challenge_data --output predictions.csv
```

### Schema & Constraint Guarantees
- Exactly 24 rows corresponding to timestamps `33000` to `39900` (step = 300 sec).
- Output columns: `timestamp,active_vm_count,aggregate_cpu_utilization`.
- `active_vm_count >= 0` (integer).
- `aggregate_cpu_utilization` clipped strictly to `[0.0, 100.0]`.
- No extraneous index column; finite numeric values only.

---

## 6. Reproduce Complete Analysis & Benchmark Metrics

To re-run the exploratory data analysis, generate all figures in `reports/figures/`, and reproduce the exact validation metrics reported in the final report:

```powershell
python scripts/run_benchmarks.py
```

Generated Figures:
- `reports/figures/target_series.png`: Chronological time series of active VMs and aggregate CPU utilization.
- `reports/figures/cpu_distribution.png`: Long-tail distribution of VM-level CPU readings and category boxplots.
- `reports/figures/autocorrelation.png`: Autocorrelation Function (ACF) showing inertia and 15-minute periodicity.
- `reports/figures/holdout_forecast_vs_actual.png`: Retrospective forecast vs. actual test observations.

Interactive EDA Notebook:
- Launch `notebooks/workload_analysis.ipynb` in VS Code or Jupyter to inspect step-by-step workload visualizations and experiments.

---

## 7. Project Architecture

```text
forecasting-challenge/
├── README.md                          # Setup, reproduction guide, and project overview
├── pyproject.toml                     # Package dependencies (numpy, pandas, matplotlib)
├── .gitignore                         # Strict exclusion of archives, caches, and datasets
├── predictions.csv                    # Validated 24-step holdout forecasts
├── src/
│   └── cwfc/
│       ├── __init__.py                # Package metadata
│       ├── data.py                    # Ingestion, validation, target aggregation & splitting
│       ├── analysis.py                # EDA profiling, workload segmentation & figure generation
│       ├── forecasting.py             # Baselines and mean-reversion forecasting rule
│       ├── evaluation.py              # Expanding-window temporal CV & peak load evaluation
│       ├── metrics.py                 # MAE, RMSE, sMAPE, and normalized MAE
│       ├── baselines.py               # Starter baseline utilities
│       ├── validation.py              # Starter TimeSplit dataclass and splitting utilities
│       └── predict.py                 # CLI entry point for prediction generation
├── scripts/
│   ├── run_benchmarks.py              # Full benchmark & figure reproduction script
│   └── generate_notebook.py           # Notebook generation utility
├── notebooks/
│   └── workload_analysis.ipynb        # Interactive EDA and model evaluation notebook
├── reports/
│   ├── final_report.md                # 9-section report
│   └── figures/                       # Generated analysis and evaluation figures
├── tests/
│   ├── test_data.py                   # Data ingestion, aggregation, and integrity tests
│   ├── test_forecasting.py            # Forecasting models, bounds, and leakage safety tests
│   ├── test_predict.py                # Schema validation and synthetic smoke dataset tests
│   ├── test_metrics.py                # Metric correctness tests
│   └── test_validation.py             # Temporal split structure tests
└── challenge_data/
    ├── README.md                      # Dataset placement instructions
    ├── SHA256SUMS                     # Expected checksums
    ├── DATA_CARD.md                   # Curated dataset documentation
    ├── vm_cpu_readings.csv.gz         # Telemetry readings (gitignored)
    └── vm_metadata.csv.gz             # VM metadata (gitignored)
```

---

## 8. Source Attribution

The underlying dataset is a curated subset of the **Microsoft Azure Public Dataset V2**:
- Upstream repository: https://github.com/Azure/AzurePublicDataset
- Trace documentation: https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md
