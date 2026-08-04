# sfs_evaluations

*** IN PROGRESS *** Eventually, a modular verification framework for evaluating NOAA SFS (Subseasonal Forecast System) monthly ocean, sea ice, and atmospheric forecasts against observational reanalyses (**ORAS5** for ocean/ice and **ERA5** for atmosphere).

---

## 📌 Project Overview

This toolset ingests cloud-hosted and local reanalysis datasets, standardizes spatial and temporal coordinates across multiple initialization months, and performs deterministic and probabilistic verification diagnostics.

### Key Capabilities
* **Cloud-Native Streaming:** Directly streams SFS Beta reforecasts from AWS S3 Zarr stores without requiring full local raw dataset downloads.
* **Tripolar Grid Transformations:** Rotates native tripolar grid ice velocity vectors (u, v) to Earth-oriented coordinates (East/North).
* **Multi-Domain Spatial Remapping:** Uses `xESMF` conservative and bilinear interpolation to standardized target grids:
  * **Sea Ice:** 0.25° x 0.25° (721 x 1440)
  * **Ocean:** 1.0° x 1.0° (181 x 360)
  * **Atmosphere:** 1.0° x 1.0° (181 x 360)
* **Schema Drift Handling:** Gracefully adapts to varying variable availability across initialization months (e.g., initializations `03` and `04` vs. `05+`).
* **Compressed Local Storage:** Precomputes and saves intermediate "verification-ready" datasets locally using Blosc/Zstandard compressed Zarr stores.
* **Multi-Initialization Support:** Loops across all SFS Beta reforecast initialization months (`03`, `04`, `05`, `06`, `07`, `08`, `11`), dynamically handling variable schema drift across runs.

---

## 🏗 Repository Architecture

The codebase will eventually follow a decoupled, modular design (**Ingest -> Preprocess -> Evaluate -> Visualize**):

```text
sfs_evaluations/
├── configs/                # Pipeline configurations (YAML)
│   └── default_config.yaml
├── sources/                # 1. DATA INGESTION
│   ├── sfs.py              # AWS S3 SFS Beta Zarr loader 
│   ├── oras5.py            # ECMWF ARCO ORAS5 Zarr loader
│   └── era5.py             # ERA5 local Zarr loader
├── preprocess/             # 2. CORE TRANSFORMATION ENGINE
│   ├── cache.py            # Retrieve/write climatology and trend (model & obs)
|   ├── climatology.py      # Lead dependent climatology and anomalies
|   ├── detrend.py          # Grid-point detrending
|   ├── engine.py           # Preprocessing/loading engine. Uses preprocess/pipeline.py
|   ├── pipeline.py         # Unified preprocessing pipeline workflows for evaluations
│   ├── regrid.py           # xESMF spatial remapping & weight caching
|   ├── temporal.py         # Temporal selection, lead time mapping, and seasonal aggregation utilities.
│   ├── time_coords.py      # Time axis parsing & alignment
|   ├── vector_rotation.py  # Tripolar grid vector rotation functions
│   └── zarr_writer.py      # Compressed local I/O helpers (**TBD**)
├── metrics/                # 3. VERIFICATION MATHEMATICS
|   ├── acc.py              # Anomaly Correlation Coefficient
│   ├── snr.py              # Signal-to-noise ratio and related concepts
│   └── (**TBD**)
├── viz/                    # 4. VISUALIZATION & PLOTTING SPECS
│   ├── styles.py           # Shared cmocean colormaps & styling defaults
│   └── maps.py             # Cartopy map generators & skill curve plots
│   └── spatial.py          # Several extremely specific multi-panel compositions
├── utils/                  # GENERIC HELPERS
│   ├── config_parser.py    # YAML parser function (**TBD**)
│   └── logging.py          # Clean logger setup (**TBD**)
├── run_preprocessing.py    # CLI runner: Cloud/Raw -> Local Processed Zarr (**TBD**)
└── run_evaluation.py       # CLI runner: Processed Zarr -> Metrics & Plots (**TBD**)
└── test_unified_snr_acc.py # Misnomer -> creates various maps related to skill, predictability, spread
└── breakdown.py            # --> creates maps of (model external variance) vs (observed variance) and
                            # (model internal variance) vs (mean squared error)
```

---

## 📊 Dataset Specifications

| Dataset | Domain | Source Location | Target Grid | Key Variables |
| :--- | :--- | :--- | :--- | :--- |
| **SFS Beta** | Sea Ice | AWS S3 Zarr | 0.25° Regular | `aice_h`, `hi_h`, `uvel_h`, `vvel_h` |
| **SFS Beta** | Ocean | AWS S3 Zarr | 1.0° Regular | `SST`, `SSH`, `ocnheat`, `dt20c`, `so` (3D) |
| **SFS Beta** | Atmosphere | AWS S3 Zarr | 1.0° Regular | `z500`, `t2m`, `prmsl`, `u10`, `v10` |
| **ORAS5** | Ice & Ocean | ECMWF / Local Zarr | Native -> Target | `ileadfra`, `iicethic`, `sosstsst`, `sossheig`, `sohtc300` |
| **ERA5** | Atmosphere | Local Zarr / CDS | 0.25° -> 1.0° | `z500`, `t2m`, `msl`, `u10`, `v10` |

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/sfs_evaluations.git
   cd sfs_evaluations
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate sfs_evaluations
   ```

3. **Configure local paths:** **TBD**
   Edit `configs/default_config.yaml` to specify your target local directories for output Zarr stores (`local_data/`) and figures (`figures/`).

---

## 🚀 Execution Pipeline

### Phase 1: Preprocessing & Regridding. **TBD**
Process raw cloud/remote datasets down to standardized, compressed local Zarr stores:
```bash
python run_preprocessing.py --config configs/default_config.yaml --domain all
```
 
### Phase 2: Evaluation & Plotting **TBD**
Compute verification metrics (Bias, RMSE, ACC) and generate diagnostic spatial maps and skill curves:
```bash
python run_evaluation.py --config configs/default_config.yaml --init-month 05
```

---

## 📝 Acknowledgment & Disclaimer

This software framework was developed with the assistance of an AI collaborator (Gemini) under human technical direction and scientific supervision. All mathematical formulations, physical grid transformations, and assembling logic were reviewed and verified by the human author.
