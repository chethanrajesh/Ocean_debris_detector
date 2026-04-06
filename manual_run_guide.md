# 🛠️ Manual Run Guide — Global Ocean Debris System

Run every step below **in order**. Each step depends on the previous one.  
All backend commands run from `d:\ocean_plastic\backend\`.  
All frontend commands run from `d:\ocean_plastic\frontend\`.

---

## ⚙️ STEP 0 — One-time Setup

### Install Python dependencies
```powershell
cd d:\ocean_plastic\backend
pip install fastapi uvicorn python-dotenv numpy scipy pandas xarray netCDF4 requests tqdm earthengine-api copernicusmarine torch torch-geometric
```

### Install Node.js dependencies
```powershell
cd d:\ocean_plastic\frontend
npm install
```

### Verify credentials are set in backend/.env
```
EARTHDATA_USERNAME=evola_chethan
EARTHDATA_PASSWORD=Evola@6360674886
CMEMS_USERNAME=crajesh
CMEMS_PASSWORD=Evola@6360674886
GEE_SERVICE_ACCOUNT_KEY=d:/ocean_plastic/backend/practical-bebop-477117-a5-eaf350af9d9b.json
GEE_SERVICE_ACCOUNT_EMAIL=earth-engine@practical-bebop-477117-a5.iam.gserviceaccount.com
```

---

## 🗺️ STEP 1 — Generate the Ocean Land Mask

> Uses ETOPO1 bathymetry from Google Earth Engine.
> Output: `data/land_mask.npy` — 455,872 real ocean nodes at 0.25°

```powershell
cd d:\ocean_plastic\backend
python -m src.simulation.land_mask
```

**Expected output:**
```
INFO  GEE initialized for ETOPO1 retrieval.
INFO  ETOPO1-based mask built successfully.
INFO  Land mask saved to data/land_mask.npy — ocean nodes: 455,872
Ocean nodes: 455,872 / Total grid: 1,036,800
```

---

## 🌊 STEP 2 — Fetch NOAA Data (Currents + Winds + Buoys)

> Downloads OSCAR ocean currents, GFS wind fields, GDP drifter buoy trajectories.
> Output: `data/ocean_currents.npy`, `data/wind_data.npy`, `data/buoy_trajectories.pkl`

```powershell
cd d:\ocean_plastic\backend
python -m src.data_pipeline.fetch_noaa
```

**Expected output:**
```
INFO  Fetching OSCAR currents: https://coastwatch.pfeg.noaa.gov/...
INFO  OSCAR currents fetched: shape (640, 1440)
INFO  GFS winds fetched: shape (721, 1440)
INFO  Downloading GDP buoy data from ERDDAP...
INFO  Saved 12,450 buoy records to data/buoy_trajectories.pkl
INFO  fetch_noaa.py complete.
```

> **Note:** OSCAR sometimes returns 500/404 (server-side). It gracefully falls back to zeros — the CMEMS step in Step 3 fills in real currents.

---

## 🌍 STEP 3 — Fetch CMEMS Currents (Copernicus Marine)

> Downloads CMEMS global physics analysis (0.083° → interpolated to 0.25°).
> Output: `data/cmems_currents.npy`

```powershell
cd d:\ocean_plastic\backend
python -m src.data_pipeline.fetch_cmems
```

**Expected output:**
```
INFO  Fetching CMEMS analysis: cmems_mod_glo_phy_anfc_0.083deg_P1D-m
INFO  Selected dataset part: "default"
INFO  Saved cmems_currents.npy — shape (720, 1440, 2)
CMEMS analysis shape: (720, 1440, 2), max |u|: 0.41
```

---

## 🛰️ STEP 4 — Fetch NASA Satellite Data

> Downloads MODIS/VIIRS surface reflectance via NASA Earthdata.
> Output: `data/nasa_reflectance.npy`

```powershell
cd d:\ocean_plastic\backend
python -m src.data_pipeline.fetch_nasa
```

**Expected output:**
```
INFO  Searching NASA CMR for MODIS MOD09GA...
INFO  Found 14 MODIS granules
INFO  Downloading granule 1/14...
INFO  NASA reflectance saved: data/nasa_reflectance.npy shape (720, 1440, 2)
```

---

## 🛸 STEP 5 — Fetch Sentinel-2 Debris Indices (via GEE)

> Computes FDI and Plastic Index from Sentinel-2 on Google Earth Engine.
> Output: `data/fdi_map.npy`, `data/pi_map.npy`

```powershell
cd d:\ocean_plastic\backend
python -m src.data_pipeline.fetch_sentinel
```

**Expected output:**
```
INFO  GEE initialized.
INFO  Computing Sentinel-2 FDI and PI composite...
INFO  Exported FDI/PI to data/fdi_map.npy — shape (720, 1440)
```

---

## 🔧 STEP 6 — Preprocess All Data into the Ocean Node Graph

> Aligns all sources to 0.25° grid, blends CMEMS+OSCAR, computes Stokes drift,
> seeds 25 river mouth source points, builds the node feature matrix.
> Output: `data/nodes.npy`, `data/stokes_drift.npy`, `data/source_points.npy`, `data/global_graph/`

```powershell
cd d:\ocean_plastic\backend
python -m src.data_pipeline.preprocess
```

**Expected output:**
```
INFO  Loaded land mask: 455,872 ocean nodes
INFO  Blending OSCAR + CMEMS currents...
INFO  Computing Stokes drift from GFS winds...
INFO  Seeding 25 known source river mouths
INFO  Building ocean node graph...
INFO  nodes.npy saved — shape (455872, 7)
INFO  edges.npy saved — 3,623,456 edges
INFO  Preprocessing complete.
```

---

## 🧠 STEP 7 — Train the GNN (Optional — skip to use physics-only mode)

> Trains OceanDebrisGNN on NOAA GDP buoy trajectories.
> Takes 30–120 minutes depending on GPU availability.
> Output: `src/models/gnn_checkpoint.pt`

```powershell
cd d:\ocean_plastic\backend
python -m src.models.train_gnn --epochs 100 --batch-size 32
```

**Expected output:**
```
INFO  Loaded 12,450 buoy records → 9,876 training patches
INFO  Epoch 001/100 | Train MSE: 0.0842 | Val MSE: 0.0791
INFO  Epoch 010/100 | Train MSE: 0.0421 | Val MSE: 0.0389
...
INFO  Early stopping at epoch 67
INFO  Best checkpoint saved: src/models/gnn_checkpoint.pt
```

> **Skip this step?** The hybrid runner detects if no checkpoint exists and runs in  
> **physics-only mode** (Lagrangian only, no GNN correction). Results are still valid.

---

## 🚀 STEP 8 — Run the Hybrid Simulation

> Runs 365 × 6h time steps (~90 days) of Lagrangian physics + GNN on all 455,872 ocean nodes.
> Takes ~20 minutes on CPU.
> Output: `data/future_predictions.npy` (455,872 × 365, ~635 MB)
>          `data/density_maps.npy`, `data/cleanup_routes.npy`

```powershell
cd d:\ocean_plastic\backend
python -m src.simulation.hybrid_runner
```

**Expected output:**
```
INFO  Ocean nodes: 455,872
INFO  Starting hybrid simulation: 365 steps, λ_GNN=0.4
INFO    t=050/365 | mean_density=0.12 | max_density=1.00 | convergence_zones=12
INFO    t=100/365 | mean_density=0.14 | max_density=1.00 | convergence_zones=15
INFO    t=150/365 | mean_density=0.15 | max_density=1.00 | convergence_zones=17
INFO    t=200/365 | mean_density=0.17 | max_density=1.00 | convergence_zones=18
INFO    t=250/365 | mean_density=0.18 | max_density=1.00 | convergence_zones=19
INFO    t=300/365 | mean_density=0.19 | max_density=1.00 | convergence_zones=20
INFO    t=350/365 | mean_density=0.20 | max_density=1.00 | convergence_zones=21
INFO  future_predictions.npy saved — shape (455872, 365)
INFO  Hybrid simulation complete.
```

---

## 🌐 STEP 9 — Start the Backend API

> Loads all .npy files, detects hotspots, serves 3 REST endpoints.
> Stays running — keep this terminal open.

```powershell
cd d:\ocean_plastic\backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO  Loading land mask...
INFO  Loading predictions (455872, 365)...
INFO  Demo data seeded (or real data loaded).
INFO  Hotspot detection complete — 12 hotspots
INFO  Application startup complete.
INFO  Uvicorn running on http://0.0.0.0:8000
```

### Quick API test:
```powershell
# Test all 3 endpoints
Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing | Select -Expand Content
Invoke-WebRequest -Uri "http://localhost:8000/hotspots" -UseBasicParsing | Select -Expand Content
Invoke-WebRequest -Uri "http://localhost:8000/predictions/0" -UseBasicParsing | Select -Expand Content
Invoke-WebRequest -Uri "http://localhost:8000/currents" -UseBasicParsing | Select -Expand Content
```

**Or with curl:**
```bash
curl http://localhost:8000/
curl http://localhost:8000/hotspots
curl http://localhost:8000/predictions/0
curl http://localhost:8000/currents
```

**API Docs (Swagger UI):**
```
http://localhost:8000/docs
```

---

## 🖥️ STEP 10 — Start the Frontend Dashboard

> Open a NEW terminal. Keep the backend terminal from Step 9 running.

```powershell
cd d:\ocean_plastic\frontend
npm run dev
```

**Expected output:**
```
▲ Next.js 16.2.2

✓ Ready in 570ms

➜  Local:   http://localhost:3000
```

**Open in browser:** [http://localhost:3000](http://localhost:3000)

---

## 🔁 Run Order Summary (Quick Reference)

```powershell
# === BACKEND (run once per dataset update) ===
cd d:\ocean_plastic\backend

python -m src.simulation.land_mask              # Step 1: ETOPO1 mask
python -m src.data_pipeline.fetch_noaa          # Step 2: OSCAR + GFS + Buoys
python -m src.data_pipeline.fetch_cmems         # Step 3: CMEMS currents
python -m src.data_pipeline.fetch_nasa          # Step 4: MODIS/VIIRS
python -m src.data_pipeline.fetch_sentinel      # Step 5: Sentinel-2 FDI/PI
python -m src.data_pipeline.preprocess          # Step 6: Build node graph
python -m src.models.train_gnn --epochs 100     # Step 7: Train GNN (optional)
python -m src.simulation.hybrid_runner          # Step 8: Run simulation

# === START SERVICES (run every session) ===
# Terminal 1:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2:
cd d:\ocean_plastic\frontend
npm run dev
```

---

## 🐳 Alternative: Run Everything with Docker

```powershell
cd d:\ocean_plastic
docker-compose up --build
```

This starts both backend (port 8000) and frontend (port 3000) in containers automatically.

---

## 🩺 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'xarray'` | `pip install xarray netCDF4` |
| `GEE hangs at ee.Initialize()` | Add `--timeout 25` — the land_mask already has a 25s timeout |
| `OSCAR returns 500` | Normal — ERDDAP server issue. CMEMS fills in real currents |
| `CMEMS: unexpected keyword 'depth_range'` | Already fixed — use `minimum_depth=0.0, maximum_depth=1.0` |
| `GDP FTP timeout` | Normal — FTP blocked. ERDDAP fallback or synthetic data used |
| Frontend shows "Backend offline" | Make sure `uvicorn` is running on port 8000 |
| Google Maps shows watermark | Add billing to GCP project for key `AIzaSyAAR0...` |
| `future_predictions.npy` doesn't load | Run Step 8 first; file is 635 MB |

---

## 📁 Data Files Generated

```
d:\ocean_plastic\backend\data\
├── land_mask.npy            (720×1440 bool)        — Step 1
├── ocean_currents.npy       (720×1440×2 float32)   — Step 2
├── wind_data.npy            (720×1440×2 float32)   — Step 2
├── buoy_trajectories.pkl    (DataFrame)            — Step 2
├── cmems_currents.npy       (720×1440×2 float32)   — Step 3
├── nasa_reflectance.npy     (720×1440×2 float32)   — Step 4
├── fdi_map.npy              (720×1440 float32)     — Step 5
├── pi_map.npy               (720×1440 float32)     — Step 5
├── nodes.npy                (455872×7 float32)     — Step 6
├── stokes_drift.npy         (720×1440×2 float32)   — Step 6
├── source_points.npy        (25×2 float32)         — Step 6
├── density_maps.npy         (720×1440 float32)     — Step 8
├── future_predictions.npy   (455872×365 float32)   — Step 8  ← 635 MB
└── cleanup_routes.npy       (N×2 float32)          — Step 8

d:\ocean_plastic\backend\src\models\
└── gnn_checkpoint.pt        — Step 7 (optional)
```
