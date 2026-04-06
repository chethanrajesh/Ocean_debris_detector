# 🌊 Global Ocean Debris Monitoring & Prediction System

> A research-grade, full-stack system for monitoring, simulating, and predicting ocean plastic accumulation globally — powered by real satellite + oceanographic data, a hybrid Lagrangian + GNN simulation engine, and an interactive live dashboard.

**Live repository:** https://github.com/chethanrajesh/Ocean_debris_detector

---

## 🗺️ Dashboard Preview

The dashboard shows:
- 🌈 **Heatmap** — plastic density gradient (blue = sparse → red = critical) from a 365-step simulation on 455,872 ocean nodes
- ▶️ **Ocean Current Arrows** — real OSCAR/NOAA current vectors; length ∝ speed
- 🔴 **Hotspot Markers** — 637 classified accumulation zones (Critical / High / Moderate)
- 🕐 **Time Slider** — scrub through 90-day future predictions (Apr → Jul 2026)

---

## 🧱 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Real Data Sources                                                  │
│  NASA Earthdata (MODIS/VIIRS) · NOAA ERDDAP (OSCAR currents)       │
│  NOAA NOMADS (GFS winds) · NOAA GDP Buoys (ERDDAP) · CMEMS         │
│  Copernicus Sentinel-2 (GEE) · ETOPO1 Bathymetry (GEE)             │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Backend  (FastAPI · Python 3.11 · PyTorch Geometric)               │
│                                                                     │
│  Data Pipeline          Simulation Engine       GNN Model           │
│  ┌─────────────────┐   ┌────────────────────┐  ┌────────────────┐  │
│  │ fetch_nasa.py   │   │ particle_drift.py  │  │ OceanDebrisGNN │  │
│  │ fetch_noaa.py   │ → │ hybrid_runner.py   │→ │ GATv2 (6 layers│  │
│  │ fetch_cmems.py  │   │ hotspot_detector   │  │ 8 heads, dim128│  │
│  │ fetch_sentinel  │   │ land_mask.py (GEE) │  │ train_gnn.py   │  │
│  └─────────────────┘   └────────────────────┘  └────────────────┘  │
│                                   ↓                                 │
│  FastAPI REST  :8000                                                │
│  GET /hotspots · /predictions/{t} · /currents                       │
└─────────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend  (Next.js 16 · TypeScript · Google Maps JS API)           │
│  GEEMap · HeatmapLayer (Canvas+Float32+LUT) · HotspotLayer         │
│  CurrentVectorLayer · TimeSlider · HotspotPanel · LegendPanel       │
│  http://localhost:3000                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📡 Data Sources

| Source | Dataset | Variables | Access |
|--------|---------|-----------|--------|
| NASA Earthdata | MODIS MOD09GA, VIIRS VNP09GA | Surface reflectance (NIR, SWIR) | Account required |
| NOAA ERDDAP | OSCAR v2 surface currents | u, v current vectors | Public |
| NOAA NOMADS | GFS 0.25° | u10, v10 winds | Public |
| NOAA AOML ERDDAP | GDP Drifter v2.00 | Buoy trajectories 1993–2024 | Public |
| CMEMS | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | uo, vo, zos (forecast) | Account required |
| CMEMS | `GLOBAL_MULTIYEAR_PHY_001_030` | Historical reanalysis | Account required |
| Google Earth Engine | `COPERNICUS/S2_SR_HARMONIZED` | FDI / PI debris indices | Service account required |
| Google Earth Engine | `NOAA/NGDC/ETOPO1` | Bathymetry + land mask | Service account required |

**Register accounts:**
- NASA Earthdata → https://urs.earthdata.nasa.gov/
- Copernicus Marine → https://marine.copernicus.eu/
- Google Earth Engine → https://earthengine.google.com/
- Google Maps API → https://console.cloud.google.com/ *(enable Maps JavaScript API)*

---

## ⚙️ Environment Variables

### `backend/.env`
```env
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password

CMEMS_USERNAME=your_copernicus_marine_username
CMEMS_PASSWORD=your_copernicus_marine_password

GEE_SERVICE_ACCOUNT_EMAIL=your-svc@your-project.iam.gserviceaccount.com
GEE_SERVICE_ACCOUNT_KEY=/app/gee_key.json
```

### `frontend/.env.local`
```env
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GEE_SERVICE_ACCOUNT_EMAIL=your-svc@your-project.iam.gserviceaccount.com
```

> 📋 Template files are provided: `backend/.env.example` and `frontend/.env.example`

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/chethanrajesh/Ocean_debris_detector.git
cd Ocean_debris_detector

# 2. Add credentials
cp backend/.env.example backend/.env
nano backend/.env          # fill in real credentials

cp frontend/.env.example frontend/.env.local
nano frontend/.env.local   # fill in Maps API key

# 3. (Optional) Add your GEE service account JSON key
cp /path/to/your-gee-key.json backend/gee_key.json

# 4. Launch
docker-compose up --build

# Dashboard  → http://localhost:3000
# API docs   → http://localhost:8000/docs
```

### Option B — Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt

# Step 1: Fetch real data (needs credentials in .env)
python -m src.data_pipeline.fetch_noaa
python -m src.data_pipeline.fetch_cmems
python -m src.data_pipeline.fetch_nasa
python -m src.data_pipeline.fetch_sentinel
python -m src.data_pipeline.preprocess

# Step 2: Generate ocean/land mask via GEE
python -m src.simulation.land_mask

# Step 3: Train the GNN on buoy trajectories
python -m src.models.train_gnn --epochs 200

# Step 4: Run the 365-step hybrid simulation
python -m src.simulation.hybrid_runner --timesteps 365

# Step 5: Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

> **Note:** Without credentials the API auto-seeds demo data so the dashboard loads immediately.

---

## 🧠 Model Details

### OceanDebrisGNN (GATv2)

```
Node encoder:   Linear(9 → 128) → LayerNorm → ReLU
Edge encoder:   Linear(3 → 64)  → LayerNorm → ReLU
Message pass:   6 × GATv2Conv (8 heads, dim=128, residual connections)
Output head:    Linear(128 → 64) → ReLU → Linear(64 → 1) → Sigmoid
```

**Node features (9-dim):** `[lat, lon, u, v, u_wind, v_wind, sst, density_t, stokes]`

**Edge features (3-dim):** `[distance_km/100, bearing/360, alignment_score]`

**Training data:** NOAA GDP buoy trajectories (1993–2024, ~25k buoys, 80/10/10 ID-split)

**Loss:** MSE + λ=0.1 × spatial smoothness regularisation

### Lagrangian Physics

```
displacement = (1.0 × ocean_current) + (0.03 × wind) + (1.0 × stokes_drift)
Time step: 6 hours
Coastline: tangential deflection at land boundary
Convergence zone: detected at |u| < 0.05 m/s
Land mask: enforced at every step via GEE ETOPO1
```

### Hybrid Blending

```
final_density = physics_density + 0.4 × gnn_correction
```

The 60/40 blend keeps deterministic physical accuracy while incorporating the GNN's learned spatial patterns from 30 years of buoy trajectories.

---

## 🗂️ API Reference

### `GET /hotspots`

Returns classified hotspot list (637 zones):

```json
[
  {
    "latitude": 32.0,
    "longitude": -141.0,
    "plastic_density": 0.91,
    "level": "critical",
    "accumulation_trend": "increasing",
    "movement_vector": { "u": 0.10, "v": -0.03 },
    "source_estimate": "Yangtze River Delta / East China Sea"
  }
]
```

### `GET /predictions/{timestep}`

Per-node plastic density for timestep 0–364 (day 0 = today, day 364 = ~1 year):

```json
{
  "timestep": 45,
  "total_timesteps": 365,
  "nodes": [
    { "lat": 32.0, "lon": -141.0, "density": 0.76 }
  ]
}
```

### `GET /currents`

Real OSCAR ocean current vectors per node:

```json
{
  "nodes": [
    { "lat": 32.0, "lon": -141.0, "u": 0.12, "v": -0.04 }
  ]
}
```

Full interactive docs available at **http://localhost:8000/docs** when running.

---

## 🗺️ Frontend Visual Guide

| Element | What it shows |
|---------|--------------|
| 🔵→🔴 Heatmap band | Plastic density: blue (low) → teal → yellow → orange → red (critical) |
| ▶️ Teal arrows | Real ocean current direction & speed (longer = faster) |
| 🔴 Circle markers | Critical hotspot zones — click for details |
| 🟠 Circle markers | High-severity zones |
| Time slider | Scrub 0–90 days into the future to see plastic drift predictions |
| + / − buttons | Zoom in (enables free pan) / zoom out (returns to world view) |

---

## 🔁 Reproducibility

- All random seeds fixed: `np.random.default_rng(42)`, `torch.manual_seed(42)`
- All credentials loaded from environment variables — zero hardcoded secrets
- GNN checkpoint saved at `backend/src/models/gnn_checkpoint.pt`
- Simulation output saved at `backend/data/predictions/future_predictions.npy`

Full reproduction from scratch:
```bash
docker-compose up --build
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Python 3.11, Uvicorn |
| GNN | PyTorch, PyTorch Geometric, GATv2 |
| Data | copernicusmarine SDK, earthaccess, xarray, netCDF4 |
| Frontend | Next.js 16, TypeScript, Google Maps JS API |
| Heatmap | Custom Canvas OverlayView (Float32 accumulator + 256-stop LUT) |
| Deployment | Docker, Docker Compose |

---

## 📄 License

MIT — for academic and research use.
