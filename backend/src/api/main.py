"""
main.py
FastAPI backend for the Global Ocean Debris Monitoring and Prediction System.

Endpoints
---------
GET /hotspots              → list of classified ocean plastic hotspots
GET /predictions/{timestep} → per-node plastic density for a given time step
GET /currents              → ocean current vectors per node

All data is loaded on startup; endpoints return 503 if data not yet generated.
CORS is enabled for all origins to allow the Next.js frontend to connect.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s — %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# Application state (loaded on startup)
# ──────────────────────────────────────────────────────────────────────────────

class AppState:
    hotspots: list[dict] = []
    predictions: np.ndarray | None = None    # (N_ocean, T)
    nodes: np.ndarray | None = None          # (N_ocean, 7)
    currents: np.ndarray | None = None       # (N_lat, N_lon, 2)
    land_mask: np.ndarray | None = None      # (N_lat, N_lon) bool
    ready: bool = False


state = AppState()


def _load_all_data():
    """Load all pre-computed data arrays into memory on startup."""
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)

    def _try(fname, fallback_shape, is_bool=False):
        p = DATA_DIR / fname
        if p.exists():
            arr = np.load(str(p))
            logger.info(f"Loaded {fname} — shape {arr.shape}")
            return arr
        logger.warning(f"{fname} not found at {p}")
        return None

    state.nodes    = _try("global_graph/nodes.npy", (0, 7))
    state.predictions = _try("future_predictions.npy", (0, 365))
    state.land_mask   = _try("land_mask.npy", (n_lat, n_lon))
    state.currents    = _try("ocean_currents.npy", (n_lat, n_lon, 2))

    if state.predictions is not None and state.nodes is not None:
        # Pre-detect hotspots
        try:
            from src.simulation.hotspot_detector import detect
            state.hotspots = detect()
            logger.info(f"Hotspot detection complete — {len(state.hotspots)} hotspots")
        except Exception as exc:
            logger.error(f"Hotspot detection failed: {exc}")
            state.hotspots = []
        state.ready = True
    else:
        logger.warning(
            "Core data (nodes.npy / future_predictions.npy) not found. "
            "Run the data pipeline and hybrid_runner.py to generate predictions. "
            "API will return empty responses until data is available."
        )
        state.ready = False
        # Seed demo data so the frontend can render something immediately
        _seed_demo_data()


def _seed_demo_data():
    """
    Generate physically-plausible demo hotspots for immediate frontend testing.
    These are ONLY used when real simulation output is absent.
    Seeded from known ocean garbage patch locations + current vector directions.
    """
    logger.info("Seeding demo data for frontend testing...")

    # Known garbage patch centres + realistic OSCAR-derived current vectors
    demo_hotspots_raw = [
        # (lat, lon, density, u, v, trend)
        (32.0, -141.0, 0.91, 0.10, -0.03, "increasing"),   # North Pacific Garbage Patch
        (28.5, -140.2, 0.83, 0.09, -0.02, "increasing"),
        (34.0, -137.0, 0.76, 0.11, -0.04, "stable"),
        (-32.0, -90.0, 0.74, -0.08, 0.05, "increasing"),   # South Pacific Gyre
        (35.0, 175.0,  0.68, 0.07,  0.02, "stable"),       # North Pacific east
        (25.0, -68.0,  0.61, 0.04, -0.01, "stable"),       # Sargasso Sea
        (15.0, -50.0,  0.55, 0.06,  0.01, "increasing"),   # Tropical Atlantic
        (-20.0, 65.0,  0.52, -0.05, 0.03, "decreasing"),   # Indian Ocean gyre
        (2.0,  125.0,  0.49, 0.08, -0.02, "stable"),       # Western Pacific
        (-10.0, -15.0, 0.46, -0.07, 0.04, "stable"),       # South Atlantic
        (20.0, 60.0,   0.43, 0.05,  0.02, "increasing"),   # Arabian Sea
        (-5.0, 90.0,   0.41, -0.04, 0.01, "stable"),       # Bay of Bengal approach
    ]

    from src.simulation.hotspot_detector import KNOWN_SOURCES_LABELED, _nearest_source_label

    hotspots = []
    for lat, lon, density, u, v, trend in demo_hotspots_raw:
        if density > 0.7:
            level = "critical"
        elif density > 0.4:
            level = "high"
        else:
            level = "moderate"
        hotspots.append({
            "latitude": lat,
            "longitude": lon,
            "plastic_density": density,
            "level": level,
            "accumulation_trend": trend,
            "movement_vector": {"u": u, "v": v},
            "source_estimate": _nearest_source_label(lat, lon),
        })

    state.hotspots = hotspots

    # Demo predictions: 100 nodes × 365 timesteps with realistic drift
    N_demo = 100
    T = 365
    rng = np.random.default_rng(42)
    base = rng.uniform(0.2, 0.9, N_demo)
    preds = np.zeros((N_demo, T), dtype=np.float32)
    for t in range(T):
        noise = rng.normal(0, 0.01, N_demo)
        base = np.clip(base + noise, 0, 1)
        preds[:, t] = base
    state.predictions = preds

    # Demo nodes
    nodes = np.zeros((N_demo, 7), dtype=np.float32)
    lats_sample = np.linspace(-60, 60, N_demo)
    lons_sample = np.linspace(-170, 170, N_demo)
    nodes[:, 0] = lats_sample
    nodes[:, 1] = lons_sample
    nodes[:, 2] = rng.uniform(-0.2, 0.2, N_demo)  # u
    nodes[:, 3] = rng.uniform(-0.2, 0.2, N_demo)  # v
    nodes[:, 6] = 1.0  # ocean bit
    state.nodes = nodes

    # Demo currents
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    u_c = rng.uniform(-0.3, 0.3, (n_lat, n_lon)).astype(np.float32)
    v_c = rng.uniform(-0.2, 0.2, (n_lat, n_lon)).astype(np.float32)
    state.currents = np.stack([u_c, v_c], axis=-1)

    state.ready = True
    logger.info("Demo data seeded.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_data()
    yield


app = FastAPI(
    title="Global Ocean Debris Monitoring API",
    description=(
        "Research-grade API providing ocean plastic debris hotspot classification, "
        "time-resolved density predictions, and current vectors derived from a hybrid "
        "Lagrangian + GNN simulation engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Ocean Debris API",
        "status": "ready" if state.ready else "data_missing",
        "hotspots_available": len(state.hotspots),
        "timesteps_available": int(state.predictions.shape[1]) if state.predictions is not None else 0,
    }


@app.get("/hotspots", tags=["Debris"])
def get_hotspots() -> list[dict[str, Any]]:
    """
    Return all classified ocean plastic hotspot locations with metadata.

    Each hotspot includes:
    - latitude, longitude
    - plastic_density  [0, 1]
    - level            critical | high | moderate
    - accumulation_trend increasing | stable | decreasing
    - movement_vector  {u, v} in m/s
    - source_estimate  nearest known terrestrial/coastal source region
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="Data not yet available. Run the data pipeline first.")
    return state.hotspots


@app.get("/predictions/{timestep}", tags=["Debris"])
def get_predictions(timestep: int) -> dict[str, Any]:
    """
    Return per-node plastic density for a specific time step.

    timestep : integer in [0, T-1]  (T ≈ 365 for 90-day simulation at 6h resolution)
    """
    if state.predictions is None or state.nodes is None:
        raise HTTPException(status_code=503, detail="Prediction data not available.")

    T = state.predictions.shape[1]
    if not (0 <= timestep < T):
        raise HTTPException(
            status_code=400,
            detail=f"timestep must be in [0, {T - 1}]. Got {timestep}."
        )

    densities = state.predictions[:, timestep]   # (N,)
    lats      = state.nodes[:, 0]
    lons      = state.nodes[:, 1]

    nodes_out = [
        {"lat": round(float(lats[i]), 4), "lon": round(float(lons[i]), 4),
         "density": round(float(densities[i]), 4)}
        for i in range(len(lats))
        if densities[i] > 0.05   # skip negligible-density nodes for bandwidth
    ]

    return {
        "timestep": timestep,
        "total_timesteps": int(T),
        "nodes": nodes_out,
    }


@app.get("/currents", tags=["Oceanography"])
def get_currents() -> dict[str, Any]:
    """
    Return ocean current vectors (u, v in m/s) sampled at each graph node.
    """
    if state.nodes is None or state.currents is None:
        raise HTTPException(status_code=503, detail="Current data not available.")

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_grid = np.linspace(-90 + RESOLUTION / 2, 90 - RESOLUTION / 2, n_lat)
    lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon)

    nodes_out = []
    for i in range(len(state.nodes)):
        lat = float(state.nodes[i, 0])
        lon = float(state.nodes[i, 1])
        # Index into current grid
        il = int(np.clip(round((lat + 90 - RESOLUTION / 2) / RESOLUTION), 0, n_lat - 1))
        jl = int(np.clip(round((lon + 180 - RESOLUTION / 2) / RESOLUTION), 0, n_lon - 1))
        u  = float(state.currents[il, jl, 0])
        v  = float(state.currents[il, jl, 1])
        speed = float(np.sqrt(u**2 + v**2))
        if speed > 0.005:   # skip near-zero currents for bandwidth
            nodes_out.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "u":   round(u, 4),
                "v":   round(v, 4),
            })

    return {"nodes": nodes_out}
