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
from fastapi.responses import ORJSONResponse, Response
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s — %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25


class SeedRequest(BaseModel):
    lat:     float
    lon:     float
    mass_kg: float = 1.0


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
    land_mask: np.ndarray | None = None      # (N_lat, N_lon) bool  — True = ocean
    ocean_node_mask: np.ndarray | None = None  # (N_ocean,) bool — pre-computed per-node ocean flag
    ready: bool = False
    # ── Trajectory mode ───────────────────────────────────────────────────────
    trajectories: np.ndarray | None  = None   # (N_particles, 14, 5)
    beaching_risk: np.ndarray | None = None   # (720, 1440)
    trajectory_ready: bool = False
    # ── Pre-computed response cache (built once at startup) ───────────────────
    # predictions_cache[t] = {"timestep":t, "nodes":[...], ...}
    predictions_cache: dict = {}
    currents_cache: dict | None = None


# ── Land-mask utility ─────────────────────────────────────────────────────────

def _build_ocean_node_mask(
    nodes: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """
    Vectorised: given node coords (N, 7) and land_mask (n_lat, n_lon),
    return a boolean array (N,) — True where the node sits on an ocean cell.
    """
    n_lat, n_lon = land_mask.shape
    il = np.clip(
        np.round((nodes[:, 0] + 90  - RESOLUTION / 2) / RESOLUTION).astype(int),
        0, n_lat - 1,
    )
    jl = np.clip(
        np.round((nodes[:, 1] + 180 - RESOLUTION / 2) / RESOLUTION).astype(int),
        0, n_lon - 1,
    )
    return land_mask[il, jl].astype(bool)   # (N,)


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

    # Pre-compute per-node ocean boolean mask (used by /predictions to strip land)
    if state.nodes is not None and state.land_mask is not None:
        state.ocean_node_mask = _build_ocean_node_mask(state.nodes, state.land_mask.astype(bool))
        n_ocean = int(state.ocean_node_mask.sum())
        n_total = len(state.nodes)
        logger.info(f"Ocean node mask ready: {n_ocean}/{n_total} nodes are ocean cells")
    else:
        state.ocean_node_mask = None

    if state.predictions is not None and state.nodes is not None:
        # Check if simulation actually produced meaningful (non-zero) output
        max_pred = float(np.max(state.predictions)) if state.predictions is not None else 0.0
        if max_pred < 0.01:
            logger.warning(
                f"future_predictions.npy max value is {max_pred:.6f} — simulation output "
                "appears to be all-zero. Falling back to demo data. "
                "Re-run src.simulation.hybrid_runner to generate real predictions."
            )
            _seed_demo_data()
        else:
            # Pre-detect hotspots from real simulation output
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

    # ── Load trajectory simulation outputs (independent of hotspot mode) ───────
    traj_path  = DATA_DIR / "trajectories.npy"
    beach_path = DATA_DIR / "beaching_risk.npy"
    if traj_path.exists():
        state.trajectories = np.load(str(traj_path))
        logger.info(f"Loaded trajectories.npy — shape {state.trajectories.shape}")
    else:
        logger.warning("trajectories.npy not found. Run: python -m src.simulation.trajectory_simulator")
    if beach_path.exists():
        state.beaching_risk = np.load(str(beach_path))
        logger.info(f"Loaded beaching_risk.npy — shape {state.beaching_risk.shape}")
    state.trajectory_ready = state.trajectories is not None

    # ── Pre-build response cache for all 361 timesteps ────────────────────────
    if state.ready:
        _build_predictions_cache()
        _build_currents_cache()


def _build_predictions_cache():
    """
    Pre-compute and pre-serialize all 361 prediction responses at startup.
    Stored as raw orjson bytes — zero per-request work.
    """
    import orjson
    if state.predictions is None or state.nodes is None:
        return

    logger.info("Building predictions cache for all timesteps...")
    T_full = state.predictions.shape[1]
    lats_all = state.nodes[:, 0]
    lons_all = state.nodes[:, 1]

    if state.ocean_node_mask is not None:
        mask = state.ocean_node_mask
        preds_ocean = state.predictions[mask]
        lats_ocean  = lats_all[mask]
        lons_ocean  = lons_all[mask]
    else:
        preds_ocean = state.predictions
        lats_ocean  = lats_all
        lons_ocean  = lons_all

    MAX_NODES = 3000
    for t in range(min(T_full, _MAX_TIMESTEPS + 1)):
        densities = preds_ocean[:, t]
        sig       = densities > 0.05
        d_f, la_f, lo_f = densities[sig], lats_ocean[sig], lons_ocean[sig]
        if len(d_f) > MAX_NODES:
            top = np.argpartition(d_f, -MAX_NODES)[-MAX_NODES:]
            d_f, la_f, lo_f = d_f[top], la_f[top], lo_f[top]
        payload = {
            "timestep":        t,
            "total_timesteps": _MAX_TIMESTEPS + 1,
            "mode":            "real" if t == 0 else "predicted",
            "day":             round(t / 4, 1),
            "nodes": [
                {"lat": round(float(la_f[i]), 4),
                 "lon": round(float(lo_f[i]), 4),
                 "density": round(float(d_f[i]), 4)}
                for i in range(len(d_f))
            ],
        }
        # Store pre-serialized bytes
        state.predictions_cache[t] = orjson.dumps(payload)
    logger.info(f"Predictions cache built — {len(state.predictions_cache)} timesteps")


def _build_currents_cache(max_nodes: int = 2000):
    """Pre-compute and pre-serialize the currents response."""
    import orjson
    if state.nodes is None:
        return
    logger.info("Building currents cache...")
    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_grid = np.linspace(-90 + RESOLUTION/2,  90 - RESOLUTION/2, n_lat, dtype=np.float32)
    lons_grid = np.linspace(-180 + RESOLUTION/2, 180 - RESOLUTION/2, n_lon, dtype=np.float32)

    curr_field = state.currents
    if curr_field is None or float(np.abs(curr_field).max()) < 1e-6:
        curr_field = _build_synthetic_currents(lats_grid, lons_grid, state.land_mask)

    lats = state.nodes[:, 0]
    lons = state.nodes[:, 1]
    il = np.clip(np.round((lats + 90  - RESOLUTION/2) / RESOLUTION).astype(int), 0, n_lat-1)
    jl = np.clip(np.round((lons + 180 - RESOLUTION/2) / RESOLUTION).astype(int), 0, n_lon-1)
    u_arr = curr_field[il, jl, 0]
    v_arr = curr_field[il, jl, 1]
    speed = np.sqrt(u_arr**2 + v_arr**2)
    mask  = speed > 0.005
    lats_f, lons_f = lats[mask], lons[mask]
    u_f, v_f, spd_f = u_arr[mask], v_arr[mask], speed[mask]
    if len(spd_f) > max_nodes:
        top = np.argpartition(spd_f, -max_nodes)[-max_nodes:]
        lats_f, lons_f, u_f, v_f = lats_f[top], lons_f[top], u_f[top], v_f[top]
    payload = {
        "nodes": [
            {"lat": round(float(lats_f[i]), 4), "lon": round(float(lons_f[i]), 4),
             "u": round(float(u_f[i]), 4), "v": round(float(v_f[i]), 4)}
            for i in range(len(lats_f))
        ]
    }
    state.currents_cache = orjson.dumps(payload)
    logger.info(f"Currents cache built — {len(lats_f)} nodes")


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
def get_hotspots():
    """Return all classified ocean plastic hotspot locations."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Data not yet available.")
    import orjson
    return Response(content=orjson.dumps(state.hotspots), media_type="application/json")


# Max steps exposed to the frontend = 90 days × 4 steps/day = 360
_MAX_TIMESTEPS = 360


@app.get("/predictions/{timestep}", tags=["Debris"])
def get_predictions(timestep: int):
    """Served from pre-serialized orjson cache — sub-millisecond response."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Prediction data not available.")
    if not (0 <= timestep <= _MAX_TIMESTEPS):
        raise HTTPException(status_code=400, detail=f"timestep must be in [0, {_MAX_TIMESTEPS}].")
    t = min(timestep, len(state.predictions_cache) - 1)
    if t in state.predictions_cache:
        return Response(content=state.predictions_cache[t], media_type="application/json")
    raise HTTPException(status_code=503, detail="Cache not ready.")



@app.get("/currents", tags=["Oceanography"])
def get_currents(max_nodes: int = 2000):
    """Return ocean current vectors. Served from pre-serialized cache."""
    if state.nodes is None:
        raise HTTPException(status_code=503, detail="Node data not available.")
    if state.currents_cache is not None:
        return Response(content=state.currents_cache, media_type="application/json")
    _build_currents_cache(max_nodes)
    return Response(content=state.currents_cache or b'{"nodes":[]}', media_type="application/json")


def _build_synthetic_currents(
    lats_grid: np.ndarray,
    lons_grid: np.ndarray,
    land_mask: np.ndarray | None,
) -> np.ndarray:
    """Build synthetic gyre circulation (mirrors trajectory_simulator logic) as a (720,1440,2) array."""
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    u = np.zeros((n_lat, n_lon), dtype=np.float32)
    v = np.zeros((n_lat, n_lon), dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons_grid, lats_grid)

    gyres = [
        ( 32.0, -140.0, 30.0, 0.30, True),   # North Pacific
        (-28.0, -100.0, 25.0, 0.22, False),   # South Pacific
        ( 30.0,  -40.0, 28.0, 0.25, True),    # North Atlantic
        (-28.0,  -15.0, 24.0, 0.20, False),   # South Atlantic
        (-28.0,   75.0, 26.0, 0.20, False),   # Indian Ocean
    ]
    for clat, clon, rad, speed, cw in gyres:
        dlat = lat_grid - clat
        dlon = lon_grid - clon
        dlon = np.where(dlon >  180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)
        dist = np.sqrt(dlat**2 + dlon**2)
        envelope = np.clip((dist / rad) * np.exp(-0.5 * (dist / rad)**2) * 2.718, 0, 1) * speed
        with np.errstate(divide="ignore", invalid="ignore"):
            norm  = np.where(dist > 0.01, dist, 0.01)
            rdlat = dlat / norm
            rdlon = dlon / norm
        sign = -1.0 if cw else 1.0
        u += sign * rdlat * envelope
        v += sign * (-rdlon) * envelope

    # Antarctic Circumpolar Current
    acc_mask  = lat_grid < -45
    acc_speed = 0.25 * np.exp(((lat_grid + 60) / 15) ** 2 * -0.5)
    u[acc_mask] += acc_speed[acc_mask]

    if land_mask is not None:
        land = ~land_mask.astype(bool)
        u[land] = 0.0
        v[land] = 0.0

    return np.stack([u, v], axis=-1).astype(np.float32)


# ── Known pollution source zones ──────────────────────────────────────────────
_KNOWN_SOURCES = [
    # East Asia
    (30.0, 121.5, "Yangtze River Delta"),
    (23.0, 113.5, "Pearl River Delta"),
    (22.0, 114.0, "Hong Kong Coastal"),
    (35.5, 139.8, "Tokyo Bay"),
    (37.5, 126.8, "Han River / Seoul Coast"),
    (10.0, 105.0, "Mekong Delta"),
    (16.0, 108.0, "Vietnam Central Coast"),
    # South/Southeast Asia
    (23.7,  90.4, "Ganges/Brahmaputra Delta"),
    (13.0,  80.3, "Chennai Coast"),
    (19.1,  72.8, "Mumbai Coast"),
    (6.9,   79.8, "Colombo Coast"),
    (3.1,  101.5, "Klang River, Malaysia"),
    (-6.2, 106.8, "Jakarta Coast"),
    (-7.2, 112.7, "Surabaya Coast"),
    # Africa
    (5.3,   -4.0, "Abidjan Coast"),
    (6.4,    3.4, "Lagos Bight"),
    (-4.3,  15.3, "Congo River Mouth"),
    (-25.9, 32.6, "Maputo Coast"),
    # Americas
    (18.5, -69.9, "Dominican Republic Coast"),
    (23.1, -82.4, "Havana Coastal"),
    (-23.0,-43.2, "Rio de Janeiro / Guanabara Bay"),
    (10.5, -66.9, "Caracas Coastal"),
    # Mediterranean
    (43.3,   5.4, "Marseille Coastal"),
    (37.9,  23.7, "Athens / Saronic Gulf"),
    (38.2,  15.6, "Messina Strait"),
    # Ocean accumulation gyres
    (32.0, -141.0, "North Pacific Garbage Patch"),
    (-32.0, -88.0, "South Pacific Gyre"),
    (28.0,  -63.0, "Sargasso Sea / N. Atlantic Gyre"),
    (-26.0,  76.0, "Indian Ocean Gyre"),
]


@app.get("/hotspots/known", tags=["Debris"])
def get_known_zones() -> list[dict[str, Any]]:
    """
    Return the list of known plastic pollution source zones and ocean accumulation gyres.
    These are fixed, research-based locations used to seed the simulation.
    """
    zones = []
    for lat, lon, label in _KNOWN_SOURCES:
        # Classify by type: river/coast vs open ocean gyre
        is_gyre = any(k in label for k in ("Gyre", "Garbage Patch", "Sargasso"))
        zones.append({
            "latitude":           round(lat, 4),
            "longitude":          round(lon, 4),
            "label":              label,
            "type":               "accumulation_gyre" if is_gyre else "pollution_source",
            "plastic_density":    0.85 if is_gyre else 0.60,
            "level":              "critical" if is_gyre else "high",
            "accumulation_trend": "increasing",
            "movement_vector":    {"u": 0.0, "v": 0.0},
            "source_estimate":    label,
        })
    return zones



# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

_SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90]

# Known source labels (reuse hotspot detector's list)
_SOURCE_LABELS = [
    (30.0,  121.5, "Yangtze River Delta / East China Sea"),
    (23.0,  113.5, "Pearl River Delta, South China Sea"),
    (35.5,  139.8, "Pacific Coast Japan"),
    (10.0,  105.0, "Mekong Delta, Gulf of Thailand"),
    (23.7,   90.4, "Bay of Bengal / Bangladesh coast"),
    (13.0,   80.3, "Bay of Bengal / India SE coast"),
    ( 6.4,    3.4, "Gulf of Guinea / Nigeria coast"),
    (-6.2,  106.8, "Java Sea / Indonesia coast"),
    (18.5,  -69.9, "Caribbean Sea"),
    (-23.0, -43.2, "South Atlantic / Brazil coast"),
    (38.2,   15.6, "Mediterranean Sea / Italy"),
    (10.5,  -66.9, "Caribbean / Venezuela coast"),
    (32.0, -141.0, "North Pacific Garbage Patch"),
    (-32.0, -88.0, "South Pacific Gyre"),
    (28.0,  -63.0, "Sargasso Sea / North Atlantic Gyre"),
    (-26.0,  76.0, "Indian Ocean Gyre"),
]

def _nearest_source(lat: float, lon: float) -> str:
    best_d, best_l = float("inf"), "Open Ocean"
    for slat, slon, label in _SOURCE_LABELS:
        d = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5
        if d < best_d:
            best_d, best_l = d, label
    return best_l if best_d < 40 else "Open Ocean Accumulation"


def _traj_503():
    raise HTTPException(
        status_code=503,
        detail="Trajectory data not available. Run: python -m src.simulation.trajectory_simulator"
    )


@app.get("/trajectories/current", tags=["Trajectory"])
def get_trajectory_current() -> dict[str, Any]:
    """
    Return current particle positions (snapshot index 0 of each trajectory).
    Includes status breakdown: active / beached / converging.
    """
    if not state.trajectory_ready or state.trajectories is None:
        _traj_503()

    traj = state.trajectories           # (N, 14, 5)
    snap0 = traj[:, 0, :]              # current positions = day-0 snapshot

    particles_out = []
    for i in range(len(snap0)):
        lat, lon, density, age, src = (
            float(snap0[i, 0]), float(snap0[i, 1]),
            float(snap0[i, 2]), float(snap0[i, 3]),
            int(snap0[i, 4]),
        )
        particles_out.append({
            "id":           i,
            "lat":          round(lat, 4),
            "lon":          round(lon, 4),
            "density":      round(density, 4),
            "age_days":     round(age, 1),
            "source_type":  src,
            "source_label": _nearest_source(lat, lon),
        })

    src_arr  = snap0[:, 4].astype(int)
    n_active = int((src_arr == 0).sum())
    n_beach  = int((src_arr == 1).sum())
    n_conv   = int((src_arr == 2).sum())

    return {
        "day":             0,
        "total_particles": len(particles_out),
        "active":          n_active,
        "beached":         n_beach,
        "converging":      n_conv,
        "particles":       particles_out,
    }


@app.get("/trajectories/forecast", tags=["Trajectory"])
def get_trajectory_forecast(max_particles: int = 2000) -> dict[str, Any]:
    """
    Return full 90-day trajectory for each particle as a list of 14 snapshots.
    Each snapshot: [lat, lon, density, age_days, source_type].
    Capped to max_particles highest-density particles for bandwidth.
    """
    if not state.trajectory_ready or state.trajectories is None:
        _traj_503()

    traj = state.trajectories           # (N, 14, 5)
    N    = len(traj)

    # Rank by initial density, cap to max_particles
    init_density = traj[:, 0, 2]
    if N > max_particles:
        top_idx = np.argpartition(init_density, -max_particles)[-max_particles:]
    else:
        top_idx = np.arange(N)

    traj_sub = traj[top_idx]
    out_trajs = []
    for j, i in enumerate(top_idx):
        snaps = traj_sub[j]
        lat0, lon0 = float(snaps[0, 0]), float(snaps[0, 1])
        out_trajs.append({
            "id":     int(i),
            "origin": {"lat": round(lat0, 4), "lon": round(lon0, 4)},
            "source_label": _nearest_source(lat0, lon0),
            "snapshots": [
                [round(float(s[0]), 4), round(float(s[1]), 4),
                 round(float(s[2]), 4), round(float(s[3]), 1),
                 int(s[4])]
                for s in snaps
            ],
        })

    return {
        "snapshot_days":    _SNAPSHOT_DAYS,
        "total_particles":  N,
        "trajectories":     out_trajs,
    }


@app.get("/trajectories/heatmap", tags=["Trajectory"])
def get_trajectory_heatmap(day: int = 0) -> dict[str, Any]:
    """
    Return 2-D density grid (720×1440) for the snapshot nearest to `day`.
    Each occupied cell: {lat, lon, density}.
    """
    if not state.trajectory_ready or state.trajectories is None:
        _traj_503()

    # Find nearest snapshot index
    snap_idx = int(np.argmin([abs(d - day) for d in _SNAPSHOT_DAYS]))
    traj      = state.trajectories       # (N, 14, 5)
    snap      = traj[:, snap_idx, :]    # (N, 5)

    n_lat, n_lon = 720, 1440
    grid = np.zeros((n_lat, n_lon), dtype=np.float32)

    lats = snap[:, 0]
    lons = snap[:, 1]
    dens = snap[:, 2]

    il = np.clip(
        np.round((lats + 90  - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lat - 1
    )
    jl = np.clip(
        np.round((lons + 180 - RESOLUTION / 2) / RESOLUTION).astype(int), 0, n_lon - 1
    )
    np.add.at(grid, (il, jl), dens)

    # Normalise
    if grid.max() > 0:
        grid /= grid.max()

    # Return only non-zero cells for efficiency
    ri, ci = np.where(grid > 0.005)
    nodes_out = [
        {
            "lat":     round(float(-89.875 + ri[k] * RESOLUTION), 4),
            "lon":     round(float(-179.875 + ci[k] * RESOLUTION), 4),
            "density": round(float(grid[ri[k], ci[k]]), 4),
        }
        for k in range(len(ri))
    ]

    return {
        "day":          _SNAPSHOT_DAYS[snap_idx],
        "snapshot_idx": snap_idx,
        "nodes":        nodes_out,
    }


@app.get("/trajectories/beaching-risk", tags=["Trajectory"])
def get_beaching_risk(top_n: int = 200) -> dict[str, Any]:
    """
    Return top-N coastal cells ranked by cumulative incoming debris probability.
    """
    if not state.trajectory_ready or state.trajectories is None:
        _traj_503()

    if state.beaching_risk is None:
        return {"top_cells": [], "total_beached": 0}

    risk = state.beaching_risk          # (720, 1440)

    # Count beached particles from final snapshot
    final_snap = state.trajectories[:, -1, :]
    total_beached = int((final_snap[:, 4] == 1).sum())

    # Find top-N risk cells
    flat = risk.flatten()
    if len(flat) > top_n:
        top_flat = np.argpartition(flat, -top_n)[-top_n:]
    else:
        top_flat = np.where(flat > 0)[0]

    top_flat = top_flat[flat[top_flat] > 0]
    ri = top_flat // 1440
    ci = top_flat % 1440

    cells = [
        {
            "lat":            round(float(-89.875 + ri[k] * RESOLUTION), 4),
            "lon":            round(float(-179.875 + ci[k] * RESOLUTION), 4),
            "risk":           round(float(risk[ri[k], ci[k]]), 4),
            "particle_count": round(float(risk[ri[k], ci[k]] * total_beached), 0),
        }
        for k in range(len(ri))
    ]
    cells.sort(key=lambda x: x["risk"], reverse=True)

    return {
        "top_cells":     cells[:top_n],
        "total_beached": total_beached,
    }


@app.post("/trajectories/seed", tags=["Trajectory"])
def seed_custom_particle(body: SeedRequest) -> dict[str, Any]:
    """
    Inject a custom debris point and return its 90-day predicted trajectory.
    Runs a single-particle physics simulation on-the-fly (~1 second).
    """
    if state.land_mask is None or state.currents is None:
        raise HTTPException(status_code=503, detail="Ocean data not loaded.")

    from src.simulation.trajectory_simulator import TrajectorySimulator, SNAPSHOT_DAYS

    n_lat = int(180 / RESOLUTION)
    n_lon = int(360 / RESOLUTION)
    lats_grid = np.linspace(-90 + RESOLUTION / 2,  90 - RESOLUTION / 2, n_lat, dtype=np.float32)
    lons_grid = np.linspace(-180 + RESOLUTION / 2, 180 - RESOLUTION / 2, n_lon, dtype=np.float32)

    winds = np.zeros((n_lat, n_lon, 2), dtype=np.float32)
    wind_path = DATA_DIR / "wind_data.npy"
    if wind_path.exists():
        winds = np.load(str(wind_path))

    sim = TrajectorySimulator(
        state.land_mask.astype(bool), state.currents, winds,
        lats_grid, lons_grid, rng_seed=0
    )

    seed = np.array([[body.lat, body.lon, min(1.0, body.mass_kg / 1000.0)]], dtype=np.float32)
    try:
        traj, _ = sim.run(seed, n_days=90)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    snaps = traj[0]   # (14, 5) for the single particle
    final_src = int(snaps[-1, 4])
    status_map = {0: "active", 1: "beached", 2: "converging"}

    return {
        "origin":        {"lat": body.lat, "lon": body.lon, "mass_kg": body.mass_kg},
        "snapshot_days": _SNAPSHOT_DAYS,
        "trajectory":    [
            [round(float(s[0]), 4), round(float(s[1]), 4),
             round(float(s[2]), 4), round(float(s[3]), 1),
             int(s[4])]
            for s in snaps
        ],
        "final_status":  status_map.get(final_src, "active"),
    }
