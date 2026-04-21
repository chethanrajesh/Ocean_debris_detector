"""
fetch_aoml.py
Fetches NOAA AOML (Atlantic Oceanographic and Meteorological Laboratory)
Global Floating Microplastics dataset and converts it into simulation seed nodes.

Data Source
-----------
NOAA AOML Surface Drifter / Microplastics concentration data available at:
  https://www.aoml.noaa.gov/phod/microplastics/
  Fallback: NCEI accession 0128141 (van Sebille et al. global accumulation zones)

Output
------
data/debris_seed_nodes.npy : (N, 3) float32 — [lat, lon, concentration_norm]
    N = number of nodes above threshold (>500 #/km² or normalised equivalent)

Notes
-----
- If the live download fails, falls back to a curated set of known high-concentration
  zones derived from published literature (Lebreton 2018, van Sebille 2015).
- concentration_norm is normalised to [0, 1] relative to the dataset max.
- When PO.DAAC CYGNSS credentials become available, this module can be swapped
  for fetch_cygnss.py without changing any downstream code.
"""
import io
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ── AOML / NCEI public endpoints ──────────────────────────────────────────────
# Primary: AOML SFER microplastics surface concentrations (CSV)
AOML_PRIMARY_URL = (
    "https://www.ncei.noaa.gov/data/oceans/microplastics/"
    "microplastics_accumulation_zones.csv"
)
# Secondary: van Sebille et al. (2015) plastic accumulation dataset via PANGAEA
PANGAEA_URL = (
    "https://store.pangaea.de/Publications/vanSebille-etal_2015/"
    "surface_concentrations.tab"
)

# Concentration threshold: nodes with concentration > this are used as seeds
CONCENTRATION_THRESHOLD = 500.0   # #/km²  (or arbitrary units — we normalise)

# ── Fallback literature-based seed locations ──────────────────────────────────
# High-confidence debris accumulation zones from published survey data:
#   Lebreton et al. (2018) – North Pacific Garbage Patch
#   Eriksen et al. (2014)  – Five global gyres
#   Maximenko et al. (2012) – Convergence zones
FALLBACK_SEEDS = [
    # North Pacific Subtropical Gyre (Great Pacific Garbage Patch)
    (32.0, -141.0, 1340.0), (31.0, -144.0, 1180.0), (29.0, -147.0, 960.0),
    (33.0, -138.0, 1050.0), (27.0, -150.0, 820.0),  (35.0, -135.0, 780.0),
    (30.0, -155.0, 690.0),  (28.0, -142.0, 1100.0), (34.0, -148.0, 890.0),
    (26.0, -145.0, 750.0),
    # South Pacific Subtropical Gyre
    (-32.0,  -88.0, 850.0), (-29.0,  -92.0, 720.0), (-35.0,  -85.0, 680.0),
    (-27.0,  -95.0, 590.0), (-33.0,  -90.0, 770.0),
    # North Atlantic Subtropical Gyre (Sargasso Sea)
    (28.0,  -63.0, 920.0),  (30.0,  -65.0, 840.0),  (26.0,  -60.0, 780.0),
    (32.0,  -58.0, 700.0),  (24.0,  -66.0, 640.0),
    # South Atlantic Subtropical Gyre
    (-28.0,  -15.0, 630.0), (-25.0,  -18.0, 580.0), (-30.0,  -12.0, 560.0),
    # Indian Ocean Subtropical Gyre
    (-26.0,   76.0, 710.0), (-23.0,   80.0, 650.0), (-28.0,   72.0, 590.0),
    (-20.0,   84.0, 540.0),
    # Western Pacific / Philippine Sea
    (18.0,  138.0, 560.0),  (15.0,  142.0, 530.0),  (20.0,  135.0, 510.0),
    # Indonesian Archipelago (high land-river input)
    (-6.0,  110.0, 880.0),  (-7.0,  107.0, 760.0),  (-5.0,  113.0, 720.0),
    (-8.0,  115.0, 690.0),
    # Bay of Bengal
    (14.0,   82.0, 620.0),  (12.0,   85.0, 580.0),  (10.0,   88.0, 540.0),
    # South China Sea
    (12.0,  114.0, 730.0),  (10.0,  112.0, 680.0),  (15.0,  117.0, 620.0),
    # Mediterranean Sea
    (36.0,   14.0, 890.0),  (38.0,   16.0, 820.0),  (34.0,   22.0, 760.0),
    (40.0,   18.0, 710.0),  (36.0,   24.0, 680.0),
    # Caribbean / Gulf of Mexico
    (20.0,  -77.0, 650.0),  (23.0,  -80.0, 610.0),  (18.0,  -74.0, 580.0),
    (25.0,  -83.0, 540.0),
    # Gulf of Guinea
    (2.0,     3.0, 590.0),  (4.0,    1.0, 550.0),   (0.0,    5.0, 520.0),
]


# ── CSV parsers ───────────────────────────────────────────────────────────────

def _parse_aoml_csv(text: str) -> pd.DataFrame:
    """
    Parse NOAA NCEI microplastics accumulation CSV.
    Expected columns (flexible — tries multiple name variants):
      latitude / lat / Latitude
      longitude / lon / Longitude
      concentration / conc / pieces_per_km2 / count
    """
    from io import StringIO
    df = pd.read_csv(StringIO(text), comment="#", skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise column names
    lat_cols  = [c for c in df.columns if "lat"  in c]
    lon_cols  = [c for c in df.columns if "lon"  in c]
    con_cols  = [c for c in df.columns if any(k in c for k in
                 ("conc", "density", "count", "piece", "number", "plastic"))]

    if not lat_cols or not lon_cols:
        raise ValueError(f"Cannot find lat/lon columns in: {list(df.columns)}")

    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    con_col = con_cols[0] if con_cols else None

    result = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "concentration": pd.to_numeric(df[con_col], errors="coerce")
                        if con_col else pd.Series(1000.0, index=df.index),
    }).dropna(subset=["lat", "lon"])

    return result


def _parse_pangaea_tab(text: str) -> pd.DataFrame:
    """
    Parse PANGAEA tab-delimited format (header lines prefixed with '*').
    """
    from io import StringIO
    lines = [l for l in text.splitlines() if not l.startswith("*")]
    if not lines:
        raise ValueError("Empty PANGAEA response")
    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    lat_col = next((c for c in df.columns if "lat" in c), None)
    lon_col = next((c for c in df.columns if "lon" in c), None)
    con_col = next((c for c in df.columns if any(k in c for k in
                    ("conc", "plastic", "piece", "density"))), None)
    if not lat_col or not lon_col:
        raise ValueError("Cannot find lat/lon in PANGAEA data")
    return pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "concentration": pd.to_numeric(df[con_col], errors="coerce")
                        if con_col else pd.Series(1000.0, index=df.index),
    }).dropna(subset=["lat", "lon"])


# ── Grid aggregation ──────────────────────────────────────────────────────────

def _aggregate_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap observations to 0.25° grid cells and sum concentrations per cell.
    Returns DataFrame with columns [lat, lon, concentration].
    """
    df = df.copy()
    df["lat_g"] = (np.round((df["lat"] + 90  - RESOLUTION / 2) / RESOLUTION)
                   * RESOLUTION) - 90  + RESOLUTION / 2
    df["lon_g"] = (np.round((df["lon"] + 180 - RESOLUTION / 2) / RESOLUTION)
                   * RESOLUTION) - 180 + RESOLUTION / 2
    df["lat_g"] = df["lat_g"].clip(-89.875, 89.875)
    df["lon_g"] = df["lon_g"].clip(-179.875, 179.875)

    grouped = (df.groupby(["lat_g", "lon_g"])["concentration"]
                 .sum()
                 .reset_index()
                 .rename(columns={"lat_g": "lat", "lon_g": "lon"}))
    return grouped


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_aoml_seeds(
    threshold: float = CONCENTRATION_THRESHOLD,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Download NOAA AOML microplastics data and produce debris seed nodes.

    Parameters
    ----------
    threshold : float
        Minimum concentration to include as a seed (default 500 #/km²).
    force_refresh : bool
        Re-download even if cached file exists.

    Returns
    -------
    seeds : (N, 3) float32 array  [lat, lon, concentration_normalised]
    """
    out_path = DATA_DIR / "debris_seed_nodes.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached AOML seeds from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame | None = None

    # ── Try NCEI primary ──────────────────────────────────────────────────────
    logger.info(f"Fetching AOML microplastics CSV from NCEI...")
    for url, parser_name, parser_fn in [
        (AOML_PRIMARY_URL,  "AOML CSV",     _parse_aoml_csv),
        (PANGAEA_URL,       "PANGAEA TAB",  _parse_pangaea_tab),
    ]:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = parser_fn(resp.text)
            logger.info(f"  Downloaded {len(df):,} rows via {parser_name}")
            break
        except Exception as exc:
            logger.warning(f"  {parser_name} failed: {exc}")

    # ── Fallback to literature seeds ──────────────────────────────────────────
    if df is None or len(df) == 0:
        logger.warning("All live sources failed — using built-in literature seed locations.")
        df = pd.DataFrame(FALLBACK_SEEDS, columns=["lat", "lon", "concentration"])

    # ── Grid aggregation + threshold ──────────────────────────────────────────
    gridded = _aggregate_to_grid(df)
    above   = gridded[gridded["concentration"] >= threshold].copy()

    if len(above) == 0:
        # Lower threshold progressively until we get at least 50 seeds
        for pct in [75, 50, 25, 10]:
            thresh_fallback = float(np.percentile(gridded["concentration"].dropna(), pct))
            above = gridded[gridded["concentration"] >= thresh_fallback].copy()
            if len(above) >= 50:
                logger.warning(f"Original threshold yielded 0 seeds; using {pct}th percentile "
                               f"({thresh_fallback:.1f}) → {len(above)} seeds")
                break

    # ── Normalise concentration to [0, 1] ────────────────────────────────────
    mx = above["concentration"].max()
    above["conc_norm"] = (above["concentration"] / mx).clip(0, 1)

    seeds = above[["lat", "lon", "conc_norm"]].values.astype(np.float32)
    np.save(str(out_path), seeds)
    logger.info(f"Saved {len(seeds):,} seed nodes → {out_path}")
    return seeds


def load_aoml_seeds() -> np.ndarray:
    """Load cached seed nodes or generate them if missing."""
    p = DATA_DIR / "debris_seed_nodes.npy"
    if p.exists():
        return np.load(str(p))
    return fetch_aoml_seeds()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    seeds = fetch_aoml_seeds(force_refresh=True)
    print(f"Seed nodes: {len(seeds):,}")
    print(f"  lat range:  [{seeds[:, 0].min():.2f}, {seeds[:, 0].max():.2f}]")
    print(f"  lon range:  [{seeds[:, 1].min():.2f}, {seeds[:, 1].max():.2f}]")
    print(f"  conc range: [{seeds[:, 2].min():.4f}, {seeds[:, 2].max():.4f}]")
