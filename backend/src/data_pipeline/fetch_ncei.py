"""
fetch_ncei.py
Fetches the NOAA NCEI Global Marine Microplastics Database (1972–present)
via the ArcGIS REST Feature Service that backs the NCEI Microplastics Portal.

Source
------
NOAA NCEI Marine Microplastics Portal:
  https://www.ncei.noaa.gov/products/microplastics
ArcGIS Feature Service (public, no auth required):
  https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/
  Marine_Microplastics_WGS84/FeatureServer/0

Dataset
-------
22,530+ in-situ observations (1972–present) with:
  - Latitude / Longitude
  - Microplastics_measurement  (concentration value)
  - Unit                       (pieces/m³, pieces/m², pieces/km², etc.)
  - Medium                     (Water Surface, Beach, Sediment, Water Column)
  - Location_Oceans            (Atlantic, Pacific, Indian, etc.)
  - Location_Regions           (sub-region label)

Output
------
data/ncei_microplastics.npy : (N, 3) float32  [lat, lon, concentration_norm]
    Only surface water observations are used as debris seeds.
    Concentration is normalised to [0, 1] relative to the dataset max.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ArcGIS Feature Service — public, no authentication required
FEATURE_SERVICE = (
    "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
    "Marine_Microplastics_WGS84/FeatureServer/0"
)

# Fields to retrieve
OUT_FIELDS = ",".join([
    "Latitude__degree_",
    "Longitude_degree_",
    "Microplastics_measurement",
    "Unit",
    "Medium",
    "Location_Oceans",
    "Location_Regions",
])

# Page size — ArcGIS default max is 1000 per request
PAGE_SIZE = 1000

# Units to normalise to pieces/m² (approximate conversion factors)
# We normalise everything to a common scale before comparing
UNIT_SCALE = {
    "pieces/m3":   1.0,       # water column — direct
    "pieces/m2":   1.0,       # surface area — comparable
    "pieces/km2":  1e-6,      # convert km² → m²
    "pieces/100m3": 0.01,     # convert 100m³ → m³
    "pieces/l":    1000.0,    # convert litres → m³
    "items/m3":    1.0,
    "items/m2":    1.0,
    "n/m3":        1.0,
    "n/m2":        1.0,
    "#/m3":        1.0,
    "#/m2":        1.0,
    "#/km2":       1e-6,
}

# Only use surface/near-surface water observations as debris seeds
# (beach and sediment data are not relevant for ocean drift simulation)
SURFACE_MEDIUMS = {
    "Water Surface",
    "Water Column",
    "Surface Water",
    "Neuston",
    "Trawl",
    "Manta Trawl",
    "Water",
}


def _fetch_all_records() -> pd.DataFrame:
    """
    Download all records from the NCEI ArcGIS Feature Service using pagination.
    Returns a DataFrame with columns: lat, lon, concentration, unit, medium, ocean, region.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (NOAA NCEI microplastics research)"})

    # First get total count
    count_url = f"{FEATURE_SERVICE}/query"
    count_params = {
        "where": "1=1",
        "returnCountOnly": "true",
        "f": "json",
    }
    r = session.get(count_url, params=count_params, timeout=15)
    r.raise_for_status()
    total = r.json().get("count", 0)
    logger.info(f"NCEI microplastics: {total:,} total records")

    all_rows = []
    offset = 0

    while offset < total:
        params = {
            "where": "1=1",
            "outFields": OUT_FIELDS,
            "f": "json",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
        }
        try:
            r = session.get(f"{FEATURE_SERVICE}/query", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])
            if not features:
                break

            for feat in features:
                a = feat.get("attributes", {})
                lat  = a.get("Latitude__degree_")
                lon  = a.get("Longitude_degree_")
                conc = a.get("Microplastics_measurement")
                unit = (a.get("Unit") or "").strip().lower()
                med  = (a.get("Medium") or "").strip()
                ocean  = (a.get("Location_Oceans") or "").strip()
                region = (a.get("Location_Regions") or "").strip()

                if lat is None or lon is None:
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue

                all_rows.append({
                    "lat":    float(lat),
                    "lon":    float(lon),
                    "conc":   float(conc) if conc is not None else None,
                    "unit":   unit,
                    "medium": med,
                    "ocean":  ocean,
                    "region": region,
                })

            offset += len(features)
            logger.info(f"  Downloaded {offset:,}/{total:,} records")

        except Exception as exc:
            logger.warning(f"  Page at offset {offset} failed: {exc}")
            offset += PAGE_SIZE  # skip failed page and continue

    logger.info(f"Downloaded {len(all_rows):,} valid records")
    return pd.DataFrame(all_rows)


def _normalise_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all concentration values to a common scale (pieces/m²-equivalent)
    then normalise to [0, 1].
    """
    df = df.copy()

    # Apply unit conversion
    def _scale(row):
        if row["conc"] is None or pd.isna(row["conc"]):
            return None
        unit = row["unit"]
        # Try exact match first, then partial match
        scale = UNIT_SCALE.get(unit)
        if scale is None:
            for key, val in UNIT_SCALE.items():
                if key in unit:
                    scale = val
                    break
        if scale is None:
            scale = 1.0  # unknown unit — use as-is
        return float(row["conc"]) * scale

    df["conc_scaled"] = df.apply(_scale, axis=1)
    df = df.dropna(subset=["conc_scaled"])
    df = df[df["conc_scaled"] > 0]

    # Normalise to [0, 1]
    mx = df["conc_scaled"].max()
    if mx > 0:
        df["conc_norm"] = (df["conc_scaled"] / mx).clip(0, 1)
    else:
        df["conc_norm"] = 0.0

    return df


def _aggregate_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Snap to 0.25° grid and take max concentration per cell."""
    df = df.copy()
    df["lat_g"] = (np.round((df["lat"] + 90  - RESOLUTION / 2) / RESOLUTION)
                   * RESOLUTION) - 90  + RESOLUTION / 2
    df["lon_g"] = (np.round((df["lon"] + 180 - RESOLUTION / 2) / RESOLUTION)
                   * RESOLUTION) - 180 + RESOLUTION / 2
    df["lat_g"] = df["lat_g"].clip(-89.875, 89.875)
    df["lon_g"] = df["lon_g"].clip(-179.875, 179.875)

    # Use max concentration per cell (most severe observation wins)
    gridded = (df.groupby(["lat_g", "lon_g"])["conc_norm"]
                 .max()
                 .reset_index()
                 .rename(columns={"lat_g": "lat", "lon_g": "lon",
                                  "conc_norm": "concentration"}))
    return gridded


def fetch_ncei_microplastics(
    surface_only: bool = True,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Download the full NOAA NCEI Marine Microplastics database and convert
    to debris seed nodes on a 0.25° grid.

    Parameters
    ----------
    surface_only : bool
        If True (default), only use surface water observations.
        Beach and sediment data are excluded as they don't represent
        ocean-drifting debris.
    force_refresh : bool
        Re-download even if cached file exists.

    Returns
    -------
    seeds : (N, 3) float32  [lat, lon, concentration_norm]
    """
    out_path = DATA_DIR / "ncei_microplastics.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached NCEI microplastics from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching NOAA NCEI Marine Microplastics database...")
    df = _fetch_all_records()

    if df.empty:
        logger.error("No records downloaded from NCEI")
        return np.zeros((0, 3), dtype=np.float32)

    logger.info(f"Raw records: {len(df):,}")
    logger.info(f"Mediums: {df['medium'].value_counts().head(10).to_dict()}")
    logger.info(f"Oceans:  {df['ocean'].value_counts().to_dict()}")

    # Filter to surface water only
    if surface_only:
        surface_mask = df["medium"].apply(
            lambda m: any(s.lower() in m.lower() for s in SURFACE_MEDIUMS)
            if isinstance(m, str) else False
        )
        df_surface = df[surface_mask].copy()
        logger.info(f"Surface water records: {len(df_surface):,} / {len(df):,}")
        if len(df_surface) < 100:
            logger.warning("Too few surface records — using all mediums")
            df_surface = df.copy()
    else:
        df_surface = df.copy()

    # Normalise concentrations
    df_norm = _normalise_concentration(df_surface)
    logger.info(f"Records with valid concentration: {len(df_norm):,}")

    # Aggregate to 0.25° grid
    gridded = _aggregate_to_grid(df_norm)
    logger.info(f"Grid cells: {len(gridded):,}")

    seeds = gridded[["lat", "lon", "concentration"]].values.astype(np.float32)
    np.save(str(out_path), seeds)
    logger.info(
        f"Saved ncei_microplastics.npy — {len(seeds):,} seed nodes  "
        f"lat=[{seeds[:,0].min():.1f}, {seeds[:,0].max():.1f}]  "
        f"lon=[{seeds[:,1].min():.1f}, {seeds[:,1].max():.1f}]"
    )
    return seeds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    seeds = fetch_ncei_microplastics(force_refresh=True)
    print(f"\nNCEI seed nodes: {len(seeds):,}")
    print(f"  lat range:  [{seeds[:,0].min():.2f}, {seeds[:,0].max():.2f}]")
    print(f"  lon range:  [{seeds[:,1].min():.2f}, {seeds[:,1].max():.2f}]")
    print(f"  conc range: [{seeds[:,2].min():.4f}, {seeds[:,2].max():.4f}]")
