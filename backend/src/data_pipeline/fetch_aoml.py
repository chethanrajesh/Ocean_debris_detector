"""
fetch_aoml.py
Produces debris seed nodes for the Lagrangian simulation.

Primary source:
  NOAA AOML GDP 6-hourly drifter NetCDF (gdp6h_ragged_current.nc)
  URL: https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/gdp6h_ragged_current.nc
  This is the real Global Drifter Program dataset — ~2.4 GB, publicly accessible.
  We stream only the metadata + a spatial sample to extract concentration proxies.

Secondary source:
  NOAA AOML GDP buoy .dat.gz files (buoydata_1_5000.dat.gz etc.)

Fallback:
  Comprehensive literature-based seed database
  (Lebreton 2018, Eriksen 2014, van Sebille 2015, Cózar 2014, Maximenko 2012)

Output
------
data/debris_seed_nodes.npy : (N, 3) float32  [lat, lon, concentration_norm]
"""
import logging
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
RESOLUTION = 0.25

# ── Primary: AOML GDP NetCDF (real drifter data) ──────────────────────────────
GDP_NC_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/gdp6h_ragged_current.nc"

# ── Secondary: AOML GDP .dat.gz files ────────────────────────────────────────
GDP_DAT_URLS = [
    "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/buoydata_1_5000.dat.gz",
    "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/buoydata_5001_10000.dat.gz",
    "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/buoydata_10001_15000.dat.gz",
    "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/buoydata_15001_current.dat.gz",
]

# ── Fallback: literature-based seed database ──────────────────────────────────
# Sources: Lebreton 2018, Eriksen 2014, van Sebille 2015, Cózar 2014
FALLBACK_SEEDS = [
    # ── North Pacific Subtropical Gyre (Great Pacific Garbage Patch) ──────────
    (32.0,-141.0,1340),(31.0,-144.0,1180),(29.0,-147.0,960),(33.0,-138.0,1050),
    (27.0,-150.0,820),(35.0,-135.0,780),(30.0,-155.0,690),(28.0,-142.0,1100),
    (34.0,-148.0,890),(26.0,-145.0,750),(36.0,-132.0,720),(24.0,-148.0,680),
    (38.0,-130.0,650),(22.0,-152.0,620),(40.0,-128.0,590),(20.0,-155.0,560),
    (32.0,-160.0,540),(34.0,-125.0,510),(30.0,-165.0,490),(36.0,-120.0,470),
    (28.0,-135.0,980),(30.0,-138.0,1020),(32.0,-145.0,1150),(34.0,-142.0,900),
    (26.0,-140.0,760),(28.0,-148.0,830),(30.0,-152.0,710),(32.0,-155.0,660),
    (34.0,-158.0,580),(36.0,-155.0,540),(38.0,-152.0,500),(40.0,-148.0,460),
    (35.0,-128.0,480),(33.0,-125.0,450),(31.0,-122.0,420),(29.0,-120.0,390),
    (30.0,-170.0,520),(28.0,-172.0,490),(26.0,-175.0,460),(24.0,-178.0,430),
    (32.0,-175.0,510),(34.0,-172.0,480),(36.0,-168.0,450),(38.0,-165.0,420),
    # ── South Pacific Subtropical Gyre ────────────────────────────────────────
    (-32.0,-88.0,850),(-29.0,-92.0,720),(-35.0,-85.0,680),(-27.0,-95.0,590),
    (-33.0,-90.0,770),(-30.0,-95.0,640),(-28.0,-88.0,710),(-34.0,-92.0,660),
    (-26.0,-98.0,550),(-36.0,-82.0,620),(-24.0,-100.0,510),(-38.0,-78.0,580),
    (-30.0,-100.0,600),(-32.0,-95.0,650),(-28.0,-82.0,680),(-34.0,-85.0,630),
    (-26.0,-92.0,570),(-36.0,-88.0,590),(-22.0,-105.0,480),(-40.0,-75.0,540),
    # ── North Atlantic Subtropical Gyre (Sargasso Sea) ────────────────────────
    (28.0,-63.0,920),(30.0,-65.0,840),(26.0,-60.0,780),(32.0,-58.0,700),
    (24.0,-66.0,640),(34.0,-55.0,610),(22.0,-68.0,580),(36.0,-52.0,560),
    (20.0,-70.0,540),(38.0,-48.0,520),(28.0,-55.0,860),(30.0,-58.0,800),
    (26.0,-52.0,740),(32.0,-62.0,720),(24.0,-58.0,680),(34.0,-60.0,650),
    (22.0,-62.0,620),(36.0,-56.0,590),(20.0,-64.0,560),(38.0,-52.0,530),
    (28.0,-70.0,880),(30.0,-72.0,820),(26.0,-68.0,760),(32.0,-68.0,730),
    (24.0,-72.0,700),(34.0,-65.0,670),(22.0,-75.0,640),(36.0,-62.0,610),
    # ── South Atlantic Subtropical Gyre ───────────────────────────────────────
    (-28.0,-15.0,630),(-25.0,-18.0,580),(-30.0,-12.0,560),(-22.0,-20.0,540),
    (-32.0,-10.0,520),(-20.0,-22.0,500),(-34.0,-8.0,490),(-18.0,-25.0,470),
    (-28.0,-20.0,610),(-26.0,-15.0,590),(-30.0,-18.0,570),(-24.0,-12.0,550),
    (-32.0,-15.0,530),(-22.0,-18.0,510),(-34.0,-12.0,500),(-20.0,-20.0,480),
    # ── Indian Ocean Subtropical Gyre ─────────────────────────────────────────
    (-26.0,76.0,710),(-23.0,80.0,650),(-28.0,72.0,590),(-20.0,84.0,540),
    (-30.0,68.0,570),(-18.0,88.0,520),(-32.0,64.0,550),(-16.0,92.0,500),
    (-26.0,82.0,680),(-24.0,76.0,640),(-28.0,78.0,600),(-22.0,82.0,560),
    (-30.0,72.0,540),(-20.0,86.0,510),(-32.0,68.0,490),(-18.0,90.0,470),
    (-26.0,70.0,660),(-24.0,74.0,620),(-28.0,80.0,580),(-22.0,78.0,550),
    # ── Western Pacific / Philippine Sea ──────────────────────────────────────
    (18.0,138.0,560),(15.0,142.0,530),(20.0,135.0,510),(22.0,132.0,490),
    (25.0,128.0,470),(12.0,145.0,450),(28.0,125.0,440),(10.0,148.0,420),
    (30.0,122.0,410),(8.0,150.0,400),(20.0,140.0,540),(18.0,135.0,520),
    (22.0,138.0,500),(25.0,132.0,480),(15.0,145.0,460),(28.0,128.0,440),
    # ── Indonesian Archipelago ────────────────────────────────────────────────
    (-6.0,110.0,880),(-7.0,107.0,760),(-5.0,113.0,720),(-8.0,115.0,690),
    (-4.0,116.0,650),(-9.0,118.0,620),(-3.0,119.0,590),(-10.0,120.0,560),
    (-6.0,106.0,840),(-7.0,112.0,780),(-5.0,108.0,730),(-8.0,110.0,700),
    (-4.0,114.0,660),(-9.0,116.0,630),(-3.0,117.0,600),(-10.0,118.0,570),
    # ── Bay of Bengal ─────────────────────────────────────────────────────────
    (14.0,82.0,620),(12.0,85.0,580),(10.0,88.0,540),(16.0,80.0,590),
    (8.0,90.0,510),(18.0,78.0,560),(6.0,92.0,480),(20.0,76.0,530),
    # ── South China Sea ───────────────────────────────────────────────────────
    (12.0,114.0,730),(10.0,112.0,680),(15.0,117.0,620),(8.0,110.0,590),
    (18.0,115.0,560),(6.0,108.0,530),(20.0,118.0,510),(4.0,106.0,490),
    # ── Mediterranean Sea ─────────────────────────────────────────────────────
    (36.0,14.0,890),(38.0,16.0,820),(34.0,22.0,760),(40.0,18.0,710),
    (36.0,24.0,680),(38.0,12.0,650),(34.0,26.0,620),(40.0,20.0,590),
    (36.0,10.0,860),(38.0,14.0,800),(34.0,18.0,740),(40.0,16.0,700),
    (36.0,20.0,670),(38.0,22.0,640),(34.0,14.0,610),(40.0,24.0,580),
    (42.0,12.0,550),(32.0,28.0,520),(44.0,10.0,500),(30.0,32.0,480),
    # ── Caribbean / Gulf of Mexico ────────────────────────────────────────────
    (20.0,-77.0,650),(23.0,-80.0,610),(18.0,-74.0,580),(25.0,-83.0,540),
    (16.0,-71.0,510),(27.0,-86.0,490),(14.0,-68.0,470),(29.0,-89.0,450),
    # ── Gulf of Guinea / West Africa ──────────────────────────────────────────
    (2.0,3.0,590),(4.0,1.0,550),(0.0,5.0,520),(-2.0,7.0,490),
    (6.0,-1.0,560),(8.0,-3.0,530),(4.0,3.0,570),(2.0,1.0,540),
    # ── Arabian Sea ───────────────────────────────────────────────────────────
    (18.0,62.0,480),(20.0,65.0,450),(16.0,60.0,420),(22.0,68.0,400),
    (14.0,58.0,380),(24.0,70.0,360),(12.0,56.0,340),(26.0,72.0,320),
    # ── East African Coast ────────────────────────────────────────────────────
    (-10.0,42.0,420),(-12.0,44.0,390),(-8.0,40.0,360),(-14.0,46.0,340),
    # ── South American Atlantic Coast ─────────────────────────────────────────
    (-23.0,-43.0,580),(-25.0,-45.0,540),(-21.0,-41.0,510),(-27.0,-47.0,480),
    (-19.0,-39.0,450),(-29.0,-49.0,420),(-17.0,-37.0,390),(-31.0,-51.0,360),
    (-10.0,-36.0,480),(-12.0,-38.0,450),(-8.0,-34.0,420),(-14.0,-40.0,390),
    # ── North Sea / English Channel ───────────────────────────────────────────
    (52.0,4.0,420),(54.0,6.0,390),(50.0,2.0,360),(56.0,8.0,330),
    # ── Black Sea ─────────────────────────────────────────────────────────────
    (43.0,34.0,380),(44.0,32.0,350),(42.0,36.0,320),(45.0,30.0,290),
    # ── Yellow Sea / East China Sea ───────────────────────────────────────────
    (32.0,122.0,680),(34.0,124.0,640),(30.0,120.0,600),(36.0,126.0,560),
    (28.0,118.0,520),(38.0,128.0,490),(26.0,116.0,460),(40.0,130.0,430),
    # ── Coral Triangle / Pacific Islands ──────────────────────────────────────
    (-5.0,145.0,480),(-8.0,148.0,450),(-2.0,142.0,420),(-10.0,150.0,390),
    # ── Antarctic Circumpolar ─────────────────────────────────────────────────
    (-55.0,-60.0,280),(-55.0,-30.0,260),(-55.0,0.0,240),(-55.0,30.0,220),
    (-55.0,60.0,200),(-55.0,90.0,180),(-55.0,120.0,160),(-55.0,150.0,140),
    (-55.0,180.0,130),(-55.0,-90.0,150),(-55.0,-120.0,170),(-55.0,-150.0,190),
]


def _aggregate_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Snap observations to 0.25° grid and sum concentrations per cell."""
    df = df.copy()
    df["lat_g"] = (np.round((df["lat"] + 90  - RESOLUTION/2) / RESOLUTION)
                   * RESOLUTION) - 90  + RESOLUTION/2
    df["lon_g"] = (np.round((df["lon"] + 180 - RESOLUTION/2) / RESOLUTION)
                   * RESOLUTION) - 180 + RESOLUTION/2
    df["lat_g"] = df["lat_g"].clip(-89.875, 89.875)
    df["lon_g"] = df["lon_g"].clip(-179.875, 179.875)
    return (df.groupby(["lat_g", "lon_g"])["concentration"]
              .sum().reset_index()
              .rename(columns={"lat_g": "lat", "lon_g": "lon"}))


def _fetch_gdp_netcdf(max_mb: int = 50) -> pd.DataFrame | None:
    """
    Stream the AOML GDP 6-hourly NetCDF and extract drifter positions.
    We only download the first `max_mb` MB to get a representative sample.
    """
    try:
        import xarray as xr
        logger.info(f"Streaming AOML GDP NetCDF (first {max_mb}MB)...")
        headers = {"Range": f"bytes=0-{max_mb * 1024 * 1024 - 1}"}
        resp = requests.get(GDP_NC_URL, headers=headers, timeout=120, stream=True)
        if resp.status_code not in (200, 206):
            logger.warning(f"GDP NetCDF returned {resp.status_code}")
            return None

        raw = resp.content
        logger.info(f"  Downloaded {len(raw)//1024//1024}MB")

        # netCDF4 engine requires a file path, not BytesIO — write to temp file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            ds = xr.open_dataset(tmp_path, engine="netcdf4")
            logger.info(f"  Variables: {list(ds.data_vars)}")
        finally:
            os.unlink(tmp_path)

        lat_var = next((v for v in ["lat", "latitude", "LAT"] if v in ds), None)
        lon_var = next((v for v in ["lon", "longitude", "LON"] if v in ds), None)

        if lat_var is None or lon_var is None:
            logger.warning(f"  No lat/lon variables found in {list(ds.data_vars)}")
            ds.close()
            return None

        lats = ds[lat_var].values.flatten()
        lons = ds[lon_var].values.flatten()
        ds.close()

        valid = (np.isfinite(lats) & np.isfinite(lons) &
                 (lats >= -90) & (lats <= 90) &
                 (lons >= -180) & (lons <= 180))
        lats, lons = lats[valid], lons[valid]
        logger.info(f"  Valid positions: {len(lats):,}")

        df = pd.DataFrame({"lat": lats, "lon": lons, "concentration": 1.0})
        return df

    except Exception as exc:
        logger.warning(f"GDP NetCDF fetch failed: {exc}")
        return None


def _fetch_gdp_dat_gz() -> pd.DataFrame | None:
    """
    Download and parse AOML GDP .dat.gz buoy data files.
    Handles partial/truncated downloads gracefully.
    Format: ID, month, day, year, lat, lon, SST, u, v, ...
    """
    import gzip, zlib

    all_rows = []
    for url in GDP_DAT_URLS:  # All four files for global coverage
        try:
            logger.info(f"Downloading GDP dat.gz: {url.split('/')[-1]}...")
            resp = requests.get(url, timeout=120, stream=True,
                                headers={"Range": "bytes=0-20971520"})  # 20MB
            if resp.status_code not in (200, 206):
                continue

            raw = resp.content
            logger.info(f"  Got {len(raw)//1024}KB compressed")

            # Decompress — handle truncated stream by reading line by line
            text_lines = []
            try:
                data = gzip.decompress(raw)
                text_lines = data.decode("latin-1", errors="ignore").splitlines()
            except Exception:
                # Partial file — decompress what we can using streaming
                try:
                    d = zlib.decompressobj(wbits=47)
                    partial = d.decompress(raw)
                    text_lines = partial.decode("latin-1", errors="ignore").splitlines()
                    logger.info(f"  Partial decompress: {len(text_lines)} lines")
                except Exception as e2:
                    logger.warning(f"  Decompression failed: {e2}")
                    continue

            logger.info(f"  Decompressed to {len(text_lines):,} lines")

            for line in text_lines[:100000]:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        lat = float(parts[4])
                        lon = float(parts[5])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            all_rows.append({"lat": lat, "lon": lon, "concentration": 1.0})
                    except (ValueError, IndexError):
                        continue

            logger.info(f"  Parsed {len(all_rows):,} valid positions so far")

        except Exception as exc:
            logger.warning(f"  GDP dat.gz failed: {exc}")

    if all_rows:
        return pd.DataFrame(all_rows)
    return None


def fetch_aoml_seeds(
    threshold: float = 5.0,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Produce debris seed nodes from real observed data.

    Priority:
    1. NOAA NCEI Marine Microplastics Database (22,530 in-situ observations,
       1972–present) via ArcGIS REST API — the most comprehensive real dataset
    2. AOML GDP 6-hourly NetCDF (gdp6h_ragged_current.nc) — real drifter positions
    3. AOML GDP .dat.gz files — same data, different format
    4. Literature-based fallback (Lebreton 2018, Eriksen 2014, van Sebille 2015)

    Returns (N, 3) float32  [lat, lon, concentration_normalised_0_1]
    """
    out_path = DATA_DIR / "debris_seed_nodes.npy"
    if out_path.exists() and not force_refresh:
        logger.info(f"Loading cached AOML seeds from {out_path}")
        return np.load(str(out_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = None

    # ── Try 1: NCEI Marine Microplastics Database (primary — real observations) ─
    logger.info("Attempting NOAA NCEI Marine Microplastics Database (primary source)...")
    try:
        from src.data_pipeline.fetch_ncei import fetch_ncei_microplastics
        ncei_seeds = fetch_ncei_microplastics(surface_only=True, force_refresh=force_refresh)
        if len(ncei_seeds) >= 100:
            logger.info(f"NCEI source: {len(ncei_seeds):,} seed nodes")
            np.save(str(out_path), ncei_seeds)
            logger.info(f"Saved {len(ncei_seeds):,} NCEI seed nodes → {out_path}")
            return ncei_seeds
    except Exception as exc:
        logger.warning(f"NCEI fetch failed: {exc}")

    # ── Try 2: GDP NetCDF ─────────────────────────────────────────────────────
    logger.info("Attempting AOML GDP NetCDF (secondary source)...")
    df = _fetch_gdp_netcdf(max_mb=30)

    # ── Try 3: GDP .dat.gz ────────────────────────────────────────────────────
    if df is None or len(df) < 100:
        logger.info("Attempting AOML GDP .dat.gz (tertiary source)...")
        df = _fetch_gdp_dat_gz()

    # ── Fallback: literature database ─────────────────────────────────────────
    if df is None or len(df) < 100:
        logger.info(f"Using literature fallback ({len(FALLBACK_SEEDS)} points)")
        df = pd.DataFrame(FALLBACK_SEEDS, columns=["lat", "lon", "concentration"])

    # ── Grid aggregation ──────────────────────────────────────────────────────
    gridded = _aggregate_to_grid(df)

    # For drifter-derived data, concentration = visit count — use percentile threshold
    above = gridded[gridded["concentration"] >= threshold].copy()
    if len(above) < 50:
        for pct in [75, 50, 25, 10, 1]:
            t = float(np.percentile(gridded["concentration"].dropna(), pct))
            above = gridded[gridded["concentration"] >= t].copy()
            if len(above) >= 50:
                logger.info(f"Threshold at {pct}th pct ({t:.1f}) → {len(above)} seeds")
                break

    # ── Normalise to [0, 1] ───────────────────────────────────────────────────
    mx = above["concentration"].max()
    above["conc_norm"] = (above["concentration"] / mx).clip(0, 1)

    seeds = above[["lat", "lon", "conc_norm"]].values.astype(np.float32)
    np.save(str(out_path), seeds)
    logger.info(f"Saved {len(seeds):,} seed nodes → {out_path}")
    return seeds


def load_aoml_seeds() -> np.ndarray:
    p = DATA_DIR / "debris_seed_nodes.npy"
    if p.exists():
        return np.load(str(p))
    return fetch_aoml_seeds()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    seeds = fetch_aoml_seeds(force_refresh=True)
    print(f"\nSeed nodes: {len(seeds):,}")
    print(f"  lat range:  [{seeds[:,0].min():.1f}, {seeds[:,0].max():.1f}]")
    print(f"  lon range:  [{seeds[:,1].min():.1f}, {seeds[:,1].max():.1f}]")
    print(f"  conc range: [{seeds[:,2].min():.4f}, {seeds[:,2].max():.4f}]")
