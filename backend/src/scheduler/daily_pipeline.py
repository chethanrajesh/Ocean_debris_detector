"""
daily_pipeline.py
Orchestrates the full data refresh pipeline in order:

  Step 1 — CMEMS ocean currents (primary current source)
  Step 2 — Open-Meteo wind fields
  Step 3 — SSH geostrophic currents (backup / blend)
  Step 4 — AOML debris seed nodes
  Step 5 — Sentinel-2 debris scores via GEE
  Step 6 — Preprocess: align, blend, build node graph
  Step 7 — Run 90-day Lagrangian trajectory simulation
  Step 8 — Run hybrid Lagrangian+GNN prediction (future_predictions.npy)
  Step 9 — Signal API reload

Usage
-----
  python -m src.scheduler.daily_pipeline
  python -m src.scheduler.daily_pipeline --skip-sentinel   # skip slow GEE step
  python -m src.scheduler.daily_pipeline --skip-simulation # data only
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("daily_pipeline")
DATA_DIR = ROOT / "data"


def _step(name, fn, *args, **kwargs):
    logger.info("=" * 60)
    logger.info(f"STEP: {name}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"  ✓ {name} complete in {elapsed:.1f}s")
        return True, result
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(f"  ✗ {name} FAILED in {elapsed:.1f}s: {exc}", exc_info=True)
        return False, None


def run_pipeline(
    skip_sentinel: bool = False,
    skip_simulation: bool = False,
    force_refresh: bool = False,
    max_particles: int = 5000,
) -> bool:
    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Ocean Debris Pipeline — {start.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    results = {}

    # ── Step 1: CMEMS currents ────────────────────────────────────────────────
    from src.data_pipeline.fetch_cmems import fetch_cmems_currents
    ok, _ = _step("CMEMS ocean currents", fetch_cmems_currents, force_refresh=force_refresh)
    results["cmems"] = ok

    # ── Step 2: Open-Meteo winds ──────────────────────────────────────────────
    from src.data_pipeline.fetch_noaa import fetch_openmeteo_winds
    ok, _ = _step("Open-Meteo wind fields", fetch_openmeteo_winds, force_refresh=force_refresh)
    results["winds"] = ok

    # ── Step 3: SSH geostrophic currents (blend) ──────────────────────────────
    from src.data_pipeline.fetch_noaa import fetch_ssh_geostrophic_currents
    ok, _ = _step("SSH geostrophic currents", fetch_ssh_geostrophic_currents, force_refresh=force_refresh)
    results["ssh_currents"] = ok

    # ── Step 4: AOML debris seeds ─────────────────────────────────────────────
    from src.data_pipeline.fetch_ncei import fetch_ncei_microplastics
    ok, _ = _step("NCEI Marine Microplastics (primary seeds)", fetch_ncei_microplastics,
                  force_refresh=force_refresh)
    results["ncei"] = ok

    from src.data_pipeline.fetch_aoml import fetch_aoml_seeds
    ok, _ = _step("AOML debris seeds (with NCEI fallback)", fetch_aoml_seeds,
                  force_refresh=force_refresh)
    results["aoml"] = ok

    # ── Step 5: Sentinel-2 via GEE ────────────────────────────────────────────
    if not skip_sentinel:
        from src.data_pipeline.fetch_sentinel import fetch_sentinel_debris_scores
        ok, _ = _step("Sentinel-2 debris scores", fetch_sentinel_debris_scores,
                      force_refresh=force_refresh)
        results["sentinel"] = ok
    else:
        logger.info("STEP: Sentinel-2 — SKIPPED")
        results["sentinel"] = None

    # ── Step 6: Preprocess ────────────────────────────────────────────────────
    from src.data_pipeline.preprocess import run_preprocessing
    ok, _ = _step("Preprocessing / node graph", run_preprocessing)
    results["preprocess"] = ok
    if not ok:
        logger.error("Preprocessing failed — cannot continue to simulation")
        return False

    # ── Step 7: Trajectory simulation ────────────────────────────────────────
    if not skip_simulation:
        from src.simulation.trajectory_simulator import run_trajectory_simulation
        ok, _ = _step(
            "90-day Lagrangian simulation",
            run_trajectory_simulation,
            n_days=90,
            max_particles=max_particles,
            force_reseed=force_refresh,
        )
        results["trajectory"] = ok

        # ── Step 8: Hybrid GNN prediction ────────────────────────────────────
        from src.simulation.hybrid_runner import run_simulation
        ok, _ = _step("Hybrid GNN prediction", run_simulation, n_timesteps=365)
        results["hybrid"] = ok
    else:
        logger.info("STEP: Simulation — SKIPPED")
        results["trajectory"] = results["hybrid"] = None

    # ── Step 9: Signal reload ─────────────────────────────────────────────────
    try:
        (DATA_DIR / "reload.flag").touch()
        logger.info("STEP: Reload flag written")
    except Exception as exc:
        logger.warning(f"Could not write reload.flag: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    ok_count  = sum(1 for v in results.values() if v is True)
    fail_count = sum(1 for v in results.values() if v is False)
    skip_count = sum(1 for v in results.values() if v is None)

    logger.info("=" * 60)
    logger.info(
        f"Pipeline done in {elapsed:.0f}s — "
        f"{ok_count} OK, {fail_count} FAILED, {skip_count} SKIPPED"
    )
    for k, v in results.items():
        icon = "✓" if v is True else ("✗" if v is False else "–")
        logger.info(f"  {icon} {k}")

    return fail_count == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocean Debris Daily Pipeline")
    parser.add_argument("--skip-sentinel",   action="store_true", help="Skip Sentinel-2/GEE step")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip trajectory simulation")
    parser.add_argument("--force-refresh",   action="store_true", help="Re-download all cached data")
    parser.add_argument("--max-particles",   type=int, default=5000)
    args = parser.parse_args()

    success = run_pipeline(
        skip_sentinel=args.skip_sentinel,
        skip_simulation=args.skip_simulation,
        force_refresh=args.force_refresh,
        max_particles=args.max_particles,
    )
    sys.exit(0 if success else 1)
