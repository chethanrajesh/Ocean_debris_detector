"""
daily_pipeline.py
Chains all data refresh steps for daily automated execution.

Steps
-----
1. Download latest AOML microplastics (falls back to cache)
2. Download latest CMEMS 7-day ocean current forecast
3. Download latest GFS 00Z wind fields (via fetch_noaa.py)
4. Re-seed particles from latest AOML/CYGNSS snapshot
5. Run 90-day Lagrangian trajectory simulation
6. Touch data/reload.flag so the FastAPI server hot-reloads trajectory data

Usage
-----
  python -m src.scheduler.daily_pipeline
  python -m src.scheduler.daily_pipeline --no-currents  # skip CMEMS/GFS fetch

Scheduling (Windows Task Scheduler)
-------------------------------------
  Register via: powershell -File backend/schedule_task.ps1
  Or use APScheduler: python -m src.scheduler.daily_pipeline --daemon
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend root to path when running as __main__
ROOT = Path(__file__).parent.parent.parent  # → backend/
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("daily_pipeline")

DATA_DIR = ROOT / "data"


def step_fetch_aoml():
    """Step 1 — AOML microplastics seed layer"""
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching AOML microplastics seed locations...")
    try:
        from src.data_pipeline.fetch_aoml import fetch_aoml_seeds
        seeds = fetch_aoml_seeds(force_refresh=True)
        logger.info(f"  ✓ AOML seeds: {len(seeds):,} nodes")
        return True
    except Exception as exc:
        logger.error(f"  ✗ AOML fetch failed: {exc}")
        return False


def step_fetch_cmems():
    """Step 2 — CMEMS 7-day ocean current forecast"""
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching CMEMS ocean current forecast...")
    try:
        from src.data_pipeline.fetch_cmems import fetch_cmems_analysis
        curr = fetch_cmems_analysis()
        logger.info(f"  ✓ CMEMS currents shape: {curr.shape}")
        return True
    except Exception as exc:
        logger.warning(f"  ✗ CMEMS fetch failed ({exc}) — using cached data")
        return False


def step_fetch_gfs():
    """Step 3 — GFS 00Z wind forecast"""
    logger.info("=" * 60)
    logger.info("STEP 3: Fetching GFS wind data...")
    try:
        from src.data_pipeline.fetch_noaa import fetch_gfs_winds
        winds = fetch_gfs_winds()
        logger.info(f"  ✓ GFS winds shape: {winds.shape}")
        return True
    except Exception as exc:
        logger.warning(f"  ✗ GFS fetch failed ({exc}) — using cached data")
        return False


def step_run_simulation(max_particles: int = 5000, n_days: int = 90):
    """Step 4+5 — Re-seed and run 90-day Lagrangian simulation"""
    logger.info("=" * 60)
    logger.info(f"STEP 4-5: Running {n_days}-day Lagrangian simulation ({max_particles:,} particles)...")
    try:
        from src.simulation.trajectory_simulator import run_trajectory_simulation
        t0 = time.time()
        traj, risk = run_trajectory_simulation(
            n_days=n_days,
            max_particles=max_particles,
            force_reseed=False,
        )
        elapsed = time.time() - t0
        logger.info(f"  ✓ Simulation complete in {elapsed:.1f}s")
        logger.info(f"  ✓ trajectories.npy: {traj.shape}")
        logger.info(f"  ✓ beaching_risk.npy: {risk.shape}")
        return True
    except Exception as exc:
        logger.error(f"  ✗ Simulation failed: {exc}", exc_info=True)
        return False


def step_signal_reload():
    """Step 6 — Touch reload.flag so FastAPI reloads trajectories"""
    logger.info("=" * 60)
    logger.info("STEP 6: Signalling API to reload trajectory data...")
    try:
        flag = DATA_DIR / "reload.flag"
        flag.touch()
        logger.info(f"  ✓ Touched {flag}")
        return True
    except Exception as exc:
        logger.warning(f"  ✗ Could not write reload.flag: {exc}")
        return False


def run_pipeline(fetch_currents: bool = True, max_particles: int = 5000):
    start = datetime.now()
    logger.info(f"{'=' * 60}")
    logger.info(f"Ocean Debris Daily Pipeline — {start.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"{'=' * 60}")

    results = {}
    results["aoml"]       = step_fetch_aoml()
    if fetch_currents:
        results["cmems"]  = step_fetch_cmems()
        results["gfs"]    = step_fetch_gfs()
    results["simulation"] = step_run_simulation(max_particles=max_particles)
    results["reload"]     = step_signal_reload()

    elapsed = (datetime.now() - start).total_seconds()
    ok  = sum(1 for v in results.values() if v)
    tot = len(results)
    logger.info(f"{'=' * 60}")
    logger.info(f"Pipeline complete: {ok}/{tot} steps succeeded in {elapsed:.1f}s")
    if not results["simulation"]:
        logger.error("CRITICAL: simulation step failed — trajectories NOT updated")
        return False
    return True


def run_daemon(interval_hours: float = 24, **kwargs):
    """Run pipeline on a schedule using APScheduler (optional daemon mode)."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.error("apscheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler()
    scheduler.add_job(lambda: run_pipeline(**kwargs), "interval",
                      hours=interval_hours, next_run_time=datetime.now())
    logger.info(f"APScheduler: running pipeline every {interval_hours}h")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocean Debris Daily Data Pipeline")
    parser.add_argument("--no-currents",   action="store_true",
                        help="Skip CMEMS + GFS fetch (use cached data)")
    parser.add_argument("--max-particles", type=int, default=5000,
                        help="Max trajectory particles (default 5000)")
    parser.add_argument("--daemon",        action="store_true",
                        help="Run continuously (every 24h) using APScheduler")
    parser.add_argument("--interval",      type=float, default=24.0,
                        help="Daemon interval in hours (default 24)")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(
            interval_hours=args.interval,
            fetch_currents=not args.no_currents,
            max_particles=args.max_particles,
        )
    else:
        success = run_pipeline(
            fetch_currents=not args.no_currents,
            max_particles=args.max_particles,
        )
        sys.exit(0 if success else 1)
