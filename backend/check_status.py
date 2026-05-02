"""Check all data files and API status."""
import numpy as np, os, requests

print("=== DATA FILES ===")
files = {
    "data/cmems_currents.npy":        "CMEMS currents",
    "data/ocean_currents.npy":        "Ocean currents (blended)",
    "data/ssh_currents.npy":          "SSH geostrophic",
    "data/wind_data.npy":             "Wind fields",
    "data/stokes_drift.npy":          "Stokes drift",
    "data/land_mask.npy":             "Land mask",
    "data/density_maps.npy":          "Density maps",
    "data/debris_seed_nodes.npy":     "Debris seeds (NCEI)",
    "data/ncei_microplastics.npy":    "NCEI microplastics raw",
    "data/global_graph/nodes.npy":    "Ocean node graph",
    "data/trajectories.npy":          "Trajectories",
    "data/beaching_risk.npy":         "Beaching risk",
    "data/future_predictions.npy":    "Future predictions",
    "data/sentinel_scores.npy":       "Sentinel-2 scores",
}
issues = []
for path, label in files.items():
    if os.path.exists(path):
        arr = np.load(path)
        mx = float(np.abs(arr).max())
        nz = int((arr != 0).sum())
        status = "OK  " if mx > 0 else "ZERO"
        print(f"  {status}  {label}: shape={arr.shape} max={mx:.4f} nonzero={nz}")
        if mx == 0:
            issues.append(f"{label} is all-zero")
    else:
        print(f"  MISS  {label}")
        issues.append(f"{label} missing")

print()
print("=== API STATUS ===")
try:
    r = requests.get("http://127.0.0.1:8000/", timeout=5)
    data = r.json()
    print(f"  Status:    {data['status']}")
    print(f"  Hotspots:  {data['hotspots_available']}")
    print(f"  Timesteps: {data['timesteps_available']}")
    if data["status"] != "ready":
        issues.append("API not ready")
except Exception as e:
    print(f"  API not running: {e}")
    issues.append("API not running")

print()
if issues:
    print("=== ISSUES ===")
    for i in issues:
        print(f"  ! {i}")
else:
    print("=== ALL OK ===")
