FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc        : compile C extensions (scipy, netCDF4)
# libhdf5    : HDF5 support for netCDF4/xarray
# libnetcdf  : NetCDF file format
# libgeos    : geometry operations (geopandas fallback)
# curl       : healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    libnetcdf-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Data directory (mounted as volume at runtime) ─────────────────────────────
RUN mkdir -p /app/data/global_graph

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

# Use 0.0.0.0 so Docker port mapping works; workers=1 keeps memory predictable
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
