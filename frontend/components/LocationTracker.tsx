"use client";
/**
 * LocationTracker.tsx
 * Floating panel to track debris from a specific beach, ocean, or river.
 * - Search by name or click on the map
 * - Runs a 90-day Lagrangian simulation from that point
 * - Shows the trajectory on the map
 */

import { useState, useRef, useEffect } from "react";
import { Search, MapPin, Waves, Anchor, Navigation, X, Loader2, RotateCcw } from "lucide-react";
import {
  searchLocations, seedCustomParticle,
  Location, SeedTrajectoryResponse, ParticleSnapshot,
} from "@/lib/api";

interface Props {
  map: google.maps.Map | null;
  onClose: () => void;
}

const TYPE_COLORS: Record<string, string> = {
  beach:  "#f59e0b",
  ocean:  "#3b82f6",
  sea:    "#06b6d4",
  river:  "#22c55e",
  gyre:   "#ef4444",
};

const TYPE_ICONS: Record<string, string> = {
  beach: "🏖️",
  ocean: "🌊",
  sea:   "🌊",
  river: "🏞️",
  gyre:  "🌀",
};

const STATUS_COLORS = { active: "#00d4c8", beached: "#ef4444", converging: "#a855f7" };
const SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90];

export default function LocationTracker({ map, onClose }: Props) {
  const [query,       setQuery]       = useState("");
  const [results,     setResults]     = useState<Location[]>([]);
  const [searching,   setSearching]   = useState(false);
  const [selected,    setSelected]    = useState<Location | null>(null);
  const [simulating,  setSimulating]  = useState(false);
  const [trajectory,  setTrajectory]  = useState<SeedTrajectoryResponse | null>(null);
  const [error,       setError]       = useState<string | null>(null);
  const [mapClickMode, setMapClickMode] = useState(false);

  // Google Maps overlays for the trajectory
  const polylineRef  = useRef<google.maps.Polyline | null>(null);
  const markersRef   = useRef<google.maps.Marker[]>([]);
  const clickListRef = useRef<google.maps.MapsEventListener | null>(null);
  const debounceRef  = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Search ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!query.trim()) { setResults([]); return; }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const r = await searchLocations(query);
        setResults(r);
      } catch { setResults([]); }
      finally { setSearching(false); }
    }, 300);
  }, [query]);

  // ── Map click mode ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map) return;
    if (mapClickMode) {
      map.setOptions({ cursor: "crosshair" });
      clickListRef.current = map.addListener("click", (e: google.maps.MapMouseEvent) => {
        if (!e.latLng) return;
        const loc: Location = {
          name: `Custom point (${e.latLng.lat().toFixed(3)}°, ${e.latLng.lng().toFixed(3)}°)`,
          lat:  e.latLng.lat(),
          lon:  e.latLng.lng(),
          type: "ocean",
        };
        setSelected(loc);
        setMapClickMode(false);
        setQuery(loc.name);
        setResults([]);
      });
    } else {
      map.setOptions({ cursor: "" });
      if (clickListRef.current) {
        google.maps.event.removeListener(clickListRef.current);
        clickListRef.current = null;
      }
    }
    return () => {
      if (clickListRef.current) {
        google.maps.event.removeListener(clickListRef.current);
        map.setOptions({ cursor: "" });
      }
    };
  }, [map, mapClickMode]);

  // ── Run simulation ────────────────────────────────────────────────────────
  const runSimulation = async (loc: Location) => {
    setSelected(loc);
    setResults([]);
    setQuery(loc.name);
    setSimulating(true);
    setError(null);
    setTrajectory(null);
    clearOverlays();

    try {
      const result = await seedCustomParticle(loc.lat, loc.lon, 1.0);
      setTrajectory(result);
      drawTrajectory(result, loc);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Simulation failed";
      setError(msg.includes("503") ? "API not ready — start the backend first" : msg);
    } finally {
      setSimulating(false);
    }
  };

  // ── Draw trajectory on map ────────────────────────────────────────────────
  const drawTrajectory = (result: SeedTrajectoryResponse, loc: Location) => {
    if (!map) return;
    clearOverlays();

    const snaps = result.trajectory as ParticleSnapshot[];
    const path  = snaps.map(s => ({ lat: s[0], lng: s[1] }));

    // Gradient polyline — colour shifts from teal (start) to red (end)
    polylineRef.current = new google.maps.Polyline({
      path,
      geodesic: true,
      strokeColor: "#00d4c8",
      strokeOpacity: 0.85,
      strokeWeight: 3,
      map,
    });

    // Origin marker
    const originMarker = new google.maps.Marker({
      position: { lat: loc.lat, lng: loc.lon },
      map,
      title: loc.name,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 8,
        fillColor: "#00d4c8",
        fillOpacity: 1,
        strokeColor: "#fff",
        strokeWeight: 2,
      },
    });
    markersRef.current.push(originMarker);

    // Snapshot markers every 7 days
    snaps.forEach((snap, i) => {
      if (i === 0) return;
      const srcType = snap[4];
      const color = srcType === 1 ? "#ef4444" : srcType === 2 ? "#a855f7" : "#00d4c8";
      const marker = new google.maps.Marker({
        position: { lat: snap[0], lng: snap[1] },
        map,
        title: `Day ${SNAPSHOT_DAYS[i]}: ${srcType === 1 ? "Beached" : srcType === 2 ? "Converging" : "Active"}`,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: i === snaps.length - 1 ? 7 : 4,
          fillColor: color,
          fillOpacity: 0.9,
          strokeColor: "#fff",
          strokeWeight: 1.5,
        },
      });
      markersRef.current.push(marker);
    });

    // Pan map to show full trajectory
    const bounds = new google.maps.LatLngBounds();
    path.forEach(p => bounds.extend(p));
    map.fitBounds(bounds, { top: 80, bottom: 80, left: 80, right: 80 });
  };

  const clearOverlays = () => {
    polylineRef.current?.setMap(null);
    polylineRef.current = null;
    markersRef.current.forEach(m => m.setMap(null));
    markersRef.current = [];
  };

  const reset = () => {
    clearOverlays();
    setTrajectory(null);
    setSelected(null);
    setQuery("");
    setResults([]);
    setError(null);
    setMapClickMode(false);
  };

  // Cleanup on unmount
  useEffect(() => () => { clearOverlays(); if (map) map.setOptions({ cursor: "" }); }, []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      width: 320,
      background: "rgba(4,15,28,0.95)",
      backdropFilter: "blur(16px)",
      border: "1px solid rgba(0,212,200,0.18)",
      borderRadius: 14,
      overflow: "hidden",
      color: "white",
      fontFamily: "var(--font-sans, sans-serif)",
      boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 16px 10px",
        borderBottom: "1px solid rgba(0,212,200,0.1)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "rgba(0,212,200,0.15)",
            border: "1px solid rgba(0,212,200,0.3)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <MapPin size={14} style={{ color: "#00d4c8" }} />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "white" }}>Location Tracker</div>
            <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              90-day debris forecast
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {trajectory && (
            <button onClick={reset} title="Reset" style={{
              width: 26, height: 26, borderRadius: 7,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              cursor: "pointer", color: "#94a3b8",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <RotateCcw size={12} />
            </button>
          )}
          <button onClick={onClose} style={{
            width: 26, height: 26, borderRadius: 7,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            cursor: "pointer", color: "#94a3b8",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Search */}
      <div style={{ padding: "12px 16px 8px" }}>
        <div style={{ position: "relative" }}>
          <Search size={13} style={{
            position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)",
            color: "#64748b", pointerEvents: "none",
          }} />
          {searching && (
            <Loader2 size={13} style={{
              position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)",
              color: "#00d4c8", animation: "spin 1s linear infinite",
            }} />
          )}
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search beach, ocean, sea, river..."
            onPointerDown={e => e.stopPropagation()}
            style={{
              width: "100%", padding: "8px 32px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(0,212,200,0.15)",
              borderRadius: 8, color: "white", fontSize: 12,
              outline: "none", boxSizing: "border-box",
            }}
          />
        </div>

        {/* Map click button */}
        <button
          onClick={() => setMapClickMode(v => !v)}
          onPointerDown={e => e.stopPropagation()}
          style={{
            marginTop: 8, width: "100%", padding: "7px 12px",
            borderRadius: 8, fontSize: 11, fontWeight: 600,
            cursor: "pointer", transition: "all 0.2s",
            background: mapClickMode ? "rgba(0,212,200,0.18)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${mapClickMode ? "rgba(0,212,200,0.4)" : "rgba(255,255,255,0.08)"}`,
            color: mapClickMode ? "#00d4c8" : "#94a3b8",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
          }}>
          <MapPin size={11} />
          {mapClickMode ? "Click anywhere on the map..." : "Or click on the map to pick a point"}
        </button>
      </div>

      {/* Search results */}
      {results.length > 0 && !trajectory && (
        <div style={{
          maxHeight: 220, overflowY: "auto",
          borderTop: "1px solid rgba(255,255,255,0.05)",
        }}>
          {results.map((loc, i) => (
            <button
              key={i}
              onClick={() => runSimulation(loc)}
              onPointerDown={e => e.stopPropagation()}
              style={{
                width: "100%", textAlign: "left",
                padding: "9px 16px",
                background: "transparent",
                border: "none", borderBottom: "1px solid rgba(255,255,255,0.04)",
                cursor: "pointer", transition: "background 0.15s",
                display: "flex", alignItems: "center", gap: 10,
              }}
              onMouseEnter={e => (e.currentTarget.style.background = "rgba(0,212,200,0.06)")}
              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            >
              <span style={{ fontSize: 16 }}>{TYPE_ICONS[loc.type] ?? "📍"}</span>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "white" }}>{loc.name}</div>
                <div style={{ fontSize: 10, color: "#64748b", marginTop: 1 }}>
                  <span style={{
                    color: TYPE_COLORS[loc.type] ?? "#94a3b8",
                    textTransform: "capitalize", marginRight: 6,
                  }}>{loc.type}</span>
                  {loc.lat.toFixed(2)}°, {loc.lon.toFixed(2)}°
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Simulating spinner */}
      {simulating && (
        <div style={{
          padding: "20px 16px", display: "flex", flexDirection: "column",
          alignItems: "center", gap: 10,
          borderTop: "1px solid rgba(255,255,255,0.05)",
        }}>
          <Loader2 size={24} style={{ color: "#00d4c8", animation: "spin 1s linear infinite" }} />
          <div style={{ fontSize: 12, color: "#94a3b8", textAlign: "center" }}>
            Running 90-day simulation from<br />
            <span style={{ color: "white", fontWeight: 600 }}>{selected?.name}</span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          margin: "0 16px 12px",
          padding: "10px 12px", borderRadius: 8,
          background: "rgba(239,68,68,0.1)",
          border: "1px solid rgba(239,68,68,0.25)",
          fontSize: 11, color: "#fca5a5",
        }}>
          {error}
        </div>
      )}

      {/* Trajectory result */}
      {trajectory && !simulating && (
        <div style={{ padding: "12px 16px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
          {/* Origin */}
          <div style={{
            background: "rgba(0,212,200,0.06)",
            border: "1px solid rgba(0,212,200,0.12)",
            borderRadius: 9, padding: "10px 12px", marginBottom: 10,
          }}>
            <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 3 }}>
              Origin
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "white" }}>{selected?.name}</div>
            <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>
              {trajectory.origin.lat.toFixed(3)}°, {trajectory.origin.lon.toFixed(3)}°
            </div>
          </div>

          {/* Stats grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
            {[
              { label: "Final Status", value: trajectory.final_status,
                color: STATUS_COLORS[trajectory.final_status as keyof typeof STATUS_COLORS] ?? "#94a3b8" },
              { label: "Snapshots",    value: `${trajectory.trajectory.length} × 7 days`, color: "#3b82f6" },
              { label: "Day 90 Lat",   value: `${(trajectory.trajectory[trajectory.trajectory.length-1][0]).toFixed(2)}°`, color: "#f59e0b" },
              { label: "Day 90 Lon",   value: `${(trajectory.trajectory[trajectory.trajectory.length-1][1]).toFixed(2)}°`, color: "#f59e0b" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 8, padding: "8px 10px",
              }}>
                <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 3 }}>{label}</div>
                <div style={{ fontSize: 12, fontWeight: 700, color, textTransform: "capitalize" }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Mini path chart */}
          <div style={{ marginBottom: 4 }}>
            <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
              Latitude over 90 days
            </div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 32 }}>
              {trajectory.trajectory.map((snap, i) => {
                const lat = snap[0];
                const srcType = snap[4];
                const color = srcType === 1 ? "#ef4444" : srcType === 2 ? "#a855f7" : "#00d4c8";
                // Normalise lat to bar height
                const allLats = trajectory.trajectory.map(s => s[0]);
                const minL = Math.min(...allLats), maxL = Math.max(...allLats);
                const h = maxL === minL ? 16 : Math.max(3, ((lat - minL) / (maxL - minL)) * 32);
                return (
                  <div key={i} title={`Day ${SNAPSHOT_DAYS[i]}: ${lat.toFixed(2)}°`}
                    style={{ flex: 1, height: h, background: color, borderRadius: 2, opacity: 0.8 }} />
                );
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3 }}>
              <span style={{ fontSize: 8, color: "#475569" }}>Day 0</span>
              <span style={{ fontSize: 8, color: "#475569" }}>Day 90</span>
            </div>
          </div>

          <div style={{ fontSize: 10, color: "#475569", textAlign: "center", marginTop: 8 }}>
            Trajectory drawn on map · drag to reposition panel
          </div>
        </div>
      )}

      {/* Empty state */}
      {!trajectory && !simulating && results.length === 0 && !error && (
        <div style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>🌊</div>
          <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>
            Search for a beach, ocean, or river to see where debris from that location will drift over 90 days
          </div>
        </div>
      )}
    </div>
  );
}
