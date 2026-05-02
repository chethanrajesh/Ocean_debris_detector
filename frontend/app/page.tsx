"use client";

/**
 * page.tsx — Global Ocean Debris Monitoring Dashboard
 *
 * Two display modes (toggle in navbar):
 *   HOTSPOT MODE  — existing hotspot/heatmap/currents view
 *   TRAJECTORY MODE — 90-day Lagrangian particle tracker
 */

import { useState, useCallback, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import {
  Waves, Zap, Activity, Globe2, RefreshCw,
  Eye, EyeOff, Layers, Navigation, Anchor, Play, Pause, MapPin,
} from "lucide-react";

import {
  getHotspots, getPredictions, getCurrents, getKnownZones,
  getTrajectoryForecast, getBeachingRisk,
  Hotspot, KnownZone, PredictionNode, CurrentNode, PredictionMode,
  ParticleTrajectory, BeachingCell, TrajectoryForecastResponse,
} from "@/lib/api";

// Lazy-load map components
const GEEMap             = dynamic(() => import("@/components/GEEMap"),             { ssr: false });
const HotspotLayer       = dynamic(() => import("@/components/HotspotLayer"),       { ssr: false });
const HeatmapLayer       = dynamic(() => import("@/components/HeatmapLayer"),       { ssr: false });
const CurrentVectorLayer = dynamic(() => import("@/components/CurrentVectorLayer"), { ssr: false });
const TimeSlider         = dynamic(() => import("@/components/TimeSlider"),         { ssr: false });
const HotspotPanel       = dynamic(() => import("@/components/HotspotPanel"),       { ssr: false });
const LegendPanel        = dynamic(() => import("@/components/LegendPanel"),        { ssr: false });
const TrajectoryLayer    = dynamic(() => import("@/components/TrajectoryLayer"),    { ssr: false });
const BeachingRiskLayer  = dynamic(() => import("@/components/BeachingRiskLayer"), { ssr: false });
const TrajectoryPanel    = dynamic(() => import("@/components/TrajectoryPanel"),    { ssr: false });
const LocationTracker    = dynamic(() => import("@/components/LocationTracker"),    { ssr: false });
import Draggable from "@/components/Draggable";

// ── Dashboard mode ───────────────────────────────────────────────────────────
type DashboardMode = "hotspot" | "trajectory";
type ColorBy = "age" | "density" | "sourceType";

const SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90];

// ── Mode banner ───────────────────────────────────────────────────────────────
function ModeBanner({ mode, day }: { mode: PredictionMode; day: number }) {
  const isReal = mode === "real";
  return (
    <div key={mode} className="animate-fade-in-up" style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "8px 18px", borderRadius: 99,
      backdropFilter: "blur(12px)",
      background: isReal ? "rgba(16,185,129,0.12)" : "rgba(168,85,247,0.12)",
      border: `1px solid ${isReal ? "rgba(16,185,129,0.40)" : "rgba(168,85,247,0.40)"}`,
      boxShadow: `0 0 20px ${isReal ? "rgba(16,185,129,0.15)" : "rgba(168,85,247,0.15)"}`,
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%",
        background: isReal ? "#10b981" : "#a855f7",
        boxShadow: `0 0 6px ${isReal ? "#10b981" : "#a855f7"}`,
        animation: "pulse 2s ease-in-out infinite",
      }} />
      <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", color: isReal ? "#10b981" : "#a855f7" }}>
        {isReal ? "📍 REAL-TIME" : "🔮 GNN PREDICTION"}
      </span>
      <span style={{ width: 1, height: 12, background: isReal ? "rgba(16,185,129,0.3)" : "rgba(168,85,247,0.3)" }} />
      <span style={{ fontSize: 10, color: "var(--text-secondary)", letterSpacing: "0.04em" }}>
        {isReal ? "Satellite & Buoy Sources · Today" : `Lagrangian + GNN · Day +${day.toFixed(0)}`}
      </span>
    </div>
  );
}

function TrajModeBanner({ snapshotIdx }: { snapshotIdx: number }) {
  const day = SNAPSHOT_DAYS[snapshotIdx] ?? 0;
  return (
    <div className="animate-fade-in-up" style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "8px 18px", borderRadius: 99,
      backdropFilter: "blur(12px)",
      background: "rgba(0,212,200,0.10)",
      border: "1px solid rgba(0,212,200,0.35)",
      boxShadow: "0 0 20px rgba(0,212,200,0.12)",
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%", background: "#00d4c8",
        boxShadow: "0 0 6px #00d4c8", animation: "pulse 2s ease-in-out infinite",
      }} />
      <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", color: "#00d4c8" }}>
        🌊 TRAJECTORY TRACKER
      </span>
      <span style={{ width: 1, height: 12, background: "rgba(0,212,200,0.3)" }} />
      <span style={{ fontSize: 10, color: "var(--text-secondary)", letterSpacing: "0.04em" }}>
        90-Day Lagrangian · Day {day}
      </span>
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, icon: Icon, color, id }: {
  label: string; value: string; icon: React.ElementType; color: string; id: string;
}) {
  return (
    <div id={id} className="glass-panel-dark px-4 py-3 flex items-center gap-3 min-w-[140px]">
      <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: `${color}22`, border: `1px solid ${color}44` }}>
        <Icon size={16} style={{ color }} />
      </div>
      <div>
        <div className="text-lg font-bold font-display text-white leading-none">{value}</div>
        <div className="text-[10px] text-[var(--text-muted)] mt-0.5 uppercase tracking-wider">{label}</div>
      </div>
    </div>
  );
}

// ── Layer toggle ──────────────────────────────────────────────────────────────
function LayerToggle({ label, active, onToggle, color, id }: {
  label: string; active: boolean; onToggle: () => void; color: string; id: string;
}) {
  return (
    <button id={id} onClick={onToggle}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border"
      style={active
        ? { background: `${color}22`, borderColor: `${color}66`, color }
        : { background: "var(--ocean-800)", borderColor: "var(--glass-border)", color: "var(--text-muted)" }
      }>
      {active ? <Eye size={11} /> : <EyeOff size={11} />}
      {label}
    </button>
  );
}

// ── Color-by toggle ───────────────────────────────────────────────────────────
function ColorByToggle({ value, onChange }: { value: ColorBy; onChange: (v: ColorBy) => void }) {
  const options: { key: ColorBy; label: string; color: string }[] = [
    { key: "age",        label: "Age",    color: "#f59e0b" },
    { key: "density",    label: "Density", color: "#3b82f6" },
    { key: "sourceType", label: "Status", color: "#a855f7" },
  ];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <span style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginRight: 4 }}>
        Color by
      </span>
      {options.map(o => (
        <button key={o.key} onClick={() => onChange(o.key)}
          style={{
            padding: "3px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600,
            cursor: "pointer", transition: "all 0.2s",
            background: value === o.key ? `${o.color}22` : "var(--ocean-800)",
            border: `1px solid ${value === o.key ? `${o.color}66` : "var(--glass-border)"}`,
            color: value === o.key ? o.color : "var(--text-muted)",
          }}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ── Dashboard mode switch ─────────────────────────────────────────────────────
function ModeSwitch({ mode, onChange }: { mode: DashboardMode; onChange: (m: DashboardMode) => void }) {
  return (
    <div style={{
      display: "flex", borderRadius: 10,
      border: "1px solid var(--glass-border)",
      overflow: "hidden", background: "var(--ocean-900)",
      position: "relative", zIndex: 9999,
      pointerEvents: "auto",
    }}>
      {(["hotspot", "trajectory"] as DashboardMode[]).map(m => (
        <button
          key={m}
          id={`mode-btn-${m}`}
          onClick={(e) => { e.stopPropagation(); onChange(m); }}
          style={{
            padding: "5px 14px", fontSize: 10, fontWeight: 700,
            letterSpacing: "0.05em", textTransform: "uppercase",
            cursor: "pointer", transition: "all 0.25s",
            background: mode === m ? "rgba(0,212,200,0.18)" : "transparent",
            color: mode === m ? "#00d4c8" : "var(--text-muted)",
            borderRight: m === "hotspot" ? "1px solid var(--glass-border)" : "none",
            border: "none", outline: "none",
            position: "relative", zIndex: 9999,
          }}>
          {m === "hotspot" ? "🔥 Hotspots" : "🌊 Trajectories"}
        </button>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [map,             setMap]           = useState<google.maps.Map | null>(null);
  // Hotspot mode state
  const [hotspots,        setHotspots]      = useState<Hotspot[]>([]);
  const [knownZones,      setKnownZones]    = useState<KnownZone[]>([]);
  const [predNodes,       setPredNodes]     = useState<PredictionNode[]>([]);
  const [currentNodes,    setCurrentNodes]  = useState<CurrentNode[]>([]);
  const [timestep,        setTimestep]      = useState(0);
  const [maxSteps,        setMaxSteps]      = useState(361);
  const [predDay,         setPredDay]       = useState(0);
  const [selectedHotspot, setSelected]      = useState<Hotspot | null>(null);
  // Trajectory mode state
  const [trajectories,    setTrajectories]  = useState<ParticleTrajectory[]>([]);
  const [beachingCells,   setBeachingCells] = useState<BeachingCell[]>([]);
  const [snapshotIdx,     setSnapshotIdx]   = useState(0);
  const [selectedTraj,    setSelectedTraj]  = useState<ParticleTrajectory | null>(null);
  const [colorBy,         setColorBy]       = useState<ColorBy>("age");
  const [isPlaying,       setIsPlaying]     = useState(false);
  const [trajStats,       setTrajStats]     = useState({ total: 0, active: 0, beached: 0 });
  // Shared state
  const [dashMode,        setDashMode]      = useState<DashboardMode>("hotspot");
  const [loading,         setLoading]       = useState(true);
  const [trajLoading,     setTrajLoading]   = useState(false);
  const [apiStatus,       setApiStatus]     = useState<"ok" | "error" | "loading">("loading");
  // Layer visibility
  const [showHotspots,    setShowHotspots]  = useState(true);
  const [showKnownZones,  setShowKnownZones]= useState(true);
  const [showHeatmap,     setShowHeatmap]   = useState(true);
  const [showCurrents,    setShowCurrents]  = useState(true);
  const [showBeaching,    setShowBeaching]  = useState(true);
  const [showTracker,     setShowTracker]   = useState(false);

  const mode: PredictionMode = timestep === 0 ? "real" : "predicted";
  const fetchRef = useRef(false);
  const playTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Initial fetch (hotspot mode) ──────────────────────────────────────────
  useEffect(() => {
    if (fetchRef.current) return;
    fetchRef.current = true;
    const go = async () => {
      setLoading(true);
      try {
        const [hs, knownZs, preds] = await Promise.all([
          getHotspots(), getKnownZones(), getPredictions(0),
        ]);
        setHotspots(hs); setKnownZones(knownZs);
        setPredNodes(preds.nodes); setMaxSteps(preds.total_timesteps || 361);
        setPredDay(preds.day ?? 0); setApiStatus("ok");
      } catch { setApiStatus("error"); } finally { setLoading(false); }
      try { const c = await getCurrents(2000); setCurrentNodes(c.nodes); } catch {}
    };
    go();
  }, []);

  // ── Fetch trajectory data when switching to trajectory mode ───────────────
  useEffect(() => {
    if (dashMode !== "trajectory" || trajectories.length > 0) return;
    setTrajLoading(true);
    Promise.all([getTrajectoryForecast(1500), getBeachingRisk(200)])
      .then(([forecast, beaching]) => {
        setTrajectories(forecast.trajectories);
        setBeachingCells(beaching.top_cells);
        setTrajStats({
          total:   forecast.total_particles,
          active:  forecast.trajectories.filter(t => t.snapshots[0][4] === 0).length,
          beached: beaching.total_beached,
        });
      })
      .catch(err => console.warn("Trajectory fetch failed:", err))
      .finally(() => setTrajLoading(false));
  }, [dashMode, trajectories.length]);

  // ── Re-fetch predictions on timestep change ───────────────────────────────
  useEffect(() => {
    if (timestep === 0 || dashMode !== "hotspot") return;
    getPredictions(timestep)
      .then(r => { setPredNodes(r.nodes); setPredDay(r.day ?? timestep / 4); })
      .catch(console.error);
  }, [timestep, dashMode]);

  // ── Playback animation ────────────────────────────────────────────────────
  useEffect(() => {
    if (isPlaying) {
      playTimer.current = setInterval(() => {
        setSnapshotIdx(i => {
          if (i >= SNAPSHOT_DAYS.length - 1) { setIsPlaying(false); return i; }
          return i + 1;
        });
      }, 600);
    } else {
      if (playTimer.current) clearInterval(playTimer.current);
    }
    return () => { if (playTimer.current) clearInterval(playTimer.current); };
  }, [isPlaying]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleMapReady       = useCallback((m: google.maps.Map) => setMap(m), []);
  const handleHotspotClick   = useCallback((h: Hotspot) => setSelected(h), []);
  const handleTimestepChange = useCallback((t: number) => setTimestep(t), []);
  const handleRefresh        = useCallback(() => {
    fetchRef.current = false; setLoading(true);
    const go = async () => {
      try {
        const [hs, knownZs, preds] = await Promise.all([
          getHotspots(), getKnownZones(), getPredictions(timestep),
        ]);
        setHotspots(hs); setKnownZones(knownZs);
        setPredNodes(preds.nodes); setPredDay(preds.day ?? timestep / 4);
        setApiStatus("ok");
      } catch { setApiStatus("error"); } finally { setLoading(false); }
      try { const c = await getCurrents(2000); setCurrentNodes(c.nodes); } catch {}
    };
    go();
  }, [timestep]);

  // ── Derived stats (hotspot mode) ──────────────────────────────────────────
  const criticalCount = hotspots.filter(h => h.level === "critical").length;
  const highCount     = hotspots.filter(h => h.level === "high").length;
  const maxDensity    = hotspots.length ? Math.max(...hotspots.map(h => h.plastic_density)) * 100 : 0;
  const trackedKm2    = hotspots.length * 8500;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <main className="relative w-screen h-screen overflow-hidden ocean-grid-bg">

      {/* Full-screen map */}
      <GEEMap onMapReady={handleMapReady} />

      {/* ── HOTSPOT MODE LAYERS ── */}
      {dashMode === "hotspot" && showHotspots && (
        <HotspotLayer map={map} hotspots={hotspots} onHotspotClick={handleHotspotClick} />
      )}
      {dashMode === "hotspot" && showKnownZones && (
        <HotspotLayer map={map} hotspots={knownZones as unknown as Hotspot[]}
          onHotspotClick={h => setSelected(h as unknown as Hotspot)} />
      )}
      {dashMode === "hotspot" && showHeatmap && (
        <HeatmapLayer map={map} nodes={predNodes} mode={mode} />
      )}
      {dashMode === "hotspot" && showCurrents && (
        <CurrentVectorLayer map={map} nodes={currentNodes} />
      )}

      {/* ── TRAJECTORY MODE LAYERS ── */}
      {dashMode === "trajectory" && !trajLoading && trajectories.length > 0 && (
        <TrajectoryLayer
          map={map}
          trajectories={trajectories}
          snapshotIndex={snapshotIdx}
          colorBy={colorBy}
          onParticleClick={setSelectedTraj}
        />
      )}
      {dashMode === "trajectory" && showBeaching && beachingCells.length > 0 && (
        <BeachingRiskLayer map={map} cells={beachingCells} />
      )}

      {/* ── NAVBAR ── */}
      <header className="absolute top-0 left-0 right-0 z-[9999] flex items-center px-5 py-3
                         bg-gradient-to-b from-[rgba(2,10,18,0.95)] to-transparent"
              style={{ pointerEvents: "auto" }}>
        {/* Logo */}
        <div className="flex items-center gap-3 flex-1">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00d4c8] to-[#0ea5e9]
                          flex items-center justify-center teal-glow-sm flex-shrink-0">
            <Waves size={18} className="text-[var(--ocean-950)]" />
          </div>
          <div>
            <h1 className="font-display text-base font-bold text-white leading-none tracking-tight">
              Ocean Debris Monitor
            </h1>
            <p className="text-[10px] text-[var(--text-muted)] mt-0.5 uppercase tracking-widest">
              Global Plastic Tracking System
            </p>
          </div>
        </div>

        {/* Mode switch */}
        <div className="mr-5">
          <ModeSwitch mode={dashMode} onChange={setDashMode} />
        </div>

        {/* Hotspot layer toggles */}
        {dashMode === "hotspot" && (
          <div className="flex items-center gap-2 mr-4">
            <div className="flex items-center gap-1.5 mr-1">
              <Layers size={12} className="text-[var(--text-muted)]" />
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Layers</span>
            </div>
            <LayerToggle id="toggle-hotspots"    label="Predicted"   active={showHotspots}   onToggle={() => setShowHotspots(v => !v)}   color="#ef4444" />
            <LayerToggle id="toggle-known-zones" label="Known Zones" active={showKnownZones} onToggle={() => setShowKnownZones(v => !v)} color="#a855f7" />
            <LayerToggle id="toggle-heatmap"     label="Heatmap"     active={showHeatmap}    onToggle={() => setShowHeatmap(v => !v)}    color="#00d4c8" />
            <LayerToggle id="toggle-currents"    label="Currents"    active={showCurrents}   onToggle={() => setShowCurrents(v => !v)}   color="#3b82f6" />
          </div>
        )}

        {/* Trajectory controls */}
        {dashMode === "trajectory" && (
          <div className="flex items-center gap-3 mr-4">
            <ColorByToggle value={colorBy} onChange={setColorBy} />
            <LayerToggle id="toggle-beaching" label="Risk Zones" active={showBeaching} onToggle={() => setShowBeaching(v => !v)} color="#ef4444" />
          </div>
        )}

        {/* API status + refresh */}
        <div className="flex items-center gap-3">
          {/* Mode banner inline in navbar */}
          <div className="mr-3">
            {dashMode === "hotspot"
              ? <ModeBanner mode={mode} day={predDay} />
              : <TrajModeBanner snapshotIdx={snapshotIdx} />
            }
          </div>
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === "ok" ? "bg-green-500" : apiStatus === "error" ? "bg-red-500" : "bg-amber-500 animate-pulse"
            }`} />
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
              {apiStatus === "ok" ? "Live" : apiStatus === "error" ? "Offline" : "Connecting"}
            </span>
          </div>
          {/* Location tracker toggle */}
          <button
            onClick={() => setShowTracker(v => !v)}
            title="Track debris from a location"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border"
            style={showTracker
              ? { background: "rgba(0,212,200,0.18)", borderColor: "rgba(0,212,200,0.4)", color: "#00d4c8" }
              : { background: "var(--ocean-800)", borderColor: "var(--glass-border)", color: "var(--text-muted)" }
            }>
            <MapPin size={11} />
            Track Location
          </button>
          <button id="refresh-btn" onClick={handleRefresh} disabled={loading}
            className="w-8 h-8 rounded-lg bg-[var(--ocean-700)] hover:bg-[var(--ocean-600)]
                       flex items-center justify-center text-[var(--text-secondary)]
                       hover:text-white transition-all disabled:opacity-40">
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </header>

      {/* Mode banner removed from below navbar — now lives inside navbar */}

      {/* ── HOTSPOT STATS (top-left, below navbar) ── */}
      {dashMode === "hotspot" && (
        <div className="absolute left-4 z-10 flex flex-col gap-2 animate-fade-in-up" style={{ top: 68 }}>
          <StatCard id="stat-critical" label="Critical Zones"  value={loading ? "—" : String(criticalCount)}                     icon={Zap}      color="#ef4444" />
          <StatCard id="stat-high"     label="High Risk Zones" value={loading ? "—" : String(highCount)}                         icon={Activity} color="#f97316" />
          <StatCard id="stat-density"  label="Peak Density"    value={loading ? "—" : `${maxDensity.toFixed(1)}%`}               icon={Waves}    color="#00d4c8" />
          <StatCard id="stat-coverage" label="Tracked Area"    value={loading ? "—" : `${(trackedKm2 / 1000).toFixed(0)}K km²`} icon={Globe2}   color="#3b82f6" />
        </div>
      )}

      {/* ── TRAJECTORY STATS (top-left, below navbar) ── */}
      {dashMode === "trajectory" && (
        <div className="absolute left-4 z-10 flex flex-col gap-2 animate-fade-in-up" style={{ top: 68 }}>
          <StatCard id="stat-particles" label="Active Particles" value={trajLoading ? "—" : String(trajStats.active)}  icon={Waves}      color="#00d4c8" />
          <StatCard id="stat-beached"   label="Beaching Events"  value={trajLoading ? "—" : String(trajStats.beached)} icon={Anchor}     color="#ef4444" />
          <StatCard id="stat-total"     label="Total Tracked"    value={trajLoading ? "—" : String(trajStats.total)}   icon={Globe2}     color="#3b82f6" />
          <StatCard id="stat-day"       label="Forecast Day"     value={`Day ${SNAPSHOT_DAYS[snapshotIdx] ?? 0}`}      icon={Navigation} color="#f59e0b" />
        </div>
      )}

      {/* ── Hotspot panel — floats top-right, clear of navbar ── */}
      {dashMode === "hotspot" && selectedHotspot && (
        <Draggable id="hotspot-panel-float" defaultPosition={() => ({ x: window.innerWidth - 356, y: 68 })}>
          <HotspotPanel hotspot={selectedHotspot} onClose={() => setSelected(null)} />
        </Draggable>
      )}

      {/* ── Trajectory panel — floats top-right, clear of navbar ── */}
      {dashMode === "trajectory" && selectedTraj && (
        <Draggable id="trajectory-panel-float" defaultPosition={() => ({ x: window.innerWidth - 316, y: 68 })}>
          <TrajectoryPanel
            trajectory={selectedTraj}
            snapshotIdx={snapshotIdx}
            onClose={() => setSelectedTraj(null)}
          />
        </Draggable>
      )}

      {/* ── HOTSPOT TIME SLIDER — floats bottom-centre ── */}
      {dashMode === "hotspot" && (
        <Draggable id="time-slider-float" defaultPosition={() => ({
          x: Math.round(window.innerWidth / 2) - 230,
          y: window.innerHeight - 120,
        })}>
          <TimeSlider value={timestep} maxSteps={maxSteps} onChange={handleTimestepChange} mode={mode} />
        </Draggable>
      )}

      {/* ── TRAJECTORY TIMELINE — floats bottom-centre ── */}
      {dashMode === "trajectory" && (
        <Draggable id="traj-timeline-float" defaultPosition={() => ({
          x: Math.round(window.innerWidth / 2) - 210,
          y: window.innerHeight - 130,
        })}>
          <div style={{
              display: "flex", alignItems: "center", gap: 12,
              padding: "12px 20px", borderRadius: 16,
              background: "rgba(4,15,28,0.88)", backdropFilter: "blur(12px)",
              border: "1px solid rgba(0,212,200,0.18)",
              boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
            }}>
              {/* Play/pause */}
              <button
                id="traj-play-btn"
                onClick={() => { setIsPlaying(p => !p); if (snapshotIdx >= SNAPSHOT_DAYS.length - 1) setSnapshotIdx(0); }}
                style={{
                  width: 32, height: 32, borderRadius: 9,
                  background: "rgba(0,212,200,0.18)", border: "1px solid rgba(0,212,200,0.4)",
                  cursor: "pointer", color: "#00d4c8",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              </button>

              {/* Timeline label */}
              <span style={{ fontSize: 10, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                TIME PROJECTION
              </span>

              {/* Scrubber */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 280 }}
                   onPointerDown={(e) => e.stopPropagation()} /* Prevent dragging while using slider */>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                  <span style={{ fontSize: 9, color: "#64748b" }}>Day 0</span>
                  <span style={{ fontSize: 9, color: "#00d4c8", fontWeight: 700 }}>
                    Day {SNAPSHOT_DAYS[snapshotIdx]}
                  </span>
                  <span style={{ fontSize: 9, color: "#64748b" }}>Day 90</span>
                </div>
                <input
                  id="traj-slider"
                  type="range"
                  min={0}
                  max={SNAPSHOT_DAYS.length - 1}
                  value={snapshotIdx}
                  onChange={e => { setSnapshotIdx(Number(e.target.value)); setIsPlaying(false); }}
                  style={{ width: "100%", accentColor: "#00d4c8", cursor: "pointer" }}
                />
                {/* Tick marks */}
                <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 2 }}>
                  {SNAPSHOT_DAYS.filter((_, i) => i % 2 === 0).map(d => (
                    <span key={d} style={{ fontSize: 8, color: "#334155" }}>{d}</span>
                  ))}
                </div>
              </div>
            </div>
          </Draggable>
      )}

      {/* ── Legend — floats bottom-left, above time slider ── */}
      <Draggable id="legend-float" defaultPosition={() => ({
        x: 16,
        y: window.innerHeight - 260,
      })}>
        <LegendPanel />
      </Draggable>

      {/* ── Location Tracker panel — floats top-right below hotspot panel ── */}
      {showTracker && (
        <Draggable id="location-tracker-float" defaultPosition={() => ({
          x: window.innerWidth - 340,
          y: 68,
        })}>
          <LocationTracker map={map} onClose={() => setShowTracker(false)} />
        </Draggable>
      )}

      {/* ── Loading overlay ── */}
      {(loading || (dashMode === "trajectory" && trajLoading)) && (
        <div className="absolute inset-0 z-30 flex items-center justify-center
                        bg-[rgba(2,10,18,0.85)] backdrop-blur-sm">
          <div className="flex flex-col items-center gap-5">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00d4c8] to-[#0ea5e9]
                            flex items-center justify-center teal-glow-lg">
              <Waves size={32} className="text-[var(--ocean-950)]" />
            </div>
            <div className="text-center">
              <p className="font-display text-xl font-bold text-white">Ocean Debris Monitor</p>
              <p className="text-sm text-[var(--text-secondary)] mt-1">
                {dashMode === "trajectory" ? "Running 90-day simulation..." : "Loading simulation data..."}
              </p>
            </div>
            <div className="flex gap-1.5">
              {[0, 1, 2].map(i => (
                <div key={i} className="w-2 h-2 rounded-full bg-teal-500"
                  style={{ animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite` }} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── API error banner ── */}
      {apiStatus === "error" && !loading && (
        <div id="api-error-banner"
          className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20
                     glass-panel px-5 py-3 flex items-center gap-3
                     border-red-500/30 bg-red-900/20 animate-fade-in-up">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-sm text-red-300">Backend offline — displaying demo data</span>
        </div>
      )}
    </main>
  );
}
