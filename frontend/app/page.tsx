"use client";

/**
 * page.tsx — Global Ocean Debris Monitoring Dashboard
 *
 * Layout:
 * ┌─────────────────────────────────────────────────────┐
 * │  [NAVBAR]                                           │
 * │  ┌────────────────────────────────────────────────┐ │
 * │  │          FULL-SCREEN MAP                       │ │
 * │  │  [HOTSPOT PANEL] (right, on click)             │ │
 * │  │                                                │ │
 * │  │  [TIME SLIDER]         (bottom-centre)         │ │
 * │  │  [LEGEND PANEL]        (bottom-left)           │ │
 * │  │  [STATS RIBBON]        (top-left)              │ │
 * │  └────────────────────────────────────────────────┘ │
 * └─────────────────────────────────────────────────────┘
 */

import { useState, useCallback, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import {
  Waves, Zap, Activity, Globe2, RefreshCw,
  Eye, EyeOff, Layers
} from "lucide-react";

import {
  getHotspots, getPredictions, getCurrents,
  Hotspot, PredictionNode, CurrentNode,
} from "@/lib/api";

// Lazy-load map components (browser-only APIs)
const GEEMap            = dynamic(() => import("@/components/GEEMap"),            { ssr: false });
const HotspotLayer      = dynamic(() => import("@/components/HotspotLayer"),      { ssr: false });
const HeatmapLayer      = dynamic(() => import("@/components/HeatmapLayer"),      { ssr: false });
const CurrentVectorLayer = dynamic(() => import("@/components/CurrentVectorLayer"), { ssr: false });
const TimeSlider        = dynamic(() => import("@/components/TimeSlider"),        { ssr: false });
const HotspotPanel      = dynamic(() => import("@/components/HotspotPanel"),      { ssr: false });
const LegendPanel       = dynamic(() => import("@/components/LegendPanel"),       { ssr: false });

// ── Stat card ────────────────────────────────────────────────────────────────
function StatCard({
  label, value, icon: Icon, color, id,
}: {
  label: string; value: string; icon: React.ElementType; color: string; id: string;
}) {
  return (
    <div id={id} className="glass-panel-dark px-4 py-3 flex items-center gap-3 min-w-[140px]">
      <div
        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: `${color}22`, border: `1px solid ${color}44` }}
      >
        <Icon size={16} style={{ color }} />
      </div>
      <div>
        <div className="text-lg font-bold font-display text-white leading-none">{value}</div>
        <div className="text-[10px] text-[var(--text-muted)] mt-0.5 uppercase tracking-wider">{label}</div>
      </div>
    </div>
  );
}

// ── Layer toggle button ───────────────────────────────────────────────────────
function LayerToggle({
  label, active, onToggle, color, id,
}: {
  label: string; active: boolean; onToggle: () => void; color: string; id: string;
}) {
  return (
    <button
      id={id}
      onClick={onToggle}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium
                 transition-all duration-200 border"
      style={
        active
          ? { background: `${color}22`, borderColor: `${color}66`, color }
          : { background: "var(--ocean-800)", borderColor: "var(--glass-border)", color: "var(--text-muted)" }
      }
    >
      {active ? <Eye size={11} /> : <EyeOff size={11} />}
      {label}
    </button>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [map,             setMap]           = useState<google.maps.Map | null>(null);
  const [hotspots,        setHotspots]      = useState<Hotspot[]>([]);
  const [predNodes,       setPredNodes]     = useState<PredictionNode[]>([]);
  const [currentNodes,    setCurrentNodes]  = useState<CurrentNode[]>([]);
  const [timestep,        setTimestep]      = useState(0);
  const [maxSteps,        setMaxSteps]      = useState(365);
  const [selectedHotspot, setSelected]      = useState<Hotspot | null>(null);
  const [loading,         setLoading]       = useState(true);
  const [apiStatus,       setApiStatus]     = useState<"ok" | "error" | "loading">("loading");

  // Layer visibility toggles
  const [showHotspots,    setShowHotspots]  = useState(true);
  const [showHeatmap,     setShowHeatmap]   = useState(true);
  const [showCurrents,    setShowCurrents]  = useState(true);

  const fetchRef = useRef(false);

  // ── Initial data fetch ──────────────────────────────────────────────────────
  useEffect(() => {
    if (fetchRef.current) return;
    fetchRef.current = true;

    const fetchAll = async () => {
      setLoading(true);
      try {
        const [hs, preds, currents] = await Promise.all([
          getHotspots(),
          getPredictions(0),
          getCurrents(),
        ]);
        setHotspots(hs);
        setPredNodes(preds.nodes);
        setMaxSteps(preds.total_timesteps || 365);
        setCurrentNodes(currents.nodes);
        setApiStatus("ok");
      } catch (err) {
        console.error("API error:", err);
        setApiStatus("error");
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, []);

  // ── Re-fetch predictions when timestep changes ─────────────────────────────
  useEffect(() => {
    if (timestep === 0) return;
    getPredictions(timestep)
      .then((r) => setPredNodes(r.nodes))
      .catch(console.error);
  }, [timestep]);

  // ── Handlers ───────────────────────────────────────────────────────────────
  const handleMapReady = useCallback((m: google.maps.Map) => setMap(m), []);
  const handleHotspotClick = useCallback((h: Hotspot) => setSelected(h), []);
  const handleTimestepChange = useCallback((t: number) => setTimestep(t), []);
  const handleRefresh = useCallback(() => {
    fetchRef.current = false;
    setLoading(true);
    // Force re-init
    setTimeout(() => { fetchRef.current = false; }, 100);
    const fetchAll = async () => {
      try {
        const [hs, preds, currents] = await Promise.all([
          getHotspots(),
          getPredictions(timestep),
          getCurrents(),
        ]);
        setHotspots(hs);
        setPredNodes(preds.nodes);
        setCurrentNodes(currents.nodes);
        setApiStatus("ok");
      } catch {
        setApiStatus("error");
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, [timestep]);

  // ── Derived stats ───────────────────────────────────────────────────────────
  const criticalCount  = hotspots.filter((h) => h.level === "critical").length;
  const highCount      = hotspots.filter((h) => h.level === "high").length;
  const maxDensity     = hotspots.length
    ? Math.max(...hotspots.map((h) => h.plastic_density)) * 100
    : 0;
  const trackedKm2     = hotspots.length * 8500; // rough estimate per hotspot

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <main className="relative w-screen h-screen overflow-hidden ocean-grid-bg">

      {/* ── Full-screen Map ── */}
      <GEEMap onMapReady={handleMapReady} />

      {/* ── Map Layers (mounted on top of map, no DOM) ── */}
      {showHotspots && (
        <HotspotLayer
          map={map}
          hotspots={hotspots}
          onHotspotClick={handleHotspotClick}
        />
      )}
      {showHeatmap && <HeatmapLayer map={map} nodes={predNodes} />}
      {showCurrents && <CurrentVectorLayer map={map} nodes={currentNodes} />}

      {/* ── Navigation Bar ── */}
      <header className="absolute top-0 left-0 right-0 z-10 flex items-center px-5 py-3
                         bg-gradient-to-b from-[rgba(2,10,18,0.95)] to-transparent">

        {/* Logo + Title */}
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

        {/* Layer toggles */}
        <div className="flex items-center gap-2 mr-4">
          <div className="flex items-center gap-1.5 mr-1">
            <Layers size={12} className="text-[var(--text-muted)]" />
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Layers</span>
          </div>
          <LayerToggle
            id="toggle-hotspots"
            label="Hotspots"
            active={showHotspots}
            onToggle={() => setShowHotspots((v) => !v)}
            color="#ef4444"
          />
          <LayerToggle
            id="toggle-heatmap"
            label="Heatmap"
            active={showHeatmap}
            onToggle={() => setShowHeatmap((v) => !v)}
            color="#00d4c8"
          />
          <LayerToggle
            id="toggle-currents"
            label="Currents"
            active={showCurrents}
            onToggle={() => setShowCurrents((v) => !v)}
            color="#3b82f6"
          />
        </div>

        {/* API status + Refresh */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div
              className={`w-2 h-2 rounded-full ${
                apiStatus === "ok"
                  ? "bg-green-500"
                  : apiStatus === "error"
                  ? "bg-red-500"
                  : "bg-amber-500 animate-pulse"
              }`}
            />
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
              {apiStatus === "ok" ? "Live" : apiStatus === "error" ? "Offline" : "Connecting"}
            </span>
          </div>
          <button
            id="refresh-btn"
            onClick={handleRefresh}
            disabled={loading}
            className="w-8 h-8 rounded-lg bg-[var(--ocean-700)] hover:bg-[var(--ocean-600)]
                       flex items-center justify-center text-[var(--text-secondary)]
                       hover:text-white transition-all disabled:opacity-40"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </header>

      {/* ── Stats Ribbon (top-left below navbar) ── */}
      <div className="absolute top-16 left-4 z-10 flex flex-col gap-2 animate-fade-in-up">
        <StatCard
          id="stat-critical"
          label="Critical Zones"
          value={loading ? "—" : String(criticalCount)}
          icon={Zap}
          color="#ef4444"
        />
        <StatCard
          id="stat-high"
          label="High Risk Zones"
          value={loading ? "—" : String(highCount)}
          icon={Activity}
          color="#f97316"
        />
        <StatCard
          id="stat-density"
          label="Peak Density"
          value={loading ? "—" : `${maxDensity.toFixed(1)}%`}
          icon={Waves}
          color="#00d4c8"
        />
        <StatCard
          id="stat-coverage"
          label="Tracked Area"
          value={loading ? "—" : `${(trackedKm2 / 1000).toFixed(0)}K km²`}
          icon={Globe2}
          color="#3b82f6"
        />
      </div>

      {/* ── Hotspot Detail Panel (right side) ── */}
      {selectedHotspot && (
        <div className="absolute top-16 right-4 z-20">
          <HotspotPanel
            hotspot={selectedHotspot}
            onClose={() => setSelected(null)}
          />
        </div>
      )}

      {/* ── Time Slider (bottom-centre) ── */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
        <TimeSlider
          value={timestep}
          maxSteps={maxSteps}
          onChange={handleTimestepChange}
        />
      </div>

      {/* ── Legend Panel (bottom-left) ── */}
      <div className="absolute bottom-6 left-4 z-10">
        <LegendPanel />
      </div>

      {/* ── Loading overlay ── */}
      {loading && (
        <div className="absolute inset-0 z-30 flex items-center justify-center
                        bg-[rgba(2,10,18,0.85)] backdrop-blur-sm">
          <div className="flex flex-col items-center gap-5">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00d4c8] to-[#0ea5e9]
                            flex items-center justify-center teal-glow-lg">
              <Waves size={32} className="text-[var(--ocean-950)]" />
            </div>
            <div className="text-center">
              <p className="font-display text-xl font-bold text-white">
                Ocean Debris Monitor
              </p>
              <p className="text-sm text-[var(--text-secondary)] mt-1">
                Loading simulation data...
              </p>
            </div>
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-2 h-2 rounded-full bg-teal-500"
                  style={{
                    animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── API error banner ── */}
      {apiStatus === "error" && !loading && (
        <div
          id="api-error-banner"
          className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20
                     glass-panel px-5 py-3 flex items-center gap-3
                     border-red-500/30 bg-red-900/20 animate-fade-in-up"
        >
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-sm text-red-300">
            Backend offline — displaying demo data
          </span>
        </div>
      )}
    </main>
  );
}
