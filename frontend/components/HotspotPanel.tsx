"use client";

/**
 * HotspotPanel.tsx
 * Detailed side panel shown when a hotspot circle is clicked.
 * Displays:
 * - Plastic density (progress bar)
 * - Severity badge
 * - Accumulation trend (icon + label)
 * - Movement direction (compass arrow)
 * - Estimated debris source region
 */

import { useEffect } from "react";
import {
  X,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Navigation,
  MapPin,
  Waves,
  Activity,
} from "lucide-react";
import { Hotspot } from "@/lib/api";

interface HotspotPanelProps {
  hotspot: Hotspot;
  onClose: () => void;
}

const SEVERITY_STYLES: Record<
  Hotspot["level"],
  { badgeClass: string; label: string; barClass: string; color: string }
> = {
  critical: {
    badgeClass: "badge-critical",
    label: "CRITICAL",
    barClass: "critical",
    color: "#ef4444",
  },
  high: {
    badgeClass: "badge-high",
    label: "HIGH",
    barClass: "high",
    color: "#f97316",
  },
  moderate: {
    badgeClass: "badge-moderate",
    label: "MODERATE",
    barClass: "moderate",
    color: "#eab308",
  },
};

const TREND_CONFIG = {
  increasing: {
    icon: TrendingUp,
    label: "Accumulating",
    color: "#ef4444",
  },
  stable: {
    icon: Minus,
    label: "Stable",
    color: "#eab308",
  },
  decreasing: {
    icon: TrendingDown,
    label: "Dispersing",
    color: "#22c55e",
  },
};

function CompassArrow({ u, v }: { u: number; v: number }) {
  const angle   = Math.atan2(u, v) * (180 / Math.PI);
  const speed   = Math.sqrt(u * u + v * v);
  const speedMs = (speed * 2).toFixed(2); // denormalize roughly

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="w-14 h-14 rounded-full bg-[var(--ocean-800)] border border-[var(--glass-border)]
                   flex items-center justify-center relative"
      >
        {/* Cardinal labels */}
        <span className="absolute top-1 left-1/2 -translate-x-1/2 text-[8px] text-[var(--text-muted)]">N</span>
        <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[8px] text-[var(--text-muted)]">S</span>
        <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[8px] text-[var(--text-muted)]">W</span>
        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[8px] text-[var(--text-muted)]">E</span>

        {/* Arrow */}
        <div style={{ transform: `rotate(${angle}deg)` }}>
          <Navigation size={20} className="text-teal" />
        </div>
      </div>
      <span className="font-mono text-xs text-[var(--text-secondary)]">
        {speedMs} m/s
      </span>
    </div>
  );
}

export default function HotspotPanel({ hotspot, onClose }: HotspotPanelProps) {
  const sev   = SEVERITY_STYLES[hotspot.level];
  const trend = TREND_CONFIG[hotspot.accumulation_trend];
  const TrendIcon = trend.icon;

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      id="hotspot-panel"
      className="glass-panel-dark w-[320px] p-5 flex flex-col gap-4 animate-slide-in"
      style={{ maxHeight: "calc(100vh - 200px)", overflowY: "auto" }}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle size={14} style={{ color: sev.color }} />
            <span
              className={`text-[10px] font-bold tracking-widest px-2 py-0.5 rounded-full ${sev.badgeClass}`}
            >
              {sev.label}
            </span>
          </div>
          <h2 className="font-display text-lg font-bold text-white leading-tight">
            Ocean Hotspot
          </h2>
          <div className="flex items-center gap-1.5 mt-1">
            <MapPin size={11} className="text-[var(--text-muted)]" />
            <span className="font-mono text-xs text-[var(--text-muted)]">
              {hotspot.latitude.toFixed(2)}°, {hotspot.longitude.toFixed(2)}°
            </span>
          </div>
        </div>
        <button
          id="hotspot-panel-close"
          onClick={onClose}
          className="w-7 h-7 rounded-full bg-[var(--ocean-700)] hover:bg-[var(--ocean-600)]
                     flex items-center justify-center text-[var(--text-secondary)]
                     hover:text-white transition-all"
        >
          <X size={14} />
        </button>
      </div>

      <hr className="divider" />

      {/* Plastic Density */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Waves size={13} className="text-[var(--text-secondary)]" />
            <span className="text-xs text-[var(--text-secondary)] font-medium">
              Plastic Density
            </span>
          </div>
          <span className="font-mono text-sm font-bold" style={{ color: sev.color }}>
            {(hotspot.plastic_density * 100).toFixed(1)}%
          </span>
        </div>
        <div className="progress-bar-track">
          <div
            className={`progress-bar-fill ${sev.barClass}`}
            style={{ width: `${hotspot.plastic_density * 100}%` }}
          />
        </div>
      </div>

      {/* Accumulation Trend + Compass (side-by-side) */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-1.5 mb-2">
            <Activity size={13} className="text-[var(--text-secondary)]" />
            <span className="text-xs text-[var(--text-secondary)] font-medium">
              Trend
            </span>
          </div>
          <div
            className="flex items-center gap-2 bg-[var(--ocean-800)] rounded-lg px-3 py-2"
          >
            <TrendIcon size={16} style={{ color: trend.color }} />
            <span className="text-sm font-semibold" style={{ color: trend.color }}>
              {trend.label}
            </span>
          </div>
        </div>

        <div className="flex-shrink-0">
          <div className="flex items-center gap-1.5 mb-2">
            <Navigation size={13} className="text-[var(--text-secondary)]" />
            <span className="text-xs text-[var(--text-secondary)] font-medium">
              Movement
            </span>
          </div>
          <CompassArrow u={hotspot.movement_vector.u} v={hotspot.movement_vector.v} />
        </div>
      </div>

      <hr className="divider" />

      {/* Source Region */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <MapPin size={13} className="text-[var(--text-secondary)]" />
          <span className="text-xs text-[var(--text-secondary)] font-medium">
            Estimated Source Region
          </span>
        </div>
        <div className="bg-[var(--ocean-800)] rounded-lg px-3 py-2.5">
          <p className="text-sm text-white font-medium leading-snug">
            {hotspot.source_estimate}
          </p>
        </div>
      </div>

      {/* Raw data */}
      <div className="bg-[var(--ocean-900)] rounded-lg px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-1.5">
          <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-medium">
            Current Vector
          </span>
        </div>
        <div className="font-mono text-xs text-[var(--text-secondary)] grid grid-cols-2 gap-1">
          <span>u = {hotspot.movement_vector.u.toFixed(4)} m/s</span>
          <span>v = {hotspot.movement_vector.v.toFixed(4)} m/s</span>
        </div>
      </div>
    </div>
  );
}
