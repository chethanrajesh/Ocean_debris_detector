"use client";

/**
 * TimeSlider.tsx
 *
 * Slider from t=0 to t=360 (= 90 days at 6h resolution, 4 steps/day).
 *
 * ─ t = 0   → 📍 REAL-TIME  mode  (green pill, confirmed satellite/buoy data)
 * ─ t > 0   → 🔮 PREDICTING mode  (purple pill, GNN + Lagrangian forecast)
 *
 * The slider track transitions from green (real) to purple (predicted).
 */

import { useMemo, useCallback } from "react";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import type { PredictionMode } from "@/lib/api";

const MAX_STEP = 360;   // 90 days × 4 steps/day

interface TimeSliderProps {
  value:     number;
  maxSteps:  number;
  onChange:  (step: number) => void;
  /** Optional: override derived mode (passed from parent). */
  mode?:     PredictionMode;
}

function stepToDate(step: number): string {
  const base  = new Date();
  const date  = new Date(base.getTime() + step * 6 * 60 * 60 * 1000);
  return date.toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });
}

function stepToDayLabel(step: number): string {
  if (step === 0) return "Now";
  const days = (step * 6) / 24;
  if (days < 1)   return `+${step * 6}h`;
  if (days === 1) return "+1 day";
  return `+${Math.round(days)} days`;
}

// ── Mode visual config ────────────────────────────────────────────────────────
const MODE_CONFIG = {
  real: {
    pill:      "📍 LIVE",
    pillBg:    "rgba(16,185,129,0.18)",
    pillBorder:"rgba(16,185,129,0.50)",
    pillColor: "#10b981",
    trackEnd:  "#10b981",
    label:     "Real-time · Satellite & Buoy Data",
    labelColor:"#10b981",
  },
  predicted: {
    pill:      "🔮 PREDICTING",
    pillBg:    "rgba(168,85,247,0.18)",
    pillBorder:"rgba(168,85,247,0.50)",
    pillColor: "#a855f7",
    trackEnd:  "#a855f7",
    label:     "GNN Forecast · Lagrangian Simulation",
    labelColor:"#a855f7",
  },
} as const;

export default function TimeSlider({
  value,
  maxSteps,
  onChange,
  mode: modeProp,
}: TimeSliderProps) {
  const cap      = Math.min(maxSteps - 1, MAX_STEP);
  const clamped  = Math.min(value, cap);
  const mode     = modeProp ?? (clamped === 0 ? "real" : "predicted");
  const cfg      = MODE_CONFIG[mode];
  const progress = (clamped / cap) * 100;
  const dateLabel = useMemo(() => stepToDate(clamped),   [clamped]);
  const dayLabel  = useMemo(() => stepToDayLabel(clamped), [clamped]);

  const dec = useCallback(() => onChange(Math.max(0, clamped - 4)),   [clamped, onChange]);
  const inc = useCallback(() => onChange(Math.min(cap, clamped + 4)), [clamped, cap, onChange]);

  // Tick positions for Now / +30d / +60d / +90d
  const ticks = [
    { label: "Now",   step: 0   },
    { label: "+30d",  step: 120 },
    { label: "+60d",  step: 240 },
    { label: "+90d",  step: 360 },
  ];

  return (
    <div className="glass-panel animate-fade-in-up"
      style={{ padding: "14px 20px", width: 440 }}>

      {/* ── Header row: icon + "Time Projection" + MODE pill ── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Calendar size={13} style={{ color: "var(--teal-500)" }} />
          <span style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase",
                         letterSpacing: "0.1em", color: "var(--teal-500)" }}>
            Time Projection
          </span>
        </div>

        {/* Mode pill */}
        <span style={{
          fontSize: 10, fontWeight: 700, padding: "2px 10px",
          borderRadius: 99, letterSpacing: "0.06em",
          background: cfg.pillBg,
          border: `1px solid ${cfg.pillBorder}`,
          color: cfg.pillColor,
          transition: "all 0.4s ease",
        }}>
          {cfg.pill}
        </span>
      </div>

      {/* ── Date + day offset + mode label ── */}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 2 }}>
        <div>
          <span style={{ fontFamily: "Syne, sans-serif", fontSize: 18, fontWeight: 700, color: "white" }}>
            {dateLabel}
          </span>
          <span style={{ marginLeft: 8, fontSize: 13, color: "var(--text-secondary)" }}>
            {dayLabel}
          </span>
        </div>
        <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10,
                       color: "var(--text-muted)", background: "var(--ocean-800)",
                       padding: "2px 7px", borderRadius: 6 }}>
          t={clamped}
        </span>
      </div>

      {/* Mode description */}
      <div style={{ fontSize: 10, color: cfg.labelColor, marginBottom: 10,
                    transition: "color 0.4s ease", letterSpacing: "0.04em" }}>
        {cfg.label}
      </div>

      {/* ── Slider + step buttons ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button id="time-slider-decrement" onClick={dec}
          style={{
            width: 28, height: 28, borderRadius: "50%", border: "none", cursor: "pointer",
            background: "var(--ocean-700)", color: "var(--text-secondary)", flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s",
          }}>
          <ChevronLeft size={14} />
        </button>

        <div style={{ flex: 1, position: "relative" }}>
          <input
            id="time-slider"
            type="range"
            min={0}
            max={cap}
            value={clamped}
            onChange={(e) => onChange(Number(e.target.value))}
            style={{
              width: "100%",
              background: `linear-gradient(to right, ${cfg.trackEnd} ${progress}%, #133558 ${progress}%)`,
              transition: "background 0.3s ease",
            }}
          />
        </div>

        <button id="time-slider-increment" onClick={inc}
          style={{
            width: 28, height: 28, borderRadius: "50%", border: "none", cursor: "pointer",
            background: "var(--ocean-700)", color: "var(--text-secondary)", flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s",
          }}>
          <ChevronRight size={14} />
        </button>
      </div>

      {/* ── Tick marks ── */}
      <div style={{ position: "relative", marginTop: 4, paddingLeft: 38, paddingRight: 38 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          {ticks.map(({ label, step }) => (
            <button
              key={label}
              onClick={() => onChange(Math.min(step, cap))}
              style={{
                fontSize: 9, border: "none", background: "none", cursor: "pointer",
                color: clamped === step ? cfg.pillColor : "var(--text-muted)",
                fontWeight: clamped === step ? 700 : 400,
                transition: "color 0.2s",
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
