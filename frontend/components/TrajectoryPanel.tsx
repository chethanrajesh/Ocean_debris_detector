"use client";
/**
 * TrajectoryPanel.tsx
 * Right-side detail panel shown when a particle trajectory is clicked.
 * Shows origin, current position, density over time, and status.
 */

import { X, Anchor, Navigation, Waves, Clock } from "lucide-react";
import type { ParticleTrajectory, SourceType } from "@/lib/api";

const SNAPSHOT_DAYS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90];

const SOURCE_LABELS: Record<number, { label: string; color: string; icon: typeof Waves }> = {
  0: { label: "Active",     color: "#00d4c8", icon: Waves   },
  1: { label: "Beached",    color: "#ef4444", icon: Anchor  },
  2: { label: "Converging", color: "#a855f7", icon: Navigation },
};

interface Props {
  trajectory:   ParticleTrajectory;
  snapshotIdx:  number;
  onClose:      () => void;
}

export default function TrajectoryPanel({ trajectory, snapshotIdx, onClose }: Props) {
  const snap     = trajectory.snapshots[Math.min(snapshotIdx, trajectory.snapshots.length - 1)];
  const lat      = snap[0].toFixed(3);
  const lon      = snap[1].toFixed(3);
  const density  = (snap[2] * 100).toFixed(1);
  const age      = snap[3].toFixed(0);
  const srcType  = snap[4] as SourceType;
  const srcInfo  = SOURCE_LABELS[srcType] ?? SOURCE_LABELS[0];
  const Icon     = srcInfo.icon;

  const finalSnap   = trajectory.snapshots[trajectory.snapshots.length - 1];
  const finalType   = finalSnap[4] as SourceType;
  const finalInfo   = SOURCE_LABELS[finalType] ?? SOURCE_LABELS[0];

  // Mini density chart values
  const densities = trajectory.snapshots.map(s => s[2]);
  const maxD      = Math.max(...densities, 0.001);

  return (
    <div
      id="trajectory-panel"
      style={{
        width: 280,
        maxHeight: "calc(100vh - 200px)",
        overflowY: "auto",
        background: "rgba(4, 15, 28, 0.92)",
        backdropFilter: "blur(16px)",
        border: "1px solid rgba(0, 212, 200, 0.15)",
        borderRadius: 14,
        padding: "16px",
        color: "white",
        fontFamily: "var(--font-sans, sans-serif)",
        boxShadow: "0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04)",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
            <div style={{
              width: 24, height: 24, borderRadius: 7,
              background: `${srcInfo.color}22`,
              border: `1px solid ${srcInfo.color}44`,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Icon size={12} style={{ color: srcInfo.color }} />
            </div>
            <span style={{ fontSize: 11, fontWeight: 700, color: srcInfo.color, letterSpacing: "0.06em", textTransform: "uppercase" }}>
              {srcInfo.label} Particle #{trajectory.id}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.04em" }}>
            {trajectory.source_label}
          </div>
        </div>
        <button
          id="trajectory-panel-close"
          onClick={onClose}
          style={{
            width: 26, height: 26, borderRadius: 7,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            cursor: "pointer", color: "#94a3b8",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <X size={12} />
        </button>
      </div>

      {/* Position row */}
      <div style={{
        background: "rgba(0,212,200,0.06)", border: "1px solid rgba(0,212,200,0.12)",
        borderRadius: 9, padding: "10px 12px", marginBottom: 10,
      }}>
        <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
          Current Position (Day {age})
        </div>
        <div style={{ fontSize: 14, fontWeight: 700, color: "white", fontVariantNumeric: "tabular-nums" }}>
          {lat}°, {lon}°
        </div>
      </div>

      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
        {[
          { label: "Density",    value: `${density}%`,           color: "#3b82f6" },
          { label: "Age",        value: `${age} days`,           color: "#f59e0b" },
          { label: "Origin",     value: `${trajectory.origin.lat.toFixed(1)}°, ${trajectory.origin.lon.toFixed(1)}°`, color: "#10b981" },
          { label: "Final State", value: finalInfo.label,        color: finalInfo.color },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 8, padding: "8px 10px",
          }}>
            <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: 12, fontWeight: 700, color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Density mini-chart */}
      <div style={{ marginBottom: 4 }}>
        <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
          Density Over 90 Days
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 36 }}>
          {densities.map((d, i) => {
            const h   = Math.max(3, (d / maxD) * 36);
            const isActive = i === Math.min(snapshotIdx, densities.length - 1);
            return (
              <div
                key={i}
                title={`Day ${SNAPSHOT_DAYS[i]}: ${(d * 100).toFixed(1)}%`}
                style={{
                  flex: 1, height: h,
                  background: isActive ? "#00d4c8" : "rgba(59,130,246,0.4)",
                  borderRadius: 2,
                  transition: "height 0.3s ease",
                }}
              />
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3 }}>
          <span style={{ fontSize: 8, color: "#475569" }}>Day 0</span>
          <span style={{ fontSize: 8, color: "#475569" }}>Day 90</span>
        </div>
      </div>
    </div>
  );
}
