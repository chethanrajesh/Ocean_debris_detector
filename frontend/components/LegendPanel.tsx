"use client";

/**
 * LegendPanel.tsx
 * Fixed bottom-left overlay showing:
 * - Hotspot severity color scale (critical / high / moderate)
 * - Heatmap density gradient key (blue → teal → yellow → red)
 * - Current vector arrow reference
 */

import { Waves, Wind, AlertTriangle } from "lucide-react";

export default function LegendPanel() {
  return (
    <div
      id="legend-panel"
      className="glass-panel-dark px-4 py-3 flex flex-col gap-3 w-[220px] animate-fade-in-up"
    >
      {/* Hotspot Severity */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <AlertTriangle size={12} className="text-[var(--text-muted)]" />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)]">
            Hotspot Severity
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {[
            { label: "Critical",  dot: "#ef4444", ring: "rgba(239,68,68,0.25)" },
            { label: "High",      dot: "#f97316", ring: "rgba(249,115,22,0.25)" },
            { label: "Moderate",  dot: "#eab308", ring: "rgba(234,179,8,0.25)" },
          ].map(({ label, dot, ring }) => (
            <div key={label} className="flex items-center gap-2">
              <div className="relative w-4 h-4 flex items-center justify-center flex-shrink-0">
                <div className="absolute inset-0 rounded-full" style={{ background: ring }} />
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ background: dot }}
                />
              </div>
              <span className="text-xs text-[var(--text-secondary)]">{label}</span>
            </div>
          ))}
        </div>
      </div>

      <hr className="divider" />

      {/* Density Heatmap */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <Waves size={12} className="text-[var(--text-muted)]" />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)]">
            Plastic Density
          </span>
        </div>
        <div className="relative h-3 rounded-full overflow-hidden"
             style={{
               background: "linear-gradient(to right, #003264, #008CC8, #00D4C8, #64E678, #F0DC00, #FF8C00, #E61E1E)"
             }}
        />
        <div className="flex justify-between mt-1">
          <span className="text-[9px] text-[var(--text-muted)]">Low</span>
          <span className="text-[9px] text-[var(--text-muted)]">High</span>
        </div>
      </div>

      <hr className="divider" />

      {/* Current vectors */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <Wind size={12} className="text-[var(--text-muted)]" />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)]">
            Ocean Currents
          </span>
        </div>
        <div className="flex items-center gap-2">
          <svg viewBox="0 0 24 24" width="16" height="16" className="text-teal flex-shrink-0">
            <polygon points="12,2 20,20 12,15 4,20" fill="#00d4c8" />
          </svg>
          <span className="text-xs text-[var(--text-secondary)]">
            Arrow length ∝ speed
          </span>
        </div>
      </div>
    </div>
  );
}
