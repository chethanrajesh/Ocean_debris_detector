"use client";
/**
 * TrajectoryLayer.tsx
 * Renders 90-day particle trajectories as animated Google Maps Polylines.
 *
 * Each particle gets:
 *   - A static polyline showing its full 14-snapshot path (faint)
 *   - An animated dot that travels along the path when playing
 *   - Click → fires onParticleClick with full trajectory data
 *
 * Colour modes (prop: colorBy):
 *   "age"        — cyan (young) → amber (old)
 *   "density"    — blue (sparse) → red (dense)
 *   "sourceType" — teal (active), red (beached), purple (converging)
 */

import { useEffect, useRef, useCallback } from "react";
import type { ParticleTrajectory, SourceType } from "@/lib/api";

// ── Colour helpers ────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * Math.clamp01(t);
}
// eslint-disable-next-line @typescript-eslint/no-namespace
declare global { interface Math { clamp01(t: number): number; } }
Math.clamp01 = (t: number) => Math.min(1, Math.max(0, t));

function ageColor(ageDays: number): string {
  // 0 days = cyan #00d4c8, 90 days = amber #f59e0b
  const t = Math.clamp01(ageDays / 90);
  const r = Math.round(lerp(0, 245, t));
  const g = Math.round(lerp(212, 158, t));
  const b = Math.round(lerp(200, 11, t));
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

function densityColor(density: number): string {
  // 0 = blue #3b82f6, 1 = red #ef4444
  const t = Math.clamp01(density);
  const r = Math.round(lerp(59, 239, t));
  const g = Math.round(lerp(130, 68, t));
  const b = Math.round(lerp(246, 68, t));
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

function sourceTypeColor(st: SourceType): string {
  switch (st) {
    case 0: return "#00d4c8"; // active  — teal
    case 1: return "#ef4444"; // beached — red
    case 2: return "#a855f7"; // converging — purple
    default: return "#94a3b8";
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  map:            google.maps.Map | null;
  trajectories:   ParticleTrajectory[];
  snapshotIndex:  number;   // 0–13 (which of the 14 snapshots to highlight)
  colorBy:        "age" | "density" | "sourceType";
  onParticleClick?: (traj: ParticleTrajectory) => void;
}

export default function TrajectoryLayer({
  map,
  trajectories,
  snapshotIndex,
  colorBy,
  onParticleClick,
}: Props) {
  // Refs: one polyline + one marker dot per particle
  const polylinesRef = useRef<google.maps.Polyline[]>([]);
  const dotsRef      = useRef<google.maps.Marker[]>([]);
  const listenersRef = useRef<google.maps.MapsEventListener[]>([]);

  // ── Cleanup helper ────────────────────────────────────────────────────────
  const clearAll = useCallback(() => {
    polylinesRef.current.forEach(p => p.setMap(null));
    dotsRef.current.forEach(d => d.setMap(null));
    listenersRef.current.forEach(l => google.maps.event.removeListener(l));
    polylinesRef.current  = [];
    dotsRef.current       = [];
    listenersRef.current  = [];
  }, []);

  // ── Draw trajectories ────────────────────────────────────────────────────
  useEffect(() => {
    if (!map || !trajectories.length) return;
    clearAll();

    const newPolylines: google.maps.Polyline[] = [];
    const newDots:      google.maps.Marker[]   = [];
    const newListeners: google.maps.MapsEventListener[] = [];

    trajectories.forEach((traj) => {
      const snaps = traj.snapshots;
      if (!snaps || snaps.length < 2) return;

      // Build path from all snapshots
      const path = snaps.map(s => ({ lat: s[0], lng: s[1] }));

      // Determine colour from the snapshot at snapshotIndex (or last available)
      const snapIdx  = Math.min(snapshotIndex, snaps.length - 1);
      const snap     = snaps[snapIdx];
      const ageDays  = snap[3];
      const density  = snap[2];
      const srcType  = snap[4] as SourceType;

      let strokeColor: string;
      switch (colorBy) {
        case "age":        strokeColor = ageColor(ageDays);        break;
        case "density":    strokeColor = densityColor(density);    break;
        case "sourceType": strokeColor = sourceTypeColor(srcType); break;
        default:           strokeColor = "#00d4c8";
      }

      // Faint full-path polyline
      const polyline = new google.maps.Polyline({
        path,
        map,
        strokeColor,
        strokeOpacity: 0.25,
        strokeWeight:  1.2,
        geodesic: true,
        zIndex: 1,
      });

      // Animated dot at current snapshot position
      const currentPos = { lat: snap[0], lng: snap[1] };
      const dot = new google.maps.Marker({
        position: currentPos,
        map,
        title: traj.source_label,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale:       srcType === 1 ? 5 : 4,  // beached slightly larger
          fillColor:   strokeColor,
          fillOpacity: srcType === 1 ? 0.95 : 0.8,
          strokeColor: "#ffffff",
          strokeWeight: srcType === 1 ? 1.5 : 1,
        },
        zIndex: 2,
        optimized: true,
      });

      // Click handler
      if (onParticleClick) {
        const listener = dot.addListener("click", () => onParticleClick(traj));
        newListeners.push(listener);
      }

      newPolylines.push(polyline);
      newDots.push(dot);
    });

    polylinesRef.current = newPolylines;
    dotsRef.current      = newDots;
    listenersRef.current = newListeners;

    return clearAll;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, trajectories, colorBy]);

  // ── Update dot positions when snapshot changes (no full redraw) ───────────
  useEffect(() => {
    dotsRef.current.forEach((dot, i) => {
      const traj  = trajectories[i];
      if (!traj) return;
      const snaps = traj.snapshots;
      const idx   = Math.min(snapshotIndex, snaps.length - 1);
      const snap  = snaps[idx];
      dot.setPosition({ lat: snap[0], lng: snap[1] });

      const srcType = snap[4] as SourceType;
      const ageDays = snap[3];
      const density = snap[2];

      let color: string;
      switch (colorBy) {
        case "age":        color = ageColor(ageDays);        break;
        case "density":    color = densityColor(density);    break;
        case "sourceType": color = sourceTypeColor(srcType); break;
        default:           color = "#00d4c8";
      }

      dot.setIcon({
        path: google.maps.SymbolPath.CIRCLE,
        scale:       srcType === 1 ? 5 : 4,
        fillColor:   color,
        fillOpacity: srcType === 1 ? 0.95 : 0.8,
        strokeColor: "#ffffff",
        strokeWeight: srcType === 1 ? 1.5 : 1,
      });
    });
  }, [snapshotIndex, colorBy, trajectories]);

  return null; // renders directly onto map via Google Maps API
}
