"use client";

/**
 * HeatmapLayer.tsx
 *
 * Custom canvas-based heatmap — replaces the deprecated
 * google.maps.visualization.HeatmapLayer (deprecated May 2025, removed May 2026).
 *
 * Approach: google.maps.OverlayView + HTMLCanvasElement
 *   - Canvas is sized to the whole map pane on every draw() call
 *   - Each node draws a radial gradient (alpha ∝ density, radius = ~28 px)
 *   - Screen pixel ← world coordinate projection via getProjection()
 *   - Colour scale: deep-blue → teal → green → yellow → orange → red
 *   - Redraws on idle / zoom / bounds_changed and when `nodes` changes
 */

import { useEffect, useRef } from "react";
import { PredictionNode, PredictionMode } from "@/lib/api";

interface HeatmapLayerProps {
  map:  google.maps.Map | null;
  nodes: PredictionNode[];
  /** "real" (t=0, satellite-seeded) or "predicted" (GNN forecast, t>0) */
  mode?: PredictionMode;
}

// ── Colour look-up table builder ─────────────────────────────────────────────
type ColorStop = [number, number, number, number];

function buildColorLUT(stops: ColorStop[], globalAlpha = 1.0): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) { lo = stops[s]; hi = stops[s + 1]; break; }
    }
    const span = hi[0] - lo[0] || 1;
    const f    = (t - lo[0]) / span;
    lut[i * 4 + 0] = Math.round(lo[1] + (hi[1] - lo[1]) * f);
    lut[i * 4 + 1] = Math.round(lo[2] + (hi[2] - lo[2]) * f);
    lut[i * 4 + 2] = Math.round(lo[3] + (hi[3] - lo[3]) * f);
    const rawAlpha  = i < 10 ? 0 : Math.min(255, Math.round(i * 0.85));
    lut[i * 4 + 3] = Math.round(rawAlpha * globalAlpha);
  }
  return lut;
}

// Real data: warm, fully saturated — «this is confirmed»
const REAL_STOPS: ColorStop[] = [
  [0.00,   0,   0,  80],
  [0.15,   0,  50, 150],
  [0.30,   0, 150, 210],
  [0.45,   0, 220, 200],
  [0.60, 120, 235, 100],
  [0.72, 250, 220,   0],
  [0.85, 255, 130,   0],
  [1.00, 230,  20,  20],
];
const REAL_LUT = buildColorLUT(REAL_STOPS, 1.0);

// Predicted data: cooler (blues/purples dominate), 80% opacity — «this is a forecast»
const PRED_STOPS: ColorStop[] = [
  [0.00,   0,   0, 100],
  [0.15,   0,  40, 160],
  [0.30,  30, 100, 210],
  [0.45,  60, 180, 220],
  [0.60, 100, 200, 190],
  [0.72, 180, 180,  50],
  [0.85, 220, 100,  30],
  [1.00, 190,  30, 180],   // magenta-red for peak predictions
];
const PRED_LUT = buildColorLUT(PRED_STOPS, 0.78);


// ── CanvasHeatmapOverlay ─────────────────────────────────────────────────────
function createCanvasOverlay(
  map: google.maps.Map,
  getNodes: () => PredictionNode[],
  getLut:   () => Uint8ClampedArray,
): google.maps.OverlayView {
  class CanvasHeatmapOverlay extends window.google.maps.OverlayView {
    private canvas: HTMLCanvasElement | null = null;
    private RADIUS = 24;

    onAdd() {
      const canvas = document.createElement("canvas");
      canvas.style.position   = "absolute";
      canvas.style.top        = "0";
      canvas.style.left       = "0";
      canvas.style.pointerEvents = "none";
      canvas.style.opacity    = "0.78";
      this.canvas = canvas;
      this.getPanes()!.overlayMouseTarget.appendChild(canvas);
    }


    draw() {
      if (!this.canvas) return;
      const proj = this.getProjection();
      if (!proj) return;

      const bounds = map.getBounds();
      if (!bounds) return;
      const mapDiv = map.getDiv();

      // ── Anchor canvas to the map viewport ───────────────────────────────────
      // The overlayMouseTarget pane has its own top/left offset relative to the
      // map container. We must subtract that offset so the canvas top-left
      // corner perfectly aligns with the map's (0,0) pixel corner.
      // This is what prevents geographic drift on zoom and pan.
      const pane       = this.getPanes()!.overlayMouseTarget as HTMLElement;
      const mapRect    = mapDiv.getBoundingClientRect();
      const paneRect   = pane.getBoundingClientRect();
      const offsetLeft = mapRect.left - paneRect.left;
      const offsetTop  = mapRect.top  - paneRect.top;
      this.canvas.style.left = `${offsetLeft}px`;
      this.canvas.style.top  = `${offsetTop}px`;

      // ── Device Pixel Ratio ────────────────────────────────────────────────
      const dpr = window.devicePixelRatio || 1;
      const W   = mapDiv.offsetWidth;
      const H   = mapDiv.offsetHeight;
      const PW  = Math.round(W * dpr);
      const PH  = Math.round(H * dpr);

      this.canvas.width        = PW;
      this.canvas.height       = PH;
      this.canvas.style.width  = `${W}px`;
      this.canvas.style.height = `${H}px`;


      const nodes = getNodes();
      if (!nodes.length) return;

      const lut = getLut();
      const R   = Math.round(this.RADIUS * dpr);

      // ── 1. Float32 accumulator ─────────────────────────────────────────────
      const acc = new Float32Array(PW * PH);

      for (const node of nodes) {
        const latLng = new window.google.maps.LatLng(node.lat, node.lon);
        const pt = proj.fromLatLngToContainerPixel(latLng);
        if (!pt) continue;

        // Convert CSS coordinates → physical canvas coordinates
        const cx = Math.round(pt.x * dpr);
        const cy = Math.round(pt.y * dpr);
        if (cx < -R || cx > PW + R || cy < -R || cy > PH + R) continue;

        const weight = node.density;
        const r2     = R * R;
        const x0 = Math.max(0, cx - R);
        const x1 = Math.min(PW - 1, cx + R);
        const y0 = Math.max(0, cy - R);
        const y1 = Math.min(PH - 1, cy + R);

        for (let py = y0; py <= y1; py++) {
          for (let px = x0; px <= x1; px++) {
            const dx = px - cx;
            const dy = py - cy;
            const d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            const falloff = Math.exp(-3 * d2 / r2);
            acc[py * PW + px] += weight * falloff;
          }
        }
      }

      // ── 2. Find peak, normalise, apply colour LUT ─────────────────────────
      let peak = 0;
      for (let i = 0; i < acc.length; i++) { if (acc[i] > peak) peak = acc[i]; }
      if (peak === 0) return;

      const ctx     = this.canvas.getContext("2d", { willReadFrequently: true })!;
      const imgData = ctx.createImageData(PW, PH);
      const pixels  = imgData.data;

      for (let i = 0; i < acc.length; i++) {
        if (acc[i] < 0.001) { pixels[i * 4 + 3] = 0; continue; }
        const t      = Math.pow(acc[i] / peak, 0.6);
        const lutIdx = Math.min(255, Math.round(t * 255)) * 4;
        pixels[i * 4]     = lut[lutIdx];
        pixels[i * 4 + 1] = lut[lutIdx + 1];
        pixels[i * 4 + 2] = lut[lutIdx + 2];
        pixels[i * 4 + 3] = lut[lutIdx + 3];
      }

      ctx.putImageData(imgData, 0, 0);
    }



    onRemove() {
      this.canvas?.parentNode?.removeChild(this.canvas);
      this.canvas = null;
    }
  }

  const overlay = new CanvasHeatmapOverlay();
  overlay.setMap(map);
  return overlay;
}

// ── React component ───────────────────────────────────────────────────────────
export default function HeatmapLayer({ map, nodes, mode = "real" }: HeatmapLayerProps) {
  const overlayRef   = useRef<google.maps.OverlayView | null>(null);
  const nodesRef     = useRef<PredictionNode[]>(nodes);
  const listenersRef = useRef<google.maps.MapsEventListener[]>([]);
  const modeRef      = useRef<PredictionMode>(mode);

  // Keep refs current
  useEffect(() => { nodesRef.current = nodes; triggerDraw(); }, [nodes]);
  useEffect(() => { modeRef.current  = mode;  triggerDraw(); }, [mode]);

  function triggerDraw() {
    if (overlayRef.current) {
      (overlayRef.current as google.maps.OverlayView & { draw: () => void }).draw?.();
    }
  }

  // Create overlay once when map is ready
  useEffect(() => {
    if (!map || !window.google?.maps) return;

    const overlay = createCanvasOverlay(
      map,
      () => nodesRef.current,
      () => modeRef.current === "real" ? REAL_LUT : PRED_LUT,
    );
    overlayRef.current = overlay;

    const events = ["idle", "zoom_changed", "bounds_changed", "resize"];
    listenersRef.current = events.map((evt) =>
      window.google.maps.event.addListener(map, evt, () => {
        (overlay as google.maps.OverlayView & { draw: () => void }).draw?.();
      })
    );

    return () => {
      listenersRef.current.forEach((l) =>
        window.google.maps.event.removeListener(l)
      );
      listenersRef.current = [];
      overlay.setMap(null);
      overlayRef.current = null;
    };
  }, [map]);

  return null;
}
