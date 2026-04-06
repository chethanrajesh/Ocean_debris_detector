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
import { PredictionNode } from "@/lib/api";

interface HeatmapLayerProps {
  map: google.maps.Map | null;
  nodes: PredictionNode[];
}

// ── Colour look-up table (256 stops, pre-computed once) ──────────────────────
function buildColorLUT(): Uint8ClampedArray {
  // Gradient stops: [position 0–1, r, g, b]
  const stops: [number, number, number, number][] = [
    [0.00,   0,   0,  80],   // deep navy  (transparent in canvas)
    [0.15,   0,  50, 140],   // dark blue
    [0.30,   0, 140, 200],   // ocean blue
    [0.45,   0, 212, 200],   // teal
    [0.60, 100, 230, 120],   // green
    [0.72, 240, 220,   0],   // yellow
    [0.85, 255, 140,   0],   // orange
    [1.00, 230,  30,  30],   // red
  ];

  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    // Find surrounding stops
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) {
        lo = stops[s];
        hi = stops[s + 1];
        break;
      }
    }
    const span = hi[0] - lo[0] || 1;
    const f = (t - lo[0]) / span;
    lut[i * 4 + 0] = Math.round(lo[1] + (hi[1] - lo[1]) * f);
    lut[i * 4 + 1] = Math.round(lo[2] + (hi[2] - lo[2]) * f);
    lut[i * 4 + 2] = Math.round(lo[3] + (hi[3] - lo[3]) * f);
    lut[i * 4 + 3] = i < 10 ? 0 : Math.min(255, Math.round(i * 0.85)); // alpha
  }
  return lut;
}

const COLOR_LUT = buildColorLUT();

// ── CanvasHeatmapOverlay ─────────────────────────────────────────────────────
function createCanvasOverlay(
  map: google.maps.Map,
  getNodes: () => PredictionNode[],
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

      // Scale the gaussian blob radius by DPR so it looks the same visually
      const R = Math.round(this.RADIUS * dpr);

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
        pixels[i * 4]     = COLOR_LUT[lutIdx];
        pixels[i * 4 + 1] = COLOR_LUT[lutIdx + 1];
        pixels[i * 4 + 2] = COLOR_LUT[lutIdx + 2];
        pixels[i * 4 + 3] = COLOR_LUT[lutIdx + 3];
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

// ── React component ──────────────────────────────────────────────────────────
export default function HeatmapLayer({ map, nodes }: HeatmapLayerProps) {
  const overlayRef  = useRef<google.maps.OverlayView | null>(null);
  // Keep a mutable ref to latest nodes so the overlay's draw() always reads current data
  const nodesRef    = useRef<PredictionNode[]>(nodes);
  const listenersRef = useRef<google.maps.MapsEventListener[]>([]);

  // Update the mutable ref whenever nodes change
  useEffect(() => {
    nodesRef.current = nodes;
    // Force a redraw if the overlay already exists
    if (overlayRef.current) {
      (overlayRef.current as google.maps.OverlayView & { draw: () => void }).draw?.();
    }
  }, [nodes]);

  // Create overlay once when map is ready
  useEffect(() => {
    if (!map || !window.google?.maps) return;

    const overlay = createCanvasOverlay(map, () => nodesRef.current);
    overlayRef.current = overlay;

    // Redraw on map events
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
