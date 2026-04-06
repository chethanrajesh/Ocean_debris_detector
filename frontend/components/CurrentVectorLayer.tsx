"use client";

/**
 * CurrentVectorLayer.tsx
 * Renders directional SVG arrow overlays on the map for ocean current vectors.
 * - Arrow length ∝ √(u² + v²)
 * - Opacity ∝ √(u² + v²)
 * - Direction from (u, v) components
 * Uses google.maps.OverlayView for pixel-accurate SVG rendering.
 */

import { useEffect, useRef } from "react";
import { CurrentNode } from "@/lib/api";

interface CurrentVectorLayerProps {
  map: google.maps.Map | null;
  nodes: CurrentNode[];
}

// ── Custom OverlayView for SVG arrows ────────────────────────────────────────
function createArrowOverlay(
  map: google.maps.Map,
  node: CurrentNode,
): google.maps.OverlayView {
  class ArrowOverlay extends window.google.maps.OverlayView {
    private div: HTMLDivElement | null = null;
    private node: CurrentNode;

    constructor(n: CurrentNode) {
      super();
      this.node = n;
    }

    onAdd() {
      this.div = document.createElement("div");
      this.div.style.position = "absolute";
      const panes = this.getPanes();
      panes?.overlayLayer.appendChild(this.div);
    }

    draw() {
      if (!this.div) return;
      const overlayProj = this.getProjection();
      const pos = overlayProj.fromLatLngToDivPixel(
        new window.google.maps.LatLng(this.node.lat, this.node.lon)
      );
      if (!pos) return;

      const { u, v } = this.node;
      const speed = Math.sqrt(u * u + v * v);
      const angle = Math.atan2(u, v) * (180 / Math.PI);
      const length = Math.min(12 + speed * 60, 28);
      const opacity = Math.min(0.3 + speed * 2, 0.85);

      this.div.style.left = `${pos.x - length / 2}px`;
      this.div.style.top  = `${pos.y - length / 2}px`;
      this.div.style.width  = `${length}px`;
      this.div.style.height = `${length}px`;
      this.div.style.opacity = String(opacity);
      this.div.style.transform = `rotate(${angle}deg)`;
      this.div.style.pointerEvents = "none";
      this.div.innerHTML = `
        <svg viewBox="0 0 24 24" width="${length}" height="${length}" xmlns="http://www.w3.org/2000/svg">
          <polygon points="12,2 20,20 12,15 4,20" fill="#00d4c8" stroke="#00d4c8" stroke-width="0.5"/>
        </svg>
      `;
    }

    onRemove() {
      this.div?.parentNode?.removeChild(this.div);
      this.div = null;
    }
  }

  const overlay = new ArrowOverlay(node);
  overlay.setMap(map);
  return overlay;
}

export default function CurrentVectorLayer({ map, nodes }: CurrentVectorLayerProps) {
  const overlaysRef = useRef<google.maps.OverlayView[]>([]);

  useEffect(() => {
    if (!map || !window.google?.maps) return;

    // Clear old overlays
    overlaysRef.current.forEach((o) => o.setMap(null));
    overlaysRef.current = [];

    // Sub-sample to max 500 arrows to keep performance OK
    const step = Math.max(1, Math.floor(nodes.length / 500));
    const sampled = nodes.filter((_, i) => i % step === 0);

    sampled.forEach((node) => {
      const overlay = createArrowOverlay(map, node);
      overlaysRef.current.push(overlay);
    });

    return () => {
      overlaysRef.current.forEach((o) => o.setMap(null));
      overlaysRef.current = [];
    };
  }, [map, nodes]);

  return null;
}
