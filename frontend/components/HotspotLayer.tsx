"use client";

/**
 * HotspotLayer.tsx
 * Renders Google Maps Circle markers for each ocean plastic hotspot.
 * - Colors:  red (critical) | orange (high) | yellow (moderate)
 * - Radius:  scaled by plastic_density × 80,000 m
 * - Click:   calls onHotspotClick(hotspot)
 */

import { useEffect, useRef } from "react";
import { Hotspot } from "@/lib/api";

interface HotspotLayerProps {
  map: google.maps.Map | null;
  hotspots: Hotspot[];
  onHotspotClick: (hotspot: Hotspot) => void;
}

const SEVERITY_CONFIG: Record<
  Hotspot["level"],
  { fillColor: string; strokeColor: string; fillOpacity: number }
> = {
  critical: { fillColor: "#ef4444", strokeColor: "#fca5a5", fillOpacity: 0.35 },
  high:     { fillColor: "#f97316", strokeColor: "#fdba74", fillOpacity: 0.30 },
  moderate: { fillColor: "#eab308", strokeColor: "#fde047", fillOpacity: 0.25 },
};

export default function HotspotLayer({
  map,
  hotspots,
  onHotspotClick,
}: HotspotLayerProps) {
  const circlesRef = useRef<google.maps.Circle[]>([]);

  useEffect(() => {
    if (!map || !window.google?.maps) return;

    // Clear previous circles
    circlesRef.current.forEach((c) => c.setMap(null));
    circlesRef.current = [];

    hotspots.forEach((hotspot) => {
      const cfg = SEVERITY_CONFIG[hotspot.level];
      const radius = hotspot.plastic_density * 90000 + 30000; // 30–120 km

      const circle = new window.google.maps.Circle({
        map,
        center: { lat: hotspot.latitude, lng: hotspot.longitude },
        radius,
        strokeColor:   cfg.strokeColor,
        strokeOpacity: 0.9,
        strokeWeight:  1.5,
        fillColor:     cfg.fillColor,
        fillOpacity:   cfg.fillOpacity,
        clickable: true,
        zIndex: hotspot.level === "critical" ? 3 : hotspot.level === "high" ? 2 : 1,
      });

      circle.addListener("click", () => onHotspotClick(hotspot));

      // Pulse marker at center for critical hotspots
      if (hotspot.level === "critical") {
        const innerCircle = new window.google.maps.Circle({
          map,
          center: { lat: hotspot.latitude, lng: hotspot.longitude },
          radius: 15000,
          strokeColor:   "#fca5a5",
          strokeOpacity: 1,
          strokeWeight:  1.5,
          fillColor:     "#ef4444",
          fillOpacity:   0.6,
          clickable: true,
          zIndex: 4,
        });
        innerCircle.addListener("click", () => onHotspotClick(hotspot));
        circlesRef.current.push(innerCircle);
      }

      circlesRef.current.push(circle);
    });

    return () => {
      circlesRef.current.forEach((c) => c.setMap(null));
      circlesRef.current = [];
    };
  }, [map, hotspots, onHotspotClick]);

  return null;
}
