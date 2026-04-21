"use client";
/**
 * BeachingRiskLayer.tsx
 * Renders coastal cells from /trajectories/beaching-risk as
 * red-gradient circles on the Google Map.
 *
 * Size and opacity scale with the normalised risk [0, 1].
 */

import { useEffect, useRef, useCallback } from "react";
import type { BeachingCell } from "@/lib/api";

interface Props {
  map:   google.maps.Map | null;
  cells: BeachingCell[];
}

export default function BeachingRiskLayer({ map, cells }: Props) {
  const markersRef = useRef<google.maps.Circle[]>([]);

  const clearAll = useCallback(() => {
    markersRef.current.forEach(c => c.setMap(null));
    markersRef.current = [];
  }, []);

  useEffect(() => {
    if (!map || !cells.length) return;
    clearAll();

    const newCircles: google.maps.Circle[] = [];

    cells.forEach((cell) => {
      const t    = Math.min(1, Math.max(0, cell.risk));
      // Colour interpolation: amber → deep red
      const r    = Math.round(239 + (180 - 239) * t);       // but clamp
      const g    = Math.round(68  * (1 - t));
      const b    = Math.round(68  * (1 - t * 0.8));
      const hex  = `#${r.toString(16).padStart(2,"0")}${g.toString(16).padStart(2,"0")}${b.toString(16).padStart(2,"0")}`;

      const circle = new google.maps.Circle({
        center:        { lat: cell.lat, lng: cell.lon },
        map,
        radius:        25_000 + t * 55_000,   // 25–80 km radius
        strokeColor:   hex,
        strokeOpacity: 0.6,
        strokeWeight:  1,
        fillColor:     hex,
        fillOpacity:   0.12 + t * 0.28,
        zIndex:        0,
      });

      newCircles.push(circle);
    });

    markersRef.current = newCircles;
    return clearAll;
  }, [map, cells, clearAll]);

  return null;
}
