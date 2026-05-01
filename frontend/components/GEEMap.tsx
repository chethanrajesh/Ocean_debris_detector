"use client";

/**
 * GEEMap.tsx
 * Mounts a Google Maps/Earth Engine map centred on the Pacific Ocean.
 * - Base layer: ETOPO1 bathymetry visualization via Earth Engine
 * - Overlay: Sentinel-2 true-colour composite
 * - Exposes mapRef so parent can attach additional layers
 */

import { useEffect, useRef } from "react";

interface GEEMapProps {
  onMapReady?: (map: google.maps.Map) => void;
}

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ee?: any;
    google?: typeof google;
    initGEEMap?: () => void;
  }
}

export default function GEEMap({ onMapReady }: GEEMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef       = useRef<google.maps.Map | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const initMap = () => {
      if (!window.google?.maps) {
        setTimeout(initMap, 300);
        return;
      }

      const map = new window.google.maps.Map(containerRef.current!, {
        mapId: "9569ac6eea69b8ef9367d3ea",
        mapTypeId: "hybrid",
        center: { lat: 10, lng: 0 },
        zoom: 3,
        minZoom: 3,
        maxZoom: 20, // Increased to allow high-resolution 3D tilting
        disableDefaultUI: false,
        
        // Explicitly enable 3D interactions
        tiltInteractionEnabled: true,
        headingInteractionEnabled: true,

        // Customised UI controls to match modern vector map layout
        zoomControl: true,
        zoomControlOptions: {
          position: window.google.maps.ControlPosition.RIGHT_BOTTOM,
        },
        fullscreenControl: true,
        fullscreenControlOptions: {
          position: window.google.maps.ControlPosition.RIGHT_BOTTOM,
        },
        tiltControl: true,
        tiltControlOptions: {
          position: window.google.maps.ControlPosition.RIGHT_BOTTOM,
        },
        streetViewControl: false, // Usually disabled for ocean trackers
        mapTypeControl: true, // Allows manually toggling Hybrid/Satellite
        mapTypeControlOptions: {
          position: window.google.maps.ControlPosition.TOP_LEFT,
        },

        // Start locked at world view; dragging enabled dynamically when zoomed in
        draggable: false,
        scrollwheel: false,
        disableDoubleClickZoom: false,
        gestureHandling: "none",
        keyboardShortcuts: false,
        isFractionalZoomEnabled: false,
        restriction: {
          latLngBounds: { north: 85, south: -85, west: -180, east: 180 },
          strictBounds: true,
        },
      });

      // ── Always freehand — drag/scroll/touch always enabled ──────────────────
      map.setOptions({
        draggable: true,
        scrollwheel: true,
        gestureHandling: "greedy",
        keyboardShortcuts: true,
      });

      // ── Anti-wrap centre clamp ──────────────────────────────────────────────
      // Keeps the viewport from crossing ±180° so the world never repeats.
      const clampCentre = () => {
        const z = map.getZoom() ?? 3;
        const W = map.getDiv().offsetWidth;
        const worldPx     = 256 * Math.pow(2, z);
        const halfViewDeg = (W / 2) * (360 / worldPx);
        const maxLng = 180 - halfViewDeg;
        const minLng = -180 + halfViewDeg;
        const c = map.getCenter();
        if (!c) return;
        const lng = c.lng();
        if (lng > maxLng || lng < minLng) {
          map.setCenter({
            lat: c.lat(),
            lng: Math.max(minLng, Math.min(maxLng, lng)),
          });
        }
      };

      map.addListener("center_changed", clampCentre);

      mapRef.current = map;
      onMapReady?.(map);

    };

    initMap();
  }, [onMapReady]);


  return (
    <div
      ref={containerRef}
      id="gee-map-container"
      className="absolute inset-0"
      style={{ background: "#020a12" }}
    />
  );
}
