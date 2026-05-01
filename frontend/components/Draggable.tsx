"use client";

/**
 * Draggable.tsx
 * Floating panel using position:fixed.
 * Drag by clicking anywhere on the panel — no visible handle bar.
 */

import React, { useState, useRef, useEffect, ReactNode } from "react";

type Position = { x: number; y: number };

interface DraggableProps {
  children: ReactNode;
  id?: string;
  defaultPosition?: Position | (() => Position);
  minVisible?: number;
}

export default function Draggable({
  children,
  id,
  defaultPosition,
  minVisible = 60,
}: DraggableProps) {
  const [pos, setPos] = useState<Position | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStart    = useRef({ mx: 0, my: 0, px: 0, py: 0 });

  // Set position after first paint — avoids SSR/hydration mismatch
  useEffect(() => {
    if (typeof defaultPosition === "function") {
      setPos(defaultPosition());
    } else if (defaultPosition) {
      setPos(defaultPosition);
    } else {
      const w = containerRef.current?.offsetWidth  ?? 0;
      const h = containerRef.current?.offsetHeight ?? 0;
      setPos({
        x: Math.round((window.innerWidth  - w) / 2),
        y: Math.round((window.innerHeight - h) / 2),
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Don't start drag if the user clicked an interactive element
    const tag = (e.target as HTMLElement).tagName;
    if (["INPUT", "BUTTON", "SELECT", "TEXTAREA"].includes(tag)) return;
    if (e.button !== 0 || !pos) return;
    e.preventDefault();
    setIsDragging(true);
    dragStart.current = { mx: e.clientX, my: e.clientY, px: pos.x, py: pos.y };
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
  };

  useEffect(() => {
    if (!isDragging) return;

    const onMove = (e: PointerEvent) => {
      const dx = e.clientX - dragStart.current.mx;
      const dy = e.clientY - dragStart.current.my;
      let nx = dragStart.current.px + dx;
      let ny = dragStart.current.py + dy;

      if (containerRef.current) {
        const { offsetWidth: w, offsetHeight: h } = containerRef.current;
        nx = Math.max(minVisible - w,  Math.min(window.innerWidth  - minVisible, nx));
        ny = Math.max(0,               Math.min(window.innerHeight - minVisible, ny));
      }

      setPos({ x: nx, y: ny });
    };

    const onUp = () => setIsDragging(false);

    document.addEventListener("pointermove",   onMove);
    document.addEventListener("pointerup",     onUp);
    document.addEventListener("pointercancel", onUp);
    return () => {
      document.removeEventListener("pointermove",   onMove);
      document.removeEventListener("pointerup",     onUp);
      document.removeEventListener("pointercancel", onUp);
    };
  }, [isDragging, minVisible]);

  return (
    <div
      ref={containerRef}
      id={id}
      onPointerDown={onPointerDown}
      style={{
        position: "fixed",
        left: pos?.x ?? -9999,
        top:  pos?.y ?? -9999,
        zIndex: isDragging ? 99999 : 1000,
        display: "inline-flex",
        flexDirection: "column",
        cursor: isDragging ? "grabbing" : "grab",
        filter: isDragging
          ? "drop-shadow(0 20px 40px rgba(0,0,0,0.7))"
          : "drop-shadow(0 8px 24px rgba(0,0,0,0.5))",
        transition: isDragging ? "none" : "filter 0.2s",
        userSelect: "none",
        visibility: pos ? "visible" : "hidden",
      }}
    >
      {children}
    </div>
  );
}
