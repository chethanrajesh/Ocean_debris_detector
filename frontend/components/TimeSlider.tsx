"use client";

/**
 * TimeSlider.tsx
 * Range slider from 0 to 364 (365 time steps = ~90 days at 6h resolution).
 * Displays the equivalent calendar date from today.
 * Calls onChange every time the value changes.
 */

import { useMemo } from "react";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";

interface TimeSliderProps {
  value: number;
  maxSteps: number;
  onChange: (step: number) => void;
}

function stepToDate(step: number): string {
  const base = new Date();
  // Each step = 6 hours
  const offsetMs = step * 6 * 60 * 60 * 1000;
  const date = new Date(base.getTime() + offsetMs);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function stepToForecastLabel(step: number): string {
  const days = (step * 6) / 24;
  if (step === 0) return "Now";
  if (days < 1) return `+${step * 6}h`;
  if (days === 1) return "+1 day";
  return `+${Math.round(days)} days`;
}

export default function TimeSlider({ value, maxSteps, onChange }: TimeSliderProps) {
  const dateLabel     = useMemo(() => stepToDate(value),          [value]);
  const forecastLabel = useMemo(() => stepToForecastLabel(value), [value]);
  const progress      = (value / (maxSteps - 1)) * 100;

  const handleDecrement = () => onChange(Math.max(0, value - 10));
  const handleIncrement = () => onChange(Math.min(maxSteps - 1, value + 10));

  return (
    <div className="glass-panel px-5 py-4 w-[420px] animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Calendar size={14} className="text-teal" />
        <span className="text-xs font-medium text-teal uppercase tracking-widest">
          Time Projection
        </span>
      </div>

      {/* Date display */}
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <span className="font-display text-xl font-bold text-white">
            {dateLabel}
          </span>
          <span className="ml-2 text-sm text-[var(--text-secondary)]">
            {forecastLabel}
          </span>
        </div>
        <span className="font-mono text-xs text-[var(--text-muted)] bg-[var(--ocean-800)] px-2 py-1 rounded-md">
          t={value}
        </span>
      </div>

      {/* Slider with stepper buttons */}
      <div className="flex items-center gap-3">
        <button
          id="time-slider-decrement"
          onClick={handleDecrement}
          className="w-7 h-7 rounded-full flex items-center justify-center
                     bg-[var(--ocean-700)] hover:bg-[var(--ocean-600)]
                     text-[var(--text-secondary)] hover:text-teal-400
                     transition-all duration-150 flex-shrink-0"
        >
          <ChevronLeft size={14} />
        </button>

        <div className="flex-1 relative">
          <input
            id="time-slider"
            type="range"
            min={0}
            max={maxSteps - 1}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full"
            style={{
              background: `linear-gradient(to right, #00d4c8 ${progress}%, #133558 ${progress}%)`,
            }}
          />
        </div>

        <button
          id="time-slider-increment"
          onClick={handleIncrement}
          className="w-7 h-7 rounded-full flex items-center justify-center
                     bg-[var(--ocean-700)] hover:bg-[var(--ocean-600)]
                     text-[var(--text-secondary)] hover:text-teal-400
                     transition-all duration-150 flex-shrink-0"
        >
          <ChevronRight size={14} />
        </button>
      </div>

      {/* Tick marks */}
      <div className="flex justify-between mt-1.5 px-10">
        {["Now", "+30d", "+60d", "+90d"].map((lbl) => (
          <span key={lbl} className="text-[10px] text-[var(--text-muted)]">
            {lbl}
          </span>
        ))}
      </div>
    </div>
  );
}
