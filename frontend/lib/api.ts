// lib/api.ts
// Axios client targeting NEXT_PUBLIC_API_URL
// Typed functions for all backend endpoints

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

// ── Types ────────────────────────────────────────────────────────────────────

export interface MovementVector {
  u: number;
  v: number;
}

export interface Hotspot {
  latitude: number;
  longitude: number;
  plastic_density: number;
  level: "critical" | "high" | "moderate";
  accumulation_trend: "increasing" | "stable" | "decreasing";
  movement_vector: MovementVector;
  source_estimate: string;
}

export interface PredictionNode {
  lat: number;
  lon: number;
  density: number;
}

export interface PredictionsResponse {
  timestep: number;
  total_timesteps: number;
  nodes: PredictionNode[];
}

export interface CurrentNode {
  lat: number;
  lon: number;
  u: number;
  v: number;
}

export interface CurrentsResponse {
  nodes: CurrentNode[];
}

// ── API calls ────────────────────────────────────────────────────────────────

export async function getHotspots(): Promise<Hotspot[]> {
  const { data } = await client.get<Hotspot[]>("/hotspots");
  return data;
}

export async function getPredictions(timestep: number): Promise<PredictionsResponse> {
  const { data } = await client.get<PredictionsResponse>(`/predictions/${timestep}`);
  return data;
}

export async function getCurrents(): Promise<CurrentsResponse> {
  const { data } = await client.get<CurrentsResponse>("/currents");
  return data;
}
