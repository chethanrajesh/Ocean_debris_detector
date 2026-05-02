// lib/api.ts
// Axios client targeting NEXT_PUBLIC_API_URL
// Typed functions for all backend endpoints

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

// ── Types ─────────────────────────────────────────────────────────────────────

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

export interface KnownZone {
  latitude: number;
  longitude: number;
  label: string;
  type: string;
  plastic_density: number;
  level: string;
  accumulation_trend: string;
  movement_vector: MovementVector;
  source_estimate: string;
}

export interface PredictionNode {
  lat: number;
  lon: number;
  density: number;
}

export type PredictionMode = "real" | "predicted";

export interface PredictionsResponse {
  timestep:        number;
  total_timesteps: number;
  /** "real" at t=0 (seeded from satellite/buoy), "predicted" at t>0 (GNN forecast) */
  mode:            PredictionMode;
  /** Convenience: timestep converted to fractional days (timestep / 4) */
  day:             number;
  nodes:           PredictionNode[];
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

// ── API calls ─────────────────────────────────────────────────────────────────

export async function getHotspots(): Promise<Hotspot[]> {
  const { data } = await client.get<Hotspot[]>("/hotspots");
  return data;
}

export async function getKnownZones(): Promise<KnownZone[]> {
  try {
    const { data } = await client.get<KnownZone[]>("/hotspots/known");
    return data;
  } catch {
    return [];   // endpoint is optional — fail silently
  }
}

/** timestep 0 = real; 1–360 = GNN prediction (4 steps/day, 90 days max) */
export async function getPredictions(
  timestep: number,
): Promise<PredictionsResponse> {
  const t = Math.max(0, Math.min(360, Math.round(timestep)));
  const { data } = await client.get<PredictionsResponse>(`/predictions/${t}`);
  return data;
}

/** Fetch up to maxNodes current vectors (default 2000) */
export async function getCurrents(
  maxNodes = 2000,
): Promise<CurrentsResponse> {
  const { data } = await client.get<CurrentsResponse>(
    `/currents?max_nodes=${maxNodes}`,
  );
  return data;
}

// ── Trajectory Types ──────────────────────────────────────────────────────────

/** One particle snapshot: [lat, lon, density, age_days, source_type] */
export type ParticleSnapshot = [number, number, number, number, number];

/** source_type encoding from simulator */
export type SourceType = 0 | 1 | 2;  // 0=active, 1=beached, 2=converging

/** A single particle's 14-snapshot trajectory */
export interface ParticleTrajectory {
  id: number;
  snapshots: ParticleSnapshot[];   // length = N_SNAPSHOTS (14)
  origin: { lat: number; lon: number };
  source_label: string;
}

/** /trajectories/current response */
export interface TrajectoriesCurrentResponse {
  day: number;
  total_particles: number;
  active: number;
  beached: number;
  converging: number;
  particles: Array<{
    id: number;
    lat: number;
    lon: number;
    density: number;
    age_days: number;
    source_type: SourceType;
    source_label: string;
  }>;
}

/** /trajectories/forecast response */
export interface TrajectoryForecastResponse {
  snapshot_days: number[];        // [0, 7, 14, …, 90]
  total_particles: number;
  trajectories: ParticleTrajectory[];
}

/** One coastal cell in the beaching-risk list */
export interface BeachingCell {
  lat: number;
  lon: number;
  risk: number;       // normalised [0, 1]
  particle_count: number;
}

/** /trajectories/beaching-risk response */
export interface BeachingRiskResponse {
  top_cells: BeachingCell[];
  total_beached: number;
}

/** /trajectories/seed POST response */
export interface SeedTrajectoryResponse {
  origin: { lat: number; lon: number; mass_kg: number };
  snapshot_days: number[];
  trajectory: ParticleSnapshot[];
  final_status: "active" | "beached" | "converging";
}

// ── Trajectory API calls ──────────────────────────────────────────────────────

/** Current particle positions (snapshot=0 of each particle's trajectory) */
export async function getTrajectoryCurrents(): Promise<TrajectoriesCurrentResponse> {
  const { data } = await client.get<TrajectoriesCurrentResponse>(
    "/trajectories/current"
  );
  return data;
}

/**
 * Full 90-day forecast for all particles.
 * @param maxParticles cap for bandwidth (default 2000)
 */
export async function getTrajectoryForecast(
  maxParticles = 2000,
): Promise<TrajectoryForecastResponse> {
  const { data } = await client.get<TrajectoryForecastResponse>(
    `/trajectories/forecast?max_particles=${maxParticles}`
  );
  return data;
}

/** Top-N coastal cells ranked by incoming debris probability */
export async function getBeachingRisk(
  topN = 200,
): Promise<BeachingRiskResponse> {
  const { data } = await client.get<BeachingRiskResponse>(
    `/trajectories/beaching-risk?top_n=${topN}`
  );
  return data;
}

/** Inject a custom debris point and get its 90-day predicted path */
export async function seedCustomParticle(
  lat: number,
  lon: number,
  mass_kg: number,
): Promise<SeedTrajectoryResponse> {
  const { data } = await client.post<SeedTrajectoryResponse>(
    "/trajectories/seed",
    { lat, lon, mass_kg }
  );
  return data;
}

// ── Location search ───────────────────────────────────────────────────────────

export interface Location {
  name: string;
  lat:  number;
  lon:  number;
  type: "beach" | "ocean" | "sea" | "river" | "gyre";
}

export async function searchLocations(q: string, type?: string): Promise<Location[]> {
  const params = new URLSearchParams({ q });
  if (type) params.set("type", type);
  const { data } = await client.get<Location[]>(`/locations/search?${params}`);
  return data;
}
