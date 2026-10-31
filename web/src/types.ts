export const SURFACES = ["Hard", "Clay", "Grass"] as const;
export type Surface = (typeof SURFACES)[number];

export interface PlayerInfo {
  name: string;
  rank: number;
  points: number;
  active: boolean; // played within the last ~9 months of the data
}

export interface EngineInfo {
  asof: string; // date of the newest match the model was trained on
  built: string; // date the model was exported
  matches: number;
  players: PlayerInfo[];
}

export interface Detail {
  name: string;
  rank: number;
  points: number;
  age: number;
  height: number;
  form: number; // win rate over the last 10 matches
  surface_rate: number; // win rate on the chosen surface
  last_match: string;
}

export interface Prediction {
  prob_a: number;
  prob_b: number;
  surface: Surface;
  a: Detail;
  b: Detail;
  h2h: [number, number]; // wins for A, wins for B
}

export interface Metrics {
  n_test: number;
  test_start: string;
  test_end: string;
  gbm_accuracy: number;
  gbm_logloss: number;
  rank_accuracy: number;
  rank_lr_logloss: number;
  logloss_gain_ci: number[];
}

// ---- the shape of public/model.json, written by python/export_model.py ----

/** One decision tree as parallel arrays. f = feature index (-1 = leaf), t = threshold, l/r = children, v = leaf value. */
export interface TreeData {
  f: number[];
  t: number[];
  l: number[];
  r: number[];
  v: number[];
}

export interface PlayerRecord {
  name: string;
  id: number;
  rank: number;
  pts: number;
  age: number;
  ht: number;
  form: number;
  surf: Record<Surface, number>;
  last: string;
  active: boolean;
}

export interface ModelData {
  version: number;
  asof: string;
  built: string;
  matches: number;
  model: { base: number; trees: TreeData[] };
  players: PlayerRecord[]; // sorted by rank
  h2h: Record<string, [number, number]>; // "lowerId-higherId" -> [wins of lower id, wins of higher id]
}
