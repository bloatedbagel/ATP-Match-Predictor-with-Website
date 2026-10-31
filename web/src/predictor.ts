// Scores a matchup using the pre-trained model exported to model.json. No training and no Python here:
// it is the same arithmetic scikit-learn does in predict_proba, plus the feature recipe from
// python/atp_predictor.py (pair_features).
import type { Detail, EngineInfo, ModelData, PlayerRecord, Prediction, Surface } from "./types";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// Start at the baseline score, walk every tree down to a leaf, and add up the leaf values.
function rawScore(model: ModelData["model"], x: number[]): number {
  let s = model.base;
  for (const t of model.trees) {
    let i = 0;
    while (t.f[i] >= 0) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
    s += t.v[i];
  }
  return s;
}

function h2hRecord(h2h: ModelData["h2h"], a: number, b: number): [number, number] {
  const rec = h2h[a < b ? `${a}-${b}` : `${b}-${a}`];
  if (!rec) return [0, 0];
  return a < b ? [rec[0], rec[1]] : [rec[1], rec[0]];
}

export function createPredictor(data: ModelData) {
  const byName = new Map(data.players.map((p) => [p.name, p]));
  const get = (name: string): PlayerRecord => {
    const p = byName.get(name);
    if (!p) throw new Error(`No matches found for player: "${name}"`);
    return p;
  };

  // Same seven features, same order, as FEATURES in atp_predictor.py. Positive values favor player a.
  const features = (a: PlayerRecord, b: PlayerRecord, surface: Surface): number[] => {
    const [wa, wb] = h2hRecord(data.h2h, a.id, b.id);
    return [
      b.rank - a.rank,
      a.pts - b.pts,
      a.age - b.age,
      a.ht - b.ht,
      a.form - b.form,
      a.surf[surface] - b.surf[surface],
      wa + wb ? wa / (wa + wb) : 0.5,
    ];
  };
  const winProb = (a: PlayerRecord, b: PlayerRecord, surface: Surface) =>
    sigmoid(rawScore(data.model, features(a, b, surface)));

  const detail = (p: PlayerRecord, surface: Surface): Detail => ({
    name: p.name,
    rank: Math.trunc(p.rank),
    points: Math.trunc(p.pts),
    age: Math.round(p.age * 10) / 10,
    height: Math.trunc(p.ht),
    form: p.form,
    surface_rate: p.surf[surface],
    last_match: p.last,
  });

  const info: EngineInfo = {
    asof: data.asof,
    built: data.built,
    matches: data.matches,
    players: data.players.map((p) => ({ name: p.name, rank: Math.trunc(p.rank), points: Math.trunc(p.pts), active: p.active })),
  };

  const predict = (nameA: string, nameB: string, surface: Surface): Prediction => {
    const a = get(nameA);
    const b = get(nameB);
    // Average both orderings so "A vs B" and "B vs A" always sum to 100%.
    const prob_a = (winProb(a, b, surface) + 1 - winProb(b, a, surface)) / 2;
    return {
      prob_a,
      prob_b: 1 - prob_a,
      surface,
      a: detail(a, surface),
      b: detail(b, surface),
      h2h: h2hRecord(data.h2h, a.id, b.id),
    };
  };

  return { info, predict };
}

export type Predictor = ReturnType<typeof createPredictor>;
