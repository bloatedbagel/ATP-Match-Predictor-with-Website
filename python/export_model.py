"""Train the model ONCE and export everything the website needs as a single static file, model.json.

The website never trains anything. It downloads model.json and does simple arithmetic in the browser.

model.json holds three things:
  1. model    the trained gradient-boosting trees, flattened into plain arrays
  2. players  every player's latest rank, points, age, height, form and per-surface win rate
  3. h2h      head-to-head win counts for every pair of players who have met

Usage (from the repo root):
    python python/export_model.py                  # downloads the CSVs, trains, writes web/public/model.json
    python python/export_model.py --out some.json  # write somewhere else
The GitHub Actions workflow runs this before every build, so the live site is always a fresh export.
"""
import argparse
import datetime
import json
import pathlib
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import atp_predictor as ap

DATA_URL = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{year}.csv"
START_YEAR = 2019
SURFACES = ["Hard", "Clay", "Grass"]  # keep in sync with SURFACES in web/src/types.ts
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


# ---------------------------------------------------------------- data download
def download_csvs(cache_dir, start_year=START_YEAR):
    """Fetch one CSV per season into cache_dir. Finished seasons are cached; the current one is always refreshed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    this_year = datetime.date.today().year
    sources = {}
    for year in range(start_year, this_year + 1):
        path = cache_dir / f"atp_matches_{year}.csv"
        if year == this_year or not path.exists():
            try:
                with urllib.request.urlopen(DATA_URL.format(year=year), timeout=60) as r:
                    path.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 404:  # season not published yet
                    continue
                raise
        sources[year] = path
    if not sources:
        raise SystemExit("Could not download any match data.")
    return sources


# ---------------------------------------------------------------- model -> arrays
def export_trees(model):
    """Flatten a fitted HistGradientBoostingClassifier into arrays a browser can walk.

    Per tree: f = feature index (-1 marks a leaf), t = split threshold, l/r = left/right child index,
    v = leaf value. Rule at each node: go left if x[f] <= t, otherwise right.
    """
    trees = []
    for per_class in model._predictors:      # one entry per boosting round
        (tree,) = per_class                  # binary classification: exactly one tree per round
        n = tree.nodes
        leaf = n["is_leaf"].astype(bool)
        trees.append({
            "f": np.where(leaf, -1, n["feature_idx"]).astype(int).tolist(),
            "t": np.where(leaf, 0.0, n["num_threshold"]).tolist(),
            "l": n["left"].astype(int).tolist(),
            "r": n["right"].astype(int).tolist(),
            "v": n["value"].tolist(),
        })
    return {"base": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def tree_raw_score(exported, x):
    """Pure-Python twin of the browser's scoring loop (web/src/predictor.ts)."""
    s = exported["base"]
    for t in exported["trees"]:
        i = 0
        while t["f"][i] >= 0:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        s += t["v"][i]
    return s


def verify_trees(engine, exported, n_players=25):
    """Fail loudly if the flattened trees disagree with scikit-learn (e.g. after a sklearn upgrade)."""
    names = list(engine.snapshots.sort_values("rank").index[:n_players])
    rows = [ap.pair_features(engine.state, engine.player(a), engine.player(b), s)
            for s in SURFACES for a in names for b in names if a != b]
    X = pd.DataFrame(rows)[ap.FEATURES]
    mine = 1 / (1 + np.exp(-np.array([tree_raw_score(exported, x) for x in X.to_numpy()])))
    expected = engine.model.predict_proba(X)[:, 1]
    worst = float(np.abs(mine - expected).max())
    if worst > 1e-9:
        raise AssertionError(f"Exported trees differ from scikit-learn by up to {worst:.2e}")
    return worst


# ---------------------------------------------------------------- everything else
def export_players(engine):
    s = engine.state
    cutoff = engine.asof - pd.Timedelta(days=ap.ACTIVE_DAYS)
    players = []
    for name, r in engine.snapshots.sort_values("rank").iterrows():   # sorted by rank, like the site expects
        pid = int(r["id"])
        players.append({
            "name": name, "id": pid,
            "rank": float(r["rank"]), "pts": float(r["pts"]), "age": float(r["age"]), "ht": float(r["ht"]),
            "form": s.form(pid),
            "surf": {surface: s.surface_winrate(pid, surface) for surface in SURFACES},
            "last": str(r["date"].date()),
            "active": bool(r["date"] >= cutoff),
        })
    return players


def export_h2h(engine, players):
    """Key is "lowerId-higherId"; value is [wins of the lower id, wins of the higher id]."""
    known = {p["id"] for p in players}
    out = {}
    for (a, b), wins in engine.state.h2h.items():
        if int(a) in known and int(b) in known:
            lo, hi = sorted((int(a), int(b)))
            out[f"{lo}-{hi}"] = [wins.get(lo, 0), wins.get(hi, 0)]
    return out


def build_payload(engine):
    exported = export_trees(engine.model)
    verify_trees(engine, exported)
    players = export_players(engine)
    return {
        "version": SCHEMA_VERSION,
        "asof": str(engine.asof.date()),                       # date of the newest match in the data
        "built": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "matches": int(engine.n_matches),
        "model": exported,
        "players": players,
        "h2h": export_h2h(engine, players),
    }


def write_payload(payload, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return out


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--out", default=str(ROOT / "web" / "public" / "model.json"))
    ap_.add_argument("--data-dir", default=str(ROOT / ".cache" / "data"), help="where downloaded CSVs are kept")
    args = ap_.parse_args()

    print("Downloading match data...")
    sources = download_csvs(pathlib.Path(args.data_dir))
    print(f"Training on seasons {min(sources)}-{max(sources)}...")
    engine = ap.build_engine(sources)
    payload = build_payload(engine)
    out = write_payload(payload, args.out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {payload['matches']:,} matches through "
          f"{payload['asof']}, {len(payload['players']):,} players, {len(payload['model']['trees'])} trees.")


if __name__ == "__main__":
    main()
