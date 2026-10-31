"""ATP match predictor engine: the notebook's pipeline as an importable module.

It never touches the network:
callers pass in CSV sources (paths, file objects or StringIO), keyed by season.
"""
import json
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
DEFAULT_HEIGHT_CM = 180
ACTIVE_DAYS = 270  # a player whose last match is older than this is flagged as inactive
FEATURES = ["rank_diff", "rank_points_diff", "age_diff", "height_diff",
            "form_diff", "surface_form_diff", "h2h_a_winrate"]
REQUIRED = ["winner_rank", "loser_rank", "surface", "winner_rank_points",
            "loser_rank_points", "winner_age", "loser_age"]
_SIDE = ["id", "name", "ht", "age", "rank", "rank_points"]
COLUMNS = {"tourney_date", "surface"} | {f"{p}_{c}" for p in ("winner", "loser") for c in _SIDE}

Player = namedtuple("Player", "id rank pts age ht")


def load_raw(sources):
    """sources: {season: path | file object}. Returns all matches sorted by date."""
    frames = [pd.read_csv(src, usecols=lambda c: c in COLUMNS) for _, src in sorted(sources.items())]
    if not frames:
        raise ValueError("No match data supplied.")
    raw = pd.concat(frames, ignore_index=True)
    raw["tourney_date"] = pd.to_datetime(raw["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    return raw.sort_values("tourney_date", kind="stable").reset_index(drop=True)


def clean_matches(raw):
    m = raw.dropna(subset=REQUIRED).copy()
    m[["winner_ht", "loser_ht"]] = m[["winner_ht", "loser_ht"]].fillna(DEFAULT_HEIGHT_CM)
    return m.reset_index(drop=True)


def latest_snapshots(matches):
    """Each player's most recent known id/rank/points/age/height, indexed by name."""
    def side(p):
        d = matches[["tourney_date", f"{p}_name", f"{p}_id", f"{p}_rank",
                     f"{p}_rank_points", f"{p}_age", f"{p}_ht"]].copy()
        d.columns = ["date", "name", "id", "rank", "pts", "age", "ht"]
        return d
    both = pd.concat([side("winner"), side("loser")]).sort_values("date", kind="stable")
    return both.drop_duplicates("name", keep="last").set_index("name")


class PlayerState:
    """Rolling win/loss history used for form, surface and head-to-head features."""

    def __init__(self, form_window=10):
        self.form_window = form_window
        self.results = {}   # player_id -> [1/0, ...]
        self.surface = {}   # (player_id, surface) -> [1/0, ...]
        self.h2h = {}       # sorted (id1, id2) -> {player_id: wins}

    def form(self, pid):
        h = self.results.get(pid, [])[-self.form_window:]
        return sum(h) / len(h) if h else 0.5

    def surface_winrate(self, pid, surface):
        h = self.surface.get((pid, surface), [])
        return sum(h) / len(h) if h else 0.5

    def h2h_record(self, a, b):
        wins = self.h2h.get(tuple(sorted((a, b))), {})
        return wins.get(a, 0), wins.get(b, 0)

    def h2h_winrate(self, a, b):
        wa, wb = self.h2h_record(a, b)
        return wa / (wa + wb) if wa + wb else 0.5

    def update(self, winner_id, loser_id, surface):
        self.results.setdefault(winner_id, []).append(1)
        self.results.setdefault(loser_id, []).append(0)
        self.surface.setdefault((winner_id, surface), []).append(1)
        self.surface.setdefault((loser_id, surface), []).append(0)
        wins = self.h2h.setdefault(tuple(sorted((winner_id, loser_id))), {})
        wins[winner_id] = wins.get(winner_id, 0) + 1


def pair_features(state, a, b, surface):
    """Feature row for 'player a vs player b'. Positive values favor player a."""
    return {
        "rank_diff": b.rank - a.rank,
        "rank_points_diff": a.pts - b.pts,
        "age_diff": a.age - b.age,
        "height_diff": a.ht - b.ht,
        "form_diff": state.form(a.id) - state.form(b.id),
        "surface_form_diff": state.surface_winrate(a.id, surface) - state.surface_winrate(b.id, surface),
        "h2h_a_winrate": state.h2h_winrate(a.id, b.id),
    }


def build_dataset(matches, seed=SEED):
    rng = np.random.default_rng(seed)
    state = PlayerState()
    rows = []
    for m in matches.itertuples(index=False):
        w = Player(m.winner_id, m.winner_rank, m.winner_rank_points, m.winner_age, m.winner_ht)
        l = Player(m.loser_id, m.loser_rank, m.loser_rank_points, m.loser_age, m.loser_ht)
        a, b, label = (w, l, 1) if rng.random() < 0.5 else (l, w, 0)
        row = pair_features(state, a, b, m.surface)
        row["label"] = label
        rows.append(row)
        state.update(w.id, l.id, m.surface)   # update AFTER recording features
    return pd.DataFrame(rows), state


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        early_stopping=True, random_state=SEED)


class Engine:
    def __init__(self, model, state, snapshots, asof, n_matches):
        self.model, self.state, self.snapshots = model, state, snapshots
        self.asof, self.n_matches = asof, n_matches

    def player(self, name):
        if name not in self.snapshots.index:
            raise ValueError(f"No matches found for player: {name!r}")
        s = self.snapshots.loc[name]
        return Player(int(s["id"]), float(s["rank"]), float(s["pts"]), float(s["age"]), float(s["ht"]))

    def win_probabilities(self, pairs, surface="Hard"):
        """P(first player beats second), averaging both orderings so A-vs-B and B-vs-A sum to 1."""
        def proba(ps):
            rows = [pair_features(self.state, self.player(a), self.player(b), surface) for a, b in ps]
            return self.model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
        forward = proba(pairs)
        return (forward + 1 - proba([(b, a) for a, b in pairs])) / 2

    def predict_match(self, a, b, surface="Hard"):
        p = float(self.win_probabilities([(a, b)], surface)[0])
        pa, pb = self.player(a), self.player(b)
        return {
            "prob_a": p, "prob_b": 1 - p, "surface": surface,
            "a": self._detail(a, pa, surface), "b": self._detail(b, pb, surface),
            "h2h": list(self.state.h2h_record(pa.id, pb.id)),
        }

    def _detail(self, name, p, surface):
        s = self.state
        return {"name": name, "rank": int(p.rank), "points": int(p.pts), "age": round(p.age, 1),
                "height": int(p.ht), "form": s.form(p.id), "surface_rate": s.surface_winrate(p.id, surface),
                "last_match": str(self.snapshots.loc[name, "date"].date())}

    def players(self):
        cutoff = self.asof - pd.Timedelta(days=ACTIVE_DAYS)
        snap = self.snapshots.sort_values("rank")
        return [{"name": n, "rank": int(r["rank"]), "points": int(r["pts"]),
                 "active": bool(r["date"] >= cutoff)} for n, r in snap.iterrows()]


def build_engine(sources):
    matches = clean_matches(load_raw(sources))
    dataset, state = build_dataset(matches)
    model = make_model().fit(dataset[FEATURES], dataset["label"])
    return Engine(model, state, latest_snapshots(matches), matches["tourney_date"].max(), len(matches))


# ---- JSON entry points used by the browser worker ----
_engine = None


def init(sources):
    global _engine
    _engine = build_engine(sources)
    return json.dumps({"asof": str(_engine.asof.date()), "matches": _engine.n_matches,
                       "players": _engine.players()})


def predict(a, b, surface):
    return json.dumps(_engine.predict_match(a, b, surface))
