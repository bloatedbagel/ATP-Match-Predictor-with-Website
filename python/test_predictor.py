"""Run: python python/test_predictor.py   (synthetic data in Sackmann's CSV format; no network)"""
import io
import numpy as np
import pandas as pd
import atp_predictor as ap
import export_model as ex

rng = np.random.default_rng(1)
N = 60
skill = {i: rng.normal() for i in range(N)}
rank = {i: r + 1 for r, i in enumerate(sorted(skill, key=lambda k: -skill[k]))}
rows = []
for d in range(1500):
    a, b = rng.choice(N, 2, replace=False)
    w, l = (a, b) if rng.random() < 1 / (1 + np.exp(skill[b] - skill[a])) else (b, a)
    date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d // 3)).strftime("%Y%m%d")
    row = {"tourney_date": int(date), "surface": rng.choice(["Hard", "Clay", "Grass"])}
    for p, i in (("winner", w), ("loser", l)):
        row.update({f"{p}_id": 1000 + i, f"{p}_name": f"Player {i}", f"{p}_ht": 185 + i % 10,
                    f"{p}_age": 20 + i % 15, f"{p}_rank": rank[i], f"{p}_rank_points": 5000 - 60 * rank[i]})
    rows.append(row)
df = pd.DataFrame(rows)
src = {2024: io.StringIO(df.to_csv(index=False))}

eng = ap.build_engine(src)
top, bottom = min(rank, key=rank.get), max(rank, key=rank.get)
fwd = eng.win_probabilities([(f"Player {top}", f"Player {bottom}")])[0]
rev = eng.win_probabilities([(f"Player {bottom}", f"Player {top}")])[0]
assert abs(fwd + rev - 1) < 1e-9 and fwd > 0.5, (fwd, rev)

m = eng.predict_match(f"Player {top}", f"Player {bottom}", "Clay")
assert abs(m["prob_a"] + m["prob_b"] - 1) < 1e-9 and len(m["h2h"]) == 2

# no leakage: a match's features don't change when later matches are added
clean = ap.clean_matches(ap.load_raw({2024: io.StringIO(df.to_csv(index=False))}))
full, _ = ap.build_dataset(clean)
prefix, _ = ap.build_dataset(clean.iloc[:301])
pd.testing.assert_series_equal(prefix.iloc[300][ap.FEATURES], full.iloc[300][ap.FEATURES], check_names=False)
assert sum(len(v) for v in ap.build_dataset(clean)[1].results.values()) == 2 * len(clean)

try:
    eng.player("Nobody Real"); raise AssertionError("expected ValueError")
except ValueError:
    pass
ap.init({2024: io.StringIO(df.to_csv(index=False))})
assert '"prob_a"' in ap.predict("Player 1", "Player 2", "Hard")

# export: flattened trees reproduce scikit-learn, and the payload is valid JSON the site can read
payload = ex.build_payload(eng)              # raises if the exported trees differ from sklearn
import json
json.loads(json.dumps(payload, allow_nan=False))
assert len(payload["players"]) == N and payload["players"][0]["rank"] == 1
assert all(a < b for a, b in (map(int, k.split("-")) for k in payload["h2h"]))
print("All tests passed.")
