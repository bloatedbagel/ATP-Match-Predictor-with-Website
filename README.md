# ATP Match Predictor

Predicts the winner of an ATP tennis match from player rankings, recent form, surface record,
head-to-head history, age, and height, using a scikit-learn gradient-boosted classifier
(`HistGradientBoostingClassifier`). Live demo: **https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/** (search any ATP player; predictions are instant)

## Results

Chronological split: the model trains on the first 80% of matches and is tested on the most recent
4,001 matches (2025-01-30 to 2026-05-25), so it is always tested on the future.

| Model | Accuracy | Log loss |
|---|---|---|
| Coin flip | 51% | 0.693 |
| Always pick the higher-ranked player | 64.2% | n/a |
| Rank-only logistic regression | n/a | 0.634 |
| **Gradient boosting (this model)** | **65.3%** | **0.628** |

Bootstrap 95% confidence intervals (2,000 resamples of the test matches):
- Log loss improvement over the rank-only model: 0.0063 (0.0016 to 0.0112)
- Accuracy edge over "pick the higher-ranked player": 1.1 points (0.1 to 2.0)

The improvement over rankings is small but statistically distinguishable from zero. Rankings already
carry most of the predictable signal in tennis, so this is expected.

## How it works (train once, predict instantly)

The model is **trained once**, not in each visitor's browser:

1. `python/export_model.py` downloads the ATP match CSVs, trains the `HistGradientBoostingClassifier`, and writes
   everything the site needs to one file, `web/public/model.json`: the 200 trained trees, every player's latest
   rank/points/age/height/form/surface win rates, and head-to-head records.
2. The website downloads `model.json` when the page opens (a static file) and scores matchups with
   a few lines of TypeScript (`web/src/predictor.ts`). That is the same arithmetic scikit-learn does in `predict_proba`.
   No Python, no training, no server.
3. GitHub Actions runs step 1 for you on every push to `main` and every Monday, then builds and deploys the site, so
   the model refreshes itself as new matches are published.

The export step checks its own work: it re-scores a few thousand matchups with the flattened trees and fails the build
if any differ from scikit-learn.

## What's in this repo

| Path | What it is |
|---|---|
| `python/atp_predictor.py` | The model pipeline: data cleaning, leakage-free features, gradient boosting, prediction |
| `python/export_model.py` | Trains the model and writes `web/public/model.json` (the "pre-trained model" the site loads) |
| `python/test_predictor.py` | Offline tests on synthetic data, including a check that the export matches scikit-learn |
| `web/public/model.json` | **Generated**, not committed. The trained model plus player stats. Created by the export script |
| `web/src/predictor.ts` | Scores a matchup from `model.json` (walks the trees, builds the 7 features, averages both orderings) |
| `web/src/usePredictor.ts` | React hook that downloads `model.json` once on page load |
| `web/src/App.tsx`, `components/` | The user interface: player search, surface picker, result, comparison table, about box |
| `web/src/types.ts` | TypeScript types, including the shape of `model.json` |
| `web/src/data/metrics.json` | Test-set results shown in the About box (from the notebook) |
| `atp_match_predictor.ipynb` | The original notebook: evaluation, baselines, bootstrap intervals, sanity checks |
| `requirements.txt` | Python packages for the export script, tests and notebook |
| `.github/workflows/deploy.yml` | Trains, exports, builds and publishes to GitHub Pages (push, weekly, or manual) |

## Deploy to GitHub Pages
1. Create a repo on GitHub and upload the contents of this folder to branch `main` (see below).
2. In the repo, go to **Settings > Pages** and set **Source** to **GitHub Actions**.
3. Open the **Actions** tab. The workflow starts on its own after the upload (or press **Run workflow**). It takes a few
   minutes; when it finishes the site is live at `https://USER.github.io/REPO/`.

## Run it locally
```bash
pip install -r requirements.txt
python python/test_predictor.py            # offline tests, no network
python python/export_model.py              # downloads data, trains, writes web/public/model.json
cd web && npm install && npm run dev       # site at http://localhost:5173
```
You only need to re-run `export_model.py` when you want fresher data or after changing the model.

## Changing the model
Edit `python/atp_predictor.py` (features, hyperparameters, anything in `make_model`), then re-run the export. If you add
or reorder features, also update `features()` in `web/src/predictor.ts` so the order matches `FEATURES`. The export
check will fail loudly if the trees and scikit-learn ever disagree.

## Design notes
- **No leakage:** each match's features use only earlier matches; player history is updated after the features are recorded.
- **Randomized sides:** each match is randomly assigned to player A or B so the model can't learn that column A always wins.
- **Symmetric predictions:** the A-vs-B and B-vs-A predictions are averaged, so probabilities always sum to 100%.
- **One engine, two homes:** the notebook and `atp_predictor.py` share the same logic. If you change the model in the
  notebook, make the same change in `atp_predictor.py`.

## Limitations
Data comes from Jeff Sackmann's ATP archive and ends at the latest published match, so rankings, form,
and head-to-head records are a snapshot as of that date. Not betting advice.
