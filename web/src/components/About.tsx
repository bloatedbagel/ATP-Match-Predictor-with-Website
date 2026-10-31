import metrics from "../data/metrics.json";
import type { EngineInfo, Metrics } from "../types";

const m = metrics as Metrics;
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function About({ info }: { info: EngineInfo }) {
  return (
    <div className="note">
      <p>
        The model was trained once, ahead of time, on {info.matches.toLocaleString()} ATP matches through {info.asof}
        (exported {info.built}), using the same Python (scikit-learn) code as the notebook. Rankings and form are as of
        each player’s latest match in that data.
      </p>
      <p>
        In the notebook’s held-out test ({m.n_test.toLocaleString()} matches, {m.test_start} to {m.test_end}) it picked the
        winner {pct(m.gbm_accuracy)} of the time with a log loss of {m.gbm_logloss.toFixed(3)}, against {pct(m.rank_accuracy)}{" "}
        and {m.rank_lr_logloss.toFixed(3)} for a ranking-only baseline. Rankings carry most of the signal, so treat these as
        probabilities, not guarantees. Not betting advice.
      </p>
    </div>
  );
}
