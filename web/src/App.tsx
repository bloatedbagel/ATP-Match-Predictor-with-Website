import { useMemo, useState } from "react";
import About from "./components/About";
import ComparisonTable from "./components/ComparisonTable";
import PlayerSearch from "./components/PlayerSearch";
import Result from "./components/Result";
import SurfacePicker from "./components/SurfacePicker";
import type { Predictor } from "./predictor";
import type { Surface } from "./types";
import { usePredictor } from "./usePredictor";

function Loading() {
  return (
    <div className="loading" role="status">
      <p>Loading the trained model…</p>
    </div>
  );
}

function PredictorView({ predictor }: { predictor: Predictor }) {
  const { info } = predictor;
  const names = info.players.map((p) => p.name);
  const [a, setA] = useState(names.includes("Jannik Sinner") ? "Jannik Sinner" : names[0]);
  const [b, setB] = useState(names.includes("Carlos Alcaraz") ? "Carlos Alcaraz" : names[1]);
  const [surface, setSurface] = useState<Surface>("Hard");

  // The model is already trained, so a prediction is instant: just recompute when the inputs change.
  const { pred, error } = useMemo(() => {
    if (a === b) return { pred: null, error: "" };
    try {
      return { pred: predictor.predict(a, b, surface), error: "" };
    } catch (e) {
      return { pred: null, error: String(e) };
    }
  }, [a, b, surface, predictor]);

  const inactive = pred ? [pred.a, pred.b].filter((d) => !info.players.find((p) => p.name === d.name)?.active) : [];

  return (
    <>
      <div className="pick">
        <PlayerSearch id="pa" label="Player A" players={info.players} value={a} onChange={setA} />
        <PlayerSearch id="pb" label="Player B" players={info.players} value={b} onChange={setB} />
      </div>
      <SurfacePicker value={surface} onChange={setSurface} />
      <section aria-live="polite">
        {a === b ? (
          <p className="verdict">Choose two different players.</p>
        ) : error ? (
          <p className="verdict">{error}</p>
        ) : pred ? (
          <>
            <Result a={pred.a} b={pred.b} probA={pred.prob_a} surface={surface} />
            {inactive.map((d) => (
              <p key={d.name} className="warn">
                {d.name} hasn’t played since {d.last_match}, so their ranking and form may be out of date.
              </p>
            ))}
            <ComparisonTable p={pred} />
          </>
        ) : null}
      </section>
      <About info={info} />
    </>
  );
}

export default function App() {
  const status = usePredictor();
  return (
    <main>
      <h1>ATP Match Predictor</h1>
      <p className="sub">
        Search any ATP player, pick a surface, and see each player’s chance of winning from a gradient-boosted model trained
        on ATP match history.
      </p>
      {status.phase === "loading" && <Loading />}
      {status.phase === "error" && <p className="warn">Something went wrong: {status.message}</p>}
      {status.phase === "ready" && <PredictorView predictor={status.predictor} />}
    </main>
  );
}
