import { useEffect, useState } from "react";
import { createPredictor, type Predictor } from "./predictor";
import type { ModelData } from "./types";

type Status = { phase: "loading" } | { phase: "ready"; predictor: Predictor } | { phase: "error"; message: string };

// Downloads the pre-trained model (public/model.json) once when the page opens. Nothing is trained in the browser.
export function usePredictor(): Status {
  const [status, setStatus] = useState<Status>({ phase: "loading" });

  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}model.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`Could not download model.json (HTTP ${r.status}).`);
        return r.json() as Promise<ModelData>;
      })
      .then((data) => { if (!cancelled) setStatus({ phase: "ready", predictor: createPredictor(data) }); })
      .catch((e) => {
        if (cancelled) return;
        const hint = import.meta.env.DEV ? " Run `python python/export_model.py` first to create it." : "";
        setStatus({ phase: "error", message: String(e) + hint });
      });
    return () => { cancelled = true; };
  }, []);

  return status;
}
