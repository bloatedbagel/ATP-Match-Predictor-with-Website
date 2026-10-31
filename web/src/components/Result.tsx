import type { Detail, Surface } from "../types";

interface Props {
  a: Detail;
  b: Detail;
  probA: number; // 0 to 1
  surface: Surface;
}

export default function Result({ a, b, probA, surface }: Props) {
  const probB = 1 - probA;
  const favorite = probA >= 0.5 ? a : b;
  const verdict =
    Math.max(probA, probB) < 0.55
      ? `Close to a toss-up on ${surface.toLowerCase()}.`
      : `The model favors ${favorite.name} on ${surface.toLowerCase()}.`;

  return (
    <>
      <div className="names">
        <div>{a.name}</div>
        <div>{b.name}</div>
      </div>
      <div className="names">
        <div className="pct">{(probA * 100).toFixed(1)}%</div>
        <div className="pct">{(probB * 100).toFixed(1)}%</div>
      </div>
      <div className="court" aria-hidden="true">
        <div className="a" style={{ width: `${probA * 100}%` }} />
        <div className="b" style={{ width: `${probB * 100}%` }} />
        <div className="ball" style={{ left: `${probA * 100}%` }} />
      </div>
      <p className="verdict">{verdict}</p>
    </>
  );
}
