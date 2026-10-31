import type { Prediction } from "../types";

export default function ComparisonTable({ p }: { p: Prediction }) {
  const { a, b } = p;
  const pct = (x: number) => `${Math.round(x * 100)}%`;
  const rows: [string, string | number, string | number][] = [
    ["Ranking", `#${a.rank}`, `#${b.rank}`],
    ["Ranking points", a.points, b.points],
    ["Age", a.age.toFixed(1), b.age.toFixed(1)],
    ["Height (cm)", a.height, b.height],
    ["Last 10 matches", pct(a.form), pct(b.form)],
    [`Win rate on ${p.surface.toLowerCase()}`, pct(a.surface_rate), pct(b.surface_rate)],
    ["Head-to-head wins", p.h2h[0], p.h2h[1]],
    ["Last match in data", a.last_match, b.last_match],
  ];
  return (
    <table>
      <thead>
        <tr>
          <th>Model inputs</th>
          <th>{a.name}</th>
          <th>{b.name}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, x, y]) => (
          <tr key={label}>
            <td>{label}</td>
            <td>{x}</td>
            <td>{y}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
