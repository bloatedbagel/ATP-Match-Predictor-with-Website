import { SURFACES, type Surface } from "../types";

interface Props {
  value: Surface;
  onChange: (surface: Surface) => void;
}

export default function SurfacePicker({ value, onChange }: Props) {
  return (
    <div className="surf" role="group" aria-label="Surface">
      {SURFACES.map((s) => (
        <button key={s} type="button" aria-pressed={s === value} onClick={() => onChange(s)}>
          {s}
        </button>
      ))}
    </div>
  );
}
