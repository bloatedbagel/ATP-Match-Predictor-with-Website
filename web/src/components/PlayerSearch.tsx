import { useMemo, useState, type KeyboardEvent } from "react";
import type { PlayerInfo } from "../types";

const norm = (s: string) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

interface Props {
  id: string;
  label: string;
  players: PlayerInfo[]; // sorted by rank
  value: string; // selected player's name
  onChange: (name: string) => void;
}

export default function PlayerSearch({ id, label, players, value, onChange }: Props) {
  const [query, setQuery] = useState<string | null>(null); // null = showing the selected name
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);

  const matches = useMemo(() => {
    const q = norm(query ?? "");
    return players.filter((p) => norm(p.name).includes(q));
  }, [players, query]);

  const choose = (p: PlayerInfo) => {
    onChange(p.name);
    setQuery(null);
    setOpen(false);
  };
  
  const move = (next: number) => {
    setHi(next);
    requestAnimationFrame(() => document.getElementById(`${id}-opt-${next}`)?.scrollIntoView({ block: "nearest" }));
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); move((hi + 1) % Math.max(matches.length, 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move((hi + matches.length - 1) % Math.max(matches.length, 1)); }
    else if (e.key === "Enter" && open && matches[hi]) { e.preventDefault(); choose(matches[hi]); }
    else if (e.key === "Escape") { setOpen(false); setQuery(null); }
  };

  return (
    <div className="search">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-activedescendant={open && matches[hi] ? `${id}-opt-${hi}` : undefined}
        autoComplete="off"
        placeholder="Search any ATP player"
        value={query ?? value}
        onFocus={(e) => { e.target.select(); setOpen(true); }}
        onBlur={() => { setOpen(false); setQuery(null); }}
        onChange={(e) => { setQuery(e.target.value); setHi(0); setOpen(true); }}
        onKeyDown={onKey}
      />
      {open && (
        <ul key={query ?? ""} id={`${id}-list`} role="listbox" onMouseDown={(e) => e.preventDefault()}>
          {matches.length === 0 && <li className="none">No player found</li>}
          {matches.map((p, i) => (
            <li
              key={p.name}
              id={`${id}-opt-${i}`}
              role="option"
              aria-selected={i === hi}
              onMouseDown={(e) => { e.preventDefault(); choose(p); }}
              onMouseEnter={() => setHi(i)}
            >
              <span>{p.name}</span>
              <small>{p.active ? `#${p.rank}` : `#${p.rank} · inactive`}</small>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
