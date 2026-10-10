"""The structured view over preferences: dimensions and their current values.

Preference entries stay free-text statements, but each sits on a dimension (its
`topic`). The layout supplies seed dimensions; the agent may add one with a
`new_dimension: <reason>` field. Showing every dimension with its current value
in each call is what lets new evidence meet the old value, so a change becomes a
supersession instead of one more unrelated preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entries import CHANGE_FIELD, Entry, parse_file
from .workspace import Layout

PREFERENCES_FILE = "state/preferences.md"
HYPOTHESES_FILE = "state/hypotheses.md"
TRANSIENT_FILE = "state/transient.md"


@dataclass
class DimensionRow:
    key: str
    description: str  # seed description, or the agent's stated reason for a custom dimension
    custom: bool
    current: list[Entry] = field(default_factory=list)
    challenged_by: list[Entry] = field(default_factory=list)
    hypotheses: list[Entry] = field(default_factory=list)  # open hypotheses tagged with this dimension
    overridden_by: list[Entry] = field(default_factory=list)  # active transient entries overriding the value


def dimension_rows(layout: Layout, preferences_text: str, hypotheses_text: str,
                   transient_text: str = "") -> list[DimensionRow]:
    prefs, _ = parse_file(PREFERENCES_FILE, preferences_text, ("P",))
    hyps, _ = parse_file(HYPOTHESES_FILE, hypotheses_text, ("H",))
    overrides = [t for t in parse_file(TRANSIENT_FILE, transient_text, ("T",))[0]
                 if t.status == "active" and t.refs().get("overrides")]
    rows = {d.key: DimensionRow(d.key, d.description, custom=False) for d in layout.dimensions}
    for p in prefs:
        key = p.fields.get("topic")
        if not key:
            continue
        if key not in rows:
            rows[key] = DimensionRow(key, "", custom=True)
        row = rows[key]
        if row.custom and not row.description and p.fields.get("new_dimension"):
            row.description = p.fields["new_dimension"]
        if p.status == "current":
            row.current.append(p)
    open_hyps = [h for h in hyps if h.status == "open"]
    for row in rows.values():
        ids = {p.id for p in row.current}
        row.challenged_by = [h for h in open_hyps if ids & set(h.refs().get("challenges", []))]
        row.hypotheses = [h for h in open_hyps if h.fields.get("topic") == row.key and h not in row.challenged_by]
        row.overridden_by = [t for t in overrides if ids & set(t.refs()["overrides"])]
    return list(rows.values())


def render_dimensions(rows: list[DimensionRow]) -> str:
    lines = [
        "<preference_dimensions>",
        "Each dimension holds one current value. New evidence on a dimension confirms, refines (supersede) "
        "or challenges (open a hypothesis against) that value.",
    ]
    seed = [r for r in rows if not r.custom]
    custom = [r for r in rows if r.custom]
    for group, title in ((seed, None), (custom, "Dimensions you added:")):
        if title and group:
            lines.append(title)
        for r in group:
            lines.append(f"{r.key}: {r.description or '(no reason given)'}")
            if r.current:
                for p in r.current:
                    lines.append(f"  current: {p.id} \"{p.statement}\"")
                    if (replaced := p.refs().get("supersedes")) and p.fields.get(CHANGE_FIELD):
                        lines.append(f"    last change: {p.date}, replaced {', '.join(replaced)}: {p.fields[CHANGE_FIELD]}")
                for t in r.overridden_by:
                    until = t.fields.get("expires") or t.fields.get("review") or "?"
                    lines.append(f"  for now, overridden by: {t.id} (until {until}) \"{t.statement}\"")
            else:
                lines.append("  current: (none yet)")
            lines += [f"  challenged by: {h.id} (open) \"{h.statement}\"" for h in r.challenged_by]
            lines += [f"  open hypothesis: {h.id} \"{h.statement}\"" for h in r.hypotheses]
    lines.append("</preference_dimensions>")
    return "\n".join(lines)
