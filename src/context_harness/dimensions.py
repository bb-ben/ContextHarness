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

EPISODES_FILE = "memory/episodes.md"
RECENT_WINDOW = 6  # tagged observations per dimension shown at revision time
PREFERENCES_FILE = "state/preferences.md"
PROFILE_FILE = "state/profile.md"
HYPOTHESES_FILE = "state/hypotheses.md"


@dataclass
class DimensionRow:
    key: str
    description: str  # seed description, or the agent's stated reason for a custom dimension
    custom: bool
    current: list[Entry] = field(default_factory=list)
    challenged_by: list[Entry] = field(default_factory=list)
    hypotheses: list[Entry] = field(default_factory=list)  # open hypotheses tagged with this dimension
    # Tagged observations no preference or hypothesis cites yet: a wish seen in one task and
    # again in another only becomes visible as a pattern if something lists them together.
    signals: list[Entry] = field(default_factory=list)
    # The dimension's latest tagged observations, cited or not: revising from the old value
    # plus one event let values grow by addition over 75 events (v3.8, item 6); revising
    # from the recent evidence lets a change that arrives gradually replace the old value.
    recent: list[Entry] = field(default_factory=list)
    recent_uncited: set[str] = field(default_factory=set)  # recent observations no belief cites yet


def uncollected_signals(observations: list[Entry], beliefs: list[Entry]) -> dict[str, list[Entry]]:
    """Observations tagged with a dimension that no preference or hypothesis cites, by dimension."""
    cited = {ref for b in beliefs for ids in b.refs().values() for ref in ids}
    out: dict[str, list[Entry]] = {}
    for o in observations:
        if o.type == "O" and o.fields.get("topic") and o.id not in cited:
            out.setdefault(o.fields["topic"], []).append(o)
    return out


def dimension_rows(layout: Layout, preferences_text: str, hypotheses_text: str,
                   episodes_text: str = "") -> list[DimensionRow]:
    prefs, _ = parse_file(PREFERENCES_FILE, preferences_text, ("P",))
    hyps, _ = parse_file(HYPOTHESES_FILE, hypotheses_text, ("H",))
    observations, _ = parse_file(EPISODES_FILE, episodes_text, ("E", "O"))
    signals = uncollected_signals(observations, prefs + hyps)
    tagged: dict[str, list[Entry]] = {}
    for o in observations:
        if o.type == "O" and o.fields.get("topic"):
            tagged.setdefault(o.fields["topic"], []).append(o)
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
        row.recent = tagged.get(row.key, [])[-RECENT_WINDOW:]
        recent_ids = {o.id for o in row.recent}
        uncited = {o.id for o in signals.get(row.key, [])}
        row.recent_uncited = recent_ids & uncited
        row.signals = [o for o in signals.get(row.key, []) if o.id not in recent_ids][-6:]
    return list(rows.values())


def render_fact_topics(layout: Layout, profile_text: str) -> str:
    """The layout's key profile facts, each with its current entries or "(none yet)"."""
    facts, _ = parse_file(PROFILE_FILE, profile_text, ("F",))
    lines = ["<profile_topics>",
             "Facts every reply must respect. Record one as soon as the person mentions it, even in passing."]
    for topic in layout.fact_topics:
        current = [f for f in facts if f.status == "current" and f.fields.get("topic") == topic.key]
        lines.append(f"{topic.key}: {topic.description}")
        lines += [f"  current: {f.id} \"{f.statement}\"" for f in current] or ["  current: (none yet)"]
    lines.append("</profile_topics>")
    return "\n".join(lines)


def render_dimensions(rows: list[DimensionRow]) -> str:
    lines = [
        "<preference_dimensions>",
        "Each dimension holds one current value (or one per scope). New evidence confirms it, or, when the "
        "recent evidence says the person wants something else, supersede it with a value written from that "
        "evidence; open a hypothesis against it when the evidence is still thin.",
    ]
    seed = [r for r in rows if not r.custom]
    custom = [r for r in rows if r.custom]
    for group, title in ((seed, None), (custom, "Dimensions you added:")):
        if title and group:
            lines.append(title)
        for r in group:
            lines.append(f"{r.key}: {r.description or '(no reason given)'}")
            if r.current:
                for p in sorted(r.current, key=lambda p: p.fields.get("scope", "")):
                    review = f" (review {p.fields['review']})" if p.fields.get("review") else ""
                    scope = f" (for {p.fields['scope']})" if p.fields.get("scope") else ""
                    lines.append(f"  current{scope}: {p.id} \"{p.statement}\"{review}")
                    if (replaced := p.refs().get("supersedes")) and p.fields.get(CHANGE_FIELD):
                        lines.append(f"    last change: {p.date}, replaced {', '.join(replaced)}: {p.fields[CHANGE_FIELD]}")
            else:
                lines.append("  current: (none yet)")
            lines += [f"  challenged by: {h.id} (open) \"{h.statement}\"" for h in r.challenged_by]
            lines += [f"  open hypothesis: {h.id} \"{h.statement}\"" for h in r.hypotheses]
            if r.recent:
                lines.append(f"  recent evidence (latest {len(r.recent)}): "
                             + "; ".join(f"{o.id} \"{o.statement}\"" for o in r.recent))
                new = [o.id for o in r.recent if o.id in r.recent_uncited]
                if new:
                    lines.append(f"    not yet in any belief: {', '.join(new)}")
            if r.signals:
                events = len({o.event for o in r.signals})
                lines.append(f"  older signals not yet in any belief ({events} event{'s' if events > 1 else ''}): "
                             + "; ".join(f"{o.id} \"{o.statement}\"" for o in r.signals))
    lines.append("</preference_dimensions>")
    return "\n".join(lines)
