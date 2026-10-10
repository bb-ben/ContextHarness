"""Deterministic memory checks, run after every event (evaluator-side; never shown
to the agent).

Each check answers one of the instrumentation questions:
- provenance: does every entry carry a valid source event and date, with evidence that exists?
- consistency: do current-state files contradict each other or the working context?
- survival: was anything deleted or rewritten that the contract says must be kept?
- access: which files did the agent read and edit (computed from tool calls)?
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .entries import (
    CHANGE_FIELD,
    CLOSED_STATUSES,
    LIVE_STATUSES,
    OBS_KINDS,
    STATUSES,
    VALUE_MAX_CHARS,
    Entry,
    ParsedWorkspace,
    Problem,
    is_valid_date,
    parse_workspace,
)
from .workspace import ARCHIVE_FILE, CONTEXT_FILE, Layout

TERMINAL = CLOSED_STATUSES
NEEDS_BASIS = {"F", "P", "G", "T"}
NEEDS_TOPIC = {"F", "P"}


@dataclass(frozen=True)
class EventInfo:
    index: int
    event_id: str  # e.g. "S-14"
    date: str  # YYYY-MM-DD


def hypothesis_ripe_at(e: Entry) -> int | None:
    """The event at which an open hypothesis first had support from two different events.

    It must be decided in that same event (confirm, refute, withdraw, or defer with
    review:). In the scenarios, "playful" and "gentler coaching" reached two events of
    support in the last event before the person's message and were left open, so no
    reply could rely on them.
    """
    events = sorted({int(ref.split("-")[1].split(".")[0]) for ref in e.refs().get("for", []) if ref.startswith("O-")})
    return events[1] if len(events) >= 2 else None


def goal_end(e: Entry) -> str | None:
    """The date by which an active goal must be reviewed: its review date if set, else its due date."""
    return e.fields.get("review") or e.fields.get("due")


_OBS_ID = re.compile(r"\bO-\d+\.\d+\b")

_GENERIC_TOKENS = {"style", "level", "preference", "preferences", "type", "mode", "pref"}


def _stems(key: str) -> set[str]:
    return {t[:5] for t in key.split("-") if t and t not in _GENERIC_TOKENS}


def check_preferences(parsed: ParsedWorkspace, layout: Layout) -> list[Problem]:
    """Dimension discipline and the evidence rule for preferences."""
    out: list[Problem] = []
    entries = parsed.entries
    prefs = [e for e in entries.values() if e.type == "P"]
    seed = {d.key for d in layout.dimensions}
    by_topic: dict[str, list[Entry]] = defaultdict(list)
    for p in prefs:
        if p.fields.get("topic"):
            by_topic[p.fields["topic"]].append(p)
    for topic, group in by_topic.items():
        if topic in seed:
            continue
        first = min(group, key=lambda e: (e.event, e.n or 0))
        if not any(e.fields.get("new_dimension") for e in group):
            out.append(Problem("undeclared_dimension", first.path, first.line,
                               f"dimension {topic!r} is not a seed dimension and has no new_dimension: reason", first.id))
        near = sorted(o for o in (seed | set(by_topic)) - {topic} if _stems(o) & _stems(topic))
        if near:
            out.append(Problem("near_duplicate_dimension", first.path, first.line,
                               f"dimension {topic!r} overlaps {', '.join(near)}", first.id))
    resolved = {h.fields.get("resolved_into") for h in entries.values() if h.type == "H"}
    for p in prefs:
        if p.status != "current":  # judged while it is in force; superseding it resolves the flag
            continue
        basis = [entries[b] for b in p.refs().get("basis", []) if b in entries and entries[b].type == "O"]
        if not basis or p.id in resolved:
            continue
        if all(b.fields.get("kind") != "stated" for b in basis) and len({b.event for b in basis}) < 2:
            out.append(Problem("unsupported_promotion", p.path, p.line,
                               "preference rests on non-stated evidence from one event and no confirmed hypothesis", p.id))
    return out


def check_state(parsed: ParsedWorkspace, events: list[EventInfo], current: int,
                layout: Layout | None = None) -> list[Problem]:
    """Provenance and consistency checks on one snapshot, as of event `current`."""
    out = list(parsed.problems)
    if layout is not None:
        out += check_preferences(parsed, layout)
    entries = parsed.entries
    now = events[current]

    def add(code: str, e: Entry, detail: str) -> None:
        out.append(Problem(code, e.path, e.line, detail, e.id))

    for e in entries.values():
        if e.event > current:
            add("future_id", e, f"created at event {e.event}, but the latest event is {current}")
        elif e.type != "O":
            if not is_valid_date(e.date):
                add("bad_date", e, f"{e.date!r} is not YYYY-MM-DD")
            elif e.date != events[e.event].date:
                add("date_mismatch", e, f"date {e.date} but event {events[e.event].event_id} is {events[e.event].date}")
        if e.type == "E" and e.fields.get("source") != events[min(e.event, current)].event_id:
            add("bad_source", e, f"source {e.fields.get('source')!r}, expected {events[min(e.event, current)].event_id!r}")
        if e.type == "O":
            if e.fields.get("kind") not in OBS_KINDS:
                add("bad_kind", e, f"kind {e.fields.get('kind')!r} is not one of {sorted(OBS_KINDS)}")
            if e.parent and int(e.parent.split("-")[1]) != e.event:
                add("parent_mismatch", e, f"listed under {e.parent}")
        if e.type in STATUSES and e.status not in STATUSES[e.type]:
            add("bad_status", e, f"status {e.status!r} is not one of {sorted(STATUSES[e.type])}")
        if e.type in NEEDS_BASIS and not e.refs().get("basis"):
            add("missing_basis", e, "no basis: observations")
        if e.type in NEEDS_TOPIC and not e.fields.get("topic"):
            add("missing_topic", e, "no topic:")
        if e.type == "T" and not (e.fields.get("expires") or e.fields.get("review")):
            add("missing_end_date", e, "no expires: or review:")
        if e.type == "G" and e.status == "active" and not (e.fields.get("due") or e.fields.get("review")):
            add("missing_end_date", e, "active goal with no due: or review:")
        for key, ids in e.refs().items():
            for ref in ids:
                if ref not in entries:
                    add("dangling_ref", e, f"{key}: {ref} does not exist")
                elif key == "basis" and entries[ref].type != "O":
                    add("basis_not_observation", e, f"basis {ref} is a {entries[ref].type} entry")
        if e.status == "superseded":
            if not (e.fields.get("superseded_by") and e.fields.get("until")):
                add("incomplete_supersession", e, "superseded without superseded_by: and until:")
            elif (new := entries.get(e.fields["superseded_by"])) and e.id not in new.refs().get("supersedes", []):
                add("supersession_mismatch", e, f"{new.id} does not list supersedes: {e.id}")
        for old_id in e.refs().get("supersedes", []):
            if (old := entries.get(old_id)) and old.status != "superseded":
                add("supersession_mismatch", e, f"supersedes {old_id}, which is still {old.status}")
        if e.path == ARCHIVE_FILE and e.status not in CLOSED_STATUSES:
            add("archived_live", e, f"archived while {e.status}")
        if e.type in ("P", "F"):
            replaced = [entries[r] for r in e.refs().get("supersedes", []) if r in entries and entries[r].type == e.type]
            if replaced and not e.fields.get(CHANGE_FIELD):
                add("missing_change_note", e, f"supersedes {replaced[0].id} without {CHANGE_FIELD}: saying what is different")
            if e.status == "current" and len(e.statement) > VALUE_MAX_CHARS:
                add("long_value", e, f"statement is {len(e.statement)} characters (limit {VALUE_MAX_CHARS})")
        if e.type == "H" and e.status == "confirmed" and not e.fields.get("resolved_into"):
            add("unresolved_confirmation", e, "confirmed without resolved_into:")
        if e.type == "T" and e.status == "active":
            end = e.fields.get("expires") or e.fields.get("review")
            if end and is_valid_date(end) and end <= now.date:
                add("overdue_transient", e, f"due {end}, still active on {now.date}")
        if e.type == "H" and e.status == "open" and (ripe := hypothesis_ripe_at(e)) is not None and ripe <= current:
            review = e.fields.get("review")
            if not (review and is_valid_date(review) and review > now.date):
                add("undecided_hypothesis", e, f"supported by two events since S-{ripe}, still open without a later review:")
        if e.type == "T" and e.status == "active":
            for target in e.refs().get("overrides", []):
                chain = [t for t in entries.values() if t.type == "T" and target in t.refs().get("overrides", [])]
                if len(chain) >= 3 and target in entries and entries[target].status == "current":
                    add("renewed_override", e, f"{target} has been overridden {len(chain)} times; the change has "
                                               "lasted, so supersede it, or let the override expire")
        if e.type == "G" and e.status == "active":
            end = goal_end(e)
            if end and is_valid_date(end) and end < now.date:  # a goal is live through its due day
                add("overdue_goal", e, f"{'review' if e.fields.get('review') else 'due'} {end} passed, still active on {now.date}")

    by_topic: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in entries.values():
        if e.type in NEEDS_TOPIC and e.status == "current" and e.fields.get("topic"):
            by_topic[(e.path, e.fields["topic"])].append(e.id)
    for (path, topic), ids in by_topic.items():
        if len(ids) > 1:
            first = entries[ids[0]]
            out.append(Problem("topic_conflict", path, first.line, f"{len(ids)} current entries for topic {topic!r}: {', '.join(ids)}", ids[0]))

    # A stated observation is the strongest evidence there is; one that no state entry,
    # hypothesis or goal note cites has been recorded but not applied.
    cited = {ref for e in entries.values() if e.type != "O" for ids in e.refs().values() for ref in ids}
    cited |= {m for e in entries.values() for note in e.notes for m in _OBS_ID.findall(note)}
    for e in entries.values():
        if e.type == "O" and e.fields.get("kind") == "stated" and e.id not in cited:
            add("unused_stated_observation", e, "stated observation cited by no state entry, hypothesis or goal note")

    if f"E-{current}" not in entries:
        out.append(Problem("missing_episode", "memory/episodes.md", 0, f"no E-{current} line for {now.event_id}"))

    ctx = parsed.context
    if ctx.as_of != now.event_id:
        out.append(Problem("context_stale", CONTEXT_FILE, 1, f"as of {ctx.as_of!r}, latest event is {now.event_id}"))
    for line, text, ids in ctx.bullets:
        if not ids:
            out.append(Problem("context_uncited", CONTEXT_FILE, line, text[:120]))
        for ref in ids:
            if ref not in entries:
                out.append(Problem("context_dangling", CONTEXT_FILE, line, f"cites {ref}, which does not exist", ref))
            elif not entries[ref].live:
                out.append(Problem("context_inactive", CONTEXT_FILE, line, f"cites {ref}, which is {entries[ref].status or entries[ref].type}", ref))
    return out


def check_retention(previous: ParsedWorkspace | None, current: ParsedWorkspace) -> list[Problem]:
    """Survival checks: nothing outside context.md may be deleted, and history fields are immutable."""
    if previous is None:
        return []
    out = []
    for old in previous.entries.values():
        new = current.entries.get(old.id)
        if new is None:
            out.append(Problem("entry_deleted", old.path, old.line, f"{old.id} was removed", old.id))
            continue
        changed = [name for name, a, b in (("statement", old.statement, new.statement), ("date", old.date, new.date),
                                           ("kind", old.fields.get("kind"), new.fields.get("kind"))) if a != b]
        lost = set(old.refs().get("basis", [])) - set(new.refs().get("basis", []))
        if lost:
            changed.append(f"basis lost {sorted(lost)}")
        if changed:
            out.append(Problem("history_rewritten", new.path, new.line, "; ".join(changed), new.id))
        if old.status in TERMINAL and new.status in LIVE_STATUSES:
            out.append(Problem("status_reverted", new.path, new.line, f"{old.status} -> {new.status}", new.id))
    return out


def access_by_file(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-file read/write/error counts from tool-call access records."""
    table: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for path in r.get("paths", []):
            table[path]["errors" if not r["ok"] else r["kind"] + "s"] += 1
    return {path: dict(c) for path, c in sorted(table.items())}


def metrics(parsed: ParsedWorkspace, events: list[EventInfo], current: int) -> dict[str, Any]:
    entries = list(parsed.entries.values())
    counts = Counter(e.type for e in entries)
    live = Counter(e.type for e in entries if e.live)
    stateful = [e for e in entries if e.type in NEEDS_BASIS | {"H"}]

    def has_provenance(e: Entry) -> bool:
        dated = is_valid_date(e.date) and e.event <= current and e.date == events[e.event].date
        evidence = e.refs().get("basis") if e.type in NEEDS_BASIS else e.refs().get("for")
        return bool(dated and evidence and all(r in parsed.entries for r in evidence))

    live_state = {e.id for e in entries if e.live and e.type in {"F", "P", "G", "T", "H"}}
    cited = {ref for _, _, ids in parsed.context.bullets for ref in ids}
    return {
        "entries": dict(counts),
        "live_entries": dict(live),
        "provenance_share": (sum(has_provenance(e) for e in stateful) / len(stateful)) if stateful else None,
        "context_bullets": len(parsed.context.bullets),
        "context_cited_ids": len(cited),
        "context_coverage_of_live_state": (len(cited & live_state) / len(live_state)) if live_state else None,
        "episodes_recorded": sum(1 for e in entries if e.type == "E"),
    }


MEMORY_KINDS = {
    "episodic": {"E", "O"},
    "state": {"F", "P", "G", "T", "H"},
    "decisions": {"D", "R"},
}


def diff_memory(previous: ParsedWorkspace | None, current: ParsedWorkspace) -> dict[str, Any]:
    """What one event added, changed, moved (archived or rolled over) and removed, per kind of memory."""
    before = previous.entries if previous else {}
    after = current.entries
    out: dict[str, Any] = {}
    for kind, types in MEMORY_KINDS.items():
        added, changed, moved, removed = [], [], [], []
        for entry_id, e in after.items():
            if e.type not in types:
                continue
            old = before.get(entry_id)
            if old is None:
                added.append({"id": e.id, "status": e.status or e.fields.get("kind") or "episode", "path": e.path,
                              "statement": e.statement})
                continue
            if old.path != e.path:
                moved.append({"id": e.id, "from": old.path, "to": e.path})
            field_changes = {k: [old.fields.get(k), v] for k, v in e.fields.items() if old.fields.get(k) != v}
            if old.status != e.status or field_changes:
                changed.append({"id": e.id, "status": [old.status, e.status], "fields": field_changes,
                                "statement": e.statement})
        removed = [{"id": i, "statement": o.statement} for i, o in before.items() if o.type in types and i not in after]
        out[kind] = {"added": added, "changed": changed, "moved": moved, "removed": removed}
    old_bullets = {text for _, text, _ in previous.context.bullets} if previous else set()
    new_bullets = {text for _, text, _ in current.context.bullets}
    old_ids = {i for _, _, ids in previous.context.bullets for i in ids} if previous else set()
    new_ids = {i for _, _, ids in current.context.bullets for i in ids}
    out["working_context"] = {
        "added": sorted(new_bullets - old_bullets),
        "removed": sorted(old_bullets - new_bullets),
        "cited_added": sorted(new_ids - old_ids),
        "cited_removed": sorted(old_ids - new_ids),
        "bullets": len(new_bullets),
        "as_of": current.context.as_of,
    }
    return out


class MemoryMonitor:
    """Runs all checks after each event and accumulates a per-item report."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.events: list[EventInfo] = []
        self.previous: ParsedWorkspace | None = None
        self.per_event: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.lifecycle: dict[str, dict[str, Any]] = {}
        self.timeline: list[dict[str, Any]] = []
        self.open: dict[tuple, int] = {}  # violation key -> index in self.violations

    @staticmethod
    def _key(p: Problem) -> tuple:
        return (p.code, p.path, p.id) if p.id else (p.code, p.path, p.detail)

    def after_event(self, event: EventInfo, snapshot: dict[str, str], access: list[dict[str, Any]]) -> dict[str, Any]:
        self.events.append(event)
        parsed = parse_workspace(snapshot, self.layout)
        problems = check_state(parsed, self.events, event.index, self.layout) + check_retention(self.previous, parsed)
        # Each distinct violation is recorded once, when it first appears, and marked
        # resolved when it disappears; per-event counts still show what is open.
        current = {}
        for p in problems:
            current.setdefault(self._key(p), p)
        new = [k for k in current if k not in self.open]
        for k in new:
            self.violations.append({"event_id": event.event_id, **asdict(current[k])})
            self.open[k] = len(self.violations) - 1
        for k in [k for k in self.open if k not in current]:
            self.violations[self.open.pop(k)]["resolved_at"] = event.event_id
        self._track(parsed, event)
        self.timeline.append({"event_id": event.event_id, "date": event.date, **diff_memory(self.previous, parsed)})
        record = {
            "event_id": event.event_id,
            "index": event.index,
            "date": event.date,
            "access": access_by_file(access),
            "violations": dict(Counter(p.code for p in current.values())),
            "new_violations": dict(Counter(current[k].code for k in new)),
            "metrics": metrics(parsed, self.events, event.index),
        }
        self.per_event.append(record)
        self.previous = parsed
        return record

    def _track(self, parsed: ParsedWorkspace, event: EventInfo) -> None:
        for e in parsed.entries.values():
            life = self.lifecycle.setdefault(e.id, {"type": e.type, "created": event.event_id, "statuses": []})
            if not life["statuses"] or life["statuses"][-1][1] != e.status:
                life["statuses"].append((event.event_id, e.status))
            life["last_seen"] = event.event_id
            life.pop("deleted", None)
        for entry_id, life in self.lifecycle.items():
            if entry_id not in parsed.entries and "deleted" not in life:
                life["deleted"] = event.event_id

    def report(self) -> dict[str, Any]:
        totals = Counter(v["code"] for v in self.violations)
        access = defaultdict(Counter)
        for record in self.per_event:
            for path, counts in record["access"].items():
                access[path].update(counts)
        return {
            "layout": self.layout.name,
            "events": len(self.per_event),
            "violations_by_code": dict(totals.most_common()),  # distinct violations, by first appearance
            "events_with_violations": sum(1 for r in self.per_event if r["violations"]),
            "events_with_new_violations": sum(1 for r in self.per_event if r.get("new_violations")),
            "unresolved_at_end": len(self.open),
            "access_totals": {p: dict(c) for p, c in sorted(access.items())},
            "final_metrics": self.per_event[-1]["metrics"] if self.per_event else {},
            "per_event": self.per_event,
            "violations": self.violations,
            "lifecycle": self.lifecycle,
            "timeline": self.timeline,
        }

    def state(self) -> dict[str, Any]:
        return {"events": [asdict(e) for e in self.events], "per_event": self.per_event,
                "violations": self.violations, "lifecycle": self.lifecycle, "timeline": self.timeline,
                "open": [[list(k), i] for k, i in self.open.items()]}

    def restore(self, state: dict[str, Any], snapshot: dict[str, str]) -> None:
        self.events = [EventInfo(**e) for e in state["events"]]
        self.per_event, self.violations, self.lifecycle = state["per_event"], state["violations"], state["lifecycle"]
        self.timeline = state.get("timeline", [])
        self.open = {tuple(k): i for k, i in state.get("open", [])}
        self.previous = parse_workspace(snapshot, self.layout) if self.events else None
