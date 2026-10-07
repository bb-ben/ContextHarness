"""Deterministic memory checks, run after every event (evaluator-side; never shown
to the agent).

Each check answers one of the instrumentation questions:
- provenance: does every entry carry a valid source event and date, with evidence that exists?
- consistency: do current-state files contradict each other or the working context?
- survival: was anything deleted or rewritten that the contract says must be kept?
- access: which files did the agent read and edit (computed from tool calls)?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .entries import (
    CLOSED_STATUSES,
    LIVE_STATUSES,
    OBS_KINDS,
    STATUSES,
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


def check_state(parsed: ParsedWorkspace, events: list[EventInfo], current: int) -> list[Problem]:
    """Provenance and consistency checks on one snapshot, as of event `current`."""
    out = list(parsed.problems)
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
        if e.type == "H" and e.status == "confirmed" and not e.fields.get("resolved_into"):
            add("unresolved_confirmation", e, "confirmed without resolved_into:")
        if e.type == "T" and e.status == "active":
            end = e.fields.get("expires") or e.fields.get("review")
            if end and is_valid_date(end) and end <= now.date:
                add("overdue_transient", e, f"due {end}, still active on {now.date}")

    by_topic: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in entries.values():
        if e.type in NEEDS_TOPIC and e.status == "current" and e.fields.get("topic"):
            by_topic[(e.path, e.fields["topic"])].append(e.id)
    for (path, topic), ids in by_topic.items():
        if len(ids) > 1:
            first = entries[ids[0]]
            out.append(Problem("topic_conflict", path, first.line, f"{len(ids)} current entries for topic {topic!r}: {', '.join(ids)}", ids[0]))

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


class MemoryMonitor:
    """Runs all checks after each event and accumulates a per-item report."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.events: list[EventInfo] = []
        self.previous: ParsedWorkspace | None = None
        self.per_event: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.lifecycle: dict[str, dict[str, Any]] = {}

    def after_event(self, event: EventInfo, snapshot: dict[str, str], access: list[dict[str, Any]]) -> dict[str, Any]:
        self.events.append(event)
        parsed = parse_workspace(snapshot, self.layout)
        problems = check_state(parsed, self.events, event.index) + check_retention(self.previous, parsed)
        for p in problems:
            self.violations.append({"event_id": event.event_id, **asdict(p)})
        self._track(parsed, event)
        record = {
            "event_id": event.event_id,
            "index": event.index,
            "date": event.date,
            "access": access_by_file(access),
            "violations": dict(Counter(p.code for p in problems)),
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
            "violations_by_code": dict(totals.most_common()),
            "events_with_violations": sum(1 for r in self.per_event if r["violations"]),
            "access_totals": {p: dict(c) for p, c in sorted(access.items())},
            "final_metrics": self.per_event[-1]["metrics"] if self.per_event else {},
            "per_event": self.per_event,
            "violations": self.violations,
            "lifecycle": self.lifecycle,
        }

    def state(self) -> dict[str, Any]:
        return {"events": [asdict(e) for e in self.events], "per_event": self.per_event,
                "violations": self.violations, "lifecycle": self.lifecycle}

    def restore(self, state: dict[str, Any], snapshot: dict[str, str]) -> None:
        self.events = [EventInfo(**e) for e in state["events"]]
        self.per_event, self.violations, self.lifecycle = state["per_event"], state["violations"], state["lifecycle"]
        self.previous = parse_workspace(snapshot, self.layout) if self.events else None
