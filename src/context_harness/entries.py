"""Parse workspace files into entries, following the format in agent.md.

    - <ID> | <status> | <date> | <field>: <value> | ... | <statement>     (F P G T H D R)
    - E-<n> | <date> | source: S-<n> | <statement>                        (episodes)
      - O-<n>.<k> | <kind> | <statement>                                  (observations)

Anything that looks like an entry but doesn't parse is reported as a problem
rather than silently skipped, because format compliance is itself measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .workspace import CONTEXT_FILE, Layout

ID_RE = re.compile(r"\b([A-Z])-(\d+)(?:\.(\d+))?\b")
_LINE = re.compile(r"^(?P<indent>\s*)- (?P<id>[A-Z]-\d+(?:\.\d+)?)\s*\|(?P<rest>.*)$")
_BULLET = re.compile(r"^(?P<indent>\s*)- (?P<text>.*)$")
_FIELD = re.compile(r"^(?P<key>[a-z_]+):\s*(?P<value>.*)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AS_OF = re.compile(r"as of .*\((S-\d+)\)")

OBS_KINDS = {"stated", "reaction", "behavior", "event"}
LIVE_STATUSES = {"current", "active", "open"}
# Closed entries may be moved to the archive; live and paused ones may not.
CLOSED_STATUSES = {"superseded", "retracted", "expired", "refuted", "withdrawn", "confirmed", "achieved", "abandoned",
                   "lapsed"}
STATUSES: dict[str, set[str]] = {
    "F": {"current", "superseded", "retracted"},
    "P": {"current", "superseded", "retracted"},
    "G": {"active", "achieved", "abandoned", "lapsed", "paused", "superseded", "retracted"},
    "T": {"active", "expired", "superseded", "retracted"},
    "H": {"open", "confirmed", "refuted", "withdrawn", "retracted"},
}  # D and R statuses are not yet specified
REF_FIELDS = ("basis", "for", "against", "supersedes", "superseded_by", "resolved_into", "challenges", "decision")


@dataclass
class Problem:
    code: str
    path: str
    line: int
    detail: str
    id: str | None = None


@dataclass
class Entry:
    id: str
    type: str
    event: int
    n: int | None
    status: str | None
    date: str | None
    fields: dict[str, str]
    statement: str
    path: str
    line: int
    parent: str | None = None  # the E entry an observation sits under
    notes: list[str] = field(default_factory=list)  # indented non-entry bullets (goal progress)

    def refs(self) -> dict[str, list[str]]:
        return {
            key: [part.strip() for part in self.fields[key].split(",") if part.strip()]
            for key in REF_FIELDS
            if key in self.fields
        }

    @property
    def live(self) -> bool:
        return self.status in LIVE_STATUSES


@dataclass
class ContextView:
    as_of: str | None  # event ID named in the heading, e.g. "S-27"
    bullets: list[tuple[int, str, list[str]]]  # (line number, text, cited IDs)


@dataclass
class ParsedWorkspace:
    entries: dict[str, Entry]
    problems: list[Problem]
    context: ContextView

    def by_type(self, entry_type: str) -> list[Entry]:
        return [e for e in self.entries.values() if e.type == entry_type]


def parse_id(entry_id: str) -> tuple[str, int, int | None] | None:
    m = ID_RE.fullmatch(entry_id.strip())
    return (m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else None) if m else None


def parse_file(path: str, text: str, allowed_types: tuple[str, ...]) -> tuple[list[Entry], list[Problem]]:
    entries: list[Entry] = []
    problems: list[Problem] = []
    current_episode: Entry | None = None
    last: Entry | None = None
    for no, line in enumerate(text.splitlines(), 1):
        m = _LINE.match(line)
        if not m:
            bullet = _BULLET.match(line)
            if bullet and bullet.group("indent") and last is not None:
                last.notes.append(bullet.group("text").strip())
            elif bullet:
                problems.append(Problem("unparsed_line", path, no, line.strip()[:160]))
            continue
        entry_type, event, n = parse_id(m.group("id"))
        segments = [s.strip() for s in m.group("rest").split("|")]
        entry, problem = _build(path, no, m.group("id"), entry_type, event, n, segments)
        if problem:
            problems.append(problem)
            continue
        if entry_type not in allowed_types:
            problems.append(Problem("wrong_file", path, no, f"{entry_type} entries don't belong here", entry.id))
        if entry_type == "O":
            if m.group("indent") == "" or current_episode is None:
                problems.append(Problem("orphan_observation", path, no, "observation is not under an E line", entry.id))
            else:
                entry.parent = current_episode.id
        if entry_type == "E":
            current_episode = entry
        entries.append(entry)
        last = entry
    return entries, problems


def _build(path, no, entry_id, entry_type, event, n, segments) -> tuple[Entry | None, Problem | None]:
    def bad(detail: str):
        return None, Problem("unparsed_entry", path, no, detail, entry_id)

    if not segments or not segments[-1]:
        return bad("missing statement")
    statement = segments[-1]
    if _FIELD.match(statement) and _FIELD.match(statement).group("key") in REF_FIELDS:
        return bad("last segment is a field, so the statement is missing")
    head = segments[:-1]
    status = date = None
    fields: dict[str, str] = {}
    if entry_type == "O":
        if len(head) != 1:
            return bad("observations need exactly '<kind> | <statement>'")
        fields["kind"] = head[0]
    else:
        if entry_type != "E":
            if not head:
                return bad("missing status")
            status, head = head[0], head[1:]
        if not head:
            return bad("missing date")
        date, head = head[0], head[1:]
        for segment in head:
            fm = _FIELD.match(segment)
            if not fm:
                return bad(f"segment {segment[:60]!r} is not a 'name: value' field")
            fields[fm.group("key")] = fm.group("value").strip()
    return Entry(entry_id, entry_type, event, n, status, date, fields, statement, path, no), None


def parse_context(text: str) -> ContextView:
    as_of = None
    bullets = []
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("#") and as_of is None and (m := _AS_OF.search(line)):
            as_of = m.group(1)
        if bm := _BULLET.match(line):
            bullets.append((no, bm.group("text").strip(), [m.group(0) for m in ID_RE.finditer(bm.group("text"))]))
    return ContextView(as_of, bullets)


def parse_workspace(snapshot: dict[str, str], layout: Layout) -> ParsedWorkspace:
    entries: dict[str, Entry] = {}
    problems: list[Problem] = []
    for spec in layout.files:
        if spec.path == CONTEXT_FILE:
            continue
        file_entries, file_problems = parse_file(spec.path, snapshot.get(spec.path, ""), spec.entry_types)
        problems += file_problems
        for e in file_entries:
            if e.id in entries:
                problems.append(Problem("duplicate_id", e.path, e.line, f"also at {entries[e.id].path}:{entries[e.id].line}", e.id))
            else:
                entries[e.id] = e
    return ParsedWorkspace(entries, problems, parse_context(snapshot.get(CONTEXT_FILE, "")))


def is_valid_date(value: str | None) -> bool:
    return bool(value) and bool(_DATE.match(value))
