"""The state workspace: a layout of markdown files with size limits.

A Layout says which files exist, their limits, and which entry types each may
hold (see agent.md). Layouts are data, so a FoodBody deployment can add files
(e.g. decisions) without code changes. The workspace root contains only the
layout's files; traces, raw logs and evaluator outputs live in sibling
directories that no tool can address.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

CONTEXT_FILE = "context.md"
EPISODES_FILE = "memory/episodes.md"
TRANSIENT_FILE = "state/transient.md"
ARCHIVE_FILE = "memory/archive.md"


@dataclass(frozen=True)
class FileSpec:
    path: str
    title: str
    limit: int
    entry_types: tuple[str, ...]  # entry ID prefixes allowed in this file; () for context.md
    preload: bool = False  # shown in full in every call when the method pre-loads state


@dataclass(frozen=True)
class Layout:
    name: str
    files: tuple[FileSpec, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)

    def spec(self, path: str) -> FileSpec | None:
        return next((f for f in self.files if f.path == path), None)

    @property
    def preloaded(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files if f.preload)

    def file_for_type(self, entry_type: str) -> FileSpec | None:
        return next((f for f in self.files if entry_type in f.entry_types), None)

    def with_limit(self, path: str, limit: int) -> "Layout":
        return replace(self, files=tuple(replace(f, limit=limit) if f.path == path else f for f in self.files))


_CORE = (
    FileSpec(CONTEXT_FILE, "Working context", 6000, ()),
    FileSpec(EPISODES_FILE, "Episodes", 60000, ("E", "O")),
    # Current state: small, bounded by archiving, and pre-loaded into every call (contract v2).
    FileSpec("state/profile.md", "Profile", 12000, ("F",), preload=True),
    FileSpec("state/preferences.md", "Preferences", 12000, ("P",), preload=True),
    FileSpec("state/goals.md", "Goals", 12000, ("G",), preload=True),
    FileSpec(TRANSIENT_FILE, "Transient state", 8000, ("T",), preload=True),
    FileSpec("state/hypotheses.md", "Hypotheses", 12000, ("H",), preload=True),
    # Closed state entries (moved by the archive tool) and old episodes (rolled over by the
    # layer), kept unchanged so every ID still resolves.
    FileSpec(ARCHIVE_FILE, "Archive", 200000, ("E", "O", "F", "P", "G", "T", "H", "D", "R")),
)

# HorizonBench has no agent decisions or outcomes, so the decisions file is absent.
HORIZONBENCH = Layout("horizonbench", _CORE)
# Reserved for event sources that include the agent's own decisions and their outcomes.
WITH_DECISIONS = Layout("with-decisions", _CORE + (FileSpec("memory/decisions.md", "Decisions", 20000, ("D", "R")),))
LAYOUTS = {layout.name: layout for layout in (HORIZONBENCH, WITH_DECISIONS)}


class WorkspaceError(Exception):
    """A rejected workspace operation. The message is shown to the agent."""


@dataclass(frozen=True)
class FileChange:
    path: str
    op: str
    before_sha256: str
    after_sha256: str
    chars_before: int
    chars_after: int
    diff: str


def template(spec: FileSpec) -> str:
    if spec.path == CONTEXT_FILE:
        return "# Working context · as of (no events yet)\n\n## Now\n\n## Watch\n\n## Active\n"
    return f"# {spec.title}\n"


class Workspace:
    def __init__(self, root: Path, layout: Layout, on_change: Callable[[FileChange], None] | None = None):
        self.root = root.resolve()
        self.layout = layout
        self.on_change = on_change

    @classmethod
    def create(cls, root: Path, layout: Layout, **kwargs) -> "Workspace":
        root.mkdir(parents=True, exist_ok=False)
        ws = cls(root, layout, **kwargs)
        for spec in layout.files:
            target = ws.root / spec.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(template(spec))
        return ws

    # ---------------------------------------------------------------- access

    def resolve(self, path: str) -> Path:
        if not isinstance(path, str):
            raise WorkspaceError("path must be a string")
        normalized = path.strip().removeprefix("./")
        if self.layout.spec(normalized) is None:
            raise WorkspaceError(f"Unknown file {path!r}. Allowed files: {', '.join(self.layout.paths)}")
        full = self.root / normalized
        # Defense in depth: the allowlist already excludes traversal, but refuse
        # symlinks or anything that resolves outside the root.
        if full.is_symlink() or not full.resolve().is_relative_to(self.root):
            raise WorkspaceError(f"Refusing to access {path!r}")
        return full

    def limit(self, path: str) -> int:
        return self.layout.spec(path.strip().removeprefix("./")).limit

    def list_files(self) -> list[dict]:
        return [{"path": f.path, "chars": len(self.read(f.path)), "limit": f.limit} for f in self.layout.files]

    def read(self, path: str) -> str:
        return self.resolve(path).read_text()

    def write(self, path: str, content: str) -> FileChange:
        return self._commit(path, "write_file", content)

    def append(self, path: str, text: str) -> FileChange:
        current = self.read(path)
        sep = "" if not current or current.endswith("\n") else "\n"
        return self._commit(path, "append_file", current + sep + text)

    def edit(self, path: str, old_text: str, new_text: str) -> FileChange:
        current = self.read(path)
        if not old_text:
            raise WorkspaceError("old_text must be non-empty")
        count = current.count(old_text)
        if count != 1:
            raise WorkspaceError(f"old_text must match exactly once in {path}; found {count} matches")
        return self._commit(path, "edit_file", current.replace(old_text, new_text, 1))

    def snapshot(self) -> dict[str, str]:
        return {p: self.read(p) for p in self.layout.paths}

    def save_snapshot(self, dest: Path) -> None:
        for path, content in self.snapshot().items():
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    def restore_snapshot(self, files: dict[str, str]) -> None:
        """Overwrite the workspace with a saved snapshot (used when resuming a run)."""
        for path, content in files.items():
            self.resolve(path).write_text(content)

    # -------------------------------------------------------------- internals

    def _commit(self, path: str, op: str, new_content: str) -> FileChange:
        if not isinstance(new_content, str):
            raise WorkspaceError("content must be a string")
        full = self.resolve(path)
        limit = self.limit(path)
        if len(new_content) > limit:
            raise WorkspaceError(
                f"Rejected: {path} would be {len(new_content)} characters, over its limit of "
                f"{limit}. Condense the content and try again."
            )
        before = full.read_text()
        full.write_text(new_content)
        clean = path.strip().removeprefix("./")
        change = FileChange(
            path=clean,
            op=op,
            before_sha256=_sha(before),
            after_sha256=_sha(new_content),
            chars_before=len(before),
            chars_after=len(new_content),
            diff="".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{clean}",
                    tofile=f"b/{clean}",
                )
            ),
        )
        if self.on_change:
            self.on_change(change)
        return change


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
