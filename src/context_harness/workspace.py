"""Per-item agent workspace: a fixed allowlist of markdown files with size budgets.

The workspace root contains only the allowlisted files. Traces, raw session logs,
and evaluator outputs live in sibling directories that no tool can address.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

CONTEXT_FILE = "context.md"

FILE_PURPOSES: dict[str, str] = {
    CONTEXT_FILE: (
        "Working context. Its full current contents are shown to you automatically at the "
        "start of every model call, and it is the only file that is. Keep here what you need "
        "most often to serve this user well right now: who they are, what is currently true "
        "about their preferences and situation, active threads, and pointers to the files "
        "that hold more detail. Replace stale content rather than appending indefinitely."
    ),
    "long_term_memory.md": (
        "Durable facts about the user that are unlikely to change soon: identity, background, "
        "relationships, and significant history, with dates. Read on demand."
    ),
    "user_state/goals.md": (
        "What the user is trying to accomplish: goals and projects with status "
        "(active / completed / abandoned) and dates."
    ),
    "user_state/persistent_state.md": (
        "Stable preferences and traits, such as how the user likes to be helped and "
        "communicated with. Note when each was first observed and last confirmed. When one "
        "changes, record the new value as current and keep the old value as superseded, "
        "with the date and the reason."
    ),
    "user_state/transient_state.md": (
        "Short-lived state: current mood, stressors, temporary constraints or circumstances, "
        "with dates and, if known, when they are expected to end."
    ),
    "user_state/progress.md": (
        "Progress on ongoing tasks and threads across sessions: what was done, what is next, "
        "open questions."
    ),
    "user_state/observations.md": (
        "Dated log of notable evidence from sessions (brief quotes or paraphrases with the "
        "session date). Evidence, not conclusions."
    ),
    "user_state/hypotheses.md": (
        "Tentative inferences about the user, each with confidence and supporting or "
        "contradicting evidence. Promote to state files when confirmed; retire when refuted."
    ),
}
WORKSPACE_FILES: tuple[str, ...] = tuple(FILE_PURPOSES)


class WorkspaceError(Exception):
    """A rejected workspace operation. The message is shown to the agent."""


@dataclass(frozen=True)
class WorkspaceBudgets:
    context_max_chars: int = 8000
    file_max_chars: int = 20000

    def limit(self, path: str) -> int:
        return self.context_max_chars if path == CONTEXT_FILE else self.file_max_chars


@dataclass(frozen=True)
class FileChange:
    path: str
    op: str
    before_sha256: str
    after_sha256: str
    chars_before: int
    chars_after: int
    diff: str


def template(path: str, budgets: WorkspaceBudgets) -> str:
    title = PurePosixPath(path).stem.replace("_", " ").capitalize()
    return (
        f"# {title}\n\n"
        f"<!-- Purpose: {FILE_PURPOSES[path]} Limit: {budgets.limit(path)} characters. -->\n"
    )


class Workspace:
    def __init__(
        self,
        root: Path,
        budgets: WorkspaceBudgets,
        on_change: Callable[[FileChange], None] | None = None,
    ):
        self.root = root.resolve()
        self.budgets = budgets
        self.on_change = on_change

    @classmethod
    def create(cls, root: Path, budgets: WorkspaceBudgets, **kwargs) -> "Workspace":
        root.mkdir(parents=True, exist_ok=False)
        (root / "user_state").mkdir()
        ws = cls(root, budgets, **kwargs)
        for path in WORKSPACE_FILES:
            (ws.root / path).write_text(template(path, budgets))
        return ws

    # ---------------------------------------------------------------- access

    def resolve(self, path: str) -> Path:
        if not isinstance(path, str):
            raise WorkspaceError("path must be a string")
        normalized = path.strip()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized not in FILE_PURPOSES:
            raise WorkspaceError(
                f"Unknown file {path!r}. Allowed files: {', '.join(WORKSPACE_FILES)}"
            )
        full = self.root / normalized
        # Defense in depth: the allowlist already excludes traversal, but refuse
        # symlinks or anything that resolves outside the root.
        if full.is_symlink() or not full.resolve().is_relative_to(self.root):
            raise WorkspaceError(f"Refusing to access {path!r}")
        return full

    def list_files(self) -> list[dict]:
        return [
            {"path": p, "chars": len(self.read(p)), "limit": self.budgets.limit(p)}
            for p in WORKSPACE_FILES
        ]

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
            raise WorkspaceError(
                f"old_text must match exactly once in {path}; found {count} matches"
            )
        return self._commit(path, "edit_file", current.replace(old_text, new_text, 1))

    def snapshot(self) -> dict[str, str]:
        return {p: self.read(p) for p in WORKSPACE_FILES}

    def save_snapshot(self, dest: Path) -> None:
        for path, content in self.snapshot().items():
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    # -------------------------------------------------------------- internals

    def _commit(self, path: str, op: str, new_content: str) -> FileChange:
        if not isinstance(new_content, str):
            raise WorkspaceError("content must be a string")
        full = self.resolve(path)
        limit = self.budgets.limit(path)
        if len(new_content) > limit:
            raise WorkspaceError(
                f"Rejected: {path} would be {len(new_content)} characters, over its limit of "
                f"{limit}. Condense the content and try again."
            )
        before = full.read_text()
        full.write_text(new_content)
        change = FileChange(
            path=path.strip().removeprefix("./"),
            op=op,
            before_sha256=_sha(before),
            after_sha256=_sha(new_content),
            chars_before=len(before),
            chars_after=len(new_content),
            diff="".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            ),
        )
        if self.on_change:
            self.on_change(change)
        return change


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
