"""Adapters for external benchmarks of memory and context management.

Each adapter replays a benchmark's history through one of our memory methods
(profile, summary, ...) and answers the benchmark's questions from the memory the
method kept, so designs are compared on the benchmark's own tasks:

- MemConflict (memconflict.py): does the agent use the right memory when memories
  disagree (a real update, a false contradiction, a condition-dependent preference)?
- DynamicMem (dynamicmem.py): does the user model stay accurate over months of app
  activity, and does it serve the user well?

Benchmark data stays outside the repo (data/external/, not committed).
"""
