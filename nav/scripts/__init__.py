"""Main entry-point scripts.

Each module here is a thin CLI wrapper that imports submodules from the
rest of the ``nav`` package and stitches them into one runnable job.
Per ``docs/refactor_plan.md`` the modules in this package should stay
thin — heavy logic belongs in :mod:`nav.eval`, :mod:`nav.harness`, etc.
"""
