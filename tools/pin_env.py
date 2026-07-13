"""Refresh dependency pins + .python-version to the CURRENT environment.

Run inside the freshly-built/trained environment (``make freeze`` does this) so
the versions committed to the repo exactly match the environment that produced
``models/*.pkl``. Matching versions is what lets the models unpickle on the
deploy target without a scikit-learn ``InconsistentVersionWarning`` ->
``AttributeError`` crash.

Usage:
    python tools/pin_env.py requirements.txt [more-requirements.txt ...]

Each requirement line is rewritten to ``name==<installed version>`` while
comments, blank lines and ``-e .`` / ``-r ...`` include lines are preserved
verbatim. A ``.python-version`` file is (re)written from the running
interpreter. Packages named in a file but not installed in the current
environment are left untouched.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version

# Matches the distribution name (and optional extras) at the start of a line.
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?")


def repin(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    out: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        # Keep blank lines, comments and pip directives (-e ., -r other.txt).
        if not stripped or stripped.startswith(("#", "-")):
            out.append(line)
            continue
        code, sep, comment = line.partition("#")
        m = _NAME.match(code.strip())
        if not m:
            out.append(line)
            continue
        name, extras = m.group(1), m.group(2) or ""
        try:
            pinned = f"{name}{extras}=={version(name)}"
        except PackageNotFoundError:
            out.append(line)  # not installed here -> leave as-is
            continue
        out.append(f"{pinned}  #{comment}" if sep else pinned)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"  pinned {path}")


def main() -> None:
    files = sys.argv[1:] or ["requirements.txt"]
    print("Refreshing pins to the current environment:")
    for path in files:
        repin(path)
    py = f"{sys.version_info.major}.{sys.version_info.minor}\n"
    with open(".python-version", "w", encoding="utf-8", newline="\n") as f:
        f.write(py)
    print(f"  wrote .python-version -> {py.strip()}")


if __name__ == "__main__":
    main()
