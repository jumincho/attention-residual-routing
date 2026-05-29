#!/usr/bin/env python3
"""Resolve every top-level import in the source tree, without executing modules.

Parses each ``.py`` file under the source roots, collects absolute (non-relative)
top-level imports, and checks that each one resolves to an importable module —
the repo's own modules via the source roots placed on ``sys.path``, third-party
packages via the installed environment. This catches a class of breakage that a
syntax-only ``compileall`` misses (e.g. importing a package that was never
declared as a dependency).

Packages that are intentionally *not* vendored in this archive can be tolerated
by passing them as arguments, in addition to a built-in set of common
externals. Usage:

    python tools/check_imports.py [extra_optional_module ...]
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

# External packages intentionally not installable from this archive alone.
OPTIONAL = {
    "MAR", "rlm", "vllm", "flash_attn", "deepspeed", "triton", "apex",
    "bitsandbytes",
} | set(sys.argv[1:])

ROOTS = [r for r in ("src", "code", "scripts", "tests") if pathlib.Path(r).is_dir()]
sys.path[:0] = ["."] + ROOTS

seen: dict[str, str] = {}
for root in ROOTS:
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"::error file={path}::syntax error: {exc}")
            sys.exit(2)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    seen.setdefault(alias.name.split(".")[0], str(path))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                seen.setdefault(node.module.split(".")[0], str(path))


def resolvable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


missing = [
    (mod, where)
    for mod, where in sorted(seen.items())
    if mod not in OPTIONAL and not resolvable(mod)
]
for mod, where in missing:
    print(f"::error file={where}::unresolved top-level import '{mod}'")
print(
    f"checked {len(seen)} top-level imports across {len(ROOTS)} source root(s); "
    f"{len(missing)} unresolved"
)
sys.exit(1 if missing else 0)
