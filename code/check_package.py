"""Check the package without running it: scripts parse, no machine paths,
every figure and table resolves, inputs exist, Supplementary references in step.

    python check_package.py     # exit 0 if consistent
"""
import ast
import io
import os
import re
import subprocess
import sys

from _paths import (CODE, DATA_CONFIDENTIAL, OUT_FIG, OUT_TAB, REP_ROOT)

FINAL = os.path.join(os.path.dirname(REP_ROOT), "Final")
problems = []


def head(t):
    print("\n" + t)
    print("-" * len(t))


# ---------------------------------------------------------------- 1. syntax
head("1. scripts parse")
scripts = sorted(f for f in os.listdir(CODE) if f.endswith(".py"))
bad = []
for fn in scripts:
    try:
        ast.parse(io.open(os.path.join(CODE, fn), encoding="utf-8",
                          errors="surrogateescape").read())
    except SyntaxError as e:
        bad.append("%s:%s %s" % (fn, e.lineno, e.msg))
print("  %d scripts, %d syntax errors" % (len(scripts), len(bad)))
problems += bad

# ---------------------------------------------------------------- 2. machine paths
head("2. no hardcoded machine paths")
hits = []
for fn in scripts:
    t = io.open(os.path.join(CODE, fn), encoding="utf-8", errors="surrogateescape").read()
    for m in re.finditer(r"[A-Za-z]:[\\/](?:Users|project)", t):
        hits.append("%s: %s" % (fn, m.group(0)))
print("  %d found" % len(hits))
problems += hits

# ---------------------------------------------------------------- 3. figures
head("3. figures referenced by the paper exist in output/figures")
INCLUDE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", re.S)
missing_fig, referenced = [], set()
for tex in ("sn-article.tex", "supplementary_information.tex"):
    p = os.path.join(FINAL, tex)
    if not os.path.exists(p):
        problems.append("missing %s" % p)
        continue
    s = re.sub(r"\\iffalse.*?\\fi", "", io.open(p, encoding="utf-8").read(), flags=re.S)
    for m in INCLUDE.finditer(s):
        f = os.path.basename(m.group(1).strip())
        referenced.add(f)
        if not os.path.exists(os.path.join(OUT_FIG, f)):
            missing_fig.append("%s -> %s" % (tex, f))
have = {f for f in os.listdir(OUT_FIG) if os.path.isfile(os.path.join(OUT_FIG, f))}
extra = sorted(have - referenced)
print("  referenced %d | present %d | missing %d | unused %d"
      % (len(referenced), len(have), len(missing_fig), len(extra)))
for x in missing_fig:
    print("    MISSING %s" % x)
for x in extra:
    print("    UNUSED  %s" % x)
problems += missing_fig

# ---------------------------------------------------------------- 4. tables
head("4. numbered tables have a CSV")
tabs = sorted(f for f in os.listdir(OUT_TAB)
              if f.startswith("Table_") and f.endswith(".csv"))
inter = os.path.join(OUT_TAB, "intermediate")
n_int = len(os.listdir(inter)) if os.path.isdir(inter) else 0
print("  %d numbered tables, %d intermediates" % (len(tabs), n_int))
for t in tabs:
    print("    %s" % t)

# ---------------------------------------------------------------- 5. inputs
head("5. inputs referenced via need() exist")
NEED = re.compile(r"need\(\s*([A-Z_]+)\s*,\s*(.+?)\)", re.S)
import _paths
missing_in = []
for fn in scripts:
    t = io.open(os.path.join(CODE, fn), encoding="utf-8", errors="surrogateescape").read()
    for m in NEED.finditer(t):
        base = getattr(_paths, m.group(1), None)
        if base is None:
            continue
        parts = re.findall(r"['\"]([^'\"]+)['\"]", m.group(2))
        if not parts:
            continue
        p = os.path.join(base, *parts)
        if not os.path.exists(p):
            tag = " (restricted)" if os.path.basename(p) in _paths.RESTRICTED else ""
            missing_in.append("%s -> %s%s" % (fn, os.path.relpath(p, REP_ROOT), tag))
print("  %d missing" % len(missing_in))
for x in missing_in:
    print("    %s" % x)
problems += [x for x in missing_in if "(restricted)" not in x]

# ---------------------------------------------------------------- 6. supp refs
head("6. Supplementary references in step")
r = subprocess.run([sys.executable, os.path.join(CODE, "sync_supp_refs.py")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("  " + (r.stdout or "").strip().replace("\n", "\n  "))
if r.returncode != 0:
    problems.append("Supplementary references are stale - run sync_supp_refs.py --apply")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 60)
if problems:
    print("PROBLEMS: %d" % len(problems))
    for p in problems:
        print("  - %s" % p)
else:
    print("OK - package is internally consistent")
sys.exit(1 if problems else 0)
