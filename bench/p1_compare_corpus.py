"""Compare the pre-P1 and HEAD-error-mode corpora: results, hashes, and MSL text.

The review's bar: any MSL diff in error mode falsifies "identical to pre-P1".
Strip only hash/path lines, per the review.
"""
import json
import pathlib
import re
import sys

a = json.load(open("bench/corpus_prep1.json"))
b = json.load(open("bench/corpus_error_head.json"))

print("=== 1. launch outcomes / values / exception messages ===")
keys = sorted(set(a["results"]) | set(b["results"]))
diffs = []
for k in keys:
    ra, rb = a["results"].get(k), b["results"].get(k)
    same = ra == rb
    if not same:
        diffs.append(k)
    print(f"  {k:30s} {'MATCH' if same else 'DIFF'}")
print(f"  -> {len(diffs)} differences" + (f": {diffs}" if diffs else ""))

print("\n=== 2. module hashes (must be IDENTICAL in error mode) ===")
hdiffs = []
for m in sorted(set(a["hashes"]) | set(b["hashes"])):
    ha, hb = a["hashes"].get(m), b["hashes"].get(m)
    same = ha == hb
    if not same:
        hdiffs.append(m)
    print(f"  {m:16s} {ha} vs {hb}  {'MATCH' if same else 'DIFF'}")
print(f"  -> {len(hdiffs)} hash differences" + (f": {hdiffs}" if hdiffs else ""))

print("\n=== 3. generated MSL text diff ===")
da = pathlib.Path("/tmp/msl_prep1/1.17.0")
db = pathlib.Path("/tmp/msl_error_head/1.17.0")


def norm(p):
    t = p.read_text(errors="replace")
    # strip only hash/path-bearing lines, per the review
    out = []
    for line in t.splitlines():
        if re.search(r"(/private/tmp/|/tmp/|msl_prep1|msl_error_head)", line):
            continue
        if re.match(r"^\s*//.*\b[0-9a-f]{7,}\b", line):
            continue
        out.append(line)
    return "\n".join(out)


fa = {p.name: p for p in da.rglob("*.metal")}
fb = {p.name: p for p in db.rglob("*.metal")}
only_a = sorted(set(fa) - set(fb))
only_b = sorted(set(fb) - set(fa))
common = sorted(set(fa) & set(fb))
print(f"  MSL files: pre-P1={len(fa)} head={len(fb)} common={len(common)}")
if only_a:
    print(f"  only in pre-P1: {only_a}")
if only_b:
    print(f"  only in HEAD:   {only_b}")
tdiffs = []
for name in common:
    if norm(fa[name]) != norm(fb[name]):
        tdiffs.append(name)
print(f"  text differences among common files: {len(tdiffs)}")
for n in tdiffs[:5]:
    print(f"    {n}")
    import difflib
    d = list(difflib.unified_diff(norm(fa[n]).splitlines(), norm(fb[n]).splitlines(), lineterm="", n=1))
    for line in d[:14]:
        print("      " + line)

verdict = not diffs and not hdiffs and not tdiffs and not only_a and not only_b
print("\nVERDICT:", "error mode is IDENTICAL to pre-P1 (results, hashes, MSL text)" if verdict
      else "NOT identical — the claim is falsified; see differences above")
sys.exit(0 if verdict else 1)
