"""
fix_csv.py — Repair combined_reduction_degree_dataset.csv
Ensures every row has exactly 18 fields. Backs up the original.
"""
import csv
import shutil
import os

IN  = "combined_reduction_degree_dataset.csv"
OUT = "combined_reduction_degree_dataset_fixed.csv"
BAK = "combined_reduction_degree_dataset.csv.bak"
N   = 18  # expected number of columns

# ---- 1. Backup ----------------------------------------------------------
if not os.path.exists(BAK):
    shutil.copy(IN, BAK)
    print(f"Backup written: {BAK}")
else:
    print(f"Backup already exists: {BAK}")

# ---- 2. Read with Python csv (handles quotes correctly) ----------------
with open(IN, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))

header = rows[0]
if len(header) != N:
    raise SystemExit(f"Header has {len(header)} fields, expected {N}:\n{header}")

print(f"Header OK ({N} columns)")

# ---- 3. Repair each row -------------------------------------------------
def repair(row):
    """Return a row with exactly N fields."""
    if len(row) == N:
        return row, "ok"
    if len(row) > N:
        r = list(row)
        # Remove empties between positions 6 and 12 (between time_min and cpr)
        # until length is back to N.
        while len(r) > N:
            removed = False
            for j in range(6, min(13, len(r))):
                if r[j] == "":
                    r.pop(j)
                    removed = True
                    break
            if not removed:
                # Nothing empty found; drop trailing extras
                r = r[:N]
        return r, "trim"
    # len(row) < N → pad with empties at the end
    return list(row) + [""] * (N - len(row)), "pad"

fixed = [header]
stats = {"ok": 0, "trim": 0, "pad": 0}
problems = []

for i, row in enumerate(rows[1:], start=2):
    if not row or all(c == "" for c in row):
        continue  # skip fully-empty trailing lines
    new_row, status = repair(row)
    fixed.append(new_row)
    stats[status] += 1
    if status != "ok":
        problems.append((i, len(row), status))

# ---- 4. Write fixed file ------------------------------------------------
with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(fixed)

print(f"\nRows total   : {len(fixed)-1}")
print(f"  OK         : {stats['ok']}")
print(f"  Trimmed    : {stats['trim']}")
print(f"  Padded     : {stats['pad']}")

if problems:
    print(f"\nFirst 20 repaired lines:")
    for lineno, n, how in problems[:20]:
        print(f"  line {lineno:>4}: had {n} fields → {how}")

# ---- 5. Sanity check with pandas ---------------------------------------
try:
    import pandas as pd
    df = pd.read_csv(OUT)
    print(f"\nPandas verification: {len(df)} rows, "
          f"{df['meta_source'].nunique()} papers, {df.shape[1]} columns")
    print("\nRows per paper (tail):")
    print(df.groupby('meta_source').size().tail(12).to_string())
except ImportError:
    print("\n(pandas not installed; skipping verification)")