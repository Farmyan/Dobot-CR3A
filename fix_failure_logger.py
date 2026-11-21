from pathlib import Path

path = Path("main.py")
source = path.read_text(encoding="utf-8")

old = 'pos = np.asarray(res.position).round(4).tolist()'
new = 'pos = np.asarray(res.cube_estimate).round(4).tolist() if res.cube_estimate is not None else None'

if old not in source:
    raise SystemExit("Expected logger line not found. No changes made.")

path.write_text(source.replace(old, new), encoding="utf-8")
print("Logger corrected: using cube_estimate.")
