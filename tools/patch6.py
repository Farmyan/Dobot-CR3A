from pathlib import Path
p = Path("config.py")
s = p.read_text(encoding="utf-8")
a = "0.005, 0.45)"
print("ok" if s.count(a) == 1 else "PROBLEM", "config.py tcp_box z max")
p.write_text(s.replace(a, "0.005, 0.55)"), encoding="utf-8")
