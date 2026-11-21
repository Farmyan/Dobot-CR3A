from pathlib import Path

p = Path("simulation/environment.py")
s = p.read_text(encoding="utf-8")
a = "            if r1 and r2:\n"
print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + "simulation/environment.py: " + a.strip())
s = s.replace(a,
    "            if (b1 == self.base_body and m.body_parentid[b2] == b1) or (b2 == self.base_body and m.body_parentid[b1] == b2):\n"
    "                continue                                   # base and first link always touch\n"
    + a)
p.write_text(s, encoding="utf-8")
