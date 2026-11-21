from pathlib import Path

p = Path("simulation/environment.py")
s = p.read_text(encoding="utf-8")

a = ("            if b1 in self.cube_bodies or b2 in self.cube_bodies:\n"
     "                if g1 or g2:\n"
     "                    continue                               # gripper may touch the cube\n")
b = ("            if b1 in self.cube_bodies or b2 in self.cube_bodies:\n"
     "                if g1 or g2 or b1 in self._held_cubes() or b2 in self._held_cubes():\n"
     "                    continue                               # gripper may touch the cube; a carried cube moves with the arm\n")
print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + "simulation/environment.py: cube rule")
s = s.replace(a, b)

a = "    def explain_collision(self, q):\n"
b = ('    def _held_cubes(self):\n'
     '        """Cube bodies currently touched by the gripper in the real simulation (i.e. being carried)."""\n'
     '        d, m = self.data, self.model\n'
     '        out = set()\n'
     '        for i in range(d.ncon):\n'
     '            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]\n'
     '            if b1 in self.cube_bodies and b2 in self.gripper_bodies:\n'
     '                out.add(b1)\n'
     '            elif b2 in self.cube_bodies and b1 in self.gripper_bodies:\n'
     '                out.add(b2)\n'
     '        return out\n\n')
print(("ok      " if s.count(a) == 1 else "PROBLEM(%d) " % s.count(a)) + "simulation/environment.py: _held_cubes method")
s = s.replace(a, b + a)
p.write_text(s, encoding="utf-8")
