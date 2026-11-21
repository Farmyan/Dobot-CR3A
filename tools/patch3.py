from pathlib import Path
def patch(path, pairs):
    p = Path(path); s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        print(("ok      " if a in s else "MISSING ") + path + ": " + a[:50])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")

patch("simulation/scene.xml", [
    ('diffuse="0.9 0.9 0.9"/>', 'diffuse="0.9 0.9 0.9" specular="0 0 0"/>'),
    ('diffuse="0.3 0.3 0.3"/>', 'diffuse="0.3 0.3 0.3" specular="0 0 0"/>'),
])
patch("perception/object_detection.py", [
    ("(0, 120, 70), (10, 255, 255)", "(0, 80, 50), (10, 255, 255)"),
    ("(170, 120, 70), (180, 255, 255)", "(170, 80, 50), (180, 255, 255)"),
    ("        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))",
     "        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))\n"
     "        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))"),
    ("cv2.minAreaRect(c)", "cv2.minAreaRect(cv2.convexHull(c))"),
])
