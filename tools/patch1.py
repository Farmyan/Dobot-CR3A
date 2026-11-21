from pathlib import Path
p = Path("tools/prepare_assets.py")
s = p.read_text(encoding="utf-8")
s = s.replace("% mesh_dir)", "% mesh_dir.as_posix())")
s = s.replace("Path(urdf_path).read_text()", "Path(urdf_path).read_text(encoding='utf-8')")
s = s.replace("tmp.write_text(text)", "tmp.write_text(text, encoding='utf-8')")
p.write_text(s, encoding="utf-8")
print("as_posix edits:", s.count("as_posix()"))
