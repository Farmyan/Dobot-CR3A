from pathlib import Path

path = Path("main.py")
source = path.read_text(encoding="utf-8")

old = '''        log.append(dict(episode=ep, n_detected=len(dets), picks_attempted=len(results), placed=placed,
                        seconds=time.time() - t0))
        print(f"ep {ep}: det={len(dets)} placed={placed}/{len(results)}")
'''

new = '''        from collections import Counter

        print("\\\\n========== PICK RESULTS ==========", flush=True)
        for i, res in enumerate(results, 1):
            status = "SUCCESS" if res.success else "FAILED"
            pos = np.asarray(res.position).round(4).tolist()
            print(
                f"Cube {i}: {status} | color={res.color} | "
                f"stage={res.stage} | position={pos} | detail={res.detail}",
                flush=True
            )

        failures = Counter(res.stage for res in results if not res.success)
        print(f"Failure stages: {dict(failures)}", flush=True)

        log.append(dict(episode=ep, n_detected=len(dets), picks_attempted=len(results), placed=placed,
                        failures=dict(failures), seconds=time.time() - t0))
        print(f"ep {ep}: det={len(dets)} placed={placed}/{len(results)}", flush=True)
'''

if old not in source:
    raise SystemExit("ERROR: Expected evaluation block was not found. No changes made.")

backup = Path("main.py.backup_before_failure_logging")
if not backup.exists():
    backup.write_text(source, encoding="utf-8")

path.write_text(source.replace(old, new), encoding="utf-8")
print("Updated main.py. Backup:", backup)
