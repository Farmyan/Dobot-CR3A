import json, collections
d = json.load(open("evaluation_results_100.json"))
steps = collections.Counter()
eps = collections.defaultdict(set)
for e in d["episodes"]:
    for k, v in e["unsafe_pairs"].items():
        steps[k] += v
        eps[k].add(e["episode"])
bad = [e for e in d["episodes"] if e["unsafe_contacts"]]
print("episodes with unsafe contacts:", len(bad), "of", len(d["episodes"]),
      " total steps:", sum(e["unsafe_contacts"] for e in bad))
for k, v in steps.most_common(15):
    print(f"{v:5d} steps in {len(eps[k]):3d} episodes: {k}   e.g. episodes {sorted(eps[k])[:6]}")
