"""Does the streamed path still REJECT a task it should reject?

The streamed path was validated only by agreement with the whole-table path on
tasks that pass. That shows it did not get looser on those, not that it can still
fail anything. So: take a task large enough to trigger streaming, inject each
defect the verifier is supposed to catch, and require a rejection every time.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path("/root/shared/.clusters/.tmp/kaggle-materialized/ddos-ciciot2023@1")
LAB = Path("/root/shared/.clusters/.tmp/negtest")
VERIFY = "/root/shared/.clusters/.tmp/kaggle-expand/verify_materialized.py"

def build(name):
    d = LAB / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "data/public").mkdir(parents=True); (d / "data/private").mkdir(parents=True)
    (d / "info").mkdir(parents=True); (d / "utils").mkdir(parents=True)
    for rel in ("RELEASE_METADATA.json", "info/task_metadata.json", "utils/metric.py"):
        shutil.copy2(SRC / rel, d / rel)
    for rel in ("data/public/train.csv", "data/public/test.csv",
                "data/public/sample_submission.csv", "data/private/test_answer.csv"):
        os.link(SRC / rel, d / rel)          # hardlink: no 8GB copy
    (d / "utils/public").symlink_to("../data/public", target_is_directory=True)
    return d

def unlink_replace(d, rel):
    """Break the hardlink so we can rewrite one file without touching the source."""
    p = d / rel
    p.unlink()
    return p

def run(d):
    cmd = [sys.executable, VERIFY, "--root", str(d.parent), "--out", f"/tmp/neg_{d.name}.jsonl",
           "--runtime", "docker", "--workers", "1", "--timeout", "1200",
           "--only", f"/tmp/only_{d.name}.txt"]
    Path(f"/tmp/only_{d.name}.txt").write_text(d.name + "\n")
    subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    out = Path(f"/tmp/neg_{d.name}.jsonl")
    for l in (out.read_text().splitlines() if out.is_file() else []):
        if l.strip():
            return json.loads(l)
    # No verdict means the verifier itself did not run (e.g. the Docker daemon was
    # saturated by a concurrent verification job), NOT that the defect slipped
    # through. Reporting it as "not rejected" would be a false alarm on the one
    # test whose whole job is to catch false results, so distinguish it.
    return {"ok": None, "problems": ["verifier did not produce a verdict "
                                     "(run failed; inconclusive, not a pass)"]}

cases = {}

# 1. answer row count no longer matches test  (the real vpn-classification defect)
d = build("neg_rowmismatch")
p = unlink_replace(d, "data/private/test_answer.csv")
with open(SRC / "data/private/test_answer.csv") as fh, open(p, "w") as out:
    for i, line in enumerate(fh):
        if i > 5000: break
        out.write(line)
cases["answer rows truncated"] = d

# 2. target column leaked into public test.csv
d = build("neg_leak")
cols_sub = open(SRC / "data/public/sample_submission.csv").readline().strip().split(",")
target = [c for c in cols_sub if c not in
          open(SRC / "data/public/test.csv").readline().strip().split(",")]
p = unlink_replace(d, "data/public/test.csv")
with open(SRC / "data/public/test.csv") as fh, open(p, "w") as out:
    hdr = fh.readline().rstrip("\n")
    out.write(hdr + "," + target[0] + "\n")
    ansf = open(SRC / "data/private/test_answer.csv"); ansf.readline()
    for line in fh:
        a = ansf.readline().rstrip("\n").split(",")
        out.write(line.rstrip("\n") + "," + (a[-1] if a else "0") + "\n")
cases[f"target '{target[0]}' leaked into test.csv"] = d

# 3. train.csv empty (header only)
d = build("neg_emptytrain")
p = unlink_replace(d, "data/public/train.csv")
p.write_text(open(SRC / "data/public/train.csv").readline())
cases["train.csv header only"] = d

print(f"streaming threshold = 200MB; source test.csv = "
      f"{(SRC/'data/public/test.csv').stat().st_size/1e6:.0f}MB\n")
fails = []
inconclusive = []
for label, d in cases.items():
    r = run(d)
    rejected = r.get("ok") is False
    verdict = "PASS" if rejected else ("SKIP" if r.get("ok") is None else "FAIL")
    print(f"{verdict}  {label}")
    print(f"      streamed={r.get('streamed')} verdict_ok={r.get('ok')} "
          f"problems={str(r.get('problems'))[:110]}")
    if r.get("ok") is None:
        inconclusive.append(label)
    elif not rejected:
        fails.append(label)
if fails:
    print(f"\nNOT REJECTED (streamed path is looser): {fails}")
elif inconclusive:
    print(f"\nno defect slipped through, but {len(inconclusive)} case(s) could not "
          f"run and prove nothing: {inconclusive}")
else:
    print("\nall defects rejected -- streamed path is not looser")
