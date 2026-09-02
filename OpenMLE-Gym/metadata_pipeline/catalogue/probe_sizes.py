#!/usr/bin/env python3
"""Measure real data size per competition via the metadata endpoint.

Needed because the catalogue's size field is missing for 2429 tasks and, where
present, is known to under-report by orders of magnitude (one dataset declared a
few MB and was actually 22 GB). Deciding which tasks to defer instead of
materializing requires real numbers.

Uses `competitions/data/list`, which returns per-file totalBytes and is a
METADATA endpoint — it does not consume the competition download quota that is
currently rate-limited. Kept single-threaded with a delay anyway: the download
quota was exhausted earlier by parallel probing, and metadata limits are
unpublished.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Files that are data, versus scaffolding a competition ships alongside it.
CODE_EXT = (".py", ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".gitignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    auth = base64.b64encode(
        f"{os.environ['KAGGLE_USERNAME']}:{os.environ['KAGGLE_KEY']}".encode()).decode()
    slugs = json.load(open(args.slugs))
    out = Path(args.out)

    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("code") != 429:
                done.add(r["slug"])
    todo = [s for s in slugs if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(slugs)} slugs, {len(done)} known, {len(todo)} to probe", flush=True)

    import collections
    tally = collections.Counter()
    t0 = time.time()
    with out.open("a") as f:
        for i, slug in enumerate(todo, 1):
            time.sleep(args.delay)
            url = f"https://www.kaggle.com/api/v1/competitions/data/list/{slug}"
            req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
            rec = {"slug": slug}
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.loads(r.read().decode("utf-8", "replace"))
                files = d.get("files") or []
                total = sum(int(x.get("totalBytes") or 0) for x in files)
                data_files = [x for x in files
                              if not str(x.get("name", "")).lower().endswith(CODE_EXT)]
                rec.update(code=200, n_files=len(files),
                           bytes=total,
                           data_bytes=sum(int(x.get("totalBytes") or 0) for x in data_files),
                           names=[x.get("name") for x in files][:10])
                tally["ok"] += 1
            except urllib.error.HTTPError as e:
                rec.update(code=e.code)
                tally[str(e.code)] += 1
                if e.code == 429:
                    # metadata endpoint is rate limiting too — back off hard
                    retry = e.headers.get("retry-after")
                    rec["retry_after"] = retry
                    print(f"  429 at {i}/{len(todo)}, retry-after={retry}; stopping",
                          flush=True)
                    f.write(json.dumps(rec) + "\n")
                    break
            except Exception as e:
                rec.update(code=f"ERR:{type(e).__name__}")
                tally["err"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                el = time.time() - t0
                gb = sum(0 for _ in ())  # placeholder, summarized at end
                print(f"  {i}/{len(todo)} {el/60:.1f}min ({i/max(el,1)*60:.0f}/min) "
                      f"{dict(tally)}", flush=True)
    print(f"done: {dict(tally)}")


if __name__ == "__main__":
    main()
