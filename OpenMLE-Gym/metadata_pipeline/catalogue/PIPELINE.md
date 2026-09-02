# Kaggle task pool expansion — state as of 2026-09-02

Expands the OpenMLE MA1 task pool from the 850 originally usable tasks by
treating `recipe` releases as what they are: complete reconstruction kits
(`prepare.py` + `metric.py` + metadata). recipe + upstream Kaggle data
reconstructs the same task a `built_task` would have shipped, which unlocks
4,343 recipes the earlier pool skipped.

## Where things stand

| | count |
|---|---|
| catalogued (metadata complete, Kaggle-independent) | **5,567** |
| **verified** — independently checked, ready to use | **1,169** |
| materialized but not yet verified | 819 |
| rejected by the verifier (real defects) | 57 |
| not yet materialized | 3,522 |
| on disk | 2,045 dirs (149 of them packed as `.tar`) |

Verification pass rate 95.4% (1169/1226). Zero residual environmental failures:
every OOM / timeout / disk-full record was re-run or purged, so a `rejected`
label means the task is genuinely defective, not that the tooling failed.

Committed to the fork as `data/kaggle-task-catalogue-20260902` →
`OpenMLE-Gym/metadata_pipeline/catalogue/`.

## Why the 57 rejections are rejections

| reason | n | meaning |
|---|---|---|
| sample_submission matches neither answer nor test | 16 | an agent cannot produce a scorable file |
| metric cannot separate random from correct answers | 15 | no gradient for the search (mostly 1-row test sets from course-setup competitions) |
| answer has fewer rows than test | 9 | some test items have no answer |
| train.csv empty | 5 | nothing to learn from |
| key mismatch | 4 | answer/submission keys do not line up with test |
| other | 8 | CSV unparseable, no prediction target |

## What is NOT done

- **3,522 tasks unmaterialized.** The blocker is the Kaggle download quota, not
  compute. The competition endpoint has been 429 for hours (`retry-after`
  21145s = ~6h); the dataset endpoint recovered and then filled the local disk.
- **819 materialized tasks unverified.** Just needs another verifier pass.
- **`continuous_worker.py` cannot consume this pool yet.** Its `prepare()` has no
  `acquisition` branch, so it still downloads + unzstds + checksums. Tasks
  materialized here are already prepared, and 149 are packed as `.tar`. The
  running collection is untouched and still on the 850 pool.
- **Key alignment is unverified above `BIG_ROWS` (2M).** The streamed path
  compares row counts and column names, not key sets. Known coverage gap,
  recorded rather than papered over.
- **The negative test covers recall, not precision.** It proves injected defects
  are still rejected; it does not prove valid tasks are not wrongly rejected.
  That direction was found by auditing the rejected pile by hand (see below).

## Things that cost real time — read before touching this

**cephfs creates small files at ~9.5 files/s and parallelism does not help.**
Measured 1 worker and 16 workers as identical; a peer independently measured
13.2 vs 12.7 files/s, i.e. 16× concurrency is slightly *slower* — the metadata
path serializes on the MDS and more threads only add contention. Sequential
writes on the same mount run at 264 MB/s. So a media task with 12,818 files took
57 minutes to publish 77% of itself. `publish()` now writes a single `.tar` above
500 files: the same task completes in 200s, ~60× faster. Do not "fix" this by
adding threads.

**podman on the H20 nodes is silently OOM-killed by the pod's memory cgroup.**
Exit code 137, no traceback, nothing on stderr — only `dmesg` shows it. Every
container invocation must pass `--memory/--memory-swap`.

**Scratch must be on node-local disk, and must be swept.** cephfs is too slow for
extraction churn, but `/data` is only 100G: leftover trees from failed tasks
filled it to 0 bytes and the next 756 tasks all died with ENOSPC — recorded as if
the *tasks* were broken. `materialize_recipe.py` now checks free space before each
task, sweeps leftovers, and stops cleanly rather than burning the inventory.

**Both Kaggle quotas are per-account and shared across every shard on every
node.** 24 shards exhausted both within minutes. A 429 now parks all shards on a
shared per-endpoint backoff and retries the task instead of consuming it; the
dataset and competition backoffs are independent so one does not stall the other.

**A transient failure must never be recorded as a task defect.** 429s, OOMs,
timeouts and ENOSPC all say "I could not check this", not "this is bad". Records
of that kind are purged and retried, and when merging verification rounds an
environmental failure is not allowed to overwrite an existing success.

**Diagnosing a hung shard.** `/proc/<pid>/stack` gives the kernel stack (a hung
shard showed `request_wait_answer → fuse_lookup`, i.e. blocked on cephfs
metadata) and `/proc/<pid>/syscall` + `/proc/<pid>/mem` recovers the path it is
blocked on. Which register holds the path depends on the syscall: `stat`(4) and
`lstat`(6) put it in arg0 = `parts[1]`, `openat`(257) and `newfstatat`(262) in
arg1 = `parts[2]`. Try each and skip the ones that raise `OSError`; require the
bytes to be printable ASCII, since an integer argument pointing at readable
memory yields a plausible-looking wrong path, which is worse than no answer.

**`pkill -f <pattern>` kills the shell running it** when the pattern matches its
own command line. Use a bracket (`mater[i]alize`) or `kill $(pgrep ...)`.

## The verifier rejected 42 valid tasks, twice, for two different reasons

Both were my bugs, both looked like task defects, and neither produced any
downstream failure — they just quietly shrank the pool.

1. **16 tasks: "Ground truth is missing required columns."** The streamed path
   loaded the answer file restricted to the *submission's* column names, but the
   answer's target is not always named the same — `beth-dataset` ships
   `sub=(row_id,score)` against `ans=(row_id,evil)`. It dropped the one column
   the metric reads. Fixed by reading all of the answer's columns (it is the
   small table; the big one is train, which the metric never sees).

2. **26 tasks: row-count equality.** Requiring test == answer == submission rows
   is simply the wrong invariant. Detection and segmentation come in two valid
   shapes: an *exploded* answer (one row per box — `annotated-potholes`: 299
   answer rows over 133 test rows) and a *packed* submission (one row per image
   with every box in a `PredictionString` — `drones@1`: test 158, sub 158,
   ans 198). The real invariant is coverage: `ans < test` is a defect; `ans > test`
   is fine; the submission must match either the answer or the test row count.

Both were found by reading the rejected pile, not by any alarm. This class of
error has no failure signature — the only way to catch it is to deliberately
review what you threw away, and the default habit is to review only what passed.

## Layout

    materialize_recipe.py       recipe/built_task -> worker-ready task dir (or .tar)
    verify_materialized.py      independent re-derivation from disk, in the canonical image
    test_streamed_rejects.py    negative test: injected defects must still be rejected
    build_catalogue.py          rebuilds task_catalogue.jsonl
    probe_sizes.py              measures competition sizes (metadata endpoint)
    node_materialize.sh         launches N shards on an H20 node
    accept_rules.py             bulk competition rules acceptance
    task_catalogue.jsonl        5567 tasks: identity, scoring contract, size, acquisition, state
    verify_final.jsonl          merged verdicts across all verification rounds
    verified_ok.json            the 1169 task_ids that passed
    hf_skip.json                15 tasks deliberately deferred (oversized / stalled downloads)

The canonical sandbox image is mandatory for `prepare.py`:
`openmle-sandbox-worker:1.0.0`. Data derived under a different pandas is not
comparable — that is what voided the first G0 baseline.

## Resuming

    # datasets (quota permitting)
    python3 materialize_recipe.py --inventory ds_resume.jsonl \
      --out /root/shared/.clusters/.tmp/kaggle-materialized \
      --scratch /data/kaggle-scratch/r0 --results ds_resume/shard0.jsonl \
      --skip hf_skip.json --runtime docker --shard 0/3

    # HuggingFace packages, no Kaggle quota at all
    INV=hf_only.jsonl TAG=hf_only N=4 bash node_materialize.sh

    # verify whatever is on disk but unverified
    python3 verify_materialized.py --root ../kaggle-materialized \
      --out verify_next.jsonl --runtime docker --workers 4 \
      --unpack-dir /data/verify-unpack

Only `ok:true` is terminal for resume, so a deliberately-deferred task must go in
`hf_skip.json` or it will be retried on every restart.
