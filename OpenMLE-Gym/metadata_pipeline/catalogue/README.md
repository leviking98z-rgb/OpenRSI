# OpenMLE task catalogue

Complete metadata for all 5,567 usable OpenMLE MA1 tasks, resolved offline.

The point of this directory: **task selection, scoring rules and provenance no
longer require a Kaggle round-trip.** Everything needed to decide *which* tasks
to run, *how* they are scored and *where* the data comes from is committed here.
The data itself is not — see "What this does not contain".

## Files

| file | rows | what it is |
|---|---|---|
| `task_catalogue.jsonl` | 5,567 | the catalogue; one record per task |
| `catalogue_report.json` | — | aggregate summary of the above |
| `recipe_meta.jsonl` | 4,343 | raw per-task metadata pulled from each recipe on HF |
| `recipe_meta_retry.jsonl` | — | second-pass fetches for records that failed the first time |
| `missing_built_meta.jsonl` | — | metadata for built_task entries absent from the recipe listing |
| `comp_sizes.jsonl` | 2,251 | measured competition data sizes via `competitions/data/list` |
| `build_catalogue.py` | — | rebuilds `task_catalogue.jsonl` from the sources above |
| `probe_sizes.py` | — | measures competition sizes (metadata endpoint, not the download quota) |

## Coverage

| field | coverage |
|---|---|
| `task_id`, `higher_is_better`, `acquisition`, `source_urls` | 5567/5567 (100%) |
| `theoretical_min` | 99.7% |
| `task`, `modality` | 99.5% |
| `size_gb` | 96.6% (190 unknown: 184 dataset, 6 competition) |
| `license` | 66.2% |
| `leaderboard_min` / `leaderboard_max` | 40.3% |

`higher_is_better` is at 100% deliberately: a task whose scoring direction is
unknown cannot be used, because an inverted direction silently trains against
the reward.

The 190 missing sizes would each cost a Kaggle range request against the
download quota. They only affect download scheduling (defer-large decisions),
not whether a task is usable, so they were left unmeasured.

## Record shape

```json
{
  "task_id": "spaceship-titanic",
  "source_list": "kaggle_competitions",
  "release_type": "recipe",
  "acquisition": {"kind": "kaggle_competition",
                  "endpoint": "api/v1/competitions/data/download-all/spaceship-titanic",
                  "needs_rules": true, "ready": true},
  "higher_is_better": true,
  "theoretical_min": 0.0, "theoretical_max": 1.0,
  "task": "classification", "modality": "tabular",
  "size_gb": 0.0012, "size_source": "measured",
  "state": "materialized", "eval_leak": false
}
```

`task` and `modality` are kept in both raw and cleaned form. The upstream fields
are LLM-annotated and dirty — values like `'Classification\n\n Let'`,
`'[TaskType]'` and `'XXXXXX'` appear verbatim, and one modality field contains an
entire reasoning transcript — so `*_raw` preserves the original and the cleaned
field holds a normalized value. `*_annotation_dirty` flags which were rewritten.

`cpu_gpu_official` is the upstream label; `cpu_gpu_reclassified` is a re-judgement
from the actual file list and prepare/metric code. They disagree often: 1,678
tasks labelled GPU are plain CPU tabular problems whose label was a default
applied alongside `modality=unknown`.

## What this does not contain

Task **data**. 4,174 of the 5,567 still need a Kaggle download (1,892 dataset +
2,282 competition); the other 1,385 are `built_task` packages fetched from
HuggingFace, which touch no Kaggle quota. `acquisition.endpoint` records where
each one comes from.

## Regenerating

```bash
python3 build_catalogue.py --out task_catalogue.jsonl --report catalogue_report.json
```

Reads the sources in this directory plus the official `task_index.jsonl` /
`package_manifest.jsonl` release listing. Offline apart from those inputs.

`probe_sizes.py` needs `KAGGLE_USERNAME` / `KAGGLE_KEY` in the environment and
hits a metadata endpoint that is separate from the download quota. Keep it
single-threaded: the download quota was exhausted earlier by parallel probing,
and the metadata endpoint's own limits are unpublished.
