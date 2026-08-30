# Private Hugging Face Checkpoint Bridge

This is the minimal cross-cluster checkpoint path for the MA1 recursive-SFT
experiment. It avoids relaying tens of gigabytes through clusterbridge:

```text
producer local NVMe -> private Hugging Face repo -> consumer local NVMe
```

Use **one private model repository per HF-format checkpoint**. Do not mix G1,
G2, optimizer state, or unrelated runs in one repository. The temporary
`Levi98/openrsi-checkpoint-bridge` repository is only the bridge smoke target.

## Validated smoke

On 2026-08-31 Asia/Beijing:

```text
source:       8x L20D node 172.16.0.115
destination:  8x H20 node 28.49.25.76
payload:      8 MiB
repository:   Levi98/openrsi-checkpoint-bridge (private)
SHA-256:      identical after download
result:       PASS
```

Observed small-file transfer timing is not a 67 GB bandwidth estimate because
connection setup, commit, and Xet finalization dominate an 8 MiB object.

## One-time node setup

The token must remain outside the repository in a mode-`0600` file.

The L20 node uses:

```text
Python:     /data2/openrsi/tools/hf-bridge-venv/bin/python
Token:      /data2/openrsi/.secrets/hf_token
Checkpoint: /data2/openrsi/experiments/.../<checkpoint>-hf
```

The H20 nodes use their existing Python environment (`huggingface_hub` is
already installed) and:

```text
Token:      /root/.cache/openrsi/.secrets/hf_token
Checkpoint: /root/.cache/openrsi/models/<checkpoint>
```

Never print a token, add it to Git, or pass it as a literal command-line
argument. Use `HF_TOKEN_FILE`, `HF_TOKEN`, or `--token-file`.

## Upload a G1 checkpoint

Choose a dedicated private repository name and rerun the same command if an
upload is interrupted:

```bash
SCRIPT=/path/to/hf_checkpoint_bridge.py
PY=/data2/openrsi/tools/hf-bridge-venv/bin/python
export HF_TOKEN_FILE=/data2/openrsi/.secrets/hf_token

"$PY" "$SCRIPT" auth
"$PY" "$SCRIPT" upload \
  /data2/openrsi/experiments/<run>/checkpoints/<g1>-hf \
  --repo-id Levi98/openrsi-ma1-g1-<run-id>
```

With `hf_xet` installed, `upload_folder` streams and chunks large files.
Rerunning the same command skips committed files and reuses uploaded chunks.
The script uploads `OPENRSI_HF_BRIDGE_MANIFEST.json` only after all checkpoint
files finish.

Upload only the converted HF inference checkpoint for rollout/eval unless the
full optimizer/trainer state is explicitly required. This keeps a normal MA1
transfer near 67 GB rather than moving the roughly 498 GB training-state
checkpoint.

## Download and verify on another cluster

```bash
SCRIPT=/path/to/hf_checkpoint_bridge.py
export HF_TOKEN_FILE=/root/.cache/openrsi/.secrets/hf_token

python3 "$SCRIPT" download \
  /root/.cache/openrsi/models/openrsi-ma1-g1-<run-id> \
  --repo-id Levi98/openrsi-ma1-g1-<run-id> \
  --download-workers 8
```

The command downloads the snapshot and verifies every file against the
uploaded size and SHA-256 manifest. A failed or interrupted download is also
resumed by rerunning the same command.

To verify an already downloaded checkpoint without contacting the Hub:

```bash
python3 "$SCRIPT" verify \
  /root/.cache/openrsi/models/openrsi-ma1-g1-<run-id>
```

## Operational boundary

- Public G0 (`FrontisAI/Frontis-MA1-35B`) should be downloaded directly from
  its pinned public revision; do not mirror it through this bridge.
- G1/G2 produced locally should use dedicated private repositories.
- Record the repository ID, immutable commit revision, manifest hash, and local
  path in the experiment README before starting SGLang.
- Pin the returned commit revision for evaluation; do not evaluate from a
  moving `main` branch.
