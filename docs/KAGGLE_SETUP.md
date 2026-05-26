# Kaggle T4 Training Setup

Step-by-step for running `scripts/kaggle/train_natscore_t4.ipynb` on a free
Kaggle T4 GPU. Targets the >70% pairwise-accuracy headline number on the
full SpeechJudge-Data train split.

## Why Kaggle (not local)

The laptop CPU baseline (`outputs/natscore-small-v0`) was trained on 500
cached pairs and lands at **52% pairwise accuracy** on dev — chance-level.
The pipeline works; the data scale is the bottleneck. Per
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md) §6.1.1:

| Approach | Where | Disk | Time / epoch |
|---|---|---|---|
| Pre-extract full features | Anywhere | 5.34 TB ❌ | one-time |
| Online encoder forward | Kaggle T4 | ~5 GB streaming cache | ~1 h |

We use the second approach. Online means the Whisper encoder runs on GPU
during training; the head trains against fresh features each batch. Total
Kaggle disk footprint stays under 20 GB.

## One-time prerequisites

You need three credentials available as Kaggle Secrets:

| Secret | What it does | Where to get it |
|---|---|---|
| `HF_TOKEN` | Streams SpeechJudge-Data parquet shards | https://huggingface.co/settings/tokens (read scope) |
| `GITHUB_TOKEN` | Clones the private natscore repo | https://github.com/settings/tokens — Personal Access Token (classic) with `repo` scope |
| `WANDB_API_KEY` *(optional)* | Logs the run | https://wandb.ai/authorize |

You also need to have **accepted the SpeechJudge-Data terms** on the
dataset page: https://huggingface.co/datasets/RMSnow/SpeechJudge-Data
(one click). Without this the parquet downloads return 403.

## Run the notebook

1. Go to https://kaggle.com/code → **New Notebook**.
2. **File → Import Notebook** → upload `scripts/kaggle/train_natscore_t4.ipynb`
   from this repo.
3. Right sidebar → **Settings**:
   - **Accelerator**: `GPU T4 x1`
   - **Persistence**: `Files only` (default; lets the working dir survive
     between kernel sessions)
   - **Internet**: `On` (required to clone the repo and stream the dataset)
4. Right sidebar → **Add-ons** → **Secrets** → add the three secrets above.
5. **Run All**.

## Expected timeline + cost

| Stage | Time | Cost |
|---|---|---|
| Pip install + clone | ~2 min | — |
| Whisper-small download (one-time) | ~1 min | — |
| Training (5 epochs × ~2600 steps) | **~5 h** | — |
| Dev eval (~6K pairs) | ~10 min | — |
| **Total** | **~5.5 h** | **$0** (free T4) |

Kaggle's free T4 quota is **30 h/week per account**; one full run uses
~6 of those.

If the kernel hits the 9 h per-session limit, the trainer's `latest.pt`
checkpoint is saved every 500 steps. Re-run the same notebook and the
trainer will resume from `latest.pt` automatically.

## Pulling results back

After the kernel finishes, everything under `/kaggle/working/outputs/` is
preserved as notebook output. Download these specifically:

- `final.pt` — trained head (~5 MB; Apache-2.0 *code*, **CC-BY-NC weights**)
- `config.yaml` — exact training config (s6.2 reproducibility)
- `eval_dev.json` — pairwise accuracy + breakdown + ECE

Locally:

```bash
# Drop the downloaded final.pt into outputs/, then:
python scripts/03_evaluate.py \
    --checkpoint outputs/natscore-small-v0-kaggle/final.pt \
    --cache cache/whisper_small_dev \
    --label "natscore-small-v0-kaggle / dev[:100]"
```

This appends the row to [`docs/BENCHMARK.md`](BENCHMARK.md) without
re-running training.

## Troubleshooting

**"403 Forbidden" on the dataset stream**
You haven't accepted the SpeechJudge-Data terms on HF. Visit the dataset
page while signed in, click **Agree and access repository**, then re-run
the notebook.

**"`fatal: could not read Username for 'https://github.com'`"**
`GITHUB_TOKEN` secret is missing or the PAT was revoked. Generate a new
token at https://github.com/settings/tokens with `repo` scope.

**Out-of-memory during the encoder forward**
Drop `batch_size` in the config cell from 16 → 8. The head is tiny, so
this only changes wall-clock time.

**The notebook times out at 9 h**
Re-run the same notebook from the top. The trainer detects
`/kaggle/working/outputs/natscore-small-v0-kaggle/latest.pt` and resumes
from the most recent 500-step checkpoint.

**`No module named 'natscore'`**
The `sys.path.insert` line in Cell 3 fires after the clone in Cell 2.
Run cells in order; don't skip Cell 2.

**The W&B logger says "not installed"**
Add `wandb` to the pip install cell (already included in the template).
Set `WANDB_API_KEY` as a Kaggle secret; if not set, training proceeds
without W&B.

## Cross-project credential hygiene

Kaggle Secrets are scoped to your account, shared across notebooks. The
notebook reads `HF_TOKEN` / `GITHUB_TOKEN` / `WANDB_API_KEY` but never
writes them. NatScore-specific artifacts (W&B project, HF repos) all use
the `natscore` prefix per the project's namespace-isolation rule.
