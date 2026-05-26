# NatScore

> **Status: pre-alpha.** Architecture, training pipeline, and evaluation
> suite all working end-to-end. Headline number pending the first
> Kaggle T4 run on the full SpeechJudge-Data train split. Not yet on
> PyPI or HuggingFace Hub. See [`STATUS.md`](STATUS.md) for the latest
> snapshot and [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for full design.

A small, preference-supervised **naturalness scorer for modern neural TTS**.

NatScore wraps a frozen Whisper-small encoder with a tiny Bradley–Terry head
trained on [SpeechJudge-Data](https://huggingface.co/datasets/RMSnow/SpeechJudge-Data)
(99K human preference pairs over CosyVoice2, F5-TTS, MaskGCT, Llasa, and others,
released Nov 2025).

```python
import natscore as ns

scorer = ns.load("natscore-small-v1")          # ~150 MB, downloads on first call

score = scorer.score("path/to/tts_output.wav")  # higher = more natural
result = scorer.compare("a.wav", "b.wav")       # pairwise comparison with confidence
```

## Why this exists

| Existing scorer | Documented weakness |
|---|---|
| **UTMOSv2** | Saturates at high quality, **negatively correlates** on conversational/expressive speech (arXiv 2603.01467) |
| **WhiSQA** | Trained on NISQA telecom/enhancement quality, not synthetic-TTS naturalness |
| **SpeechJudge-GRM** | Excellent, but 7B params and GPU-only |

NatScore targets the gap: **<500K trainable params, CPU-deployable, trained on
human preferences over modern neural-TTS output.**

## Licensing

- **Code:** Apache-2.0 (this repo)
- **Model weights:** CC-BY-NC-4.0 (inherited from SpeechJudge-Data — see
  [`MODEL_LICENSE.md`](MODEL_LICENSE.md))

Research and non-commercial use only for the weights. The code supports
retraining on a permissively-licensed dataset if commercial use is required.

## Roadmap

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §8 for the milestone breakdown:

- [x] **M0** — Repo scaffold, package skeleton, CI ([`e578473`](../../commit/e578473))
- [x] **M1** — SpeechJudge-Data inspection + schema dump ([`d084774`](../../commit/d084774))
- [x] **M2** — Frozen Whisper feature extraction + cache ([`d261cd5`](../../commit/d261cd5))
- [x] **M3** — BT head + training loop (sanity-validated) ([`d0cceee`](../../commit/d0cceee))
- [x] **M4** — Evaluation suite with bootstrap CI + ECE + breakdown ([`34ad250`](../../commit/34ad250))
- [x] **M5a** — Kaggle T4 online-training notebook ready ([`d9fa459`](../../commit/d9fa459))
- [ ] **M5b** — Headline run on full 42K train (kicked off on Kaggle) → pairwise acc > 70%
- [ ] **M5c** — Ablation grid (high-consensus, regular↔expressive, layer-wise probe, magnitude-weighted BT, Gemini-CoT distillation)
- [ ] **M6** — Packaging + PyPI + HF Hub release
- [ ] **M7** — Workshop paper draft + HF Spaces demo

## Current benchmark numbers

| Run | n_pairs | Pairwise acc | 95% CI | ECE | Notes |
|---|---|---|---|---|---|
| `natscore-small-v0` / dev[:100] | 100 | **52.00%** | [42.98, 62.00] | 31.30% | laptop-CPU pipeline validation only; 500 training pairs |

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full breakdown and
per-language slices. The 52% is a pipeline-validation baseline — it
includes 50% (chance) in the CI. The first real headline number lands
when the Kaggle T4 run completes on the full 42K train split.

## Running the headline training run on Kaggle (the next step)

See [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md) for the full
step-by-step. The five-minute version:

1. **One-time:** accept the SpeechJudge-Data terms at
   https://huggingface.co/datasets/RMSnow/SpeechJudge-Data
2. **One-time:** create a new Kaggle notebook, set **Accelerator =
   GPU T4 x1**, add three secrets (`HF_TOKEN`, `GITHUB_TOKEN`, optionally
   `WANDB_API_KEY`).
3. **Per run:** **File → Import Notebook** the file at
   `scripts/kaggle/train_natscore_t4.ipynb` and click **Run All**.
4. ~5.5 h later, download
   `/kaggle/working/outputs/natscore-small-v0-kaggle/final.pt` and the
   companion `eval_dev.json`.

The trainer auto-resumes from `latest.pt` if the Kaggle kernel hits
its 9 h timeout — re-run the same notebook and it picks up at the
last 500-step checkpoint.

## Development

```bash
git clone https://github.com/harrrshall/natscore.git
cd natscore
python -m venv .venv && source .venv/bin/activate
# CPU-only torch (smaller wheel, faster install)
pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.3,<2.7" torchaudio
pip install -e ".[dev,train]"
pytest -q          # 83 pass, 3 Whisper-gated skip
```

Resume protocol after a session interruption: read
[`STATUS.md`](STATUS.md) → check `git log --oneline | head` →
`pytest -q` → continue from `STATUS.md`'s **Next concrete action**.

## Citation

Pending model release.
