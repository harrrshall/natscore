# NatScore

> **Status: pre-alpha (Milestone 0 scaffold).** Not yet installable from PyPI. No
> trained model exists yet. See `PROJECT_PLAN.md` for the full design and
> implementation milestones.

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

- [x] **M0** — Repo scaffold, package skeleton, CI
- [ ] **M1** — SpeechJudge-Data inspection + schema dump
- [ ] **M2** — Frozen Whisper feature extraction + disk cache
- [ ] **M3** — BT head training loop
- [ ] **M4** — Evaluation suite (SpeechJudge-Eval + VoiceMOS + NISQA + SOMOS)
- [ ] **M5** — Ablations (high-consensus, regular↔expressive, encoder ablations, layer-wise probe)
- [ ] **M6** — Packaging + PyPI + HF Hub release
- [ ] **M7** — Workshop paper draft + HF Spaces demo

## Citation

Pending model release.
