# NatScore — Benchmark

> Headline metric: pairwise accuracy on SpeechJudge-Eval. Target >70%
> (beats half the field), stretch >73% (beats SpeechJudge-BTRM, the
> closest comparable). Aspirational >77% (matches SpeechJudge-GRM with
> ~1/14000th of its trainable parameters).

## Runs

| Run | Checkpoint | n_pairs | Pairwise acc | 95% CI | Mean margin | ECE |
|---|---|---|---|---|---|---|
| natscore-small-v0 / dev[:100] | `final.pt` | 100 | 52.00% | [42.98, 62.00] | +0.132 | 31.30% |

## Per-language breakdown (natscore-small-v0 / dev[:100])

| Language setting | n_pairs | Accuracy |
|---|---|---|
| `zh2en` | 22 | 68.18% |
| `en2en` | 23 | 56.52% |
| `en2zh` | 25 | 48.00% |
| `zh2zh` | 30 | 40.00% |

## How to read these numbers

This is a **pipeline-validation baseline**, not a publication-quality
result. The training set was **500 pairs cached on CPU** (the
laptop-feasible slice per [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) §6.1.1),
not the full ~42K train split. Specifically:

- Pairwise accuracy on dev[:100] is **52%** with a 95% CI that includes
  50% (chance). At this training scale the model has not learned a
  statistically meaningful naturalness signal.
- Training accuracy reached 100% by step ~170 (overfit on 500 pairs is
  expected with ~400K trainable params).
- The per-language breakdown is more interesting than the overall
  number: 68% on zh2en vs 40% on zh2zh suggests the encoder is doing
  most of the work and the head's added value is small at this data
  scale.
- ECE = 31% reflects that overfit + small data ⇒ the model is
  confidently wrong on many held-out pairs. Anchor-regression
  calibration (`PROJECT_PLAN.md` §3.2) becomes a real option later.

**What unlocks the headline number**: extracting features on Kaggle T4
and training on the full 42K split. Online encoder extraction (~10
min/epoch on T4) is the realistic path; full-cache pre-extraction is
infeasible (5.34 TB). See [`PROJECT_PLAN.md`](../PROJECT_PLAN.md)
§6.1.1 for the corrected strategy.

## To-be-added once Kaggle T4 runs land

- `natscore-small-v0 / dev (full ~6K pairs)` once trained on the full
  42K train split
- `natscore-small-v0 / test (~50K pairs)` — the headline number
- UTMOSv2 baseline on the same audio
- WhiSQA baseline on the same audio
- SpeechJudge-BTRM (72.7%) and SpeechJudge-GRM (77.2%) reference points
- Ablations from [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) §10.1:
  high-consensus vs full, regular ↔ expressive, encoder choice,
  layer-wise probe, Gemini-CoT distillation
