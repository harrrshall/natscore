# NatScore

> **A small, preference-supervised naturalness scorer for modern neural TTS.**
> Frozen Whisper-small encoder, ~400K trainable parameters, Bradley-Terry trained on 99K human preference pairs.

[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](STATUS.md)
[![License (code)](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![License (weights)](https://img.shields.io/badge/weights-CC--BY--NC--4.0-yellow)](MODEL_LICENSE.md)

**Status (2026-05-29):** Architecture, training pipeline, and evaluation suite all working end-to-end. Headline Kaggle T4 x2 run finished at **71.3% pairwise accuracy on dev[:1000]** (95% CI 68.6 to 74.1, ECE 2.27%), clearing the >70% target for M5b. Trained in ~4h52m across two days on T4 x2 + DataParallel with mid-run checkpoint resume. Checkpoint live on HF Hub: [harrrshall/natscore-small-v0](https://huggingface.co/harrrshall/natscore-small-v0). See [`STATUS.md`](STATUS.md) for the handoff snapshot.

---

## 1. The problem

Modern neural TTS (CosyVoice2, F5-TTS, MaskGCT, Llasa, XTTS-v2, etc.) generates speech that crosses the threshold where the dominant failure mode is no longer artifacts; it's **subtle unnaturalness**: prosody glitches, expressive overshoot, speaker-clone drift, breath placement, code-switching mismatches. The existing automatic naturalness scorers were not trained on this kind of failure surface.

Concretely:

- **UTMOSv2** (the VoiceMOS 2024 winner) was trained on read-speech MOS labels. It saturates at high quality and is documented to produce *negative correlations* with human judgment on conversational and expressive speech (arXiv 2603.01467).
- **WhiSQA** was designed for telecom and speech-enhancement quality (NISQA training data). It is intentionally not a synthetic-TTS scorer.
- **DNSMOS, NISQA-TTS, and the rest of the legacy stack** predate modern neural TTS and lack the distribution coverage.
- **SpeechJudge-GRM** (released Nov 2025) is excellent, but it is a 7B-parameter LALM. ~$0.001 per score on Modal A100. Unusable inside a TTS training loop or for large-scale offline evaluation.

The data that fixes the distribution gap, **SpeechJudge-Data**, was released in **November 2025**: 99K human-labeled TTS preference pairs across CosyVoice2, F5-TTS, MaskGCT, Llasa, and others, in en/zh + code-switching, with both regular and expressive splits. As of writing, no clean public artifact combines this data with a small, deployable, CPU-runnable scorer.

NatScore fills that gap.

---

## 2. Why this is the best fit for modern-TTS evaluation

A direct landscape comparison (verified May 2026):

| Tool | Backbone | Training data | Trainable params | Strengths | Known weakness for our task |
|---|---|---|---|---|---|
| **UTMOSv2** (Baba et al., SLT 2024) | wav2vec 2.0 + MFCC image-classifier ensemble | VoiceMOS 2022/2024 (read speech MOS) | ~300M | Won 7/16 VoiceMOS 2024 metrics | Saturates at high quality; *negative correlation* on conversational/expressive (arXiv 2603.01467); read-speech only |
| **WhiSQA** (Close et al., SPECOM 2025) | Frozen Whisper + weighted-sum head | NISQA + simulated distortions | ~5M | Beats DNSMOS on enhancement quality | Trained on enhancement, not TTS naturalness; intentional domain mismatch |
| **NISQA-TTS** (Mittag et al., 2021) | CNN-LSTM | Multilingual MOS | ~5M | Only widely-used multilingual MOS predictor | Predates modern neural TTS; lower accuracy |
| **SpeechJudge-BTRM** (Zhang et al., Nov 2025) | Qwen2.5-Omni-7B + BT head | SpeechJudge-Data | ~7B | 72.7% on SpeechJudge-Eval | Massive; sparse documentation; no clean release |
| **SpeechJudge-GRM** (Zhang et al., Nov 2025) | Qwen2.5-Omni-7B + SFT/RL with CoT | SpeechJudge-Data + rationales | ~7B | **77.2%** on SpeechJudge-Eval (gold standard) | 7B GPU-only; ~$0.001/score; hard to integrate |
| **Audiobox-Aesthetics** (Meta, 2025) | Custom small | Internal Meta data | <100M | 4-axis (PQ/PC/CE/CU); fast | Closed data; CC-BY-NC; 4 fixed axes; no TTS-naturalness target |
| **NatScore** (this work) | Frozen Whisper-small + BT head | SpeechJudge-Data | **~400K** | Drop-in pip API; CPU-deployable; targets the SpeechJudge naturalness distribution directly | Pre-alpha; headline run mid-flight |

The intended sweet spot:

- **Open and reproducible.** Trains on the public, dated CC-BY-NC SpeechJudge-Data; the head is a single file you can read in 200 lines.
- **CPU-deployable.** Frozen Whisper-small encoder + a ~400K head means real-time on a laptop.
- **Distribution-matched.** Trained on pairs from the exact modern TTS systems users want to evaluate.
- **Drop-in API.** `score(wav) -> float` and `compare(a, b) -> Pair`. Usable inside a TTS training loop for offline reward modeling, or as a fast eval gate in CI.

---

## 3. Architecture

```
audio waveform (16 kHz)
    │
    ▼
[Whisper-small encoder, FROZEN]            ← openai/whisper-small (244M, encoder ~88M)
    │ hidden states across 13 layers
    │ shape (B, T, 768) per layer
    ▼
[Layer-weighted sum head, TRAINABLE]
    │ learned softmax weights α_0..α_12
    │ output (B, T, 768)
    ▼
[Attention pooling, TRAINABLE]
    │ (768 → 256 → 1) attention scores
    │ pools (B, T, 768) → (B, 768)
    ▼
[Score head, TRAINABLE]
    │ 2-layer MLP, 768 → 256 → 1
    ▼
scalar logit s (∈ ℝ)
```

| Component | Parameters | Trainable |
|---|---|---|
| Whisper-small encoder | ~88M | No (frozen) |
| Layer weights α | 13 | Yes |
| Attention pooler | ~200K | Yes |
| Score MLP head | ~200K | Yes |
| **Total deployed** | ~88M | |
| **Total trainable** | **~400K** | |

### Why this architecture (not the alternatives)

| Alternative | Why we did not pick it |
|---|---|
| Fine-tune full Whisper encoder | Loses generalization, blows compute, requires careful regularization. Frozen-encoder + small-head is the modern best practice (WhiSQA, Whisper-PMFA, SimWhisper-Codec) |
| wav2vec 2.0 / WavLM / HuBERT encoder | Comparable per layer-probing literature. Whisper wins on (a) supervised pretraining produces more content-aligned features for naturalness (arXiv 2509.04830), (b) 99-language coverage out of the box, (c) user-familiar checkpoints. HuBERT variant is on the ablation list |
| MOS regression head (MSE on raw scores) | SpeechJudge-Data is pairwise, not pointwise. Converting via BT inference would lose information. Train pairwise; expose pointwise score as a derived API |
| Generative judge with CoT (like SpeechJudge-GRM) | Defeats the goal. NatScore is the small, fast, deterministic complement |
| Transformer head | Overkill at this trainable-param budget. Attention pooler + 2-layer MLP is the WhiSQA pattern and works |

---

## 4. Training methodology

### 4.1 Loss

Bradley-Terry pairwise loss:

```
L = -log σ(s_chosen - s_rejected)
```

Both clips share the same target text (from `target_text` in SpeechJudge-Data). The model produces a scalar logit per clip; the loss penalizes ordering errors.

### 4.2 Auxiliary losses (ablation-ready)

- **Ordinal-magnitude-weighted BT.** `naturalness_annotation` carries per-rater ordinal magnitudes like `["B+2", "B+1"]`. The average magnitude becomes a per-pair confidence weight `w_i ∈ {0.5, 1, 1.5, 2}` multiplying the BT term: `L_i = w_i · -log σ(s_chosen - s_rejected)`. Strictly stronger signal than the binary `chosen` flag.
- **Margin loss** on the high-consensus subset (`chosen == True`): `L_margin = max(0, m − (s_chosen − s_rejected))`. Tighter calibration.
- **Anchor regression** on UTMOSv2 score for natural reference clips. Biases the absolute score scale without distorting pairwise ranking.
- **Order-swap consistency.** Penalizes position-dependent score drift on identical-clip synthetic pairs.

### 4.3 Data

**SpeechJudge-Data** (`huggingface.co/datasets/RMSnow/SpeechJudge-Data`), released Nov 2025:

| Field | Value |
|---|---|
| Size | 99K pairs |
| Train split | ~42K pairs |
| High-consensus subset (`chosen == True`, rater agreement >40%) | ~31K pairs |
| Avg annotations per pair | 2.49 |
| TTS systems sampled | CosyVoice2, F5-TTS, MaskGCT, Llasa, others |
| Languages | en→en, en→zh, zh→zh, zh→en, en→mixed, zh→mixed (mixed = code-switching) |
| Subsets | `regular` (Emilia-style) + `expressive` (emotional/accented/whispered/game) |
| License | CC-BY-NC |

The 15-column schema is fully documented in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §4.1. Notably, each row also contains a **`gemini-2.5-flash`** chain-of-thought rationale (~1KB) which is a free distillation target (see ablation grid).

### 4.4 Training setup (the active headline run)

| Setting | Value |
|---|---|
| Encoder | `openai/whisper-small` (frozen) |
| Train pairs | ~42K (full train split) |
| Online feature extraction | yes (full pre-extraction would need 5.34 TB cache) |
| Hardware | Kaggle T4 x2 with DataParallel |
| Effective batch | 16 pairs (8 per GPU) |
| Optimizer | AdamW, lr 1e-4 → cosine decay |
| Total steps | 13,125 (5 epochs) |
| Throughput | ~17 steps/min (audio I/O bound) |
| Resume scaffold | yes, latest.pt versioned to a Kaggle Dataset across 9h walls |
| Logging | W&B run `harshalsingh1223-gladium-ai/natscore/yjypnef4` |

Multi-day Kaggle workflow (DataParallel patch + checkpoint-dataset round-trip + resume scaffold) is documented in [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md).

---

## 5. Benchmark

### 5.1 Headline (final, 2026-05-29)

| Run | Stage | Pairwise acc | Mean margin | ECE | Notes |
|---|---|---|---|---|---|
| Kaggle T4 x2 (`yjypnef4`) | Final, dev[:1000] | **71.3%** (95% CI 68.6 – 74.1) | +1.010 | 2.27% | Cleared >70% target. Trained 13,250 steps / 5 epochs in ~4h52m on T4 x2 + DataParallel, resumed mid-run from step 8000 |

**Breakdowns.** Regular subset 74.0%, expressive 69.5% (expressive prosody is the harder slice, as expected). Per-language ordering: en→zh 87.3%, zh→zh 83.2%, zh→en 66.2%, en→en 63.9%, zh→mixed 61.3%, en→mixed 52.5% (mixed-language code-switching is the obvious tail to investigate next).

### 5.2 Reference points

| System | Test set | Pairwise acc | Trainable params | Notes |
|---|---|---|---|---|
| **SpeechJudge-BTRM** | SpeechJudge-Eval | 72.7% | ~7B | Direct baseline; we target beating this with ~1/17000th the params |
| **SpeechJudge-GRM** | SpeechJudge-Eval | 77.2% | ~7B | Gold-standard reference; aspirational matching target |
| **UTMOSv2** | (out of distribution) | not directly comparable | ~300M | Documented negative correlation on conversational speech |
| **WhiSQA** | (out of distribution) | not directly comparable | ~5M | NISQA-trained, not TTS-naturalness |

Targets for NatScore on SpeechJudge-Eval:

- **Realistic floor:** >70% (beats UTMOSv2 / WhiSQA / NISQA-TTS for this task)
- **Stretch:** >73% (beats SpeechJudge-BTRM, the closest comparable)
- **Aspirational:** >77% (matches SpeechJudge-GRM with ~1/17000th the trainable params)

### 5.3 Pipeline-validation baseline (laptop CPU, 500 training pairs)

This is the baseline that shipped before the Kaggle run; it validates that the pipeline produces a non-broken model, not that the model is good.

| Run | n_pairs | Pairwise acc | 95% CI | ECE | Notes |
|---|---|---|---|---|---|
| `natscore-small-v0` / dev[:100] | 100 | 52.00% | [42.98, 62.00] | 31.30% | laptop-CPU pipeline validation; 500 training pairs |

Full per-language breakdown is in [`docs/BENCHMARK.md`](docs/BENCHMARK.md). The dev-set CI includes 50% (chance), as expected at 500 training pairs.

### 5.4 Planned benchmark slices (post-headline)

- `natscore-small-v0 / dev (full ~6K pairs)` once trained on the full 42K
- `natscore-small-v0 / test (~50K pairs)`, the headline number
- UTMOSv2 and WhiSQA baselines on the same audio
- Per-language slices (en→en, en→zh, zh→zh, zh→en, code-switch)
- Regular vs expressive subset breakdown
- Per-system breakdown (CosyVoice2, F5-TTS, MaskGCT, Llasa)
- Calibration plots (reliability + ECE)

---

## 6. Installation

```bash
git clone https://github.com/cybernovas/natscore.git
cd natscore
python -m venv .venv && source .venv/bin/activate

# CPU-only torch (smaller wheel, faster install)
pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.3,<2.7" torchaudio

pip install -e ".[dev,train]"
pytest -q          # 104 pass, 3 Whisper-gated skip
```

No PyPI release planned for v0. Install from source and load the trained checkpoint directly from [HF Hub](https://huggingface.co/harrrshall/natscore-small-v0).

---

## 7. How to use

### 7.1 Python API

```python
import natscore as ns

# Load the released v0 checkpoint from HF Hub.
# First call downloads ~290 MB Whisper-small + ~5 MB NatScore head; subsequent
# calls hit the local HF cache.
scorer = ns.load()                       # defaults to "harrrshall/natscore-small-v0"
# Equivalents:
# scorer = ns.load("natscore-small-v0")            # short alias
# scorer = ns.load("harrrshall/natscore-small-v0") # full HF repo id
# scorer = ns.load(device="cuda", dtype=torch.float16)  # fp16 encoder on GPU

# Pointwise score (higher = more natural)
s: float = scorer.score("path/to/tts_output.wav")
s: float = scorer.score(audio_bytes)     # also accepts bytes, numpy arrays, torch tensors

# Pairwise comparison (recommended; uses the BT structure the model was trained on)
result: ns.Pair = scorer.compare("a.wav", "b.wav")
# result.winner       -> "a" | "b" | "tie"
# result.score_a, result.score_b
# result.margin       = score_a - score_b
# result.prob_a_wins  = sigmoid(margin)

# Batched
scores: list[float] = scorer.batch_score(["clip1.wav", "clip2.wav", "clip3.wav"])
```

### 7.2 CLI

```bash
# Single file
natscore score audio.wav

# Pairwise compare
natscore compare a.wav b.wav

# Batch over a directory
natscore batch --input-dir ./tts_outputs/ --output scores.jsonl
```

### 7.3 Integration patterns

- **Offline TTS evaluation gate.** Score every TTS output in a release candidate; flag the bottom-percentile clips for human review.
- **Inside a TTS training loop.** Run NatScore on validation samples once per N steps; track naturalness trajectory next to MEL-spec / FAD / SIM-O. Cheap enough to keep on by default.
- **Pair selection for preference fine-tuning.** Use NatScore to filter or weight preference pairs collected for DPO-style TTS fine-tuning.
- **A/B test arbiter.** When two TTS systems disagree, NatScore.compare gives a fast, deterministic tiebreaker that beats either system's internal heuristics.

---

## 8. Reproducing the training run

The complete reproduction path is in [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md). The five-minute version:

1. **One-time:** accept the SpeechJudge-Data terms at https://huggingface.co/datasets/RMSnow/SpeechJudge-Data
2. **One-time:** open a Kaggle notebook with **Accelerator = GPU T4 x2** and add three secrets (`HF_TOKEN`, `GITHUB_TOKEN`, optional `WANDB_API_KEY`)
3. **Per run:** **File → Import Notebook** the file at `scripts/kaggle/train_natscore_t4.ipynb` and click **Run All**
4. After completion, pull `final.pt` and `eval_dev.json` from `/kaggle/working/outputs/natscore-small-v0-kaggle/`

The trainer auto-resumes from `latest.pt` if Kaggle hits its 9h timeout (re-run the same notebook; it picks up at the last 500-step checkpoint). Multi-day resume across kernels uses the versioned Kaggle Dataset `harshalsinghcn/natscore-checkpoint`.

### Ablation grid (post-headline)

These dispatch via `train_natscore_ablation.ipynb` with `ABLATION_CONFIG=<name>`:

| Ablation | Config | Question it answers |
|---|---|---|
| High-consensus subset | `configs/train.high_consensus.yaml` | Does noisy preference data hurt, or does volume win? |
| Magnitude-weighted BT | `configs/train.magnitude.yaml` | Should strong preferences get more gradient than ties? |
| Regular-only subset | `configs/train.regular_only.yaml` | Are adversarial / edge-case pairs helping or confusing? |
| Layer-probe L6 | `configs/train.layer_probe_L6.yaml` | Where in the Whisper stack does naturalness signal live? (one of 5 layer probes) |
| Encoder swap (HuBERT, WavLM, Distil-Whisper) | TBD | Is Whisper actually the best encoder for this? |
| Gemini-CoT distillation | TBD | Does the included CoT rationale add signal as an auxiliary head? |

---

## 9. Project structure

```
natscore/
├── README.md                       # this file
├── STATUS.md                       # always-current handoff doc
├── PROJECT_PLAN.md                 # full design doc (read for context)
├── MODEL_LICENSE.md                # CC-BY-NC for weights
├── LICENSE                         # Apache-2.0 for code
├── pyproject.toml                  # package + dev deps
│
├── src/natscore/
│   ├── __init__.py                 # ns.load, ns.Pair
│   ├── cli.py                      # `natscore` command
│   ├── features.py                 # frozen Whisper extractor + DP wrapper
│   ├── model.py                    # layer-weighted head + attention pool + MLP
│   ├── loss.py                     # BT loss (+ magnitude-weighted variant)
│   ├── pair_dataset.py             # SpeechJudge-Data adapter
│   └── train/
│       ├── online_trainer.py       # streaming-feature trainer
│       └── ablation.py             # subset filters + layer-probe configs
│
├── scripts/
│   ├── 01_inspect_dataset.py       # M1 dataset schema dump
│   ├── 02_extract_features.py      # offline cache builder (for laptop testing)
│   ├── 03_evaluate.py              # eval suite (bootstrap CI + ECE + breakdown)
│   ├── 04_compute_baselines.py     # UTMOSv2 / WhiSQA on the same audio
│   ├── 05_aggregate_benchmark.py   # writes/updates docs/BENCHMARK.md
│   └── kaggle/
│       ├── train_natscore_t4.ipynb     # headline run notebook
│       ├── train_natscore_ablation.ipynb
│       ├── save_checkpoint_to_dataset.sh
│       └── kernel-metadata.train.json
│
├── configs/
│   ├── train.kaggle.yaml           # the headline config
│   ├── train.high_consensus.yaml
│   ├── train.magnitude.yaml
│   ├── train.regular_only.yaml
│   └── train.layer_probe_L6.yaml
│
├── tests/                          # 104 pass, 3 Whisper-gated skip
├── docs/
│   ├── BENCHMARK.md                # current numbers + planned slices
│   ├── KAGGLE_SETUP.md             # full reproduction guide
│   └── ARCHITECTURE.md             # deeper architectural notes
│
├── cache/                          # gitignored, feature caches
└── outputs/                        # gitignored, checkpoints + eval JSON
```

---

## 10. Roadmap

| Milestone | Status |
|---|---|
| M0 Scaffold, package skeleton, CI | done |
| M1 SpeechJudge-Data inspection + schema dump | done |
| M2 Frozen Whisper feature extraction + cache | done |
| M3 BT head + trainer + 30-pair sanity | done |
| M3.5 500-pair local CPU training | done |
| M4 Eval suite (bootstrap CI + ECE + breakdown) | done |
| M5a Kaggle online-training notebook | done |
| **M5b Headline run (Kaggle T4 x2 + DP)** | **done. 71.3% pairwise on dev[:1000]** |
| M5b Tier-1 ablation infrastructure | shipped |
| M5c Tier-1 ablation runs (high-consensus, magnitude, regular-only, layer probe) | post 2026-05-30 Kaggle quota reset |
| M6 HF Hub release (v0) | **done. [harrrshall/natscore-small-v0](https://huggingface.co/harrrshall/natscore-small-v0)** |
| M7 Workshop paper draft + HF Spaces demo | pending |

See [`STATUS.md`](STATUS.md) for the live state and [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §8 for the full milestone breakdown.

---

## 11. Licensing

- **Code:** [Apache-2.0](LICENSE)
- **Model weights:** [CC-BY-NC-4.0](MODEL_LICENSE.md), inherited from SpeechJudge-Data

The CC-BY-NC inheritance is non-negotiable; the weights are research-only. The training code in this repo supports retraining on a permissively-licensed dataset if commercial use is required.

---

## 12. Citation

A workshop-paper-length writeup is in flight (M7). Until then, cite as:

```bibtex
@software{singh2026natscore,
  author  = {Singh, Harshal},
  title   = {NatScore: A small, preference-supervised naturalness scorer for modern neural TTS},
  year    = {2026},
  url     = {https://github.com/cybernovas/natscore}
}
```

And the underlying training data:

```bibtex
@dataset{zhang2025speechjudge,
  author  = {Zhang, et al.},
  title   = {SpeechJudge-Data},
  year    = {2025},
  url     = {https://huggingface.co/datasets/RMSnow/SpeechJudge-Data}
}
```

---

## 13. Acknowledgements

- The **SpeechJudge** authors for releasing 99K human-labeled TTS pairs under CC-BY-NC; this project would not exist without that release.
- **Hugging Face** for the `transformers` + `datasets` infrastructure and the audio open-source culture that makes a project like this trivially reproducible.
- **Kaggle** for the free T4 x2 quota that turns "needs a budget" into "needs a weekend".

---

## 14. Resume protocol (cold start)

If you are picking this up in a new session:

1. Read `~/.claude/projects/-home-cybernovas-Desktop-2026-experiments-NatScore/memory/MEMORY.md`
2. Read [`STATUS.md`](STATUS.md)
3. `git log --oneline | head -10`, confirm latest commit
4. `git status`, surface uncommitted work
5. `.venv/bin/python -m pytest -q`, confirm 104 pass / 3 skip
6. Resume from `STATUS.md` **Next concrete action**
