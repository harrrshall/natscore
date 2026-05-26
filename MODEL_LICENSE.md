# Model Weight License (CC-BY-NC-4.0)

All trained model checkpoints distributed by this project — including but
not limited to `natscore-small-v1`, `natscore-base-v1`, and any other
checkpoint hosted on HuggingFace Hub under the `natscore` namespace — are
released under **Creative Commons Attribution-NonCommercial 4.0
International (CC-BY-NC-4.0)**.

## Why CC-BY-NC and not Apache-2.0

The training data, **SpeechJudge-Data** (Zhang et al., Nov 2025,
`huggingface.co/datasets/RMSnow/SpeechJudge-Data`), is licensed under
CC-BY-NC-4.0. Model weights trained on CC-BY-NC data are themselves
considered derivative works and inherit the non-commercial restriction.

This is consistent with how the original SpeechJudge authors released
their own checkpoints (SpeechJudge-BTRM, SpeechJudge-GRM), and with
common practice for HuggingFace dataset license inheritance.

## What this means for users

You **may**:

- Use the model weights for academic research, personal projects,
  benchmarking, and non-commercial evaluation of TTS systems.
- Cite the model and dataset in publications.
- Build derivative models for non-commercial research purposes.

You **may not**:

- Use the model weights in any commercial product, service, or workflow
  that generates revenue.
- Use the model weights to evaluate TTS output that is part of a
  commercial pipeline (even internally).
- Re-license the weights under a more permissive license.

If you need a commercially-usable naturalness scorer, retrain on a
permissively-licensed dataset. The code in this repository (Apache-2.0)
supports that — only the released checkpoints are restricted.

## Citation

If you use these weights, cite both this repository and the SpeechJudge
dataset:

```bibtex
@misc{speechjudge2025,
  title = {SpeechJudge: A Large-Scale Human Preference Benchmark for
           Naturalness Evaluation of Modern Neural TTS},
  author = {Zhang and others},
  year = {2025},
  url = {https://arxiv.org/abs/2511.07931}
}
```
