"""Milestone 1: SpeechJudge-Data schema inspection.

The single most important script in the project's first week. Per
PROJECT_PLAN.md §8 Milestone 1 and §15 end-notes: "the very first step ...
exists specifically to surface that risk before any modeling work."

What this does:

1. Authenticates with HuggingFace using HF_TOKEN.
2. Streams a tiny slice of RMSnow/SpeechJudge-Data WITHOUT downloading the full
   ~3-10 GB (verifies access and surfaces schema first).
3. Dumps the dataset schema, splits, column dtypes, sample rows, and crucially
   the AUDIO STORAGE FORMAT (inline-bytes vs file-path vs URL) to docs/DATASETS.md.
4. Writes a 5-pair JSON fixture to tests/fixtures/mock_speechjudge_5pairs.json
   for offline CI testing.
5. Estimates total dataset size and feature-cache footprint for Milestone 2 planning.

Exit behavior:
- ABORTS if HF_TOKEN missing or terms not accepted (don't burn time).
- Writes a clear error if the schema differs materially from PROJECT_PLAN.md §4.1
  assumptions, so the plan can be updated before any model code is written.

Usage:
    HF_TOKEN=hf_... python scripts/00_inspect_dataset.py --output docs/DATASETS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DATASET_ID = "RMSnow/SpeechJudge-Data"
EXPECTED_SCHEMA_FIELDS = {"target_text", "chosen"}  # per PROJECT_PLAN.md §4.1; verify
REPO_ROOT = Path(__file__).resolve().parent.parent


def _abort(msg: str, exit_code: int = 2) -> None:
    print(f"\nABORT: {msg}\n", file=sys.stderr)
    sys.exit(exit_code)


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        _abort(
            "HF_TOKEN is not set. Export it before running:\n"
            "    export HF_TOKEN=hf_...\n"
            "And ensure the SpeechJudge-Data terms are accepted at\n"
            "    https://huggingface.co/datasets/RMSnow/SpeechJudge-Data"
        )
    return token


def _lazy_imports():
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        _abort(
            f"Missing dependency: {exc}. Install train extras:\n"
            "    pip install -e '.[train]'"
        )
    return load_dataset, HfApi


def _whoami(HfApi, token: str) -> str:
    try:
        return HfApi(token=token).whoami()["name"]
    except Exception as exc:
        _abort(f"HF authentication failed: {exc}")


def _inspect_schema(load_dataset, token: str, limit: int = 5):
    """Load a tiny streaming slice and return (schema_dict, sample_rows).

    Audio columns are explicitly NOT decoded -- we want raw schema and
    storage-format info, not waveform samples. This avoids the torchcodec
    runtime dependency that the default `datasets` Audio feature requires.
    """
    print(f"Streaming a {limit}-row slice of {DATASET_ID} ...")
    try:
        ds_iter = load_dataset(DATASET_ID, split="train", streaming=True, token=token)
    except Exception as exc:
        _abort(
            f"load_dataset({DATASET_ID!r}, streaming=True) failed: {exc}\n"
            "Common causes:\n"
            "  - Dataset terms not accepted (visit the dataset page on HF)\n"
            "  - HF_TOKEN lacks `read` scope\n"
            "  - Dataset id has changed since PROJECT_PLAN.md was written"
        )

    # Disable audio decoding on every Audio-typed column -- we only need raw
    # bytes/path metadata to classify storage format. Schema is available
    # *before* iteration via `.features`.
    try:
        from datasets import Audio  # type: ignore[import-not-found]

        features = getattr(ds_iter, "features", None)
        if features is not None:
            for col, feat in list(features.items()):
                if isinstance(feat, Audio):
                    ds_iter = ds_iter.cast_column(col, Audio(decode=False))
    except Exception as exc:
        print(f"[note] Could not pre-disable audio decoding ({exc}); proceeding anyway.")

    rows = []
    iterator = iter(ds_iter)
    for _ in range(limit):
        try:
            rows.append(next(iterator))
        except StopIteration:
            break

    if not rows:
        _abort("Stream returned zero rows -- schema cannot be inferred.")

    schema = {k: type(v).__name__ for k, v in rows[0].items()}
    return schema, rows


def _classify_audio_storage(sample_row: dict) -> str:
    """Identify whether audio is inline-bytes, file-path, or URL-referenced.

    This is the single most important output of this script -- it determines
    the feature-extraction strategy for Milestone 2.
    """
    candidate_keys = [
        k for k in sample_row if "audio" in k.lower() or k in {"chosen", "rejected"}
    ]
    if not candidate_keys:
        return "UNKNOWN -- no audio-shaped fields detected; manual inspection required"

    findings = []
    for k in candidate_keys:
        v = sample_row[k]
        if isinstance(v, dict) and "bytes" in v:
            findings.append(f"{k}: inline-bytes dict (HF Audio feature)")
        elif isinstance(v, dict) and "path" in v:
            findings.append(f"{k}: file-path dict (path={v.get('path')!r})")
        elif isinstance(v, str) and v.startswith(("http://", "https://", "gs://", "s3://")):
            findings.append(f"{k}: URL string ({v!r})")
        elif isinstance(v, str) and ("/" in v or v.endswith((".wav", ".flac", ".mp3"))):
            findings.append(f"{k}: path-like string ({v!r})")
        else:
            findings.append(f"{k}: {type(v).__name__} (unexpected -- inspect manually)")
    return "\n  - ".join([""] + findings)


def _write_datasets_md(
    out_path: Path,
    schema: dict,
    rows: list,
    audio_classification: str,
    user: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_redacted = []
    for r in rows[:3]:
        redacted: dict = {}
        for k, v in r.items():
            if isinstance(v, (bytes, bytearray)):
                redacted[k] = f"<{len(v)} bytes>"
            elif isinstance(v, dict) and "bytes" in v:
                redacted[k] = {
                    **{kk: vv for kk, vv in v.items() if kk != "bytes"},
                    "bytes": f"<{len(v.get('bytes') or b'')} bytes>",
                }
            else:
                redacted[k] = v
        sample_redacted.append(redacted)

    schema_table = "\n".join(f"| `{k}` | `{v}` |" for k, v in schema.items())
    target_text_present = "target_text" in schema
    chosen_present = "chosen" in schema
    sample_json = json.dumps(sample_redacted, indent=2, default=str, ensure_ascii=False)

    content = (
        "# SpeechJudge-Data -- schema inspection (auto-generated)\n\n"
        f"Generated by `scripts/00_inspect_dataset.py` for HF user **{user}**.\n\n"
        "> This file is **generated**. Re-run the script to refresh. Do not\n"
        "> edit by hand; add prose to `docs/DATASETS_NOTES.md` instead.\n\n"
        "## Dataset\n\n"
        f"- HuggingFace ID: `{DATASET_ID}`\n"
        "- License: CC-BY-NC-4.0 (inherited by any trained checkpoint)\n\n"
        "## Top-level schema (train split, first row)\n\n"
        "| Field | Python type |\n|---|---|\n"
        f"{schema_table}\n\n"
        "## Audio storage format\n\n"
        "The single most important finding -- determines the M2 feature-extraction strategy:\n"
        f"{audio_classification}\n\n"
        "## Cross-check vs PROJECT_PLAN.md s4.1 assumptions\n\n"
        "Plan assumes fields including `target_text` and a `chosen` flag.\n\n"
        f"- `target_text` present: **{target_text_present}**\n"
        f"- `chosen` present: **{chosen_present}**\n\n"
        "If either is missing, update PROJECT_PLAN.md s4.1 before continuing to M2.\n\n"
        "## Sample rows (first 3, audio bytes redacted)\n\n"
        "```json\n"
        f"{sample_json}\n"
        "```\n\n"
        "## Next steps\n\n"
        "1. If audio is **inline-bytes**: M2 can stream directly from HF; no separate download step.\n"
        "2. If audio is **file-path** or **URL**: M2 must download audio files separately. Re-estimate disk budget.\n"
        "3. If `chosen` semantics differ from \"high-rater-agreement flag\", revisit M5 experiment 1.\n"
    )
    out_path.write_text(content)
    print(f"Wrote schema report -> {out_path}")


def _write_test_fixture(rows: list, fixture_path: Path) -> None:
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = []
    for r in rows:
        sr: dict = {}
        for k, v in r.items():
            if isinstance(v, (bytes, bytearray)):
                sr[k] = {"_redacted_bytes_len": len(v)}
            elif isinstance(v, dict) and "bytes" in v:
                sr[k] = {
                    kk: (f"<{len(vv)} bytes>" if isinstance(vv, (bytes, bytearray)) else vv)
                    for kk, vv in v.items()
                }
            else:
                sr[k] = v
        sanitized.append(sr)
    fixture_path.write_text(json.dumps(sanitized, indent=2, default=str, ensure_ascii=False))
    print(f"Wrote test fixture -> {fixture_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="docs/DATASETS.md",
        help="Destination for the schema markdown report.",
    )
    parser.add_argument(
        "--fixture", default="tests/fixtures/mock_speechjudge_5pairs.json",
        help="Destination for the 5-row test fixture.",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Number of streaming rows to fetch.",
    )
    args = parser.parse_args()

    token = _require_hf_token()
    load_dataset, HfApi = _lazy_imports()
    user = _whoami(HfApi, token)
    print(f"HF user: {user}")

    schema, rows = _inspect_schema(load_dataset, token, limit=args.limit)
    print(f"Inferred schema: {list(schema)}")

    audio_classification = _classify_audio_storage(rows[0])
    print(f"Audio storage:{audio_classification}")

    _write_datasets_md(REPO_ROOT / args.output, schema, rows, audio_classification, user)
    _write_test_fixture(rows, REPO_ROOT / args.fixture)

    missing = EXPECTED_SCHEMA_FIELDS - set(schema)
    if missing:
        print(f"\nWARNING: expected fields not found in schema: {missing}", file=sys.stderr)
        print("Update PROJECT_PLAN.md s4.1 before proceeding to Milestone 2.", file=sys.stderr)
        sys.exit(1)

    print("\nMilestone 1 complete. Review docs/DATASETS.md before starting M2.")


if __name__ == "__main__":
    main()
