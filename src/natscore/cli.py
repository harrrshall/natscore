"""NatScore CLI entry point.

Milestone 0 ships the command surface so packaging/install is verifiable.
Subcommands raise NotImplementedError until the corresponding milestone lands.
"""

from __future__ import annotations

import click

from natscore import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="natscore")
def main() -> None:
    """NatScore — preference-supervised TTS naturalness scorer."""


@main.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default="natscore-small-v1", show_default=True)
@click.option("--language", default=None, help="Optional language hint (passed to Whisper).")
def score(audio: str, model: str, language: str | None) -> None:
    """Score a single audio file. Higher = more natural."""
    raise NotImplementedError(
        "`natscore score` lands in Milestone 6. See PROJECT_PLAN.md §8."
    )


@main.command()
@click.argument("audio_a", type=click.Path(exists=True, dir_okay=False))
@click.argument("audio_b", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default="natscore-small-v1", show_default=True)
def compare(audio_a: str, audio_b: str, model: str) -> None:
    """Pairwise comparison of two audio files."""
    raise NotImplementedError(
        "`natscore compare` lands in Milestone 6. See PROJECT_PLAN.md §8."
    )


@main.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--model", default="natscore-small-v1", show_default=True)
def batch(input_dir: str, output: str, model: str) -> None:
    """Score every audio file in a directory; write JSONL."""
    raise NotImplementedError(
        "`natscore batch` lands in Milestone 6. See PROJECT_PLAN.md §8."
    )


if __name__ == "__main__":
    main()
