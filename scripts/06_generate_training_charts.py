"""Generate the headline training charts into graphs/.

Six PNGs, dark aesthetic, ~30s end-to-end. Each tells a self-contained
story about the M5b headline run (W&B run y2o2g3yh -> yjypnef4, dataset
`harshalsinghcn/natscore-checkpoint`, model `harrrshall/natscore-small-v0`):

    01_loss_curve_with_resume_seam.png  - 2-day training loss, resume boundary visible
    02_per_language_accuracy.png        - bar chart of per-language pairwise acc on dev[:1000]
    03_train_accuracy_trajectory.png    - per-batch accuracy with EMA overlay
    04_mean_margin_growth.png           - BT margin trajectory (0 -> 2.37)
    05_lr_schedule.png                  - cosine-with-warmup + Day-2 resume seam
    06_param_efficiency.png             - NatScore vs SpeechJudge-{BTRM,GRM} (log-x)

Run:
    python scripts/06_generate_training_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = REPO_ROOT / "graphs"
EVAL_JSON = REPO_ROOT / "outputs" / "natscore-small-v0-kaggle" / "eval_dev.json"

WANDB_PROJECT = "harshalsingh1223-gladium-ai/natscore"
DAY1_RUN = "y2o2g3yh"   # single-T4, step 0 -> 8000, hit 9h wall
DAY2_RUN = "yjypnef4"   # T4 x2 + DataParallel, resumed step 8000 -> 13250

# ---------------------------------------------------------------- aesthetic

BG = "#0a0e14"
PANEL = "#11151c"
TEXT = "#e6e8ec"
MUTED = "#8b95a7"
GRID = "#1f2530"

CYAN = "#4cc9f0"
AMBER = "#ffb86b"
MAGENTA = "#ff79c6"
MINT = "#50fa7b"
CORAL = "#ff6b6b"
LAVENDER = "#bd93f9"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.dpi": 140,
})


def _fetch_wandb_history(run_id: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    rows = run.history(samples=100_000, pandas=False)
    df = pd.DataFrame(rows)
    return df.sort_values("step").reset_index(drop=True)


def _ema(series: pd.Series, alpha: float = 0.1) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def _annotate_corner(ax, text: str, color: str = MUTED) -> None:
    ax.text(
        0.99, 0.02, text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=color, alpha=0.85, family="monospace",
    )


# ============================================================ 01: loss curve

def chart_loss(day1: pd.DataFrame, day2: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Raw (light) + EMA (bold) per day
    for day_df, day_color, day_label in [
        (day1, CYAN, "Day 1 (single T4)"),
        (day2, MAGENTA, "Day 2 (T4 x2 + DataParallel)"),
    ]:
        ax.plot(day_df["step"], day_df["loss"],
                color=day_color, alpha=0.18, linewidth=1.0)
        ax.plot(day_df["step"], _ema(day_df["loss"], 0.08),
                color=day_color, linewidth=2.2, label=day_label)

    # Resume seam at step ~8000
    resume_step = max(day1["step"].max(), 8000)
    ax.axvspan(day1["step"].max(), day2["step"].min(),
               color=AMBER, alpha=0.10)
    ax.axvline(resume_step, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.annotate(
        "9 h wall hit\ncheckpoint saved\nresumed next day",
        xy=(resume_step, ax.get_ylim()[1] * 0.95),
        xytext=(resume_step + 600, 0.85),
        color=AMBER, fontsize=9, ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color=AMBER, alpha=0.6, lw=0.8),
    )

    # Final loss callout
    final = day2.iloc[-1]
    ax.scatter([final["step"]], [final["loss"]], s=80, color=MINT,
               zorder=5, edgecolor=BG, linewidth=1.5)
    ax.annotate(
        f"  final loss\n  {final['loss']:.3f}",
        xy=(final["step"], final["loss"]),
        xytext=(final["step"] - 1800, final["loss"] + 0.05),
        color=MINT, fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color=MINT, alpha=0.6, lw=0.8),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("Bradley-Terry loss")
    ax.set_title("Training loss across the 2-day resumed run", pad=14)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    _annotate_corner(ax, "y2o2g3yh -> yjypnef4 | 13,250 steps | 5 epochs")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# =================================================== 02: per-language acc

def chart_per_language(eval_data: dict, out: Path) -> None:
    per_lang = eval_data["per_language"]
    rows = sorted(
        [(k, v["accuracy"], v["n_pairs"]) for k, v in per_lang.items()],
        key=lambda r: r[1], reverse=True,
    )
    labels = [r[0].replace("2", " -> ") for r in rows]
    accs = np.array([r[1] for r in rows])
    ns = [r[2] for r in rows]

    # Color by tier
    def tier_color(a: float) -> str:
        if a >= 0.80: return MINT
        if a >= 0.70: return CYAN
        if a >= 0.60: return AMBER
        return CORAL
    colors = [tier_color(a) for a in accs]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars = ax.barh(labels, accs, color=colors, edgecolor=BG, linewidth=1.2)
    # Value + N label at the end of each bar
    for bar, a, n in zip(bars, accs, ns):
        ax.text(a + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{100*a:.1f}%   n={n}",
                va="center", ha="left", color=TEXT, fontsize=10)

    # Chance baseline
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.5, len(labels) - 0.4, " chance",
            color=MUTED, fontsize=9, va="bottom")

    # Headline baseline (overall 71.3%)
    overall = eval_data["pairwise_accuracy"]
    ax.axvline(overall, color=LAVENDER, linestyle=":", linewidth=1.4, alpha=0.9)
    ax.text(overall, -0.6, f" overall {100*overall:.1f}%",
            color=LAVENDER, fontsize=9, va="top")

    ax.set_xlim(0.40, 1.02)
    ax.set_xlabel("pairwise accuracy on SpeechJudge dev[:1000]")
    ax.set_title("Per-language accuracy: code-switching is the tail", pad=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
    _annotate_corner(ax, "natscore-small-v0 | 394K trainable params")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# ============================================== 03: train acc trajectory

def chart_train_acc(day1: pd.DataFrame, day2: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for day_df, day_color, day_label in [
        (day1, CYAN, "Day 1"),
        (day2, MAGENTA, "Day 2"),
    ]:
        ax.plot(day_df["step"], day_df["accuracy"],
                color=day_color, alpha=0.16, linewidth=1.0)
        ax.plot(day_df["step"], _ema(day_df["accuracy"], 0.05),
                color=day_color, linewidth=2.2, label=day_label + " (EMA)")

    # 50% chance line
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(day1["step"].iloc[5], 0.51, "chance",
            color=MUTED, fontsize=9, va="bottom")

    # M5b target
    ax.axhline(0.70, color=AMBER, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(day2["step"].iloc[-30], 0.715, "M5b target (>70%)",
            color=AMBER, fontsize=9, va="bottom")

    ax.set_xlabel("training step")
    ax.set_ylabel("per-batch pairwise accuracy")
    ax.set_title("Train-batch accuracy: chance to 80%+ over 13K steps", pad=14)
    ax.set_ylim(0.40, 1.02)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")
    _annotate_corner(ax, "translucent = raw, bold = EMA smoothed")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# ============================================== 04: mean margin growth

def chart_margin(day1: pd.DataFrame, day2: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    full = pd.concat([day1, day2]).sort_values("step").reset_index(drop=True)

    ax.plot(full["step"], full["mean_margin"],
            color=LAVENDER, alpha=0.20, linewidth=1.0)
    ax.plot(full["step"], _ema(full["mean_margin"], 0.05),
            color=LAVENDER, linewidth=2.4, label="mean margin (EMA)")

    # Fill area under EMA
    ema = _ema(full["mean_margin"], 0.05)
    ax.fill_between(full["step"], 0, ema, color=LAVENDER, alpha=0.10)

    # Final callout
    final_step, final_margin = full["step"].iloc[-1], ema.iloc[-1]
    ax.scatter([final_step], [final_margin], s=80, color=MINT,
               zorder=5, edgecolor=BG, linewidth=1.5)
    ax.annotate(
        f"final mean margin\n{full['mean_margin'].iloc[-1]:.2f}",
        xy=(final_step, final_margin),
        xytext=(final_step - 3500, final_margin + 0.15),
        color=MINT, fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color=MINT, alpha=0.6, lw=0.8),
    )

    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("score_chosen - score_rejected   (logit units)")
    ax.set_title("BT margin: model confidence grows from 0 to 2.37", pad=14)
    ax.grid(True, axis="y")
    _annotate_corner(ax, "sigmoid(2.37) ~ 91% confidence that chosen > rejected")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# ============================================================ 05: LR schedule

def chart_lr(day1: pd.DataFrame, day2: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    full = pd.concat([day1, day2]).sort_values("step").reset_index(drop=True)

    ax.plot(full["step"], full["lr"] * 1000,
            color=CYAN, linewidth=2.2, label="learning rate")
    ax.fill_between(full["step"], 0, full["lr"] * 1000,
                    color=CYAN, alpha=0.10)

    # Resume seam: LR was restored from the checkpoint, so there's a
    # small jump if the cosine wasn't exactly preserved across the wall.
    resume_step = day2["step"].min()
    ax.axvline(resume_step, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(resume_step + 100, ax.get_ylim()[1] * 0.85,
            "  resume", color=AMBER, fontsize=9, va="top")

    # Warmup annotation
    warmup_end = 500
    ax.axvspan(0, warmup_end, color=MAGENTA, alpha=0.08)
    ax.text(warmup_end / 2, ax.get_ylim()[1] * 0.5, "warmup\n500 steps",
            color=MAGENTA, fontsize=9, ha="center", va="center")

    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate  (x 10$^{-3}$)")
    ax.set_title("Cosine LR schedule with warmup, preserved across the resume", pad=14)
    ax.grid(True, axis="y")
    _annotate_corner(ax, "AdamW | peak lr 1e-3 | cosine decay to 0")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# =========================================================== 06: param eff

def chart_param_efficiency(eval_data: dict, out: Path) -> None:
    natscore_acc = eval_data["pairwise_accuracy"]

    # Three systems, all measured on SpeechJudge-Eval (apples to apples).
    systems = [
        # (name, params, accuracy, color, marker)
        ("NatScore-small-v0\n(this work)", 394_255, natscore_acc, MINT, "o"),
        ("SpeechJudge-BTRM\n(Zhang+ 2025)", 7_000_000_000, 0.727, CYAN, "s"),
        ("SpeechJudge-GRM\n(Zhang+ 2025)", 7_000_000_000, 0.772, LAVENDER, "D"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    for name, params, acc, color, marker in systems:
        ax.scatter([params], [acc * 100], s=260, c=color,
                   marker=marker, edgecolor=BG, linewidth=2.0, zorder=5)
        # Label offset based on position
        if "NatScore" in name:
            xytext = (params * 2.5, acc * 100 - 1.5)
            ha = "left"
        else:
            xytext = (params * 0.45, acc * 100 + 1.0)
            ha = "right"
        ax.annotate(
            name, xy=(params, acc * 100), xytext=xytext,
            color=color, fontsize=10.5, ha=ha, va="center",
            fontweight="bold",
        )

    # Highlight the parameter-efficiency story
    # NatScore is 17,500x smaller than BTRM at ~the same accuracy
    ax.annotate(
        "",
        xy=(394_255, 71.3), xytext=(7_000_000_000, 72.7),
        arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.2, alpha=0.6,
                        linestyle="--"),
    )
    ax.text(
        np.sqrt(394_255 * 7_000_000_000), 75.5,
        "  17,500x fewer trainable params\n  -1.4 pp accuracy",
        color=AMBER, fontsize=10, ha="center", va="bottom",
        fontstyle="italic",
    )

    ax.set_xscale("log")
    ax.set_xlim(5e4, 5e10)
    ax.set_ylim(50, 85)
    ax.set_xlabel("trainable parameters  (log scale)")
    ax.set_ylabel("pairwise accuracy on SpeechJudge dev  (%)")
    ax.set_title("Parameter efficiency on the same benchmark", pad=14)
    ax.grid(True, which="both", alpha=0.4)
    ax.axhline(50, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(6e4, 50.8, "chance", color=MUTED, fontsize=9)
    _annotate_corner(ax, "NatScore measured on dev[:1000]; baselines on full eval set")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# ============================================================ main

def main() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] fetching W&B history (Day 1 + Day 2)...")
    day1 = _fetch_wandb_history(DAY1_RUN)
    day2 = _fetch_wandb_history(DAY2_RUN)
    print(f"        Day 1: {len(day1)} rows, steps {day1['step'].min()}-{day1['step'].max()}")
    print(f"        Day 2: {len(day2)} rows, steps {day2['step'].min()}-{day2['step'].max()}")

    print("[2/3] loading eval_dev.json...")
    eval_data = json.loads(EVAL_JSON.read_text())
    print(f"        overall: {100*eval_data['pairwise_accuracy']:.2f}% on {eval_data['n_pairs']} pairs")

    print("[3/3] rendering charts...")
    charts = [
        ("01_loss_curve_with_resume_seam.png",
         lambda p: chart_loss(day1, day2, p)),
        ("02_per_language_accuracy.png",
         lambda p: chart_per_language(eval_data, p)),
        ("03_train_accuracy_trajectory.png",
         lambda p: chart_train_acc(day1, day2, p)),
        ("04_mean_margin_growth.png",
         lambda p: chart_margin(day1, day2, p)),
        ("05_lr_schedule.png",
         lambda p: chart_lr(day1, day2, p)),
        ("06_param_efficiency.png",
         lambda p: chart_param_efficiency(eval_data, p)),
    ]
    for name, fn in charts:
        out_path = GRAPHS_DIR / name
        fn(out_path)
        print(f"        wrote {out_path.relative_to(REPO_ROOT)}")

    print(f"\nall charts in {GRAPHS_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
