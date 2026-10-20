"""Bar charts for the criteria-vs-steps comparison and the temperature runs.

Reads the point-estimate CSV from classification_metrics and the CI CSV
from bootstrap_ci. Evidence (ordinal) comes from the ordinal run. Evidence
(binary), suggestion, and connection come from the binary run of the same
prompt family. That is how the paper figure was assembled.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

STAT_KEYS = ["accuracy", "f1", "precision", "recall"]
STAT_LABELS = {"accuracy": "Accuracy", "f1": "F1", "precision": "Precision", "recall": "Recall"}
COMBO_METRICS = [
    ("Evidence (Ordinal)", "ordinal", "evidence"),
    ("Evidence (Binary)", "binary", "evidence"),
    ("Suggestion", "binary", "suggestion"),
    ("Connection", "binary", "connection"),
]


def _ci_bounds(ci_frame: pd.DataFrame, variant: str, metric: str, stat: str) -> tuple[float, float]:
    """Look up one interval.

    Args:
        ci_frame: analysis_ci.csv contents.
        variant: r1, r2, both, or consensus.
        metric: Column prefix used in the point-estimate file.
        stat: accuracy, precision, recall, or f1.

    Returns:
        Low and high bounds.
    """
    match = ci_frame[(ci_frame["variant"] == variant) & (ci_frame["metric"] == metric)]
    if match.empty:
        raise ValueError(f"No CI row for variant={variant} metric={metric}")
    row = match.iloc[0]
    return float(row[f"{stat}_low"]), float(row[f"{stat}_high"])


def _point(frame: pd.DataFrame, variant: str, metric: str, stat: str) -> float:
    """Look up one point estimate.

    Args:
        frame: analysis_results.csv contents.
        variant: r1, r2, both, or consensus.
        metric: Column prefix.
        stat: accuracy, precision, recall, or f1.

    Returns:
        The point estimate.
    """
    match = frame[frame["variant"] == variant]
    if match.empty:
        raise ValueError(f"No row for variant={variant}")
    return float(match.iloc[0][f"{metric}_{stat}"])


def plot_combo(
    ordinal_csv: Path,
    binary_csv: Path,
    ordinal_ci: Path,
    binary_ci: Path,
    variant: str,
    title: str,
    output_path: Path,
) -> None:
    """Save the four-panel comparison chart.

    Args:
        ordinal_csv: Point estimates from the ordinal-evidence run.
        binary_csv: Point estimates from the binary-evidence run.
        ordinal_ci: CI CSV for the ordinal run.
        binary_ci: CI CSV for the binary run.
        variant: Which reporting row to plot.
        title: Figure title.
        output_path: PNG path.
    """
    points = {"ordinal": pd.read_csv(ordinal_csv), "binary": pd.read_csv(binary_csv)}
    intervals = {"ordinal": pd.read_csv(ordinal_ci), "binary": pd.read_csv(binary_ci)}
    fig, axes = plt.subplots(1, len(STAT_KEYS), figsize=(12, 3.6), sharey=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for axis, stat in zip(axes, STAT_KEYS):
        values = []
        lows = []
        highs = []
        labels = []
        for label, source, metric in COMBO_METRICS:
            value = _point(points[source], variant, metric, stat)
            low, high = _ci_bounds(intervals[source], variant, metric, stat)
            values.append(value)
            lows.append(value - low)
            highs.append(high - value)
            labels.append(label)
        positions = list(range(len(labels)))
        axis.bar(positions, values, color=colors, yerr=[lows, highs], capsize=3, ecolor="black")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(STAT_LABELS[stat])
    axes[0].set_ylabel("Score")
    fig.suptitle(f"{title} ({variant})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_temperature(
    runs: Sequence[tuple[str, Path, Path]],
    variant: str,
    output_path: Path,
) -> None:
    """Save grouped bars for several temperatures.

    Args:
        runs: (label, point-estimate csv, ci csv) triples, already ordered.
        variant: Which reporting row to plot.
        output_path: PNG path.
    """
    metric_order = [
        ("Evidence (Ordinal)", "evidence_ord"),
        ("Evidence (Binary)", "evidence_bin"),
        ("Suggestion", "suggestion"),
        ("Connection", "connection"),
    ]
    loaded = [(label, pd.read_csv(points), pd.read_csv(ci)) for label, points, ci in runs]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    width = 0.8 / max(len(loaded), 1)
    for axis, stat in zip(axes.ravel(), STAT_KEYS):
        for index, (label, points, intervals) in enumerate(loaded):
            values = []
            yerr_low = []
            yerr_high = []
            for _metric_label, metric_key in metric_order:
                value = _point(points, variant, metric_key, stat)
                low, high = _ci_bounds(intervals, variant, metric_key, stat)
                values.append(value)
                yerr_low.append(value - low)
                yerr_high.append(high - value)
            positions = [i + (index - (len(loaded) - 1) / 2) * width for i in range(len(metric_order))]
            axis.bar(
                positions,
                values,
                width=width,
                label=label,
                yerr=[yerr_low, yerr_high],
                capsize=2,
                ecolor="black",
            )
        axis.set_xticks(range(len(metric_order)))
        axis.set_xticklabels([label for label, _key in metric_order], rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(STAT_LABELS[stat])
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(loaded))
    fig.suptitle(f"Temperature ablation ({variant})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def zip_temperature_runs(
    labels: Sequence[str],
    points: Sequence[Path],
    intervals: Sequence[Path],
) -> List[tuple[str, Path, Path]]:
    """Pair labels, point-estimate files, and CI files in the order given.

    Args:
        labels: One label per run.
        points: One point-estimate CSV per run.
        intervals: One CI CSV per run.

    Returns:
        Triples in the given order.

    Raises:
        ValueError: If the three lists have different lengths.
    """
    if len(labels) != len(points) or len(labels) != len(intervals):
        raise ValueError(
            "Pass the same number of --label, --points, and --ci "
            f"({len(labels)} labels, {len(points)} point files, {len(intervals)} CI files)."
        )
    return list(zip(labels, points, intervals))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot QuAL LLM judge results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    combo = subparsers.add_parser("combo", help="Criteria or evaluation-steps comparison")
    combo.add_argument("--ordinal-csv", required=True, type=Path)
    combo.add_argument("--binary-csv", required=True, type=Path)
    combo.add_argument("--ordinal-ci", required=True, type=Path)
    combo.add_argument("--binary-ci", required=True, type=Path)
    combo.add_argument("--variant", default="consensus")
    combo.add_argument("--title", required=True)
    combo.add_argument("--output", required=True, type=Path)

    temperature = subparsers.add_parser("temperature", help="Temperature ablation bars")
    temperature.add_argument(
        "--label",
        action="append",
        required=True,
        help="Bar label. Repeat once per run, in the same order as --points and --ci.",
    )
    temperature.add_argument(
        "--points",
        action="append",
        required=True,
        type=Path,
        help="Point-estimate CSV. Repeat once per run, in the same order as --label.",
    )
    temperature.add_argument(
        "--ci",
        action="append",
        required=True,
        type=Path,
        help="CI CSV. Repeat once per run, in the same order as --label.",
    )
    temperature.add_argument("--variant", default="consensus")
    temperature.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    try:
        if args.command == "combo":
            plot_combo(
                args.ordinal_csv,
                args.binary_csv,
                args.ordinal_ci,
                args.binary_ci,
                args.variant,
                args.title,
                args.output,
            )
        else:
            plot_temperature(
                zip_temperature_runs(args.label, args.points, args.ci),
                args.variant,
                args.output,
            )
    except (ValueError, OSError, KeyError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
