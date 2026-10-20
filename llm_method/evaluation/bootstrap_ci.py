"""Snippet bootstrap confidence intervals for a scored run.

1,000 resamples. Pass the run's base seed (default 42). Each variant and
metric uses that seed plus variant_index * 1,000 plus metric_index.
Resampling is by row. For the ``both`` variant each draw keeps both rater
comparisons of the sampled snippets.

For a 4-class metric, precision, recall, and F1 are computed for each class
and then averaged, with each class weighted by its number of true examples
(scikit-learn average='weighted'). Weighted recall equals accuracy. The macro
values are the unweighted average (scikit-learn average='macro'). Binary
metrics use the positive class.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from evaluation.classification_metrics import (
    F1_MACRO,
    F1_WEIGHTED,
    PRECISION_MACRO,
    PRECISION_WEIGHTED,
    RECALL_MACRO,
    RECALL_WEIGHTED,
    compute_binary_metrics,
    compute_multiclass_metrics,
    filter_full_agreement,
    get_metric_arrays,
    is_binary_metric,
    load_data,
    should_binarize_gt,
)

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
VARIANT_ORDER = ["r1", "r2", "both", "consensus"]
# Plain "evidence" is left at 99 so it sorts after suggestion and connection.
# That order is fixed so the metric seed offset stays stable.
METRIC_PRIORITY = {"evidence_bin": 0, "evidence_ord": 1, "suggestion": 2, "connection": 3}


def detect_metric_names(frame: pd.DataFrame) -> List[str]:
    """List metric names from ``*_llm`` columns, in seed order.

    Args:
        frame: Scored records.

    Returns:
        Metric base names.
    """
    names: List[str] = []
    for column in frame.columns:
        if column.endswith("_llm") and column[:-4] not in names:
            names.append(column[:-4])
    names.sort(key=lambda name: METRIC_PRIORITY.get(name.lower(), 99))
    return names


def read_bin_evidence(config_path: Path) -> bool:
    """Read ``bin_evidence`` or the older ``bin_evid`` flag.

    Args:
        config_path: Run config.yaml.

    Returns:
        True when evidence ground truth should be binarized.
    """
    if not config_path.is_file():
        return False
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "bin_evidence" in config:
        return bool(config["bin_evidence"])
    return bool(config.get("bin_evid", False))


def build_variant_arrays(
    frame: pd.DataFrame,
    variant_name: str,
    metric_name: str,
    binarize_gt: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build label and prediction arrays for one variant.

    Args:
        frame: Rows to score. For ``both``, one row per snippet.
        variant_name: r1, r2, both, or consensus.
        metric_name: Metric base name.
        binarize_gt: Whether to map evidence labels 0-1 to 0 and 2-3 to 1.

    Returns:
        Ground truth and predictions.
    """
    gt_base = metric_name.lower().replace("_bin", "").replace("_ord", "")
    if variant_name == "both":
        y_true_r1, y_pred_r1, _ = get_metric_arrays(frame, metric_name, f"{gt_base}_r1", binarize_gt)
        y_true_r2, y_pred_r2, _ = get_metric_arrays(frame, metric_name, f"{gt_base}_r2", binarize_gt)
        if len(y_true_r1) == 0 and len(y_true_r2) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if len(y_true_r1) == 0:
            return y_true_r2, y_pred_r2
        if len(y_true_r2) == 0:
            return y_true_r1, y_pred_r1
        return np.concatenate([y_true_r1, y_true_r2]), np.concatenate([y_pred_r1, y_pred_r2])

    gt_col = f"{gt_base}_r1" if variant_name == "consensus" else f"{gt_base}_{variant_name}"
    y_true, y_pred, _ = get_metric_arrays(frame, metric_name, gt_col, binarize_gt)
    return y_true, y_pred


def summarize(y_true: np.ndarray, y_pred: np.ndarray, is_binary: bool) -> Dict[str, float]:
    """Accuracy and the precision, recall, and F1 used in the paper tables.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        is_binary: Use binary metrics when true, 4-class metrics otherwise.

    Returns:
        The four scores.
    """
    if is_binary:
        metrics = compute_binary_metrics(y_true, y_pred, y_score=None)
        return {
            "Accuracy": float(metrics["Accuracy"]),
            "Precision": float(metrics["Precision"]),
            "Recall": float(metrics["Recall"]),
            "F1": float(metrics["F1"]),
        }
    metrics = compute_multiclass_metrics(y_true, y_pred, y_score=None, labels=[0, 1, 2, 3])
    return {
        "Accuracy": float(metrics["Accuracy"]),
        "Precision": float(metrics[PRECISION_WEIGHTED]),
        "Recall": float(metrics[RECALL_WEIGHTED]),
        "F1": float(metrics[F1_WEIGHTED]),
        "Precision_macro": float(metrics[PRECISION_MACRO]),
        "Recall_macro": float(metrics[RECALL_MACRO]),
        "F1_macro": float(metrics[F1_MACRO]),
    }


def bootstrap_metric(
    frame: pd.DataFrame,
    variant_name: str,
    metric_name: str,
    is_binary: bool,
    binarize_gt: bool,
    n_bootstrap: int,
    seed: int,
) -> Tuple[int, Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Point estimates and percentile 95% intervals.

    Args:
        frame: Rows for this variant.
        variant_name: r1, r2, both, or consensus.
        metric_name: Metric base name.
        is_binary: Binary metric flag.
        binarize_gt: Ground-truth binarization flag.
        n_bootstrap: Number of draws.
        seed: Numpy seed for this metric.

    Returns:
        Sample size, point estimates, and (low, high) intervals.
    """
    y_true, y_pred = build_variant_arrays(frame, variant_name, metric_name, binarize_gt)
    if len(y_true) == 0:
        return 0, {}, {}
    point = summarize(y_true, y_pred, is_binary)
    draws: Dict[str, List[float]] = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    sample_size = len(frame)
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, sample_size, size=sample_size)
        sampled = frame.iloc[chosen]
        y_true_draw, y_pred_draw = build_variant_arrays(sampled, variant_name, metric_name, binarize_gt)
        if len(y_true_draw) == 0:
            continue
        stats = summarize(y_true_draw, y_pred_draw, is_binary)
        for name, value in stats.items():
            draws[name].append(value)

    intervals: Dict[str, Tuple[float, float]] = {}
    for name, values in draws.items():
        if not values:
            intervals[name] = (float("nan"), float("nan"))
        else:
            intervals[name] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
    return int(len(y_true)), point, intervals


def format_ci(point: float, low: float, high: float) -> str:
    """Format one cell as ``0.1234 [0.1000, 0.1500]``."""
    if np.isnan(low) or np.isnan(high):
        return f"{point:.4f} [N/A]"
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def render_section(
    by_variant: Dict[str, Dict[str, Dict[str, object]]],
    n_bootstrap: int,
    random_seed: int,
) -> str:
    """Render the CI block appended to analysis_summary_CI.txt.

    Args:
        by_variant: Nested results from ``bootstrap_metric``.
        n_bootstrap: Draws requested.
        random_seed: Base seed recorded in the header.

    Returns:
        The text block.
    """
    lines = [
        "=" * 90,
        "BOOTSTRAP 95% CONFIDENCE INTERVALS",
        "=" * 90,
        "Method: Non-parametric bootstrap over samples (with replacement).",
        "For variant=both, resampling is by snippet row and includes both rater comparisons.",
        f"Replicates: {n_bootstrap}",
        f"Seed: {random_seed}",
        "For a 4-class metric, Precision, Recall, and F1 are the per-class scores",
        "averaged with each class weighted by its number of true examples",
        "(scikit-learn average='weighted'). Weighted recall equals accuracy.",
        "Macro is the unweighted average of those per-class scores",
        "(scikit-learn average='macro'). Binary metrics use the positive class.",
    ]
    for variant_name in VARIANT_ORDER:
        variant_metrics = by_variant.get(variant_name, {})
        if not variant_metrics:
            continue
        lines.extend(
            [
                "",
                "#" * 90,
                f"# VARIANT: {variant_name}",
                "#" * 90,
                f"{'Metric':<15} {'N':>6} {'Accuracy [95% CI]':>29} "
                f"{'Precision [95% CI]':>29} {'Recall [95% CI]':>29} {'F1 [95% CI]':>29}",
                "-" * 140,
            ]
        )
        ordered = sorted(variant_metrics, key=lambda name: METRIC_PRIORITY.get(name.lower(), 99))
        for metric_name in ordered:
            item = variant_metrics[metric_name]
            point = item["point"]
            interval = item["ci"]
            cells = [
                format_ci(point[name], interval[name][0], interval[name][1])
                for name in ("Accuracy", "Precision", "Recall", "F1")
            ]
            lines.append(
                f"{metric_name:<15} {item['N']:>6} {cells[0]:>29} {cells[1]:>29} {cells[2]:>29} {cells[3]:>29}"
            )
            if "Precision_macro" in point:
                macro_cells = [
                    format_ci(point[name], interval[name][0], interval[name][1])
                    for name in ("Precision_macro", "Recall_macro", "F1_macro")
                ]
                lines.append(
                    f"{'macro':<15} {'':>6} {'':>29} {macro_cells[0]:>29} "
                    f"{macro_cells[1]:>29} {macro_cells[2]:>29}"
                )
        lines.append("-" * 140)
    return "\n".join(lines)


def write_csv(path: Path, by_variant: Dict[str, Dict[str, Dict[str, object]]]) -> None:
    """Write one row per variant and metric.

    Args:
        path: Destination CSV.
        by_variant: Nested results.
    """
    rows = []
    for variant_name in VARIANT_ORDER:
        for metric_name, item in by_variant.get(variant_name, {}).items():
            point = item["point"]
            interval = item["ci"]
            row = {"variant": variant_name, "metric": metric_name, "N": item["N"]}
            for name in ("Accuracy", "Precision", "Recall", "F1"):
                key = name.lower()
                row[key] = f"{point[name]:.4f}"
                row[f"{key}_low"] = f"{interval[name][0]:.4f}"
                row[f"{key}_high"] = f"{interval[name][1]:.4f}"
            for name, key in (
                ("Precision_macro", "precision_macro"),
                ("Recall_macro", "recall_macro"),
                ("F1_macro", "f1_macro"),
            ):
                if name not in point:
                    continue
                row[key] = f"{point[name]:.4f}"
                row[f"{key}_low"] = f"{interval[name][0]:.4f}"
                row[f"{key}_high"] = f"{interval[name][1]:.4f}"
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_bootstrap(
    evaluated_path: Path,
    bin_evidence: bool,
    n_bootstrap: int,
    base_seed: int,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Bootstrap every variant and metric.

    Args:
        evaluated_path: Scored JSON.
        bin_evidence: Binarize plain evidence ground truth.
        n_bootstrap: Draws per metric.
        base_seed: Run seed. Each variant and metric adds
            ``variant_index * 1_000 + metric_index``.

    Returns:
        Nested results.
    """
    frame = load_data(str(evaluated_path))
    consensus = filter_full_agreement(frame)
    metric_names = detect_metric_names(frame)
    by_variant: Dict[str, Dict[str, Dict[str, object]]] = {}
    for variant_index, variant_name in enumerate(VARIANT_ORDER):
        variant_frame = consensus if variant_name == "consensus" else frame
        by_variant[variant_name] = {}
        for metric_index, metric_name in enumerate(metric_names):
            seed = base_seed + variant_index * 1_000 + metric_index
            n_value, point, interval = bootstrap_metric(
                variant_frame,
                variant_name,
                metric_name,
                is_binary_metric(metric_name, bin_evidence),
                should_binarize_gt(metric_name, bin_evidence),
                n_bootstrap,
                seed,
            )
            if n_value == 0:
                continue
            by_variant[variant_name][metric_name] = {"N": n_value, "point": point, "ci": interval}
    return by_variant


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs for a scored QuAL run")
    parser.add_argument("--input", required=True, help="evaluated_test_cases.json")
    parser.add_argument("--output-dir", help="Defaults to the input file's directory")
    parser.add_argument("--bin-evidence", action="store_true")
    parser.add_argument("--config", help="config.yaml to read the binary-evidence flag from")
    parser.add_argument("--draws", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Run base seed. Each variant and metric adds variant_index * 1000 + metric_index.",
    )
    args = parser.parse_args()

    evaluated_path = Path(args.input)
    if not evaluated_path.is_file():
        print(f"Input not found: {evaluated_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else evaluated_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_evidence = args.bin_evidence
    config_path = Path(args.config) if args.config else evaluated_path.parent / "config.yaml"
    if config_path.is_file() and not args.bin_evidence:
        bin_evidence = read_bin_evidence(config_path)

    by_variant = run_bootstrap(
        evaluated_path,
        bin_evidence=bin_evidence,
        n_bootstrap=args.draws,
        base_seed=args.seed,
    )
    section = render_section(by_variant, args.draws, args.seed)
    summary_path = output_dir / "analysis_summary.txt"
    ci_text_path = output_dir / "analysis_summary_CI.txt"
    if summary_path.is_file() and summary_path.resolve() != ci_text_path.resolve():
        ci_text_path.write_text(summary_path.read_text(encoding="utf-8").rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    else:
        ci_text_path.write_text(section + "\n", encoding="utf-8")
    csv_path = output_dir / "analysis_ci.csv"
    write_csv(csv_path, by_variant)
    print(f"Wrote {ci_text_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
