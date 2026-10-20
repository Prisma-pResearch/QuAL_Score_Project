"""Classification metrics analysis for LLM evaluator performance.

Computes standard classification metrics (Accuracy, Precision, Recall, F1, AUROC)
comparing LLM predictions against rater labels under multiple reporting variants:
- r1: evaluate against rater 1
- r2: evaluate against rater 2
- both: treat each example as two examples (vs r1 and vs r2)
- consensus: subset where r1 and r2 agree on all three metrics

Usage:
    uv run python -m evaluation.classification_metrics --input path/to/evaluated_test_cases.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


# 4-class Evidence. Precision, recall, and F1 are computed for each class
# and then averaged. Weighted: each class is weighted by its number of true
# (reference) examples (sklearn average='weighted'). Weighted recall equals
# accuracy. Macro: each class has equal weight (sklearn average='macro').
PRECISION_WEIGHTED = "Precision (weighted by true class count)"
RECALL_WEIGHTED = "Recall (weighted by true class count; equals accuracy)"
F1_WEIGHTED = "F1 (weighted by true class count)"
PRECISION_MACRO = "Precision (macro, unweighted)"
RECALL_MACRO = "Recall (macro, unweighted)"
F1_MACRO = "F1 (macro, unweighted)"
FOUR_CLASS_NOTE = (
    "4-class precision, recall, and F1 are computed for each class (0, 1, 2, 3) "
    "and then averaged. The weighted average weights each class by its number of "
    "true (reference) examples (scikit-learn average='weighted'). Weighted recall "
    "equals accuracy. The macro average gives every class the same weight "
    "(scikit-learn average='macro'). Binary precision, recall, and F1 are the "
    "positive class."
)


class Tee:
    """Helper to write to multiple files (stdout and file) simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_data(input_path: str) -> pd.DataFrame:
    """Load evaluated test cases from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def filter_full_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only samples where r1 and r2 agree on ALL three metrics.
    
    Returns:
        DataFrame containing only rows where evidence_r1 == evidence_r2 AND
        suggestion_r1 == suggestion_r2 AND connection_r1 == connection_r2
    """
    mask = (
        (df['evidence_r1'] == df['evidence_r2']) &
        (df['suggestion_r1'] == df['suggestion_r2']) &
        (df['connection_r1'] == df['connection_r2'])
    )
    return df[mask].copy()


def binarize_evidence(score: int) -> int:
    """Binarize evidence score: 0-1 → 0 (low), 2-3 → 1 (high)."""
    return 0 if score <= 1 else 1


def is_binary_metric(metric_name: str, bin_evid: bool) -> bool:
    """Return whether a metric should be evaluated as binary.

    Args:
        metric_name: Base metric name (derived from `<metric_name>_llm` column).
        bin_evid: Whether the run treats evidence as binary (applies to plain `evidence`).

    Returns:
        True if the metric is binary, False otherwise.
    """
    name = metric_name.lower()
    if name in {"suggestion", "connection"}:
        return True
    if name.endswith("_bin"):
        return True
    if name.endswith("_ord"):
        return False
    if name == "evidence":
        return bin_evid
    # Default: treat unknown metrics as binary (conservative for judge-style scores).
    return True


def should_binarize_gt(metric_name: str, bin_evid: bool) -> bool:
    """Return whether the ground truth should be binarized using evidence mapping.

    Important: Only evidence labels should ever be binarized. Suggestion/Connection are
    already binary and must not be passed through the evidence binarizer.

    Args:
        metric_name: Base metric name (derived from `<metric_name>_llm` column).
        bin_evid: Whether the run treats evidence as binary (applies to plain `evidence`).

    Returns:
        True if the GT should be binarized with `binarize_evidence`.
    """
    name = metric_name.lower()
    if name.endswith("_bin"):
        return True
    if name == "evidence":
        return bin_evid
    return False


def compute_consensus(df: pd.DataFrame, bin_evid: bool = False) -> pd.DataFrame:
    """Compute consensus ground truth from r1 and r2 scores.

    Consensus = round((r1 + r2) / 2)
    - For evidence: 0, 1, 2, or 3 (or 0, 1 if bin_evid=True)
    - For suggestion/connection: 0 or 1

    Args:
        df: DataFrame with rater scores
        bin_evid: If True, binarize evidence (0-1 → 0, 2-3 → 1)
    """
    result = df.copy()

    # Evidence consensus (0-3, or 0-1 if binarized)
    mask = result['evidence_r1'].notna() & result['evidence_r2'].notna()
    evidence_consensus = np.round(
        (result.loc[mask, 'evidence_r1'] + result.loc[mask, 'evidence_r2']) / 2
    ).astype(int)
    if bin_evid:
        evidence_consensus = evidence_consensus.apply(binarize_evidence)
    result.loc[mask, 'evidence_gt'] = evidence_consensus

    # Suggestion consensus (0-1)
    mask = result['suggestion_r1'].notna() & result['suggestion_r2'].notna()
    result.loc[mask, 'suggestion_gt'] = np.round(
        (result.loc[mask, 'suggestion_r1'] + result.loc[mask, 'suggestion_r2']) / 2
    ).astype(int)

    # Connection consensus (0-1)
    mask = result['connection_r1'].notna() & result['connection_r2'].notna()
    result.loc[mask, 'connection_gt'] = np.round(
        (result.loc[mask, 'connection_r1'] + result.loc[mask, 'connection_r2']) / 2
    ).astype(int)

    return result


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    """Compute binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_score: Predicted scores for AUROC (optional)
    
    Returns:
        Dict with accuracy, precision, recall, f1, auroc
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # AUROC requires both classes present
    auroc = np.nan
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = np.nan
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUROC': auroc,
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None, labels=None) -> dict:
    """Compute multi-class classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Predicted probability scores for AUROC (optional, shape: n_samples x n_classes or n_samples for raw score)
        labels: List of class labels
    
    Returns:
        Dict with accuracy, macro/weighted precision, recall, f1, auroc
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Unweighted mean of the per-class scores (equal weight per class).
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=labels
    )

    # Mean of the per-class scores, weighted by the number of true examples.
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, labels=labels
    )
    
    # AUROC (one-vs-rest, macro average)
    auroc = np.nan
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            # For raw scores, we need to convert to pseudo-probabilities
            # Use one-vs-rest with the raw score
            auroc = roc_auc_score(
                y_true, y_score, multi_class='ovr', average='macro',
                labels=labels
            )
        except ValueError:
            auroc = np.nan
    
    return {
        'Accuracy': accuracy,
        PRECISION_MACRO: prec_macro,
        PRECISION_WEIGHTED: prec_weighted,
        RECALL_MACRO: rec_macro,
        RECALL_WEIGHTED: rec_weighted,
        F1_MACRO: f1_macro,
        F1_WEIGHTED: f1_weighted,
        'AUROC (OvR macro)': auroc,
    }


def print_table(title: str, metrics: dict, width: int = 50):
    """Print a formatted metrics table."""
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)
    
    max_key_len = max(len(k) for k in metrics.keys())
    for key, value in metrics.items():
        if isinstance(value, float):
            if np.isnan(value):
                val_str = "N/A"
            else:
                val_str = f"{value:.4f}"
        else:
            val_str = str(value)
        print(f"  {key:<{max_key_len}}  {val_str}")
    
    print("-" * width)


def print_confusion_matrix(title: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print()
    print(f"Confusion Matrix: {title}")
    print("(rows=true, cols=predicted)")
    print()
    
    # Header
    header = "      " + "  ".join(f"{l:>5}" for l in labels)
    print(header)
    print("      " + "-" * (len(labels) * 7))
    
    # Rows
    for i, label in enumerate(labels):
        row = f"  {label:>2} |" + "  ".join(f"{cm[i, j]:>5}" for j in range(len(labels)))
        print(row)
    print()


def get_metric_arrays(df, metric_name, gt_col, binarize_gt):
    """Extract y_true, y_pred, y_score arrays for a metric, filtering NaNs."""
    llm_col = f'{metric_name}_llm'
    llm_raw_col = f'{metric_name}_llm_raw'

    # Check if cols exist
    if llm_col not in df.columns:
        return np.array([]), np.array([]), np.array([])
    
    if gt_col not in df.columns:
         # Fallback for consensus: map evidence_ord -> evidence_gt, evidence_bin -> evidence_gt
         base_gt = gt_col.replace('_bin', '').replace('_ord', '')
         if base_gt in df.columns:
             gt_col = base_gt
         else:
             print(f"Warning: GT col {gt_col} not found")
             return np.array([]), np.array([]), np.array([])

    mask = df[gt_col].notna() & df[llm_col].notna()
    subset = df[mask]

    y_true = subset[gt_col].astype(int).values
    y_pred = subset[llm_col].astype(int).values
    y_score = subset[llm_raw_col].values if llm_raw_col in subset.columns else None

    if binarize_gt:
        y_true = np.array([binarize_evidence(v) for v in y_true])

    return y_true, y_pred, y_score


def analyze_metric(metric_name, y_true, y_pred, y_score, is_binary):
    """Analyze a single metric."""
    if len(y_true) == 0:
        return {}
        
    if not is_binary:
        metrics = compute_multiclass_metrics(y_true, y_pred, y_score=None, labels=[0, 1, 2, 3])
        metrics['N'] = len(y_true)
        print(FOUR_CLASS_NOTE)
        print_table(f"{metric_name.upper()} (4-class)", metrics)
        print_confusion_matrix(metric_name, y_true, y_pred, labels=[0, 1, 2, 3])
    else:
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        metrics['N'] = len(y_true)
        print_table(f"{metric_name.upper()} (binary)", metrics)
        labels = [0, 1]
        print_confusion_matrix(metric_name, y_true, y_pred, labels=labels)

    return metrics


def print_summary_table_dynamic(variant_name, metrics_by_metric):
    """Print a summary table for one variant dynamically."""
    print()
    print("=" * 110)
    print(f" SUMMARY: {variant_name}")
    print("=" * 110)
    print(
        "Prec/Recall/F1 are the true-count-weighted class average for 4-class "
        "metrics (weighted recall equals accuracy) and the positive class for "
        "binary metrics. Prec_mac/Recall_mac/F1_mac are the unweighted class average."
    )
    print(f"{'Metric':<15} {'N':>6} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'Prec_mac':>8} {'Recall_mac':>8} {'F1_mac':>8} {'AUROC':>8}")
    print("-" * 110)

    for metric_name, m in metrics_by_metric.items():
        if not m: continue
        
        # 4-class rows report the true-count-weighted average and the macro average.
        is_multiclass = PRECISION_WEIGHTED in m

        if is_multiclass:
            prec = m[PRECISION_WEIGHTED]
            rec = m[RECALL_WEIGHTED]
            f1 = m[F1_WEIGHTED]
            auroc_str = "N/A"
        else:
            prec = m['Precision']
            rec = m['Recall']
            f1 = m['F1']
            auroc_str = f"{m['AUROC']:.4f}" if not np.isnan(m['AUROC']) else "N/A"

        if is_multiclass:
            macro = (
                f"{m[PRECISION_MACRO]:>8.4f} {m[RECALL_MACRO]:>8.4f} {m[F1_MACRO]:>8.4f}"
            )
        else:
            macro = f"{'':>8} {'':>8} {'':>8}"
        print(f"{metric_name:<15} {m['N']:>6} {m['Accuracy']:>8.4f} "
              f"{prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {macro} {auroc_str:>8}")

    print("-" * 110)


def write_csv_summary(csv_path: Path, all_variant_metrics: dict):
    """Write a CSV with one row per variant and columns for each metric's scores."""
    
    # Determine all metric names present in the data (from the first variant)
    first_variant = list(all_variant_metrics.keys())[0]
    metric_names = list(all_variant_metrics[first_variant].keys())
    
    # Sort them (Evidence_Bin, Evidence_Ord, Suggestion, Connection)
    priority = {'evidence_bin': 0, 'evidence_ord': 1, 'suggestion': 2, 'connection': 3}
    metric_names.sort(key=lambda x: priority.get(x.lower(), 99))

    # Build header. Existing precision/recall/f1 columns stay the true-count-weighted
    # average for 4-class metrics. Macro columns are added beside them.
    header = ['variant']
    multiclass_metrics = set()
    for m in metric_names:
        sample = all_variant_metrics[first_variant].get(m, {})
        header.extend([f'{m}_N', f'{m}_accuracy', f'{m}_precision', f'{m}_recall', f'{m}_f1'])
        if PRECISION_WEIGHTED in sample:
            multiclass_metrics.add(m)
            header.extend([f'{m}_precision_macro', f'{m}_recall_macro', f'{m}_f1_macro'])

    rows = []
    for variant_name in ['r1', 'r2', 'both', 'consensus']:
        row = [variant_name]
        for m in metric_names:
            metrics = all_variant_metrics[variant_name].get(m, {})
            
            if not metrics:
                blank = 8 if m in multiclass_metrics else 5
                row.extend(['N/A'] * blank)
                continue

            is_multiclass = PRECISION_WEIGHTED in metrics

            if is_multiclass:
                prec = metrics[PRECISION_WEIGHTED]
                rec = metrics[RECALL_WEIGHTED]
                f1 = metrics[F1_WEIGHTED]
            else:
                prec = metrics['Precision']
                rec = metrics['Recall']
                f1 = metrics['F1']

            row.extend([
                metrics['N'],
                f"{metrics['Accuracy']:.4f}",
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
            ])
            if m in multiclass_metrics:
                row.extend([
                    f"{metrics[PRECISION_MACRO]:.4f}",
                    f"{metrics[RECALL_MACRO]:.4f}",
                    f"{metrics[F1_MACRO]:.4f}",
                ])
        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute classification metrics for LLM evaluator"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to evaluated_test_cases.json file"
    )
    parser.add_argument(
        "--bin-evid",
        action="store_true",
        help="Binarize evidence: 0-1 → 0 (low), 2-3 → 1 (high)"
    )
    args = parser.parse_args()

    # Load and prepare data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Setup Tee to write to both stdout and a file
    output_path = input_path.parent / "analysis_summary.txt"
    original_stdout = sys.stdout

    with open(output_path, "w") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            print(f"Loading data from {input_path}...")
            df = load_data(input_path)
            
            # Count how many actually have LLM scores (check any _llm col)
            llm_cols = [c for c in df.columns if c.endswith('_llm')]
            if llm_cols:
                # Use the first LLM column found to check completeness
                scored_mask = df[llm_cols[0]].notna()
                scored_count = scored_mask.sum()
            else:
                scored_count = 0
            
            total_rows = len(df)
            
            if scored_count < total_rows:
                print("\n" + "!" * 80)
                print("!!! WARNING: INCOMPLETE DATASET !!!")
                print(f"!!! Only {scored_count} out of {total_rows} samples have LLM evaluations.")
                print("!!! The metrics below represent only the successfully evaluated subset.")
                print("!" * 80 + "\n")
            
            total_samples = total_rows # Legacy variable name support
            print(f"Loaded {total_rows} total rows (Scored: {scored_count})")

            if args.bin_evid:
                print("Evidence Mode: Binarized (0-1 → 0 low, 2-3 → 1 high)")

            # Prepare consensus subset (all-three-agree)
            df_consensus = filter_full_agreement(df)
            print(f"Consensus subset: {len(df_consensus)} samples with full rater agreement "
                  f"({len(df_consensus)/total_samples*100:.1f}%)")

            # Run all 4 variants
            all_variant_metrics = {}
            for variant_name, variant_df in [
                ('r1', df),
                ('r2', df),
                ('both', df),
                ('consensus', df_consensus),
            ]:
                # Dynamic metric detection
                available_metrics = []
                for col in df.columns:
                    if col.endswith('_llm'):
                        metric_base = col[:-4] # strip _llm
                        if metric_base not in available_metrics:
                            available_metrics.append(metric_base)
                
                # Sort to ensure consistent order
                # Prefer: Evidence_Bin, Evidence_Ord, Suggestion, Connection
                priority = {'evidence_bin': 0, 'evidence_ord': 1, 'suggestion': 2, 'connection': 3}
                available_metrics.sort(key=lambda x: priority.get(x.lower(), 99))

                variant_metrics = {}
                print()
                print("#" * 70)
                print(f"# VARIANT: {variant_name}")
                print("#" * 70)
                
                for metric_name in available_metrics:
                    # Determine ground truth column
                    # Evidence_Bin -> evidence_r1 (binarized)
                    # Evidence_Ord -> evidence_r1 (raw)
                    # Suggestion -> suggestion_r1
                    
                    gt_base = metric_name.lower().replace('_bin', '').replace('_ord', '')
                    
                    if variant_name == 'consensus':
                        # Consensus subset: r1 == r2 across all metrics, so r1 is sufficient.
                        gt_col = f'{gt_base}_r1'
                    else:
                        if variant_name == 'both':
                             # Special handling inside get_metric_arrays
                             gt_col = gt_base 
                        else:
                             gt_col = f'{gt_base}_{variant_name}'

                    binary_metric = is_binary_metric(metric_name, args.bin_evid)
                    binarize_gt = should_binarize_gt(metric_name, args.bin_evid)
                    
                    if variant_name == 'both':
                        # Handle both r1 and r2
                        r1_col = f'{gt_base}_r1'
                        r2_col = f'{gt_base}_r2'
                        y_true_r1, y_pred_r1, y_score_r1 = get_metric_arrays(
                            variant_df, metric_name, r1_col, binarize_gt
                        )
                        y_true_r2, y_pred_r2, y_score_r2 = get_metric_arrays(
                            variant_df, metric_name, r2_col, binarize_gt
                        )
                        y_true = np.concatenate([y_true_r1, y_true_r2])
                        y_pred = np.concatenate([y_pred_r1, y_pred_r2])
                        if y_score_r1 is not None and y_score_r2 is not None:
                            y_score = np.concatenate([y_score_r1, y_score_r2])
                        else:
                            y_score = None
                    else:
                        y_true, y_pred, y_score = get_metric_arrays(
                            variant_df, metric_name, gt_col, binarize_gt
                        )
                    
                    # Analyze
                    variant_metrics[metric_name] = analyze_metric(
                        metric_name, y_true, y_pred, y_score, binary_metric
                    )

                all_variant_metrics[variant_name] = variant_metrics
                # Print summary table for this variant
                print_summary_table_dynamic(variant_name, variant_metrics)

            print()

            # Write CSV summary
            csv_path = input_path.parent / "analysis_results.csv"
            write_csv_summary(csv_path, all_variant_metrics)
            print(f"CSV summary saved to {csv_path}")

        finally:
            sys.stdout = original_stdout
            print(f"\nAnalysis summary saved to {output_path}")


if __name__ == "__main__":
    main()
