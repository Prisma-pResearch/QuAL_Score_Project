"""Score feedback comments with gpt-oss-120b, one GEval call per QuAL dimension.

Reads a JSON list of comments, runs the chosen prompt, and writes a run
folder. The folder gets the evaluated JSON, the config, the system prompt
that was sent, and a copy of the prompt module.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from judges import DEFAULT_MAX_TOKENS, convert_score, get_judge, get_metrics_config
from judges.system_prompt import FEEDBACK_EVALUATION_SYSTEM_PROMPT
from prompts import PROMPT_IDS, load_prompt

MAX_RETRIES = 10
RETRY_SLEEP_SECONDS = 60
MAX_CONCURRENT = 100
THROTTLE_SECONDS = 0.0
TASK_TIMEOUT_SECONDS = "600"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PACKAGE_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Score feedback comments with a QuAL LLM judge")
    parser.add_argument("--input", required=True, help="JSON file of comments to score")
    parser.add_argument("--prompt", required=True, choices=PROMPT_IDS, help="Prompt setup to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Completion token cap. Default 2048.",
    )
    parser.add_argument(
        "--bin-evidence",
        action="store_true",
        help="Map a metric named evidence from the 0-1 model score with the binary threshold. "
        "Ground-truth binarization (0-1 -> 0, 2-3 -> 1) happens later, in the metrics script.",
    )
    parser.add_argument("--metrics", nargs="*", help="Metric names to run. Default: every metric in the prompt.")
    parser.add_argument("--examples", help="Development JSON used by in_context_binary")
    parser.add_argument("--output", help="Evaluated JSON path. Default: a new folder under --run-root.")
    parser.add_argument("--run-dir", help="Write into this folder instead of creating one")
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Parent directory for new run folders. Default: runs/ in this folder.",
    )
    parser.add_argument("--note", default="", help="Short note stored in config.yaml")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=RETRY_SLEEP_SECONDS)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--skip-analysis", action="store_true", help="Do not run classification metrics afterward")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON list of comment records.

    Args:
        path: Input JSON path.

    Returns:
        Records as dicts.

    Raises:
        ValueError: If the file is not a list or a record has no id.
    """
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Input must be a JSON list: {path}")
    missing = [index for index, record in enumerate(records) if "id" not in record]
    if missing:
        raise ValueError(f"Records missing id at positions: {missing[:10]}")
    return records


def normalize_context(value: Any) -> List[str]:
    """Turn a record's context field into a list of strings.

    Args:
        value: None, a string, or a list.

    Returns:
        Non-empty strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        items = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def build_test_cases(records: List[Dict[str, Any]], shared_context: List[str]) -> List[LLMTestCase]:
    """Build DeepEval cases. The record id is stored for later matching.

    Args:
        records: Input records.
        shared_context: Extra context appended to every case.

    Returns:
        LLM test cases.
    """
    cases = []
    for record in records:
        context_items = normalize_context(record.get("context"))
        context_items.extend(shared_context)
        kwargs: Dict[str, Any] = {
            "input": record.get("input", ""),
            "actual_output": record.get("actual_output", ""),
            "additional_metadata": {"id": record["id"]},
        }
        if context_items:
            kwargs["context"] = context_items
        cases.append(LLMTestCase(**kwargs))
    return cases


def choose_metrics(prompt_module: Any, judge: Any, requested: Optional[List[str]]) -> List[Any]:
    """Build metrics and optionally keep a subset.

    Args:
        prompt_module: Prompt module with ``get_feedback_metrics``.
        judge: DeepEval judge.
        requested: Metric names, or None for all.

    Returns:
        Metrics to run.

    Raises:
        ValueError: If a requested name is not in the prompt.
    """
    all_metrics = prompt_module.get_feedback_metrics(model=judge)
    if not requested:
        return all_metrics
    by_name = {metric.name.lower(): metric for metric in all_metrics}
    missing = [name for name in requested if name.lower() not in by_name]
    if missing:
        available = ", ".join(metric.name for metric in all_metrics)
        raise ValueError(f"Unknown metrics: {', '.join(missing)}. Available: {available}")
    return [by_name[name.lower()] for name in requested]


def create_run_dir(run_root: Path, prompt_id: str, model_name: str, bin_evidence: bool) -> Path:
    """Create a timestamped run directory.

    Args:
        run_root: Parent directory.
        prompt_id: Prompt id, used in the folder name.
        model_name: Served model name, used in the folder name.
        bin_evidence: Whether binary evidence mapping is on.

    Returns:
        New directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    parts = [timestamp, prompt_id, safe_model]
    if bin_evidence:
        parts.append("bin")
    folder = run_root / "_".join(parts)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def resolve_output_dir(args: argparse.Namespace, model_name: str) -> tuple[Path, Path]:
    """Pick the run directory and the evaluated JSON path.

    Args:
        args: Parsed arguments.
        model_name: Served model name, used when creating a new folder.

    Returns:
        Run directory and output JSON path.
    """
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(args.output) if args.output else run_dir / "evaluated_test_cases.json"
        return run_dir, output_path
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.parent, output_path
    run_dir = create_run_dir(Path(args.run_root), args.prompt, model_name, args.bin_evidence)
    return run_dir, run_dir / "evaluated_test_cases.json"


def save_run_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    input_path: Path,
    prompt_module: Any,
    judge: Any,
    metrics: List[Any],
) -> None:
    """Write config, prompts, and a copy of the input next to the results.

    The config records paths and settings. It does not record API keys.

    Args:
        run_dir: Destination folder.
        args: Parsed arguments.
        input_path: Original JSON path.
        prompt_module: Prompt module that was loaded.
        judge: Judge instance.
        metrics: Metrics that were run.
    """
    config = {
        "input": str(input_path),
        "prompt": args.prompt,
        "model_name": judge.get_model_name(),
        "base_url": judge.base_url,
        "max_tokens": judge.max_tokens,
        "temperature": judge.temperature,
        "bin_evidence": args.bin_evidence,
        "metrics": [metric.name for metric in metrics],
        "examples": args.examples,
        "note": args.note,
        "max_concurrent": args.max_concurrent,
        "max_retries": args.max_retries,
        "command_line": " ".join(sys.argv),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    system_prompt = getattr(judge, "system_prompt", FEEDBACK_EVALUATION_SYSTEM_PROMPT)
    (run_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    prompt_path = Path(prompt_module.__file__)
    shutil.copy(prompt_path, run_dir / "criteria.py")
    shutil.copy(input_path, run_dir / "input_test_cases.json")

    metrics_config = get_metrics_config(metrics)
    metrics_config["judge"] = judge.get_model_name()
    metrics_config["bin_evidence"] = args.bin_evidence
    with (run_dir / "metrics_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_config, handle, indent=2)


def evaluate_with_retries(
    cases: List[LLMTestCase],
    metrics: List[Any],
    max_retries: int,
    retry_sleep: float,
    max_concurrent: int,
) -> Dict[Any, Any]:
    """Run DeepEval, retrying cases whose metrics returned an error.

    Args:
        cases: Test cases.
        metrics: Metrics.
        max_retries: Attempts, including the first.
        retry_sleep: Seconds to wait between attempts.
        max_concurrent: DeepEval concurrency cap.

    Returns:
        Map from record id to a successful test result.
    """
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = TASK_TIMEOUT_SECONDS
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = TASK_TIMEOUT_SECONDS
    pending = {case.additional_metadata["id"] for case in cases}
    results: Dict[Any, Any] = {}
    async_config = AsyncConfig(
        run_async=True,
        throttle_value=THROTTLE_SECONDS,
        max_concurrent=max_concurrent,
    )
    started = time.time()

    for attempt in range(max_retries):
        if not pending:
            break
        if attempt > 0:
            print(f"Retry {attempt}/{max_retries - 1} for {len(pending)} cases. Waiting {retry_sleep:.0f}s.")
            time.sleep(retry_sleep)
        batch = [case for case in cases if case.additional_metadata["id"] in pending]
        try:
            evaluation = evaluate(
                test_cases=batch,
                metrics=metrics,
                async_config=async_config,
                error_config=ErrorConfig(ignore_errors=True),
            )
        except Exception as exc:
            print(f"Evaluation batch failed: {exc}")
            traceback.print_exc()
            continue
        for test_result in evaluation.test_results:
            record_id = test_result.additional_metadata.get("id")
            if record_id is None or not test_result.metrics_data:
                continue
            errors = [item for item in test_result.metrics_data if getattr(item, "error", None)]
            if errors:
                for item in errors:
                    print(f"Case {record_id}: {item.name} failed: {item.error}")
                continue
            results[record_id] = test_result
            pending.discard(record_id)

    elapsed = time.time() - started
    print(f"Scored {len(results)}/{len(cases)} cases in {elapsed:.1f}s.")
    if pending:
        print(f"Incomplete run. Missing ids: {sorted(pending, key=str)}")
    return results


def merge_scores(
    records: List[Dict[str, Any]],
    results: Dict[Any, Any],
    binarize_evidence: bool,
) -> None:
    """Write ``<metric>_llm`` fields back onto the input records.

    Args:
        records: Input records, edited in place.
        results: Successful DeepEval results keyed by id.
        binarize_evidence: Passed through to ``convert_score``.
    """
    by_id = {record["id"]: record for record in records}
    for record_id, test_result in results.items():
        record = by_id.get(record_id)
        if record is None:
            print(f"Warning: result for unknown id {record_id}")
            continue
        for metric_data in test_result.metrics_data:
            metric_name = metric_data.name.lower().split("[")[0].strip()
            normalized = metric_data.score
            record[f"{metric_name}_llm"] = convert_score(metric_name, normalized, binarize_evidence)
            record[f"{metric_name}_llm_raw"] = normalized
            record[f"{metric_name}_llm_reason"] = metric_data.reason or ""


def print_averages(records: List[Dict[str, Any]], metrics: List[Any]) -> None:
    """Print the mean integer score for each metric.

    Args:
        records: Scored records.
        metrics: Metrics that were requested.
    """
    print("Averages:")
    for metric in metrics:
        name = metric.name.lower()
        scores = [record[f"{name}_llm"] for record in records if f"{name}_llm" in record]
        if not scores:
            print(f"  {name}: no scores")
            continue
        print(f"  {name}: {sum(scores) / len(scores):.2f} (n={len(scores)})")


def run_analysis(output_path: Path, bin_evidence: bool) -> None:
    """Run classification metrics on the file just written.

    Args:
        output_path: Evaluated JSON.
        bin_evidence: Forwarded to the metrics script.
    """
    from evaluation.classification_metrics import main as metrics_main

    argv = ["classification_metrics.py", "--input", str(output_path)]
    if bin_evidence:
        argv.append("--bin-evid")
    previous = sys.argv
    sys.argv = argv
    try:
        metrics_main()
    finally:
        sys.argv = previous


def main() -> None:
    """Score the input file and write a run folder."""
    load_dotenv(override=False)
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    prompt_module = load_prompt(args.prompt)
    try:
        shared_context = prompt_module.load_shared_context(args.examples)
    except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        print(f"Could not load examples: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.max_tokens < 1:
        print("--max-tokens must be at least 1", file=sys.stderr)
        sys.exit(1)
    try:
        judge = get_judge(temperature=args.temperature, max_tokens=args.max_tokens)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    try:
        metrics = choose_metrics(prompt_module, judge, args.metrics)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    records = load_records(input_path)
    cases = build_test_cases(records, shared_context)
    run_dir, output_path = resolve_output_dir(args, judge.get_model_name())
    print(
        f"Judge {judge.get_model_name()} base_url={judge.base_url} "
        f"temperature={judge.temperature} max_tokens={judge.max_tokens}"
    )
    print(f"Prompt {args.prompt}: {', '.join(metric.name for metric in metrics)}")
    print(f"Writing {output_path}")
    save_run_metadata(run_dir, args, input_path, prompt_module, judge, metrics)

    results = evaluate_with_retries(
        cases,
        metrics,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        max_concurrent=args.max_concurrent,
    )
    merge_scores(records, results, binarize_evidence=args.bin_evidence)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print_averages(records, metrics)

    if not args.skip_analysis:
        run_analysis(output_path, args.bin_evidence)


if __name__ == "__main__":
    main()
