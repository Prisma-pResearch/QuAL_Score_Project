"""Run one paper configuration from configs/paper_runs.yaml.

The command always uses the one OpenAI-compatible client. That client
defaults to the local vLLM server started by scripts/serve_vllm.sh
(LLM_BASE_URL http://127.0.0.1:8000/v1, LLM_API_KEY dummy). ``--max-tokens``
defaults to 2048.

The four main 2x2 runs were produced on a hosted gateway. The temperature
and in-context runs used self-hosted vLLM. This command calls the local server.

Each run's ``seed`` is the bootstrap base. This command prints that seed.

This does not pick a data file for you. Pass --input. The development and
test JSON files are not in this folder.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_TOKENS = 2048
# These four paper runs used a hosted gateway. The command still calls local vLLM.
HOSTED_GATEWAY_RUNS = frozenset(
    {"criteria_ordinal", "criteria_binary", "steps_ordinal", "steps_binary"}
)


def load_config(path: Path) -> dict:
    """Load the paper-run YAML.

    Args:
        path: Config path.

    Returns:
        Parsed mapping with ``defaults`` and ``runs``.
    """
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "runs" not in config:
        raise ValueError(f"Config needs a runs list: {path}")
    return config


def find_run(config: dict, run_id: str) -> dict:
    """Return one run entry.

    Args:
        config: Parsed config.
        run_id: ``id`` field to find.

    Returns:
        The run mapping.

    Raises:
        ValueError: If the id is missing.
    """
    for run in config["runs"]:
        if run["id"] == run_id:
            return run
    known = ", ".join(run["id"] for run in config["runs"])
    raise ValueError(f"Unknown run {run_id}. Known: {known}")


def build_command(run: dict, args: argparse.Namespace, defaults: dict) -> list:
    """Build the run_judge.py command for one config entry.

    Args:
        run: One item from the runs list.
        args: CLI arguments.
        defaults: Shared settings from the config.

    Returns:
        Argument vector.
    """
    command = [
        sys.executable,
        str(ROOT / "run_judge.py"),
        "--input",
        args.input,
        "--prompt",
        run["prompt"],
        "--temperature",
        str(run["temperature"]),
        "--max-tokens",
        str(defaults.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "--max-concurrent",
        str(defaults.get("max_concurrent", 100)),
        "--max-retries",
        str(defaults.get("max_retries", 10)),
        "--retry-sleep",
        str(defaults.get("retry_sleep_seconds", 60)),
    ]
    if run.get("bin_evidence"):
        command.append("--bin-evidence")
    if args.examples:
        command.extend(["--examples", args.examples])
    if args.run_dir:
        command.extend(["--run-dir", args.run_dir])
    else:
        command.extend(["--run-root", args.run_root])
    if args.skip_analysis:
        command.append("--skip-analysis")
    return command


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run one paper config against the local vLLM server."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_runs.yaml"))
    parser.add_argument("--run", required=True, help="Run id from the config")
    parser.add_argument("--input", help="Comments JSON. Required unless --dry-run.")
    parser.add_argument("--examples", help="Development JSON for the in-context run")
    parser.add_argument("--run-dir", help="Write into this folder")
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "runs"),
        help="Parent directory for new run folders. Default: runs/ in this folder.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
        run = find_run(config, args.run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if run.get("examples_required") and not args.examples and not args.dry_run:
        print(f"{run['id']} needs --examples (development JSON). It is not bundled.", file=sys.stderr)
        sys.exit(1)
    if not args.input and not args.dry_run:
        print("--input is required", file=sys.stderr)
        sys.exit(1)
    if args.dry_run and not args.input:
        args.input = "<input.json>"

    defaults = config.get("defaults") or {}
    command = build_command(run, args, defaults)
    print(
        "Client: local vLLM from scripts/serve_vllm.sh. "
        f"LLM_BASE_URL defaults to {DEFAULT_LOCAL_BASE_URL}."
    )
    if run["id"] in HOSTED_GATEWAY_RUNS:
        print(
            f"{run['id']} was produced via a hosted OpenAI-compatible gateway. "
            "This command calls the local vLLM server."
        )
    if run.get("seed") is not None:
        print(f"Bootstrap seed: {run['seed']}")
    print(" ".join(command))
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
