"""Import and metric checks. Does not call a model API."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from deepeval.models import DeepEvalBaseLLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judges.scores import convert_score, normalized_to_binary_score, normalized_to_evidence_score
from prompts import PROMPT_IDS, load_prompt


class DummyJudge(DeepEvalBaseLLM):
    """Judge that records calls and never sends requests to the network."""

    def __init__(self) -> None:
        self.calls = 0
        self.temperature = 0.0
        self.system_prompt = "unused"

    def load_model(self) -> "DummyJudge":
        return self

    def generate(self, prompt: str, *args: object, **kwargs: object) -> str:
        del prompt, args, kwargs
        self.calls += 1
        raise AssertionError("generate was called")

    async def a_generate(self, prompt: str, *args: object, **kwargs: object) -> str:
        del prompt, args, kwargs
        self.calls += 1
        raise AssertionError("a_generate was called")

    def get_model_name(self) -> str:
        return "dummy"


def check_scores() -> None:
    """Check the 0-3 and 0/1 cut points."""
    assert normalized_to_evidence_score(0.0) == 0
    assert normalized_to_evidence_score(0.10) == 0
    assert normalized_to_evidence_score(0.11) == 1
    assert normalized_to_evidence_score(0.35) == 1
    assert normalized_to_evidence_score(0.36) == 2
    assert normalized_to_evidence_score(0.65) == 2
    assert normalized_to_evidence_score(0.66) == 3
    assert normalized_to_binary_score(0.49) == 0
    assert normalized_to_binary_score(0.5) == 1
    assert convert_score("evidence_ord", 0.8) == 3
    assert convert_score("evidence", 0.8, binarize_evidence=True) == 1
    assert convert_score("suggestion", 1.0) == 1


def check_prompts() -> None:
    """Build every prompt with a dummy model and check the snapshot choices."""
    for prompt_id in PROMPT_IDS:
        module = load_prompt(prompt_id)
        judge = DummyJudge()
        metrics = module.get_feedback_metrics(model=judge)
        assert metrics, prompt_id
        assert judge.calls == 0, prompt_id
        names = [metric.name for metric in metrics]
        if prompt_id == "eval_steps_combo":
            assert names == ["Evidence_Bin", "Evidence_Ord", "Suggestion", "Connection"]
        else:
            assert names == ["Evidence", "Suggestion", "Connection"]
        if prompt_id == "criteria_only_binary":
            assert metrics[0].criteria.startswith("SYSTEM_PROMPT:\n")
            assert metrics[0].evaluation_steps in (None, [])
        if prompt_id == "criteria_only_ordinal":
            assert not str(metrics[0].criteria).startswith("SYSTEM_PROMPT:")
            assert metrics[0].evaluation_steps in (None, [])
        if prompt_id == "eval_steps_binary":
            assert metrics[0].criteria in (None, "")
            assert len(metrics[0].evaluation_steps) == 3
        if prompt_id == "in_context_binary":
            context = module.load_shared_context(ROOT / "data" / "example_input.json")
            assert len(context) == 2
            payload = json.loads(context[1])
            assert len(payload) == 5
            assert "foam-model" in payload[2]["feedback"]

    ordinal = load_prompt("criteria_only_ordinal")
    assert ordinal.load_shared_context(None) == []


def write_scored_example(path: Path) -> None:
    """Add fake model scores to the synthetic comments."""
    records = json.loads((ROOT / "data" / "example_input.json").read_text(encoding="utf-8"))
    for record in records:
        record["evidence_llm"] = record["evidence_r1"]
        record["evidence_llm_raw"] = record["evidence_r1"] / 3
        record["evidence_llm_reason"] = "synthetic"
        record["suggestion_llm"] = record["suggestion_r1"]
        record["suggestion_llm_raw"] = float(record["suggestion_r1"])
        record["suggestion_llm_reason"] = "synthetic"
        record["connection_llm"] = int(record["connection_r1"])
        record["connection_llm_raw"] = float(record["connection_r1"])
        record["connection_llm_reason"] = "synthetic"
    path.write_text(json.dumps(records), encoding="utf-8")


def check_metrics_and_figures() -> None:
    """Run classification, a short bootstrap, and the figure script."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        scored = folder / "evaluated_test_cases.json"
        write_scored_example(scored)
        subprocess.run(
            [sys.executable, "-m", "evaluation.classification_metrics", "--input", str(scored)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.bootstrap_ci",
                "--input",
                str(scored),
                "--draws",
                "20",
                "--seed",
                "1",
            ],
            cwd=ROOT,
            check=True,
        )
        results_csv = folder / "analysis_results.csv"
        assert results_csv.is_file()
        assert (folder / "analysis_ci.csv").is_file()
        header = results_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "evidence_precision_macro" in header
        assert "evidence_recall_macro" in header
        assert "evidence_f1_macro" in header
        assert "suggestion_precision_macro" not in header
        rows = results_csv.read_text(encoding="utf-8").splitlines()[1:]
        r1 = rows[0].split(",")
        # Weighted recall for 4-class evidence equals accuracy.
        accuracy = r1[header.index("evidence_accuracy")]
        recall = r1[header.index("evidence_recall")]
        assert accuracy == recall
        # The combo figure needs two runs. Reuse this one for both sides.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.figures",
                "combo",
                "--ordinal-csv",
                str(folder / "analysis_results.csv"),
                "--binary-csv",
                str(folder / "analysis_results.csv"),
                "--ordinal-ci",
                str(folder / "analysis_ci.csv"),
                "--binary-ci",
                str(folder / "analysis_ci.csv"),
                "--title",
                "synthetic",
                "--output",
                str(folder / "combo.png"),
            ],
            cwd=ROOT,
            check=True,
        )
        assert (folder / "combo.png").is_file()
        points = folder / "temp_points.csv"
        ci = folder / "temp_ci.csv"
        points.write_text(
            "variant,evidence_ord_accuracy,evidence_ord_f1,evidence_ord_precision,evidence_ord_recall,"
            "evidence_bin_accuracy,evidence_bin_f1,evidence_bin_precision,evidence_bin_recall,"
            "suggestion_accuracy,suggestion_f1,suggestion_precision,suggestion_recall,"
            "connection_accuracy,connection_f1,connection_precision,connection_recall\n"
            "consensus,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.9,0.9,0.9,0.9,0.7,0.7,0.7,0.7\n",
            encoding="utf-8",
        )
        ci.write_text(
            "variant,metric,accuracy_low,accuracy_high,f1_low,f1_high,"
            "precision_low,precision_high,recall_low,recall_high\n"
            "consensus,evidence_ord,0.3,0.5,0.3,0.5,0.3,0.5,0.3,0.5\n"
            "consensus,evidence_bin,0.7,0.9,0.7,0.9,0.7,0.9,0.7,0.9\n"
            "consensus,suggestion,0.8,1.0,0.8,1.0,0.8,1.0,0.8,1.0\n"
            "consensus,connection,0.6,0.8,0.6,0.8,0.6,0.8,0.6,0.8\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.figures",
                "temperature",
                "--label",
                "T=0.0",
                "--points",
                str(points),
                "--ci",
                str(ci),
                "--label",
                "T=1.0",
                "--points",
                str(points),
                "--ci",
                str(ci),
                "--variant",
                "consensus",
                "--output",
                str(folder / "temperature.png"),
            ],
            cwd=ROOT,
            check=True,
        )
        assert (folder / "temperature.png").is_file()
        mismatch = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.figures",
                "temperature",
                "--label",
                "T=0.0",
                "--points",
                str(points),
                "--ci",
                str(ci),
                "--label",
                "T=1.0",
                "--output",
                str(folder / "mismatch.png"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert mismatch.returncode != 0
        assert "2 labels, 1 point files, 1 CI files" in mismatch.stderr
        help_text = subprocess.run(
            [sys.executable, "-m", "evaluation.figures", "temperature", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--label" in help_text.stdout
        assert "--points" in help_text.stdout
        assert "--ci" in help_text.stdout
        assert "::" not in help_text.stdout


def check_default_client() -> None:
    """The unset client points at local vLLM and does not rewrite a supplied URL."""
    import os

    from judges.openai_compatible import (
        DEFAULT_API_KEY,
        DEFAULT_BASE_URL,
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_NAME,
        openai_compatible_judge,
    )

    keys = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        judge = openai_compatible_judge()
        assert judge.base_url == DEFAULT_BASE_URL == "http://127.0.0.1:8000/v1"
        assert judge.api_key == DEFAULT_API_KEY == "dummy"
        assert judge.max_tokens == DEFAULT_MAX_TOKENS == 2048
        assert judge.get_model_name() == DEFAULT_MODEL_NAME == "gpt-oss-120b"
        os.environ["LLM_BASE_URL"] = "http://example.test"
        os.environ["LLM_API_KEY"] = "local-test"
        supplied = openai_compatible_judge(max_tokens=512)
        assert supplied.base_url == "http://example.test"
        assert supplied.max_tokens == 512
        assert not supplied.base_url.endswith("/v1")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_dry_run() -> None:
    """The reproduce script should print a local-vLLM command and not start a judge."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--run", "criteria_ordinal", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run_judge.py" in completed.stdout
    assert "criteria_ordinal" in completed.stdout
    assert "criteria_only_ordinal" in completed.stdout
    assert "scripts/serve_vllm.sh" in completed.stdout
    assert "http://127.0.0.1:8000/v1" in completed.stdout
    assert "--max-tokens 2048" in completed.stdout
    assert "hosted OpenAI-compatible gateway" in completed.stdout
    assert "Bootstrap seed: 42" in completed.stdout
    assert "4096" not in completed.stdout
    assert "--judge" not in completed.stdout
    assert "sk-" not in completed.stdout
    temp = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--run", "temp_0_0", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--max-tokens 2048" in temp.stdout
    assert "local vLLM" in temp.stdout
    assert "Bootstrap seed: 40042" in temp.stdout
    assert "does not reproduce" not in temp.stdout
    assert "sk-" not in temp.stdout
    check_bootstrap_seeds()


def check_bootstrap_seeds() -> None:
    """Published runs store a base seed, and the bootstrap CLI takes that seed."""
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "paper_runs.yaml").read_text(encoding="utf-8"))
    expected = {
        "criteria_ordinal": 42,
        "criteria_binary": 10042,
        "steps_ordinal": 20042,
        "steps_binary": 30042,
        "temp_0_0": 40042,
        "temp_0_5": 50042,
        "temp_1_0": 60042,
        "in_context_steps": 42,
    }
    found = {run["id"]: run["seed"] for run in config["runs"]}
    assert found == expected
    raw = (ROOT / "configs" / "paper_runs.yaml").read_text(encoding="utf-8")
    assert "ci_run_index" not in raw
    help_text = subprocess.run(
        [sys.executable, "-m", "evaluation.bootstrap_ci", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--seed" in help_text.stdout
    assert "--run-index" not in help_text.stdout


def main() -> None:
    """Run the checks."""
    check_scores()
    check_prompts()
    check_default_client()
    check_metrics_and_figures()
    check_dry_run()
    print("smoke test ok")


if __name__ == "__main__":
    main()
