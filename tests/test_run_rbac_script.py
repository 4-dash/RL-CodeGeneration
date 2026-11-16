"""Exercise output preservation without Conda, models, GPUs, or benchmark code."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_rbac.sh"


class RunRbacScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rbac preservation ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        project = self.root / "project with spaces"
        (project / "eval").mkdir(parents=True)
        (project / "eval/rbac.py").touch()
        benchmark = self.root / "benchmark with spaces"
        evaluator = benchmark / "rbac-exp/evaluation/evaluate_protocol_v2.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.touch()
        for dataset in ("spider", "bird"):
            data = benchmark / f"data/selected/{dataset}/column_level_rbac_dataset_{dataset}_v3_no_sm.json"
            data.parent.mkdir(parents=True)
            data.write_text("[{}]")
        self.databases = self.root / "databases"
        self.databases.mkdir()
        self.results = self.root / "results"
        self.log = self.root / "calls.jsonl"
        conda = self.root / "mock-conda"
        conda.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["MOCK_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if "-m" in args:
    out = pathlib.Path(args[args.index("--output-dir") + 1])
    (out / "predictions.jsonl").write_text('{"prediction": "SELECT 1"}\\n')
    (out / "role_data.json").write_text('[{"input": "test"}]')
    if os.environ.get("MOCK_FAIL") == "generation":
        sys.exit(17)
else:
    assert pathlib.Path(args[args.index("--role_json") + 1]).is_file()
    assert pathlib.Path(args[args.index("--prediction_path") + 1]).is_file()
    out = pathlib.Path(args[args.index("--output_dir") + 1])
    (out / "metrics.json").write_text('{"score": 1}')
    if os.environ.get("MOCK_FAIL") == "evaluation":
        sys.exit(19)
''')
        conda.chmod(0o755)
        # Deliberately freeze timestamps: uniqueness must come from reservation.
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        date = bin_dir / "date"
        date.write_text("#!/bin/sh\nprintf '%s\\n' 20260101T000000Z\n")
        date.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("RBAC_") and key != "MOCK_FAIL"}
        self.env.update(
            PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
            CONDA_EXE=str(conda), RBAC_PROJECT_DIR=str(project),
            RBAC_BENCHMARK_DIR=str(benchmark), RBAC_RESULTS_DIR=str(self.results),
            RBAC_SPIDER_DB_DIR=str(self.databases), RBAC_BIRD_DB_DIR=str(self.databases),
            MOCK_CALLS=str(self.log),
        )

    def run_script(self, mode="all", expected=0, **settings):
        result = subprocess.run(
            ["bash", str(SCRIPT), mode], env=dict(self.env, **settings),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def latest(self):
        return Path((self.results / "latest_generation.txt").read_text().strip())

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    @staticmethod
    def snapshot(directory):
        return {path.relative_to(directory): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def assert_preserved(self, directory, original):
        for relative, contents in original.items():
            self.assertEqual((directory / relative).read_bytes(), contents, str(relative))

    def test_repeated_runs_preserve_all_old_files_with_identical_timestamps(self):
        self.run_script()
        first = self.latest()
        original = self.snapshot(first)
        self.run_script()
        second = self.latest()
        self.assertNotEqual(first, second)
        self.assert_preserved(first, original)
        for dataset in ("spider", "bird"):
            self.assertTrue((second / dataset / "predictions.jsonl").is_file())

    def test_evaluation_reuses_predictions_and_preserves_previous_scores(self):
        self.run_script()
        run = self.latest()
        original = self.snapshot(run)
        self.run_script("evaluate")
        self.assertEqual(len(self.calls()), 6)  # No second generation.
        self.assertTrue(all("-m" not in args for args in self.calls()[-2:]))
        self.assert_preserved(run, original)
        self.assertEqual(self.latest(), run)
        for dataset in ("spider", "bird"):
            self.assertEqual(len(list((run / dataset / "evaluations").iterdir())), 2)

    def test_explicit_older_run_and_recorded_single_dataset(self):
        self.run_script("generate", RBAC_DATASETS="bird")
        older = self.latest()
        self.run_script("generate")
        newer = self.latest()
        self.run_script("evaluate", RBAC_OUTPUT_DIR=str(older))
        args = self.calls()[-1]
        self.assertEqual(args[args.index("--dataset") + 1], "bird")
        self.assertEqual(self.latest(), newer)

    def test_existing_explicit_directory_is_never_used_for_generation(self):
        existing = self.root / "legacy results"
        existing.mkdir()
        (existing / "predictions.jsonl").write_text("valuable old predictions")
        (existing / "generation.log").write_text("valuable old log")
        original = self.snapshot(existing)
        result = self.run_script("generate", expected=1, RBAC_OUTPUT_DIR=str(existing))
        self.assertIn("requires a NEW directory", result.stderr)
        self.assertEqual(self.snapshot(existing), original)
        self.assertFalse(self.log.exists())

    def test_failed_generation_retains_partial_files_without_changing_latest(self):
        self.run_script("generate")
        previous = self.latest()
        failed = self.root / "failed generation"
        self.run_script("generate", expected=17, RBAC_OUTPUT_DIR=str(failed), MOCK_FAIL="generation")
        self.assertEqual(self.latest(), previous)
        self.assertTrue((failed / "spider/predictions.jsonl").is_file())
        self.assertTrue((failed / "spider/generation.log").is_file())

    def test_failed_evaluation_can_be_retried_without_regeneration(self):
        self.run_script(expected=19, MOCK_FAIL="evaluation")
        run = self.latest()  # Both datasets finished generation before evaluation.
        original = self.snapshot(run)
        self.run_script("evaluate")
        self.assert_preserved(run, original)
        self.assertEqual(sum("-m" in args for args in self.calls()), 2)

    def test_legacy_flat_predictions_can_be_reevaluated(self):
        legacy = self.root / "old flat results"
        legacy.mkdir()
        (legacy / "predictions.jsonl").write_text('{"prediction": "SELECT 1"}\n')
        (legacy / "role_data.json").write_text("[{}]")
        (legacy / "evaluation").mkdir()
        (legacy / "evaluation/metrics.json").write_text("old scores")
        original = self.snapshot(legacy)
        self.run_script("evaluate", RBAC_OUTPUT_DIR=str(legacy), RBAC_DATASETS="spider")
        self.assert_preserved(legacy, original)
        self.assertEqual(len(list((legacy / "evaluations").iterdir())), 1)

    def test_two_generations_competing_for_one_directory_do_not_overwrite(self):
        output = self.root / "shared output"
        env = dict(self.env, RBAC_OUTPUT_DIR=str(output), RBAC_DATASETS="spider")
        processes = [subprocess.Popen(["bash", str(SCRIPT), "generate"], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        output_logs = [process.communicate(timeout=20) for process in processes]
        self.assertEqual(sorted(process.returncode for process in processes), [0, 1], output_logs)
        self.assertEqual(len(self.calls()), 1)
        self.assertTrue((output / "spider/predictions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
