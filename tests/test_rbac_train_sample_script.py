"""Exercise subset selection and launch isolation without models or GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "run_rbac_train_sample.sh"


class SampleTrainingScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.benchmark = self.root / "benchmark"
        policy = self.benchmark / "rbac-exp/evaluation/prediction_policy.py"
        policy.parent.mkdir(parents=True)
        policy.touch()
        self.database = self.benchmark / "data/spider/database/demo/demo.sqlite"
        self.database.parent.mkdir(parents=True)
        self.database.touch()
        self.rows = [
            {"id": i, "db_id": "demo", "policy": {"t": ["x"]},
             "metadata": {"permission": "allowed" if i % 2 else "denied"}}
            for i in range(300)
        ]
        self.source = self.benchmark / "data/selected/spider/column_level_rbac_dataset_spider_train_v3_no_sm.json"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(json.dumps(self.rows))
        binary = self.root / "bin/accelerate"
        binary.parent.mkdir()
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "sample = pathlib.Path(os.environ['RBAC_TRAIN_DATA'])\n"
            "(sample.parent / 'launch.json').write_text(json.dumps({"
            "'args': sys.argv[1:], 'sample': str(sample), "
            "'wandb_mode': os.environ['WANDB_MODE']}))\n"
            "print('fake training output', flush=True)\n"
            "sys.exit(int(os.environ.get('SAMPLE_TEST_EXIT', '0')))\n"
        )
        binary.chmod(0o755)
        self.runs = self.root / "runs"
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith('RBAC_') and key != 'WANDB_MODE'
        }
        self.env.update({
            "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
            "RBAC_PYTHON": sys.executable,
            "RBAC_BENCHMARK_DIR": str(self.benchmark),
            "RBAC_TRAIN_RUNS_DIR": str(self.runs),
        })

    def run_script(self, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT)], cwd=self.root,
            env={**self.env, **overrides}, text=True, capture_output=True,
        )

    def test_reproducible_subset_and_separate_run_outputs(self):
        for _ in range(2):
            result = self.run_script()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = sorted(self.runs.iterdir())
        self.assertEqual(len(runs), 2)
        samples = [json.loads((run / "sample.json").read_text()) for run in runs]
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(len(samples[0]), 256)
        self.assertEqual(len({row['id'] for row in samples[0]}), 256)
        for run in runs:
            launch = json.loads((run / "launch.json").read_text())
            args = launch["args"]
            self.assertEqual(args[args.index("--num_processes") + 1], "2")
            self.assertEqual(args[args.index("--output-dir") + 1], str(run / "checkpoints"))
            self.assertEqual(launch["sample"], str(run / "sample.json"))
            self.assertEqual(launch["wandb_mode"], "offline")
            manifest = json.loads((run / "sample_manifest.json").read_text())
            self.assertEqual(samples[0], [self.rows[i] for i in manifest["sample_indices"]])
            self.assertIn('fake training output', (run / 'training.log').read_text())
        self.assertEqual(json.loads(self.source.read_text()), self.rows)

    def test_full_dataset_and_missing_database_do_not_launch(self):
        result = self.run_script(RBAC_SAMPLE_SIZE="300")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('smaller than the source', result.stderr)
        self.database.unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing training database', result.stderr)
        self.assertEqual(list(self.runs.glob('*/launch.json')), [])

    def test_launcher_failure_preserves_sample_and_log(self):
        result = self.run_script(SAMPLE_TEST_EXIT="7", RBAC_SAMPLE_SIZE="32", RBAC_NUM_GPUS="4")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        run, = self.runs.iterdir()
        self.assertEqual(len(json.loads((run / "sample.json").read_text())), 32)
        launch = json.loads((run / "launch.json").read_text())
        args = launch['args']
        self.assertEqual(args[args.index('--num_processes') + 1], '4')
        self.assertIn('fake training output', (run / 'training.log').read_text())


if __name__ == '__main__':
    unittest.main()
