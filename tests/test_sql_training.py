"""SQL regression checks without downloading models or starting training."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch

from eval.rbac import build_messages
from rl_datasets.sql_dataset import resolve_training_data


PROJECT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, PROJECT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Scalar scoring doesn't use torch. Tensor tests below use real torch if present.
try:
    import torch
except ImportError:
    torch = None
with patch.dict(sys.modules, {"torch": torch or SimpleNamespace()}):
    sql_rewards = load_file("sql_rewards_test", "rewards/sql_rewards.py")


class TrainingLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="training logs ")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run with spaces"
        # Load the logging helper without importing models or the training stack.
        tree = ast.parse((PROJECT / "wrappers/wrapper_utils.py").read_text())
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "start_training_log")
        self.source = "import atexit, os, subprocess, sys\n" + ast.unparse(helper)

    def run_logged_process(self, rank, ending=""):
        source = self.source + '''
start_training_log(sys.argv[1])
rank = os.environ.get("RANK", "0")
print(f"python stdout rank={rank}")
print(f"python stderr rank={rank}", file=sys.stderr)
os.write(1, f"native stdout rank={rank}\\n".encode())
os.write(2, f"native stderr rank={rank}\\n".encode())
''' + ending
        env = dict(os.environ)
        env.pop("RANK", None)
        if rank is not None:
            env["RANK"] = str(rank)
        return subprocess.run(
            [sys.executable, "-c", source, str(self.output)],
            env=env, capture_output=True, text=True, timeout=20,
        )

    def test_various_rank_counts_keep_combined_and_individual_logs(self):
        root = self.output
        for count in (1, 2, 4, 8):
            with self.subTest(ranks=count):
                self.output = root / f"{count} ranks"
                with ThreadPoolExecutor(max_workers=count) as pool:
                    results = list(pool.map(self.run_logged_process, range(count)))
                combined = (self.output / "train.log").read_text()
                for rank, result in enumerate(results):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    contents = (self.output / f"rank{rank}.log").read_text()
                    for message in ("python stdout", "python stderr", "native stdout", "native stderr"):
                        text = f"{message} rank={rank}"
                        self.assertIn(text, contents)
                        self.assertIn(text, combined)
                        self.assertIn(text, result.stdout)
                    for other in range(count):
                        if other != rank:
                            self.assertNotIn(f"python stdout rank={other}\n", contents)
                self.assertEqual(len(list(self.output.glob("*.log"))), count + 1)

    def test_single_process_without_rank_defaults_to_rank_zero(self):
        result = self.run_logged_process(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "train.log").read_text(),
                         (self.output / "rank0.log").read_text())

    def test_traceback_is_saved_and_failure_exit_status_is_preserved(self):
        result = self.run_logged_process(0, 'raise RuntimeError("training failed")\n')
        self.assertEqual(result.returncode, 1)
        self.assertIn("RuntimeError: training failed", result.stdout)
        self.assertIn("RuntimeError: training failed", (self.output / "train.log").read_text())
        self.assertIn("RuntimeError: training failed", (self.output / "rank0.log").read_text())

    def test_native_output_is_saved_even_without_python_cleanup(self):
        result = self.run_logged_process(1, 'os._exit(23)\n')
        self.assertEqual(result.returncode, 23)
        self.assertIn("native stderr", (self.output / "rank1.log").read_text())
        self.assertIn("native stderr", (self.output / "train.log").read_text())

    def test_existing_log_contents_are_preserved(self):
        self.output.mkdir()
        (self.output / "train.log").write_text("previous output\n")
        result = self.run_logged_process(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "train.log").read_text().startswith("previous output\n"))


class SQLScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        database = self.root / "demo/demo.sqlite"
        database.parent.mkdir()
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE t (x INTEGER)")
            connection.execute("INSERT INTO t VALUES (1)")
            connection.commit()
        policy_file = self.root / "rbac-exp/evaluation/prediction_policy.py"
        policy_file.parent.mkdir(parents=True)
        policy_file.touch()
        checker = Mock()
        checker.check.return_value = {"status": "PASS", "not_checked": []}
        dependencies = {
            "data_process.response_cleaner": SimpleNamespace(clean_model_response=Mock(side_effect=self.clean)),
            "evaluation.prediction_policy": SimpleNamespace(PolicyChecker=Mock(return_value=checker)),
            "evaluation.exec_eval": SimpleNamespace(result_eq=lambda result, gold, order_matters: result == gold),
            "sqlglot": SimpleNamespace(errors=SimpleNamespace(ParseError=SyntaxError, TokenError=SyntaxError)),
            "sqlfluff": SimpleNamespace(lint=Mock(return_value=[])),
        }
        with patch.dict(sys.modules, dependencies), patch.object(sys, "path", sys.path.copy()):
            self.scorer = sql_rewards.SQLScorer(self.root, self.root)
        # SQL parsing and policy checking are external dependencies; execution
        # below uses actual read-only SQLite, not a mock result.
        self.scorer.read_query = Mock(return_value=SimpleNamespace(args={}))

    @staticmethod
    def clean(response):
        if response.startswith("SELECT"):
            return response
        if response in ("I cannot answer this request.", "Sorry, I cannot answer."):
            return "Sorry, I cannot answer."
        return "empty response"

    @staticmethod
    def row(permission="allowed", gold="SELECT x FROM t"):
        return {"db_id": "demo", "policy": {"t": ["x"]},
                "metadata": {"permission": permission, "gold_sql": gold}}

    def test_invalid_and_empty_outputs_are_negative_for_both_permissions(self):
        for permission in ("allowed", "denied"):
            for response in ("", "  ", "nonsense", "empty response", "```sql\n```"):
                with self.subTest(permission=permission, response=response):
                    self.assertEqual(self.scorer.score(response, self.row(permission)), -1)
        self.scorer.checker.check.assert_not_called()

    def test_real_refusal_requires_denied_permission(self):
        for response in ("I cannot answer this request.", "Sorry, I cannot answer."):
            self.assertEqual(self.scorer.score(response, self.row("denied")), 1)
            self.assertEqual(self.scorer.score(response, self.row()), -1)

    def test_refusal_check_ignores_case_and_outer_whitespace(self):
        for response in ("  sorry, i cannot answer.\n", "\tSORRY, I CANNOT ANSWER.  ",
                         "SoRrY, I CaNnOt AnSwEr."):
            with self.subTest(response=response), patch.object(self.scorer, "clean", return_value=response):
                self.assertEqual(self.scorer.score(response, self.row("denied")), 1)
                self.assertEqual(self.scorer.score(response, self.row()), -1)
        self.scorer.read_query.assert_not_called()
        self.scorer.checker.check.assert_not_called()

    def test_sql_containing_refusal_words_is_scored_as_sql(self):
        sql = "SELECT 'sorry, I cannot answer.'"
        self.assertAlmostEqual(self.scorer.score(sql, self.row(gold=sql)), 1.0)
        self.assertEqual(self.scorer.score(sql, self.row("denied", gold=sql)), -1)

    def test_sql_requires_policy_and_correct_execution_results(self):
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 1.0)
        self.assertAlmostEqual(self.scorer.score("SELECT 2", self.row()), -0.45)
        self.scorer.checker.check.return_value = {"status": "BLOCK"}
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), -0.7)

    def test_invalid_syntax_skips_policy_and_all_execution(self):
        for error in (ValueError("unsupported SQL"), SyntaxError("invalid SQL")):
            with self.subTest(error=error), patch.object(self.scorer, "execute") as execute:
                self.scorer.read_query.side_effect = error
                self.assertEqual(self.scorer.score("SELECT broken", self.row()), -0.9)
                self.scorer.checker.check.assert_not_called()
                execute.assert_not_called()
                self.scorer.lint_sql.assert_not_called()

    def test_failed_or_incomplete_policy_skips_all_execution(self):
        for access in ({"status": "BLOCK"}, {"status": "PASS", "not_checked": ["unknown"]}):
            with self.subTest(access=access), patch.object(self.scorer, "execute") as execute:
                self.scorer.checker.check.return_value = access
                self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), -0.7)
                execute.assert_not_called()
                self.scorer.lint_sql.assert_not_called()

    def test_total_rewards_match_current_stage_weights(self):
        totals = [self.scorer.score("", self.row()),
                  self.scorer.score("Sorry, I cannot answer.", self.row())]
        with patch.object(self.scorer, "read_query", side_effect=ValueError("invalid SQL")):
            totals.append(self.scorer.score("SELECT broken", self.row()))
        self.scorer.checker.check.return_value = {"status": "BLOCK"}
        totals.append(self.scorer.score("SELECT x FROM t", self.row()))
        self.scorer.checker.check.return_value = {"status": "PASS", "not_checked": []}
        totals.append(self.scorer.score("SELECT 2", self.row()))
        self.scorer.lint_sql.return_value = [{"code": rule} for rule in self.scorer.sqlfluff_rules]
        totals.append(self.scorer.score("SELECT x FROM t", self.row()))
        self.scorer.lint_sql.return_value = []
        totals.append(self.scorer.score("SELECT x FROM t", self.row()))
        # Output and refusal failures tie; later stages keep prior rewards.
        for actual, expected in zip(totals, [-1.0, -1.0, -0.9, -0.7, -0.45, 0.9, 1.0]):
            self.assertAlmostEqual(actual, expected)

    def test_refusal_and_denied_sql_skip_sql_checks(self):
        for response, permission, expected in (
            ("Sorry, I cannot answer.", "denied", 1),
            ("Sorry, I cannot answer.", "allowed", -1),
            ("SELECT x FROM t", "denied", -1),
        ):
            with self.subTest(response=response, permission=permission), \
                 patch.object(self.scorer, "execute") as execute:
                self.assertEqual(self.scorer.score(response, self.row(permission)), expected)
                self.scorer.read_query.assert_not_called()
                self.scorer.checker.check.assert_not_called()
                execute.assert_not_called()
                self.scorer.lint_sql.assert_not_called()

    def test_execution_error_keeps_earlier_rewards(self):
        self.assertAlmostEqual(self.scorer.score("SELECT missing FROM t", self.row()), -0.45)
        self.scorer.lint_sql.assert_called_once()

    def test_incorrect_results_keep_sqlfluff_rewards(self):
        for failed_rules, expected in (([], -0.45), (self.scorer.sqlfluff_rules, -0.55)):
            with self.subTest(failed_rules=failed_rules):
                self.scorer.lint_sql.reset_mock()
                self.scorer.lint_sql.return_value = [{"code": rule} for rule in failed_rules]
                self.assertAlmostEqual(self.scorer.score("SELECT 2", self.row()), expected)
                self.scorer.lint_sql.assert_called_once()

    def test_sqlfluff_runs_before_final_execution(self):
        self.assertEqual(list(self.scorer.reward_steps)[-2:], ["sqlfluff", "execution"])
        calls = Mock()
        calls.attach_mock(self.scorer.lint_sql, "lint")
        with patch.object(self.scorer, "execute", wraps=self.scorer.execute) as execute:
            calls.attach_mock(execute, "execute")
            self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 1.0)
        self.assertEqual([call[0] for call in calls.mock_calls], ["lint", "execute", "execute"])
        self.scorer.lint_sql.assert_called_once_with(
            "SELECT x FROM t", dialect="sqlite", rules=["LT01", "CP01", "LT06", "CP03"],
        )
        self.scorer.lint_sql.return_value = [{"code": "LT01"}]
        self.assertAlmostEqual(self.scorer.score("SELECT  x FROM t", self.row()), 0.975)

    def test_sqlfluff_rewards_each_passing_rule_once(self):
        for codes, expected in (([], 1.0), (["LT01", "LT01"], 0.975),
                                (["LT01", "CP01"], 0.95),
                                (["LT01", "CP01", "LT06"], 0.925),
                                (["LT01", "CP01", "LT06", "CP03"], 0.9)):
            with self.subTest(codes=codes):
                self.scorer.lint_sql.return_value = [{"code": code} for code in codes]
                self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), expected)

    def test_sqlfluff_parse_errors_do_not_earn_style_points(self):
        for code in ("PRS", "TMP", "LXR"):
            with self.subTest(code=code):
                self.scorer.lint_sql.return_value = [{"code": code}]
                self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.9)

    def test_sqlfluff_bonus_is_split_across_configured_rules(self):
        self.scorer.sqlfluff_rules = ["LT01", "CP01"]
        self.scorer.lint_sql.return_value = [{"code": "LT01"}]
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.95)

    def test_empty_sqlfluff_rules_disable_the_bonus(self):
        self.scorer.sqlfluff_rules = []
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.95)
        self.scorer.lint_sql.assert_not_called()

    def test_sqlfluff_score_is_configurable(self):
        check, _, _ = self.scorer.reward_steps["sqlfluff"]
        self.scorer.reward_steps["sqlfluff"] = (check, 0.02, -0.01)
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.97)
        self.scorer.lint_sql.return_value = [{"code": "CP01"}]
        self.assertAlmostEqual(self.scorer.score("SELECT x from t", self.row()), 0.9625)
        self.scorer.lint_sql.return_value = [{"code": rule} for rule in self.scorer.sqlfluff_rules]
        self.assertAlmostEqual(self.scorer.score("SELECT x from t", self.row()), 0.94)

    def test_real_sqlfluff_spacing_and_capitalisation_rules(self):
        try:
            from sqlfluff import lint
        except ImportError:
            self.skipTest("sqlfluff is not installed; run in the training environment")
        self.scorer.lint_sql = lint
        for sql, expected in (("SELECT x FROM t", 1.0),
                              ("SELECT  x FROM t", 0.975),
                              ("SELECT x from t", 0.975),
                              ("SELECT COUNT (x) FROM t", 0.95),  # Both LT01 and LT06 fail.
                              ("SELECT COUNT(x) + count(x) FROM t", 0.975),
                              ("SELECT  COUNT (x) + count(x) from t", 0.9)):
            with self.subTest(sql=sql):
                self.assertAlmostEqual(self.scorer.score(sql, self.row(gold=sql)), expected)

    def test_reference_error_is_not_scored_as_a_model_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot execute reference SQL"):
            self.scorer.score("SELECT x FROM t", self.row(gold="SELECT missing FROM t"))

    def test_reference_results_are_cached(self):
        with patch.object(self.scorer, "execute", wraps=self.scorer.execute) as execute:
            self.scorer.score("SELECT 1", self.row())
            self.scorer.score("SELECT 2", self.row())
        self.assertEqual([call.args[1] for call in execute.call_args_list],
                         ["SELECT x FROM t", "SELECT 1", "SELECT 2"])

    def test_scores_can_be_changed_without_changing_checks(self):
        check, _, _ = self.scorer.reward_steps["execution"]
        self.scorer.reward_steps["execution"] = (check, 0.2, -0.8)
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.75)
        self.assertAlmostEqual(self.scorer.score("SELECT 2", self.row()), -0.25)

    def test_zero_sqlglot_scores_still_prevent_execution(self):
        check, _, _ = self.scorer.reward_steps["sqlglot"]
        self.scorer.reward_steps["sqlglot"] = (check, 0.0, 0.0)
        self.scorer.read_query.side_effect = ValueError("invalid SQL")
        with patch.object(self.scorer, "execute") as execute:
            self.assertEqual(self.scorer.score("SELECT broken", self.row()), 0.1)
            execute.assert_not_called()

    def test_reward_steps_can_be_added_and_removed(self):
        del self.scorer.reward_steps["execution"]
        extra = Mock(return_value=(0.25, True))
        self.scorer.reward_steps["extra"] = (extra, 0.25, -0.25)
        with patch.object(self.scorer, "execute") as execute:
            self.assertAlmostEqual(self.scorer.score("SELECT x FROM t", self.row()), 0.8)
            execute.assert_not_called()
        extra.assert_called_once()

    def test_result_comparison_receives_prediction_ordering(self):
        self.scorer.read_query.return_value = SimpleNamespace(args={"order": object()})
        self.scorer.result_eq = Mock(return_value=True)
        self.assertAlmostEqual(self.scorer.score("SELECT x FROM t ORDER BY x", self.row()), 1.0)
        self.scorer.result_eq.assert_called_once_with([(1,)], [(1,)], order_matters=True)


@unittest.skipIf(torch is None, "torch is not installed; tensor checks require the training environment")
class SQLTokenRewardTests(unittest.TestCase):
    def test_empty_and_finished_responses_keep_one_terminal_penalty(self):
        responses = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0]])
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            decode=lambda tokens, **kwargs: "" if tokens.tolist() == [1] else "valid",
        )
        scorer = Mock()
        scorer.score.side_effect = lambda text, row: -1.0 if not text else 1.0
        rewards = sql_rewards.sql_outcome_reward(
            responses, tokenizer, None, None, ["{}"] * 3, scorer,
        )
        self.assertEqual(rewards.tolist(), [[-1, 0, 0], [-1, 0, 0], [0, 1, 0]])
        self.assertEqual(scorer.score.call_count, 3)
        # Match the trainer's mask: PAD-only responses retain the first penalty.
        lengths = torch.where(responses == 0, torch.arange(3), 3).min(dim=1).values - 1
        lengths = lengths.clamp(min=0)
        mask = torch.arange(3) > lengths[:, None]
        self.assertEqual(rewards.masked_fill(mask, 0).sum(dim=1).tolist(), [-1, -1, 1])

    def test_pad_only_response_uses_configured_empty_output_score(self):
        scorer = Mock()
        scorer.score.return_value = -0.75
        tokenizer = SimpleNamespace(pad_token_id=0, decode=Mock())
        rewards = sql_rewards.sql_outcome_reward(
            torch.tensor([[0, 0]]), tokenizer, None, None, ['{"db_id": "demo"}'], scorer,
        )
        self.assertEqual(rewards.tolist(), [[-0.75, 0]])
        scorer.score.assert_called_once_with("", {"db_id": "demo"})
        tokenizer.decode.assert_not_called()


class SQLDatasetTests(unittest.TestCase):
    def prepare_rows(self, rows, **environment):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, apply_chat_template=Mock())
        tokenizer.apply_chat_template.return_value = [2, 3]
        dataset = Mock(column_names=["row_json"])
        processed = []
        serialized = {}

        def from_dict(columns):
            serialized.update(columns)
            return dataset

        def map_rows(prepare, **kwargs):
            processed.extend(prepare({"row_json": row}) for row in serialized["row_json"])
            result = Mock()
            result.__getitem__ = Mock(side_effect=lambda key: [row[key] for row in processed])
            return result

        dataset.__len__ = Mock(return_value=len(rows))
        dataset.map.side_effect = map_rows
        scorer = Mock()
        dependencies = {
            "accelerate": SimpleNamespace(PartialState=lambda: SimpleNamespace(local_main_process_first=nullcontext)),
            "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_dict=from_dict), DatasetDict=dict),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
                AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(max_position_embeddings=10000)),
            ),
            "rewards.sql_rewards": SimpleNamespace(SQLScorer=lambda *args: scorer),
        }
        with patch.dict(sys.modules, dependencies):
            module = load_file("sql_dataset_test", "rl_datasets/sql_dataset.py")
            with patch.dict(os.environ, {"RBAC_BENCHMARK_DIR": "/unused", "RBAC_SAMPLE_SIZE": "5000",
                                         "RBAC_SAMPLE_SEED": "42", **environment}):
                with patch.object(module, "iter_dataset", return_value=iter(rows)), \
                     patch.object(module, "resolve_training_data", return_value=Path("/training.json")) as resolve:
                    helper = module.SQLDatasetHelper("unused-model", "/unused-cache")
        resolve.assert_called_once_with("/unused-cache")
        return helper, processed, tokenizer, serialized, scorer

    def test_original_policy_reaches_prompt_and_reward_without_added_keys(self):
        rows = [
            {"db_id": "demo", "instruction": "schema", "role": "reader", "input": "question",
             "policy": policy, "metadata": {"permission": "allowed", "gold_sql": "SELECT 1"}}
            for policy in ({"alpha": ["x"]}, {"beta": []})
        ]
        _, processed, tokenizer, serialized, _ = self.prepare_rows(rows)
        self.assertEqual(len(processed), len(rows))
        self.assertTrue(all(isinstance(row, str) for row in serialized["row_json"]))
        for original, prepared, call in zip(rows, processed, tokenizer.apply_chat_template.call_args_list):
            self.assertEqual(call.args[0], build_messages(original))
            self.assertEqual(json.loads(prepared["reward_metadata"])["policy"], original["policy"])
        self.assertEqual(rows[0]["policy"], {"alpha": ["x"]})

    def test_default_training_sample_has_5000_reproducible_random_rows(self):
        rows = [{"id": i, **SQLScoringTests.row()} for i in range(6000)]
        selections = []
        for seed in ("42", "42", "43"):
            _, processed, _, serialized, scorer = self.prepare_rows(rows, RBAC_SAMPLE_SEED=seed)
            indices = [json.loads(row)["id"] for row in serialized["row_json"]]
            self.assertEqual(len(processed), 5000)
            self.assertEqual(len(set(indices)), 5000)
            self.assertEqual(scorer.validate_reference.call_count, 5000)
            self.assertNotEqual(indices, list(range(5000)))
            selections.append(indices)
        self.assertEqual(selections[0], selections[1])
        self.assertNotEqual(selections[0], selections[2])
        self.assertEqual([row["id"] for row in rows], list(range(6000)))

    def test_explicit_smaller_sample_size_is_respected(self):
        rows = [{"id": i, **SQLScoringTests.row()} for i in range(20)]
        _, processed, _, serialized, _ = self.prepare_rows(rows, RBAC_SAMPLE_SIZE="7")
        self.assertEqual(len(processed), 7)
        self.assertEqual(len({json.loads(row)["id"] for row in serialized["row_json"]}), 7)

    def test_nonpositive_sample_size_is_rejected(self):
        for size in ("0", "-1"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "RBAC_SAMPLE_SIZE must be positive"):
                self.prepare_rows([SQLScoringTests.row()], RBAC_SAMPLE_SIZE=size)

    def test_empty_training_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RBAC training data is empty"):
            self.prepare_rows([])


class TrainingDataSourceTests(unittest.TestCase):
    def test_default_downloads_only_the_training_file_to_configured_cache(self):
        download = Mock(return_value="/cache/spider/training.json")
        with patch.dict(os.environ, {"RBAC_TRAIN_DATA": ""}), \
             patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            path = resolve_training_data("/configured/cache")
        self.assertEqual(path, Path("/cache/spider/training.json"))
        download.assert_called_once_with(
            repo_id="sharkiefff/RBAC-Text2SQL-Benchmark", repo_type="dataset",
            filename="spider/column_level_rbac_dataset_spider_train_v3_no_sm.json",
            cache_dir="/configured/cache",
        )

    def test_local_override_bypasses_hugging_face(self):
        download = Mock(side_effect=AssertionError("Local files must not download anything"))
        with patch.dict(os.environ, {"RBAC_TRAIN_DATA": "~/training.jsonl"}), \
             patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            path = resolve_training_data("/unused/cache")
        self.assertEqual(path, Path("~/training.jsonl").expanduser().resolve())
        download.assert_not_called()

    def test_download_failure_is_not_replaced_with_evaluation_data(self):
        download = Mock(side_effect=ConnectionError("offline and not cached"))
        with patch.dict(os.environ, {"RBAC_TRAIN_DATA": ""}), \
             patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            with self.assertRaisesRegex(ConnectionError, "not cached"):
                resolve_training_data("/configured/cache")
        download.assert_called_once()


def load_save_model():
    # Compile the actual method in isolation: importing the full trainer would
    # require CUDA/TRL and unrelated robotics dependencies for these file checks.
    path = PROJECT / "wrappers/ppo_wrapper.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "PPOWrapper")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "save_model")
    namespace = {"os": os, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["save_model"]


class CheckpointSavingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "checkpoint"
        self.model = SimpleNamespace(policy=Mock(), value_model=Mock())
        self.trainer = SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=True, unwrap_model=Mock(return_value=self.model)),
            model=object(), deepspeed=object(), sql_task=True,
            processing_class=Mock(), args=SimpleNamespace(output_dir=str(self.output)),
        )
        self.save = load_save_model()

    def test_non_main_rank_writes_nothing(self):
        self.trainer.accelerator.is_main_process = False
        self.save(self.trainer)
        self.assertFalse(self.output.exists())
        self.trainer.accelerator.unwrap_model.assert_not_called()
        self.model.policy.save_pretrained.assert_not_called()
        self.model.value_model.save_pretrained.assert_not_called()
        self.trainer.processing_class.save_pretrained.assert_not_called()

    def test_main_rank_saves_unwrapped_policy_value_and_tokenizer(self):
        original = self.trainer.model
        deepspeed = self.trainer.deepspeed
        self.save(self.trainer)
        self.trainer.accelerator.unwrap_model.assert_called_once_with(original)
        self.model.policy.save_pretrained.assert_called_once_with(str(self.output / "policy"))
        self.model.value_model.save_pretrained.assert_called_once_with(str(self.output / "value"))
        self.trainer.processing_class.save_pretrained.assert_called_once_with(str(self.output / "policy"))
        self.assertIs(self.trainer.model, original)
        self.assertIs(self.trainer.deepspeed, deepspeed)

    def test_other_tasks_do_not_save_the_sql_tokenizer(self):
        self.trainer.sql_task = False
        self.save(self.trainer)
        self.model.policy.save_pretrained.assert_called_once()
        self.model.value_model.save_pretrained.assert_called_once()
        self.trainer.processing_class.save_pretrained.assert_not_called()

    def test_save_failure_does_not_replace_training_model(self):
        original = self.trainer.model
        deepspeed = self.trainer.deepspeed
        self.model.policy.save_pretrained.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.save(self.trainer, str(self.output))
        self.assertIs(self.trainer.model, original)
        self.assertIs(self.trainer.deepspeed, deepspeed)


if __name__ == '__main__':
    unittest.main()
