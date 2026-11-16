"""SQL regression checks without downloading models or starting training."""

import ast
from contextlib import closing, nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
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
        self.scorer = sql_rewards.SQLScorer.__new__(sql_rewards.SQLScorer)
        self.scorer.db_dir = self.root
        self.scorer.references = {}
        self.scorer.clean = Mock(side_effect=self.clean)
        self.scorer.checker = Mock()
        self.scorer.checker.check.return_value = {"status": "PASS", "not_checked": []}
        self.scorer.result_eq = lambda result, gold, order_matters: result == gold
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

    def test_sql_containing_refusal_words_is_scored_as_sql(self):
        sql = "SELECT 'sorry, I cannot answer.'"
        self.assertEqual(self.scorer.score(sql, self.row(gold=sql)), 1)
        self.assertEqual(self.scorer.score(sql, self.row("denied", gold=sql)), -1)

    def test_sql_requires_policy_and_correct_execution_results(self):
        self.assertEqual(self.scorer.score("SELECT x FROM t", self.row()), 1)
        self.assertEqual(self.scorer.score("SELECT 2", self.row()), -1)
        self.scorer.checker.check.return_value = {"status": "BLOCK"}
        self.assertEqual(self.scorer.score("SELECT x FROM t", self.row()), -1)


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
        self.assertEqual(scorer.score.call_count, 2)
        # Match the trainer's mask: PAD-only responses retain the first penalty.
        lengths = torch.where(responses == 0, torch.arange(3), 3).min(dim=1).values - 1
        lengths = lengths.clamp(min=0)
        mask = torch.arange(3) > lengths[:, None]
        self.assertEqual(rewards.masked_fill(mask, 0).sum(dim=1).tolist(), [-1, -1, 1])


class SQLDatasetTests(unittest.TestCase):
    def test_original_policy_reaches_prompt_and_reward_without_added_keys(self):
        rows = [
            {"db_id": "demo", "instruction": "schema", "role": "reader", "input": "question",
             "policy": policy, "metadata": {"permission": "allowed", "gold_sql": "SELECT 1"}}
            for policy in ({"alpha": ["x"]}, {"beta": []})
        ]
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, apply_chat_template=Mock())
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: list(range(len(str(messages))))
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
            with patch.dict(os.environ, {"RBAC_BENCHMARK_DIR": "/unused"}):
                with patch.object(module, "iter_dataset", return_value=iter(rows)), \
                     patch.object(module, "resolve_training_data", return_value=Path("/training.json")) as resolve:
                    module.SQLDatasetHelper("unused-model", "/unused-cache")
        resolve.assert_called_once_with("/unused-cache")
        self.assertTrue(all(isinstance(row, str) for row in serialized["row_json"]))
        for original, prepared, call in zip(rows, processed, tokenizer.apply_chat_template.call_args_list):
            self.assertEqual(call.args[0], build_messages(original))
            self.assertEqual(json.loads(prepared["reward_metadata"])["policy"], original["policy"])
        self.assertEqual(rows[0]["policy"], {"alpha": ["x"]})


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
