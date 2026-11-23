"""SQL SynCode wiring with real tensors, without models or GPU dependencies."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None


PROJECT = Path(__file__).resolve().parents[1]
WRAPPERS = ("wrappers/ppo_wrapper.py", "ppowrapperwithlogs.py")


def trainer_methods(filename):
    tree = ast.parse((PROJECT / filename).read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOWrapper")
    return {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}


@unittest.skipIf(torch is None, "torch is not installed; run in the training environment")
class SQLSynCodeTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.is_valid.return_value = True
        self.grammar = Mock(side_effect=lambda language: language)
        engine = self.engine

        class Processor:
            def __init__(self, grammar, tokenizer, **kwargs):
                self.tokenizer = tokenizer
                engine.tokenizer = tokenizer
                self.grammar_engine = engine

        dependencies = {
            "syncode": SimpleNamespace(SyncodeLogitsProcessor=Processor, Grammar=self.grammar, language_model=None),
            "syncode.grammar_mask.grammar_constrainer": SimpleNamespace(GrammarConstrainer=object),
            "transformers": SimpleNamespace(PreTrainedTokenizer=object),
        }
        spec = importlib.util.spec_from_file_location("syncode_test", PROJECT / "wrappers/syncode_wrapper.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(self.module)
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1,
                                         batch_decode=Mock(return_value=[""]))
        self.processor = self.module.SyncodeLogitsListWrapper(self.tokenizer, 2, language="sql")

    def test_sql_grammar_and_logits_are_unchanged(self):
        self.grammar.assert_called_once_with("sql")
        logits = torch.randn(2, 12)
        original = logits.clone()
        self.assertIs(self.processor(torch.tensor([[5], [6]]), logits), logits)
        torch.testing.assert_close(logits, original)
        self.assertEqual(self.engine.is_valid.call_count, 2)

    def test_sql_uses_shared_helper_without_waiting_for_python_fences(self):
        self.assertEqual(len(self.processor.logit_processors), 2)
        for helper in self.processor.logit_processors:
            self.assertIsInstance(helper, self.module.SyncodeLogitsProcessorHelper)
        self.processor(torch.tensor([[50], [51]]), torch.randn(2, 12))
        torch.testing.assert_close(self.processor.get_scores(), torch.ones(2, 1))
        self.tokenizer.batch_decode.assert_not_called()

    def test_valid_and_invalid_token_rewards_use_python_scale_without_normalization(self):
        self.engine.is_valid.side_effect = [True, False, True, False]
        self.processor(torch.tensor([[50], [51]]), torch.randn(2, 12))
        self.processor(torch.tensor([[50, 2], [51, 3]]), torch.randn(2, 12))
        torch.testing.assert_close(self.processor.get_scores(), torch.tensor([[1., 1.], [-1., -1.]]))

    def test_sql_refusals_have_no_special_exemption(self):
        self.tokenizer.batch_decode.return_value = ["Sorry, I cannot answer."]
        self.engine.is_valid.return_value = False
        self.processor(torch.tensor([[50, 9]]), torch.randn(1, 12))
        torch.testing.assert_close(self.processor.get_scores(), torch.tensor([[-1.]]))

    def test_smaller_rollout_resets_existing_helpers(self):
        self.processor(torch.tensor([[50], [51]]), torch.randn(2, 12))
        torch.testing.assert_close(self.processor.get_scores(), torch.ones(2, 1))
        self.processor(torch.tensor([[52]]), torch.randn(1, 12))
        torch.testing.assert_close(self.processor.get_scores(), torch.ones(1, 1))
        self.assertEqual(self.engine.reset.call_count, 2)

    def test_python_default_keeps_existing_code_block_behavior(self):
        self.tokenizer.batch_decode = Mock(return_value=[""])
        processor = self.module.SyncodeLogitsListWrapper(self.tokenizer, 2)
        self.assertEqual(self.grammar.call_args.args, ("python",))
        logits = torch.randn(2, 12)
        self.assertIs(processor(torch.tensor([[50], [51]]), logits), logits)
        torch.testing.assert_close(processor.get_scores(), torch.zeros(2, 1))
        self.engine.is_valid.assert_not_called()

    def test_sql_processor_is_enabled_and_ruff_stays_disabled_in_both_wrappers(self):
        for filename in WRAPPERS:
            for sql_task, coefficient in ((True, 0.1), (True, 0.0), (False, 0.1)):
                with self.subTest(filename=filename, sql_task=sql_task, coefficient=coefficient):
                    init = trainer_methods(filename)["__init__"]
                    start = next(i for i, node in enumerate(init.body)
                                 if isinstance(node, ast.Assign)
                                 and ast.unparse(node.targets[0]) == "self.syncode_logits_processor")
                    factory, ruff = Mock(), Mock()
                    trainer = SimpleNamespace(sql_task=sql_task, syncode_coef=coefficient,
                                              args=SimpleNamespace(local_rollout_forward_batch_size=2))
                    namespace = {"self": trainer, "dataset_helper": SimpleNamespace(tokenizer=self.tokenizer),
                                 "SyncodeLogitsListWrapper": factory, "ruff_linter_reward": ruff}
                    exec(compile(ast.Module(body=init.body[start:], type_ignores=[]), filename, "exec"), namespace)
                    factory.assert_called_once_with(self.tokenizer, 2, language="sql" if sql_task else "python")
                    self.assertIs(trainer.linter_function, None if sql_task else ruff)

    def test_sql_rollout_receives_processor_in_both_wrappers(self):
        for filename in WRAPPERS:
            with self.subTest(filename=filename):
                train = trainer_methods(filename)["train"]
                assignment = next(node for node in ast.walk(train)
                                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                                  and ast.unparse(node.value.func) == "adjusted_batch_generation")
                generate = Mock(return_value=(None, None, None))
                namespace = {
                    "self": SimpleNamespace(sql_task=True, syncode_logits_processor=self.processor),
                    "unwrapped_model": SimpleNamespace(policy=object()), "queries": torch.tensor([[50]]),
                    "args": SimpleNamespace(local_rollout_forward_batch_size=2),
                    "processing_class": self.tokenizer, "generation_config": object(),
                    "adjusted_batch_generation": generate,
                }
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), filename, "exec"), namespace)
                self.assertEqual(generate.call_args.args[-1], [self.processor])

    def test_sql_training_uses_python_weighting_without_running_ruff(self):
        for filename in WRAPPERS:
            with self.subTest(filename=filename):
                train = trainer_methods(filename)["train"]
                branch = next(node for node in ast.walk(train)
                              if isinstance(node, ast.If) and ast.unparse(node.test) == "self.sql_task"
                              and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "self.linter_function"
                                      for statement in node.orelse for call in ast.walk(statement)))
                ruff = Mock(side_effect=AssertionError("Ruff must not run for SQL"))
                responses = torch.tensor([[2, 3, 1]])
                syntax = torch.tensor([[1., -1., 1.]])
                namespace = {"torch": torch, "self": SimpleNamespace(sql_task=True, syncode_coef=0.1,
                                                                      linter_function=ruff),
                             "postprocessed_response": responses, "syncode_score": syntax,
                             "processing_class": self.tokenizer}
                exec(compile(ast.Module(body=[branch], type_ignores=[]), filename, "exec"), namespace)
                torch.testing.assert_close(namespace["score"], torch.tensor([[0.1, -0.1, 0.1]]))
                ruff.assert_not_called()
                namespace["self"] = SimpleNamespace(sql_task=False, syncode_coef=0.1, linter_coef=0.1,
                                                    linter_function=Mock(return_value=(torch.full_like(syntax, 2), None)))
                exec(compile(ast.Module(body=[branch], type_ignores=[]), filename, "exec"), namespace)
                torch.testing.assert_close(namespace["score"], torch.tensor([[0.3, 0.1, 0.3]]))

    def test_rollout_helper_collects_existing_per_token_scores(self):
        filename = "wrappers/wrapper_utils.py"
        tree = ast.parse((PROJECT / filename).read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "adjusted_batch_generation")
        def generate(model, query, *args):
            self.processor(query, torch.randn(1, 12))
            self.processor(torch.cat((query, torch.tensor([[2]])), dim=1), torch.randn(1, 12))
            sequence = torch.cat((query, torch.tensor([[2, 1]])), dim=1)
            return sequence, torch.zeros(1, 2, 12)

        namespace = {"torch": torch, "generate": generate, "GenerationConfig": object,
                     "LogitsProcessorList": list, "StoppingCriteriaList": list,
                     "pad": lambda tensors, **kwargs: torch.stack(tensors)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
        with patch.object(self.processor, "get_scores", wraps=self.processor.get_scores) as score:
            result = namespace["adjusted_batch_generation"](
                object(), torch.tensor([[50, 51], [52, 53]]), 1, 0, object(), [self.processor],
            )
        self.assertEqual(score.call_count, 2)
        for call in score.call_args_list:
            self.assertEqual(call.args, ())
        torch.testing.assert_close(result[2], torch.ones(2, 2))

    def test_sql_syntax_weight_is_enabled_without_ruff(self):
        framework = json.loads((PROJECT / "config/hyperparams.json").read_text())["framework"]
        sql = framework["sql"]
        self.assertEqual(sql["syncode_coef"], framework["code_gen"]["syncode_coef"])
        self.assertEqual(sql["linter_coef"], 0.0)


if __name__ == "__main__":
    unittest.main()
