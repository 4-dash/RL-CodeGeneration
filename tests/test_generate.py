"""Check base-model and optional checkpoint selection without downloading weights."""

import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import generate


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.tokenizer = Mock()
        self.transformers = SimpleNamespace(
            AutoModelForCausalLM=Mock(), AutoTokenizer=Mock(),
        )
        self.transformers.AutoModelForCausalLM.from_pretrained.return_value = self.model
        self.transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer

    def test_default_loads_pretrained_model_without_peft(self):
        with patch.dict(sys.modules, {"transformers": self.transformers, "peft": None}):
            model, tokenizer = generate.load_generation_model()
        self.assertIs(model, self.model)
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.args,
                         (generate.DEFAULT_MODEL,))
        self.model.eval.assert_called_once()

    def test_custom_pretrained_model(self):
        with patch.dict(sys.modules, {"transformers": self.transformers, "peft": None}):
            generate.load_generation_model(model_name="example/model")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.args,
                         ("example/model",))

    def test_explicit_device_map_pins_the_entire_model_to_one_gpu(self):
        device_map = {"": "cuda:1"}
        with patch.dict(sys.modules, {"transformers": self.transformers, "peft": None}):
            generate.load_generation_model(device_map=device_map)
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["device_map"],
                         device_map)

    def test_checkpoint_loads_its_base_model_and_adapter(self):
        peft = SimpleNamespace(PeftConfig=Mock(), PeftModel=Mock())
        peft.PeftConfig.from_pretrained.return_value.base_model_name_or_path = "checkpoint/base"
        with patch.dict(sys.modules, {"transformers": self.transformers, "peft": peft}):
            model, _ = generate.load_generation_model(adapter="checkpoint/policy")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.args,
                         ("checkpoint/base",))
        peft.PeftModel.from_pretrained.assert_called_once_with(
            self.model, "checkpoint/policy", is_trainable=False,
        )
        self.assertIs(model, peft.PeftModel.from_pretrained.return_value)
        model.eval.assert_called_once()

    def test_cli_defaults_to_base_model(self):
        with patch.object(generate, "load_generation_model", return_value=(self.model, self.tokenizer)) as load:
            with patch.object(generate, "generate_response", return_value="def f(): pass"):
                with contextlib.redirect_stdout(io.StringIO()):
                    generate.main(["Write a function"])
        load.assert_called_once_with(adapter=None, model_name=None)


if __name__ == "__main__":
    unittest.main()
