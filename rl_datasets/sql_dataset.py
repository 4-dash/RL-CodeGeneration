"""RBAC training data using the same prompt as RBAC evaluation."""

import json
import os
from pathlib import Path

from eval.rbac import build_messages, iter_dataset


def resolve_training_data(cache_dir):
    local_path = os.environ.get("RBAC_TRAIN_DATA")
    if local_path:
        return Path(local_path).expanduser().resolve()

    from huggingface_hub import hf_hub_download

    # Read raw JSON to avoid Arrow adding null entries to policy dictionaries.
    return Path(hf_hub_download(
        repo_id="sharkiefff/RBAC-Text2SQL-Benchmark",
        repo_type="dataset",
        filename="spider/column_level_rbac_dataset_spider_train_v3_no_sm.json",
        cache_dir=cache_dir,
    ))


class SQLDatasetHelper:
    def __init__(self, model_name, cache_dir):
        # The sample script also uses this module to locate data without loading
        # the training stack. Import training dependencies only when needed.
        from accelerate import PartialState
        from datasets import Dataset, DatasetDict
        from transformers import AutoConfig, AutoTokenizer
        from rewards.sql_rewards import SQLScorer

        benchmark = Path(os.environ["RBAC_BENCHMARK_DIR"]).expanduser().resolve()
        db_dir = Path(os.environ.get(
            "RBAC_TRAIN_DB_DIR", str(benchmark / "data/spider/database")
        )).expanduser().resolve()
        scorer = SQLScorer(benchmark, db_dir)
        self.reward_kwargs = {"sql_scorer": scorer}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer.padding_side = "left"
        # Keep PAD distinct from EOS without adding an embedding to the base model.
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token = "<|fim_pad|>"
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            raise ValueError("SQL training requires a tokenizer with distinct PAD and EOS tokens")
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.context_limit = config.max_position_embeddings
        self.target_feature_name = "output"

        def prepare(example):
            row = json.loads(example["row_json"])
            permission = row["metadata"]["permission"]
            if permission not in ("allowed", "denied"):
                raise ValueError(f"Unknown permission label: {permission}")
            gold = row["metadata"]["gold_sql"]
            scorer.validate_reference(row["db_id"], gold)
            ids = self.tokenizer.apply_chat_template(
                build_messages(row), tokenize=True, add_generation_prompt=True,
            )
            if len(ids) >= self.context_limit:
                raise ValueError(f"Prompt for {row['db_id']} exceeds model context; no rows were truncated")
            metadata = {key: row[key] for key in ("db_id", "policy", "metadata")}
            return {"input_ids": ids, "attention_mask": [1] * len(ids),
                    # PPO needs this field; SQL rewards use the untruncated metadata.
                    "labels": [self.tokenizer.eos_token_id],
                    "reward_metadata": json.dumps(metadata)}

        with PartialState().local_main_process_first():
            data_path = resolve_training_data(cache_dir)
            # Arrow unifies dictionary keys across rows. Keep raw rows serialized
            # until prompting/scoring metadata is prepared so policies stay exact.
            dataset = Dataset.from_dict({
                "row_json": [json.dumps(row) for row in iter_dataset(data_path)]
            })
            if not len(dataset):
                raise ValueError("RBAC training data is empty")
            dataset = dataset.map(prepare, remove_columns=dataset.column_names)
        self.model_input_size = max(len(ids) for ids in dataset["input_ids"])
        self.tokenized_dataset = DatasetDict(train=dataset)

    def preprocess_dataset(self):
        return self.tokenized_dataset
