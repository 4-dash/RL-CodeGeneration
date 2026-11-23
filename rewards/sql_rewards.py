"""Combine configurable checks into one RBAC SQL training reward."""

import json
from contextlib import closing
import sqlite3
import sys
import time
from pathlib import Path

import torch


class SQLScorer:
    def __init__(self, benchmark_dir, db_dir):
        experiment = Path(benchmark_dir) / "rbac-exp"
        if not (experiment / "evaluation/prediction_policy.py").is_file():
            raise FileNotFoundError(f"Current RBAC benchmark code not found: {experiment}")
        sys.path.insert(0, str(experiment))
        from data_process.response_cleaner import clean_model_response
        from evaluation.prediction_policy import PolicyChecker
        from evaluation.exec_eval import result_eq
        import sqlglot
        from sqlfluff import lint

        self.sqlglot = sqlglot
        self.lint_sql = lint
        self.clean = clean_model_response
        self.result_eq = result_eq
        self.db_dir = Path(db_dir).resolve()
        self.checker = PolicyChecker(str(self.db_dir))
        self.references = {}
        # The response cleaner flattens lines, so check spacing and capitalization only.
        self.sqlfluff_rules = ["LT01", "CP01", "LT06", "CP03"]

        # Each check returns (reward, continue_scoring). Add new checks here.
        # Keep prerequisite checks in order; (0.0, 0.0) disables their score only.
        # Name: (check, success reward, failure reward).
        # Failure totals improve with progress: -1.0, -0.9, -0.8, -0.6, -0.3.
        # Correct SQL totals 0.95 before the small style adjustment.
        self.reward_steps = {
            "output":    (self.output_reward,    0.0,         -1.0 ),
            "refusal":   (self.refusal_reward,   [0.1, 1.0],  -1.0 ),
            "sqlglot":   (self.sqlglot_reward,   0.2,         -1.0 ),
            "policy":    (self.policy_reward,    0.2,         -1.0 ),
            "sqlfluff":  (self.sqlfluff_reward,  0.05,        -0.05),
            "execution": (self.execution_reward, 0.45,        -1.0 ),
        }

    def database(self, db_id):
        path = (self.db_dir / db_id / f"{db_id}.sqlite").resolve()
        if not path.is_relative_to(self.db_dir) or not path.is_file():
            raise FileNotFoundError(f"Training database not found: {path}")
        return path

    def read_query(self, sql):
        trees = self.sqlglot.parse(sql, read="sqlite")
        exp = self.sqlglot.exp
        if len(trees) != 1 or not isinstance(trees[0], exp.Query):
            raise ValueError("Only a single SELECT query is supported in this iteration")
        tree = trees[0]
        if any(isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Command))
               or node.args.get("into") or node.args.get("locks") for node in tree.walk()):
            raise ValueError("Write-state checking is deferred; modifying SQL is unsupported")
        return tree

    def validate_reference(self, db_id, gold):
        self.database(db_id)
        self.read_query(gold)

    def execute(self, db_id, sql):
        deadline = time.monotonic() + 5
        with closing(sqlite3.connect(self.database(db_id).as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            return connection.execute(sql).fetchall()

    def score(self, response, row):
        """Sum applicable checks, stopping when later checks no longer apply."""
        context = {"prediction": response, "row": row}
        total = 0.0
        for check, success, failure in self.reward_steps.values():
            reward, proceed = check(context, success, failure)
            total += reward
            if not proceed:
                break
        return total

    def output_reward(self, context, success, failure):
        """Clean the response; reject empty or unrecognized output."""
        response = context["prediction"]
        if not response.strip() or response.strip().lower() == "empty response":
            return failure, False
        prediction = self.clean(response)
        # Upstream's broad refusal detector includes the invalid-output sentinel
        # and SQL string literals containing "sorry". Neither is a model refusal.
        if not prediction or prediction.strip().lower() == "empty response":
            return failure, False
        context["prediction"] = prediction
        return success, True

    def refusal_reward(self, context, success, failure):
        """Score refusals/denied requests; allowed SQL continues without points."""
        prediction, row = context["prediction"], context["row"]
        allowed = row["metadata"]["permission"] == "allowed"
        if prediction.strip().lower() == "sorry, i cannot answer.":
            return (failure if allowed else success[1]), False
        if not allowed:
            return failure, False
        return success[0], True

    def sqlglot_reward(self, context, success, failure):
        """Reward supported SELECT syntax; invalid SQL skips policy/execution."""
        try:
            context["tree"] = self.read_query(context["prediction"])
        except (ValueError, self.sqlglot.errors.ParseError, self.sqlglot.errors.TokenError):
            return failure, False
        return success, True

    def policy_reward(self, context, success, failure):
        """Reward a complete policy check; never execute unauthorized SQL."""
        row = context["row"]
        access = self.checker.check(row["db_id"], context["prediction"], row["policy"])
        if access["status"] != "PASS" or access.get("not_checked"):
            return failure, False
        return success, True

    def execution_reward(self, context, success, failure):
        """Compare results with cached gold results; broken references still raise."""
        row = context["row"]
        db_id, gold = row["db_id"], row["metadata"]["gold_sql"]
        key = (db_id, gold)
        if key not in self.references:
            try:
                self.references[key] = self.execute(db_id, gold)
            except sqlite3.Error as error:
                raise RuntimeError(f"Cannot execute reference SQL for {db_id}") from error
        try:
            result = self.execute(db_id, context["prediction"])
        except sqlite3.Error:
            return failure, False
        correct = self.result_eq(result, self.references[key], order_matters=bool(context["tree"].args.get("order")))
        return (success if correct else failure), correct

    def sqlfluff_reward(self, context, success, failure):
        """Split the style reward equally across enabled rules."""
        if not self.sqlfluff_rules:
            return 0.0, True
        violations = self.lint_sql(context["prediction"], dialect="sqlite", rules=self.sqlfluff_rules)
        failed_rules = {violation["code"] for violation in violations}
        # Parse/templating errors mean the style checks could not complete.
        if not failed_rules.issubset(self.sqlfluff_rules):
            return failure, True
        passed = sum(rule not in failed_rules for rule in self.sqlfluff_rules)
        # success/failure are the totals for all rules passing/failing.
        reward = (passed * success + (len(self.sqlfluff_rules) - passed) * failure) / len(self.sqlfluff_rules)
        return reward, True


def sql_outcome_reward(postprocessed_responses, processing_class, scoress, labels,
                       reward_metadata, sql_scorer, **kwargs):
    rewards = torch.zeros_like(postprocessed_responses, dtype=torch.bfloat16)
    for index, (response, metadata) in enumerate(zip(postprocessed_responses, reward_metadata)):
        valid = (response == processing_class.pad_token_id).cumsum(0) == 0
        positions = valid.nonzero(as_tuple=True)[0]
        if not len(positions):
            rewards[index, 0] = sql_scorer.score("", json.loads(metadata))
            continue
        text = processing_class.decode(response[valid], skip_special_tokens=True)
        rewards[index, positions[-1]] = sql_scorer.score(text, json.loads(metadata))
    return rewards
