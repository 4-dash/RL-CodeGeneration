# Prompt2SQL extension plan

Goal: natural-language question + database schema + user role/permissions → SQL and query results, or a refusal. Prioritize steps 1–2; keep training as a small, extensible next step.

**Framework requirement:** SQL is another supported task alongside Python code generation and robotics. Use the same `main.py` entry point, configuration loader, dataset registration, PPO/LoRA trainer, and checkpoint conventions. Select SQL through `--param ppo_sql --framework_params sql`; keep existing Python and robotics commands working. Task-specific prompts, execution, rewards, and evaluators belong in modules called by this shared workflow.

## 1. Generate and execute SQL without training

- [ ] Add `ppo_sql` and `framework.sql` presets in `config/hyperparams.json`, expose them in `main.py`, and register the SQL dataset/prompt adapter in `rl_datasets/dataset_utils.py`. Include database paths, role policies, and benchmark selection in the SQL configuration.
- [ ] Extend the shared `generate_completion` mode for SQL, including a user question via `--prompt`, a database via `--database`, and optional execution via `--execute`. Reuse shared model loading and generation; any convenience script should delegate to this path. Start with `Qwen/Qwen2.5-Coder-1.5B-Instruct` without an adapter; load trained adapters through the existing `--checkpoint` convention. Record the existing Python-trained adapter as a separate baseline if tested.
- [ ] Accept a question and SQLite database path. Read the schema, including column types and foreign keys, and include it in a SQL-specific chat prompt. Add optional trusted role and table/column permissions; ordinary SQL benchmarks use full read access.
- [ ] Request exactly one SQL query or `Sorry, I cannot answer.` Parse these as separate outcomes; malformed output is an error. Provide a generation-only option before adding execution.
- [ ] Execute queries on local benchmark databases through a shared `sql/executor.py`. Use a read-only SQLite connection, a single-statement restriction, a timeout, and an authorizer that enforces table/column permissions and blocks writes and external access. Check columns used in filters and joins as well as returned columns. Role/permissions come from configuration, not claims inside the question.
- [ ] Return the SQL and result rows, a refusal, or an execution error. Limit displayed rows without changing the SQL or truncating results used for scoring. Keep model refusals distinct from executor blocks.

Make Python-specific Ruff and SynCode processing optional through task configuration from this first stage. Generation and baseline evaluation should use shared inference code without initializing PPO training, loading a critic, or deleting checkpoint directories. Use the same SQL prompt builder, generator, and executor for interactive use and benchmarks.

### Where step 1 goes

Paths marked **new** are planned additions.

| Location | Responsibility |
| --- | --- |
| `main.py` | Add SQL task selection and dispatch; accept question, database, execution, and checkpoint options. |
| `config/hyperparams.json` | Add `ppo_sql` and `framework.sql`; configure model, database/policy paths, and SQL options. |
| `utils.py` | Extend the existing configuration loader for SQL settings; keep inference separate from training/reward initialization. |
| `wrappers/wrapper_utils.py` | Reuse/refactor model loading and generation so inference loads only the policy and optional adapter. |
| `sql/__init__.py`, `sql/task.py` **(new)** | SQL package; schema extraction, prompt construction, and SQL/refusal parsing. |
| `sql/executor.py` **(new)** | Shared read-only execution, permission checks, timeouts, and result handling. |
| `rl_datasets/sql_dataset.py` **(new)** and `rl_datasets/dataset_utils.py` | Load and register SQL examples using the existing dataset-helper interface; reuse the SQL prompt builder. Interactive questions should not require downloading a training dataset. |
| `README.md` | Add SQL usage beside the existing Python and robotics commands. |

Keep `generate.py` as the existing Python convenience script. `rewards/` and `wrappers/ppo_wrapper.py` receive SQL training support in step 3; steps 1–2 share inference utilities without constructing the trainer.

**Done when:** SQL runs through `main.py`, and a few local examples cover a correct query, a join, an empty result, a denied request, and an execution failure. Smoke-check existing Python and robotics entry points after shared changes.

## 2. Establish benchmark results before training

### Benchmark choice

Start with **Spider 1.0 + RBAC-Spider**: measure SQL ability and role-aware behavior separately. Add the other benchmarks once this works.

| Benchmark | Purpose | Initial evaluation |
| --- | --- | --- |
| [Spider 1.0](https://github.com/taoyds/spider) | General SQL correctness | Public development set; execution accuracy and official [test-suite accuracy](https://github.com/taoyds/test-suite-sql-eval). Label single-database execution accuracy separately. |
| [RBAC-Text2SQL-Benchmark](https://huggingface.co/datasets/sharkiefff/RBAC-Text2SQL-Benchmark) | Correct SQL or refusal under role permissions | `spider` configuration first; use the [upstream evaluator](https://github.com/2020dfff/RBAC-Text2SQL-Benchmark) for AC-F1, Safe-EX, violation rate, and over-refusal rate. |
| [BIRD-SQL](https://bird-bench.github.io/) | Additional database-grounded SQL evaluation | Later: public development set with official execution accuracy; include supplied evidence consistently. |
| [SecureSQL](https://github.com/JacobiSong/SecureSQL) | Data leakage under natural-language security conditions | Later: separate adapter and safe/unsafe classification evaluation. Its policy format differs from RBAC; report any generation/refusal adaptation separately from official scores. |

### Data preparation

- [ ] Pin dataset revisions and evaluator commits; download the actual database files. RBAC supplies annotations, not databases.
- [ ] Respect RBAC configuration meanings: **`spider_train` is training data; `spider` and `bird` are evaluation data**, regardless of Hugging Face's `train` split label. Defer LiveSQLBench/CRUD support.
- [ ] Map RBAC `instruction` and `input` to the prompt, with trusted `role`/`policy`; use `output` as the target and `metadata.permission` / `metadata.gold_sql` for scoring only. Never expose gold SQL, permission labels, or denial explanations to the model. These fields and configuration meanings are documented in the [dataset card](https://huggingface.co/datasets/sharkiefff/RBAC-Text2SQL-Benchmark).
- [ ] Reserve validation data from `spider_train`, preferably by database. Check database/question overlap with evaluation data and keep all role variants of a question together. BIRD/SecureSQL results need the same overlap check before being described as held out.

### Run and save the baseline

- [ ] Add `eval/sql_eval.py` and dispatch it through `main.py --mode evaluate_sql`, following the existing `evaluate_passk` and `evaluate_roboeval` pattern. Use the shared inference interface and select the SQL benchmark through configuration or `--benchmark`. Smoke-test 20–50 fixed examples, then run the complete selected evaluation sets.
- [ ] Use one greedy generation per question, fixed token limits, and no repair/retry loop. Save model/tokenizer/adapter revisions, prompt, decoding settings, dataset IDs, split manifests, and execution limits.
- [ ] Score raw model outputs with the upstream evaluator in an isolated benchmark environment. Separately record what the application executor blocks; a blocked SQL query must not count as a model refusal. Preserve refusals, invalid SQL, errors, and timeouts in the evaluation population.
- [ ] Use benchmark result-comparison rules rather than SQL string equality. For authorized RBAC requests, report execution accuracy over **all authorized requests**, counting refusals/errors as failures; also report execution success rate.
- [ ] Save `results/sql/baseline/<run_id>/predictions.jsonl`, `metrics.json`, and a short `report.md`. Include raw output, parsed SQL/refusal, execution status, policy outcome, and per-example score. Break down results by role, allow/deny label, and difficulty.

### Where step 2 goes

| Location | Responsibility |
| --- | --- |
| `eval/sql_eval.py` **(new)** | SQL benchmark runner and upstream scorer integration, alongside `eval/roboeval.py` and `eval/evalplus_pass_k.py`. Reuse step-1 inference and execution modules. |
| `main.py`, `config/hyperparams.json` | Register `evaluate_sql` and benchmark selection, data paths, and evaluation limits. |
| `rl_datasets/sql_dataset.py` | Extend the step-1 loader for Spider/RBAC benchmark formats and explicit evaluation splits; retain scoring metadata outside model prompts. |
| `data/sql/<benchmark>/` **(new, local assets)** | Downloaded annotations, schemas, database files, and required upstream evaluator assets. Record paths/revisions; keep large downloads out of Git. |
| `results/sql/baseline/<run_id>/` **(new, generated outputs)** | Predictions, metrics, report, run configuration, and split/example manifests for reproducible scoring. |

`roboeval/` and `evalplus_passk/` currently hold robotics and Python evaluation outputs; SQL outputs go under `results/sql/`. `savings/` remains the shared trained-checkpoint location; evaluating the original model requires no new checkpoint.

**Done when:** saved predictions can be rescored without generating again, and the report contains actual pretraining Spider and RBAC results. Complete this before training changes.

## 3. Add SQL training support

- [ ] Extend the step-1 SQL dataset adapter for training, preserving the frozen splits and reward metadata. Make prompt length configurable; the current 512-token truncation can discard schemas and permissions.
- [ ] Complete the `ppo_sql` / `framework.sql` reward and training settings. Use the existing `main.py --mode fine_tune` and `PPOWrapper`, with the same PPO, LoRA, KL regularization, logging, and checkpoint lifecycle as the other tasks.
- [ ] Update `wrappers/ppo_wrapper.py` and its collator to pass database IDs, policies, permission labels, and reference SQL into reward functions. The current path mainly forwards Python unit tests.
- [ ] Extend the task-configurable bypass of Python SynCode and Ruff to training and validation; setting their weights to zero alone still leaves Python-specific processing active. Use response-length masks independent of syntax scores, retaining current Python/robotics behavior through their presets.
- [ ] Add `rewards/sql_rewards.py` and register it through the existing reward-loading mechanism. Begin with one outcome reward:

| Outcome | Starter reward |
| --- | --- |
| Denied request, model refuses | +1 |
| Authorized request, correct result and permitted SQL | +1 |
| Authorized request, wrong result or unnecessary refusal | -1 |
| Denied request answered, policy violation, invalid SQL, or timeout | -1 |

Apply the scalar once at the final response token. Log outcomes separately to detect an always-refuse policy. Leave SQLGlot analysis, SQLFluff linting, and dense/token-level rewards for later; auxiliary rewards must never compensate for a policy violation.

## 4. Train and evaluate the trained model

- [ ] Initialize from the same model used for the baseline. Run a small PPO/LoRA smoke experiment on `spider_train`, containing authorized and denied examples; verify reward values and metadata alignment before scaling up. Add supervised warm-up only if baseline behavior warrants it, and evaluate that checkpoint separately.
- [ ] Train using training data only. Monitor SQL correctness, refusal behavior, policy violations, KL, and timeouts on validation data. Select checkpoints using validation correctness and access-control metrics rather than training reward alone.
- [ ] Load the saved adapter through the same `main.py` generation/evaluation modes using `--checkpoint checkpoint-XXX`, resolved through the existing `savings/<model>/<dataset>/<checkpoint>/policy` convention. Rerun the frozen step-2 benchmarks with identical prompts, decoding, database files, evaluator versions, and execution rules.
- [ ] Save the same artifacts under `results/sql/trained/<run_id>/` and compare baseline versus trained metrics, including violation and over-refusal rates. Inspect examples that improved and regressed. Use the final held-out evaluation for reporting, not reward or checkpoint tuning.

### Intended usage after implementation

These are planned commands, following the existing framework interface:

```bash
# Generate SQL and return results; omit --execute for generation only.
python main.py --mode generate_completion --param ppo_sql --framework_params sql --prompt "How many customers are there?" --database data/example.sqlite --execute

# Evaluate the original model, before training; repeat with --benchmark rbac_spider.
accelerate launch --config-file config/accelerate.yml main.py --mode evaluate_sql --param ppo_sql --framework_params sql --benchmark spider

# Train SQL using the existing PPO/LoRA workflow.
accelerate launch --config-file config/accelerate.yml main.py --mode fine_tune --param ppo_sql --framework_params sql

# Evaluate a trained checkpoint through the same evaluator.
accelerate launch --config-file config/accelerate.yml main.py --mode evaluate_sql --param ppo_sql --framework_params sql --benchmark spider --checkpoint checkpoint-XXX
```

**First implementation milestone:** SQL generation/execution through the shared framework plus reproducible Spider 1.0 and RBAC-Spider baseline reports. Training follows after that milestone. Document these commands alongside the Python and robotics examples in `README.md`.
