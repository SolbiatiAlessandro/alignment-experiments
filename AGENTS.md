# Alignment Experiments

This repo contains small, practical alignment experiments used as working
evidence and demos for Alessandro's OpenAI interview research presentation.
The broader research framing, interview context, and presentation synthesis live
in `../alignment-knowledge-base`; treat that KB as the source of truth for how
these experiments should be explained.

## How we work together

- Responses are displayed in a terminal. Use terminal-friendly plain text.
- Be short by default. Lead with the answer or action, then expand only when it
  helps.
- Tag nontrivial research claims by confidence: verified with local/source
  evidence, fairly sure, or a guess. Do not present guesses as facts.
- Test Alessandro's ideas before building on them. Say plainly when a premise is
  solid, partly right, or wrong.
- Use plain words and standard terms. If a framing is your own synthesis, say so.
- Do not invent weird, complex labels or phrases and then keep reusing them as
  if they were established terminology. Prefer the ordinary phrase for the
  thing.
- Do not use LaTeX commands or display-math delimiters in chat replies. Write
  formulas in readable ASCII and put code in fenced blocks.

## Relationship to the Alignment KB

- For research explanation, presentation narrative, source discipline, and OAI
  interview framing, read `../alignment-knowledge-base/AGENTS.md` first.
- Useful KB entry points are `../alignment-knowledge-base/wiki/_index.md`,
  `wiki/concepts/`, `wiki/topics/`, `wiki/roles/`, and `wiki/prep/`.
- If an experiment here becomes part of the presentation, update or suggest an
  update in the KB rather than treating this repo as the canonical writeup.
- Keep claims about papers, OpenAI/Anthropic work, and interview expectations
  grounded in KB sources or fresh source checks.

## Repo layout

```text
.
├── AGENTS.md
├── pyproject.toml
├── uv.lock
├── model-organisms-emergent-misalignment/
└── towards-understanding-sycophancy/
```

`model-organisms-emergent-misalignment/` is a small educational replication of
the "Model Organisms for Emergent Misalignment" setup. It fine-tunes
`Qwen/Qwen2.5-1.5B-Instruct` with hand-written LoRA modules on bad medical
advice, then checks for out-of-domain degradation. It has its own local
`AGENTS.md`; follow that file inside the subfolder.

`towards-understanding-sycophancy/` contains compact Inspect/OpenRouter
experiments inspired by "Towards Understanding Sycophancy in Language Models":

- `3_1_biased_feedback/`: identical content under different user framings,
  measuring sentiment shift in model feedback.
- `swaying/`: two-turn factual QA with neutral pushback, measuring whether the
  model defends, hedges, concedes, or flips.

These experiments are diagnostic and presentation-oriented, not polished
statistical replications. Treat small-N results as directional unless expanded
and rechecked.

## Environment

Use the shared Python environment from the repo root:

```bash
uv sync
source .venv/bin/activate
```

Experiments that call hosted models usually need `.env` loaded:

```bash
set -a
source .env
set +a
```

Common dependencies are managed in `pyproject.toml`: `inspect-ai`, `openai`,
`python-dotenv`, `transformers`, `torch`, `modal`, `wandb`, `weave`,
`matplotlib`, and notebook tooling.

## Running experiments

For sycophancy evals:

```bash
source .venv/bin/activate
set -a
source .env
set +a
cd towards-understanding-sycophancy
inspect view
```

Read the relevant subfolder README before running batches:

- `towards-understanding-sycophancy/3_1_biased_feedback/README.md`
- `towards-understanding-sycophancy/swaying/README.md`

For model-organism LoRA work:

```bash
cd model-organisms-emergent-misalignment
uv run modal run train_lora.py::main --config training_configs/test_config.json
```

Modal and W&B setup may be required:

```bash
uv run modal setup
uv run modal secret create wandb-secret WANDB_API_KEY=...
```

## Data and generated artifacts

- Do not casually commit secrets, API keys, large generated logs, model weights,
  or private data.
- `.env`, generated Inspect logs, response dumps, local model checkpoints, and
  tokenized training data should be treated as local artifacts unless there is a
  specific reason to version them.
- Prefer keeping final human-readable reports (`report.pdf`, selected HTML
  reports, or presentation figures) when they are useful evidence for the KB or
  interview presentation.

## Coding conventions

- Preserve the local style in each subproject. These are research experiments,
  so clarity and inspectability matter more than premature abstraction.
- Keep changes narrow. Avoid unrelated refactors while debugging a run or
  preparing a result.
- Prefer structured parsing for logs, JSONL, and eval outputs. Avoid fragile
  string scraping when Inspect/Python data structures are available.
- For notebooks, put reusable logic into scripts when it becomes part of a
  repeatable pipeline.
- When changing evals or graders, record enough context in README/report text so
  old and new results are not accidentally compared as the same setup.

## Verification

Choose checks based on the touched area:

- Python syntax/import smoke tests for script edits.
- Small Inspect dry runs where cost is acceptable.
- Report rebuilds after changing report scripts.
- Modal dry runs or short configs for training changes.
- Manual review of generated examples when LLM judges are involved.

When reporting results, separate:

- what was actually run,
- what the grader or metric says,
- what interpretation is your synthesis,
- and what remains uncertain.
