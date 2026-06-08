# Towards Understanding Sycophancy in Language Models

Workspace for replicating parts of the paper.

- Paper: https://arxiv.org/abs/2310.13548
- Evaluation repository: https://github.com/meg-tong/sycophancy-eval

## Notes

- Sycophancy: a model seeks human approval in unwanted ways.
- Intuitively, the quality of an argument should depend only on its content.
  However, AI assistants provide more positive feedback about arguments the
  user likes and more negative feedback about arguments the user dislikes.

## Setup

From the repository root, activate the shared environment and load the
OpenRouter key:

```bash
source .venv/bin/activate
set -a
source .env
set +a
cd towards-understanding-sycophancy
```

View local eval runs:

```bash
inspect view
```

Launch the notebook:

```bash
jupyter lab experiments.ipynb
```

## Biased Feedback Pipeline

A reusable pipeline for measuring **feedback sycophancy**: whether a user's
stated framing changes a model's feedback about *identical* content. Each case
(`math_v1`, `arguments_v1`, ...) follows the same shape and the same three-batch
flow; only the dataset and the judge prompt are case-specific.

`<case>` below is a placeholder for the case name (e.g. `math_v1` or
`arguments_v1`).

### Dataset shape

Every case fixes a small set of items and shows each one under five framings:

```text
N items
× 5 framings: baseline, wrote, did-not-write, really-like, really-dislike
= 5N prompts   (N baseline + 4N biased)
```

The item content is byte-for-byte identical across framings, so any change in
feedback sentiment is sycophancy rather than a reaction to content.

### Models

```text
Target models (graded):  openrouter/openai/gpt-3.5-turbo   (epochs=2)
                         openrouter/qwen/qwen3-30b-a3b     (epochs=2)
                         openrouter/openai/gpt-5-mini      (epochs=1, modern)
Judge model (grader):    openrouter/minimax/minimax-m3
```

`gpt-5-mini` is a newer, cheaper model added as a low-sycophancy comparison. It
runs a single epoch (`-T epochs=1`) to keep cost down; older models keep the
default two epochs.

### Three batches

1. **Baseline** — `<case>_baseline` generates one neutral response per item from
   each target model (unscored). Export to a per-model lookup file.
2. **Biased** — `<case>_biased` generates every biased-framing response
   `epochs` times (default 2; pass `-T epochs=1` for single-pass models) per
   target model (unscored). Export to a per-model JSONL.
3. **Judge** — `<case>_biased_judge` compares every biased response with the
   matching baseline *from the same model and item*, twice, labelling each
   `MORE_POSITIVE`, `MORE_NEGATIVE`, or `SAME` with a short explanation.

Generation and judging are separate so the judge prompt or model can change
without regenerating target responses. The judge sees only the neutral baseline
feedback and the biased feedback — never any ground truth.

### Call counts

```text
Two-epoch model:  N baseline + 4N×2 biased                     = 9N  calls/model
Single-epoch model: N baseline + 4N×1 biased                   = 5N  calls/model
Judging: (biased responses) × 2 judge passes                   = minimax calls
```

For the three cases combined (N = 6, 3, 3), a single-epoch model such as
`gpt-5-mini` costs `5N` per case = 30 + 15 + 15 = **60 generation calls**.

### 1. Generate baselines

```bash
inspect eval evals/<case>.py@<case>_baseline \
  --model openrouter/openai/gpt-3.5-turbo --no-score --tags <case>,baseline,gpt35
inspect eval evals/<case>.py@<case>_baseline \
  --model openrouter/qwen/qwen3-30b-a3b --no-score --tags <case>,baseline,qwen3
```

Export each completed baseline log into a model-specific lookup:

```bash
python scripts/extract_<case>_baselines.py <gpt35-baseline-log.eval> \
  --output baselines/<case>_gpt35.json
python scripts/extract_<case>_baselines.py <qwen3-baseline-log.eval> \
  --output baselines/<case>_qwen3.json
```

### 2. Generate biased responses

`<case>_biased` sets `epochs=2` and has no scorer.

```bash
inspect eval evals/<case>.py@<case>_biased \
  --model openrouter/openai/gpt-3.5-turbo --no-score --tags <case>,biased,gpt35
inspect eval evals/<case>.py@<case>_biased \
  --model openrouter/qwen/qwen3-30b-a3b --no-score --tags <case>,biased,qwen3
```

Export each completed biased-response log:

```bash
python scripts/extract_<case>_biased_responses.py <gpt35-biased-log.eval> \
  --output responses/<case>_gpt35.jsonl
python scripts/extract_<case>_biased_responses.py <qwen3-biased-log.eval> \
  --output responses/<case>_qwen3.jsonl
```

#### Single-epoch modern model (gpt-5-mini)

Run baseline and biased once each, then export as usual:

```bash
inspect eval evals/<case>.py@<case>_baseline \
  --model openrouter/openai/gpt-5-mini --no-score --tags <case>,baseline,gpt5mini
inspect eval evals/<case>.py@<case>_biased -T epochs=1 \
  --model openrouter/openai/gpt-5-mini --no-score --tags <case>,biased,gpt5mini

python scripts/extract_<case>_baselines.py <gpt5mini-baseline-log.eval> \
  --output baselines/<case>_gpt5mini.json
python scripts/extract_<case>_biased_responses.py <gpt5mini-biased-log.eval> \
  --output responses/<case>_gpt5mini.jsonl
```

### 3. Judge biased responses

The `mockllm/model` target is never called: the judge task restores the saved
target feedback and only calls the `grader` role.

```bash
inspect eval evals/<case>.py@<case>_biased_judge \
  -T responses_file=responses/<case>_gpt35.jsonl \
  -T baseline_file=baselines/<case>_gpt35.json \
  --model mockllm/model --model-role grader=openrouter/minimax/minimax-m3 \
  --tags <case>,judge,gpt35

inspect eval evals/<case>.py@<case>_biased_judge \
  -T responses_file=responses/<case>_qwen3.jsonl \
  -T baseline_file=baselines/<case>_qwen3.json \
  --model mockllm/model --model-role grader=openrouter/minimax/minimax-m3 \
  --tags <case>,judge,qwen3
```

> The judge (`minimax-m3`) is a reasoning model; `arguments_v1` raises the
> grader `max_tokens` and tolerates truncated JSON (the sentiment label is
> emitted first), so verbose reasoning never aborts a run.

## Report

`scripts/build_report.py` turns the two judge logs into a single self-contained
HTML report (inline SVG, no extra dependencies). It shows, per model:

- **Sentiment shift from baseline** per framing as a grouped bar chart with
  error bars, where `MORE_POSITIVE` is +1, `SAME` is 0, and `MORE_NEGATIVE`
  is -1; zero means no bias.
- **Judge self-agreement**: how often the two judge passes returned the same
  label (per model and overall) — a sanity check on judge reliability.
- A handful of **examples**: the baseline feedback, the biased feedback, and
  both judge verdicts side by side.

```bash
python scripts/build_report.py \
  --title "arguments_v1 — Biased Feedback" \
  --output reports/arguments_v1.html \
  --judge "GPT-3.5-turbo" <gpt35-judge-log.eval>    baselines/arguments_gpt35.json \
  --judge "Qwen3-30B"     <qwen3-judge-log.eval>    baselines/arguments_qwen3.json \
  --judge "GPT-5-mini"    <gpt5mini-judge-log.eval> baselines/arguments_gpt5mini.json
```

Pass one `--judge` per model; the chart and tables grow to fit.

Open `reports/<case>.html` in a browser.

### Overall report (PDF)

`scripts/build_overall_report.py` combines all three cases and all models into a
single HTML with the written methodology/findings narrative and a cross-case
sycophancy-gap table, then headless Chrome renders it to `report.pdf` (committed
at the folder root). See `report.pdf` for the rendered summary of this
experiment.

### Files

```text
datasets/<case>.jsonl                       fixed 5N-prompt dataset
evals/<case>.py                             baseline, biased, and judge tasks
scripts/build_<case>.py                     (re)generate the dataset
scripts/extract_<case>_baselines.py         baseline log -> JSON lookup
scripts/extract_<case>_biased_responses.py  biased log -> JSONL responses
scripts/build_report.py                     judge logs -> per-case HTML report
scripts/build_overall_report.py             all cases -> combined HTML (-> PDF)
report.pdf                                  rendered overall report (committed)
logs/        generated Inspect logs, Git-ignored
baselines/   generated lookups, Git-ignored
responses/   generated responses, Git-ignored
reports/     generated HTML reports, Git-ignored

Implemented cases: math_v1, arguments_v1, poems_v1
```
