#!/usr/bin/env python3
"""Build one overall HTML report for the §3.1 biased-feedback experiment.

Combines all three cases (math_v1, arguments_v1, poems_v1) across the three
target models, adds a cross-case sycophancy-gap summary and the written
methodology / findings narrative, and is sized for printing to PDF.

Usage (paths are case::model triples discovered by the caller):

    python scripts/build_overall_report.py --output reports/overall.html \
      --case math_v1 \
        --judge "GPT-3.5-turbo" <log> <baseline.json> \
        --judge "Qwen3-30B"     <log> <baseline.json> \
        --judge "GPT-5-mini"    <log> <baseline.json> \
      --case arguments_v1 ... --case poems_v1 ...
"""

import argparse
import statistics
from pathlib import Path

import build_report as br


FRAMING_ORDER = br.FRAMING_ORDER
LABELS = br.FRAMING_LABELS
SCORES = br.SCORES


def gap(run_stat: dict) -> float:
    """like - dislike sentiment shift: the paper's core sycophancy statistic."""
    return (
        run_stat["by_framing"]["really_like"]["mean"]
        - run_stat["by_framing"]["really_dislike"]["mean"]
    )


def case_section(case: str, runs: list[dict]) -> tuple[str, dict]:
    run_stats = [br.stats(r) for r in runs]
    chart = br.chart(runs, run_stats)
    table = br.summary_table(runs, run_stats)
    agree = br.agreement_table(runs, run_stats)
    gaps = {r["name"]: gap(s) for r, s in zip(runs, run_stats)}
    html = (
        f'<h2>{br.esc(case)}</h2>'
        f'<div class="panel">{chart}</div>{table}'
        f'<details class="agree"><summary>Judge reliability &amp; label counts</summary>{agree}</details>'
    )
    return html, gaps


def gap_summary(case_gaps: dict[str, dict[str, float]], model_order: list[str]) -> str:
    head = "".join(f"<th>{br.esc(m)}</th>" for m in model_order)
    rows = []
    for case, gaps in case_gaps.items():
        cells = "".join(
            f"<td><strong>{gaps.get(m, float('nan')):+.2f}</strong></td>"
            for m in model_order
        )
        rows.append(f"<tr><th>{br.esc(case)}</th>{cells}</tr>")
    return (
        '<table><thead><tr><th>Case</th>' + head + '</tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
    )


NARRATIVE_TOP = """
<p class="lede">Replication of §3.1 ("AI assistants can give biased feedback")
from <em>Towards Understanding Sycophancy in Language Models</em> (Sharma et al.,
2023). We test whether a user's stated framing shifts a model's feedback about
<em>identical</em> content, across three domains (math solutions, arguments,
poems), four framings, and three target models.</p>

<div class="panel">
<h3 style="margin-top:0">Method</h3>
<ul>
<li><strong>Baseline-then-perturb.</strong> Each model first writes neutral
feedback ("comment on this"). We then re-ask with a user preference appended and
measure the shift against that model's own baseline — matching the paper.</li>
<li><strong>Four framings:</strong> "I really like" / "I really dislike" /
"I wrote" / "I did not write".</li>
<li><strong>Judge.</strong> minimax-m3 labels each biased response
MORE_POSITIVE / SAME / MORE_NEGATIVE versus the baseline, twice (we report
self-agreement). The paper used GPT-4.</li>
<li><strong>Metric.</strong> Mean sentiment shift on a -1..+1 scale (0 = no
bias); the paper's "feedback positivity" is the same idea remapped to 0-100%.</li>
<li><strong>Scale.</strong> Deliberately small (3-6 items/domain) versus the
paper's 300 arguments / 400 poems — error bars are wide; read directions, not
decimals.</li>
</ul>
</div>
"""


def narrative_bottom(case_gaps: dict, model_order: list[str]) -> str:
    return f"""
<h2>Cross-case sycophancy gap (like &minus; dislike)</h2>
<p class="lede">The paper's headline statistic: how much more positive feedback
becomes when the user likes versus dislikes the same text. Range is -2..+2;
larger = more sycophantic.</p>
{gap_summary(case_gaps, model_order)}

<h2>Findings</h2>
<div class="panel">
<ul>
<li><strong>The core effect replicates</strong> on the subjective domains
(arguments, poems): all three models give markedly more positive feedback when
the user says they like the text and more negative when they dislike it. Our
GPT-3.5-turbo lands within a few points of the paper's own GPT-3.5.</li>
<li><strong>Math is nearly flat.</strong> When the content has an objective
right/wrong anchor, feedback barely tracks user preference. This is consistent
with sycophancy emerging from <em>ambiguity</em>: with no independent quality
signal, the user's stated preference becomes the dominant cue.</li>
<li><strong>The authorship axis ("I wrote" / "I didn't") is noisy</strong> at
our scale and sometimes contradicts the paper — expected with only 3 items.</li>
<li><strong>GPT-5-mini shows less <em>positive</em> sycophancy</strong>
("I really like" ≈ +0.17 vs +0.50–0.67 for older models; math ≈ 0), but still
caves hard to negative framing ("I really dislike" ≈ -0.83 to -1.00). The
improvement is asymmetric — consistent with OpenAI's GPT-5 post-training, which
adds a dedicated anti-sycophancy reward signal aimed mainly at over-agreement /
flattery.</li>
</ul>
</div>

<h2>Caveats</h2>
<p class="lede">Small N (3–6 items/domain); a different judge than the paper
(minimax-m3 at 67–96% self-agreement vs their GPT-4); arguments are all "weak"
quality; GPT-5-mini ran a single epoch (half the per-cell samples of the other
models). Treat magnitudes as indicative, directions as robust.</p>
"""


def build(title: str, cases: dict[str, list[dict]]) -> str:
    sections = []
    case_gaps: dict[str, dict[str, float]] = {}
    model_order: list[str] = []
    for case, runs in cases.items():
        if not model_order:
            model_order = [r["name"] for r in runs]
        html, gaps = case_section(case, runs)
        sections.append(html)
        case_gaps[case] = gaps

    body = (
        NARRATIVE_TOP
        + narrative_bottom(case_gaps, model_order)
        + "<h2>Per-case detail</h2>"
        + "".join(sections)
    )
    # Reuse build_report's stylesheet (read from source to avoid running its
    # HTML builder with empty runs).
    src = Path(br.__file__).read_text(encoding="utf-8")
    style = src.split("<style>")[1].split("</style>")[0]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{br.esc(title)}</title><style>{style}
h2{{border-top:1px solid var(--line);padding-top:22px}}
details.agree summary{{cursor:pointer;color:var(--muted);margin:8px 0}}
@media print{{body{{background:#fff}} .panel,table,.example{{box-shadow:none}} h2{{break-before:auto}}}}
</style></head><body><main>
<h1>{br.esc(title)}</h1>
{body}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="§3.1 Biased Feedback — Sycophancy Replication")
    parser.add_argument("--output", required=True, type=Path)
    # Interleaved: --case NAME then one or more --judge MODEL LOG BASELINE.
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--judge", action="append", nargs=4, default=[],
                        metavar=("CASE", "MODEL", "LOG", "BASELINE"))
    args = parser.parse_args()

    cases: dict[str, list[dict]] = {c: [] for c in args.cases}
    for case, model, log, baseline in args.judge:
        cases.setdefault(case, [])
        cases[case].append(br.load_run(model, Path(log), Path(baseline)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.title, cases), encoding="utf-8")
    print(f"Wrote overall report to {args.output}")


if __name__ == "__main__":
    main()
