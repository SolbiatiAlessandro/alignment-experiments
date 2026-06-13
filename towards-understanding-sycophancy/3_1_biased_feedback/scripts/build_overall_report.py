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
    label = case.removesuffix("_v1").title()
    html = (
        f'<section class="case-section"><div class="case-heading">'
        f'<p class="eyebrow">Domain detail</p><h3>{br.esc(label)}</h3></div>'
        f'<div class="panel chart-panel">{chart}</div><div class="table-wrap">{table}</div>'
        f'<details class="agree"><summary>Judge reliability &amp; label counts</summary>{agree}</details>'
        f'</section>'
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
        '<div class="table-wrap gap-table"><table><thead><tr><th>Case</th>'
        + head + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )


NARRATIVE_TOP = """
<p class="lede">Replication of §3.1 ("AI assistants can give biased feedback")
from <em>Towards Understanding Sycophancy in Language Models</em> (Sharma et al.,
2023). We test whether a user's stated framing shifts a model's feedback about
<em>identical</em> content, across three domains (math solutions, arguments,
poems), four framings, and three target models.</p>

<div class="panel method-panel">
<p class="eyebrow">Experiment design</p>
<h2>Method</h2>
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
<section class="report-section"><p class="eyebrow">Headline metric</p>
<h2>Cross-case sycophancy gap <span>(like &minus; dislike)</span></h2>
<p class="lede">The paper's headline statistic: how much more positive feedback
becomes when the user likes versus dislikes the same text. Range is -2..+2;
larger = more sycophantic.</p>
{gap_summary(case_gaps, model_order)}
</section>

<section class="report-section"><p class="eyebrow">Interpretation</p>
<h2>Findings</h2>
<div class="panel findings-panel">
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
</section>

<aside class="caveat"><p class="eyebrow">Read with care</p><h2>Caveats</h2>
<p>Small N (3–6 items/domain); a different judge than the paper
(minimax-m3 at 67–96% self-agreement vs their GPT-4); arguments are all "weak"
quality; GPT-5-mini ran a single epoch (half the per-cell samples of the other
models). Treat magnitudes as indicative, directions as robust.</p>
</aside>
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
        + '<section class="detail-intro"><p class="eyebrow">Full results</p>'
        + '<h2>Per-case detail</h2><p class="lede">Mean sentiment shift from each model’s neutral baseline. Error bars show standard error.</p></section>'
        + "".join(sections)
    )
    # Reuse build_report's stylesheet (read from source to avoid running its
    # HTML builder with empty runs).
    src = Path(br.__file__).read_text(encoding="utf-8")
    style = src.split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{br.esc(title)}</title><style>{style}
:root{{--ink:#18202f;--muted:#667085;--line:#e3e8ef;--paper:#fff;--bg:#f5f7fa;--accent:#3157d5;--accent-soft:#eef2ff}}
body{{background:linear-gradient(180deg,#eef2f9 0,#f7f8fb 360px,var(--bg) 100%);letter-spacing:-.005em}}
main{{max-width:1160px;padding:64px 28px 100px}}
.hero{{position:relative;overflow:hidden;padding:40px 44px;margin-bottom:28px;background:linear-gradient(135deg,#17213b,#263b72);color:#fff;border-radius:24px;box-shadow:0 24px 60px #17213b26}}
.hero:after{{content:"";position:absolute;width:300px;height:300px;right:-80px;top:-150px;border-radius:50%;background:#708cff33}}
.hero h1{{position:relative;max-width:760px;font-size:clamp(36px,6vw,62px);line-height:1.03;letter-spacing:-.045em;margin:8px 0 16px}}
.hero .lede{{position:relative;max-width:820px;color:#d8e0f4;font-size:17px;margin:0}}
.eyebrow{{margin:0 0 7px;color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}
.hero .eyebrow{{color:#9db0ff}}
h2{{font-size:27px;line-height:1.2;letter-spacing:-.025em;margin:0 0 10px}}
h2 span{{color:var(--muted);font-size:.7em;font-weight:550}}
h3{{font-size:24px;letter-spacing:-.02em;margin:0}}
.panel{{border-radius:18px;padding:28px;box-shadow:0 8px 26px #1720330a}}
.method-panel{{display:grid;grid-template-columns:190px 1fr;column-gap:28px;align-items:start}}
.method-panel .eyebrow,.method-panel h2{{grid-column:1}}
.method-panel ul{{grid-column:2;grid-row:1 / span 2;margin:0;padding-left:20px}}
.report-section{{margin-top:58px}}
.findings-panel{{border-left:4px solid var(--accent);background:linear-gradient(110deg,#fff,#fafbff)}}
.findings-panel li::marker{{color:var(--accent)}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:#fff;box-shadow:0 8px 24px #17203308}}
table{{min-width:680px;border:0}}
th,td{{padding:14px 17px}}
thead th{{background:#f8fafc;color:#475467;font-size:12px;letter-spacing:.035em;text-transform:uppercase}}
tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
tbody tr:hover{{background:#f8faff}}
.gap-table td strong{{display:inline-block;padding:4px 9px;border-radius:999px;background:var(--accent-soft);color:#2949b5}}
.caveat{{margin-top:48px;padding:24px 28px;background:#fffaf0;border:1px solid #f5dfb4;border-radius:18px}}
.caveat .eyebrow{{color:#a56305}} .caveat h2{{font-size:21px}} .caveat p:last-child{{margin-bottom:0;color:#6f5628}}
.detail-intro{{margin-top:76px;padding-top:32px;border-top:2px solid var(--ink)}}
.case-section{{margin-top:40px;padding:30px;background:#fff;border:1px solid var(--line);border-radius:22px;box-shadow:0 12px 32px #1720330a}}
.case-heading{{margin-bottom:18px}} .chart-panel{{padding:14px;margin:0 0 18px;background:#fbfcfe;box-shadow:none}}
.case-section .table-wrap{{box-shadow:none}}
details.agree{{margin-top:12px;overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fbfcfe}}
details.agree summary{{cursor:pointer;color:#475467;padding:13px 16px;font-weight:650}}
details.agree[open] summary{{border-bottom:1px solid var(--line)}}
details.agree .table-wrap{{border:0}} details.agree table{{border-radius:0}}
svg rect{{transition:opacity .15s}} svg rect:hover{{opacity:.72}}
@media(max-width:720px){{main{{padding:20px 12px 60px}}.hero{{padding:28px 22px;border-radius:18px}}.hero h1{{font-size:38px}}.method-panel{{display:block}}.method-panel ul{{padding-left:18px}}.panel,.case-section{{padding:20px}}.case-section{{border-radius:16px}}}}
@media print{{body{{background:#fff}}main{{padding:0}}.hero{{background:#fff;color:var(--ink);box-shadow:none;border:1px solid var(--line)}}.hero .lede{{color:var(--muted)}}.hero .eyebrow{{color:var(--accent)}}.panel,table,.example,.case-section,.table-wrap{{box-shadow:none}}.case-section{{break-inside:avoid}}}}
</style></head><body><main>
<header class="hero"><p class="eyebrow">Replication report · Section 3.1</p>
<h1>{br.esc(title)}</h1></header>
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
