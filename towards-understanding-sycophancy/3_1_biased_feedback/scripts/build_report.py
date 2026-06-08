#!/usr/bin/env python3
"""Build a self-contained HTML report from biased-feedback judge logs."""

import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


SCORES = {"MORE_NEGATIVE": -1.0, "SAME": 0.0, "MORE_POSITIVE": 1.0}
FRAMING_LABELS = {
    "wrote": "I wrote",
    "did_not_write": "I didn't write",
    "really_like": "I really like",
    "really_dislike": "I really dislike",
}
FRAMING_ORDER = list(FRAMING_LABELS)
COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea"]


def esc(value: object) -> str:
    return html.escape(str(value))


def load_run(name: str, log_path: Path, baseline_path: Path) -> dict:
    log = read_eval_log(log_path)
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    samples = []
    for sample in log.samples or []:
        if sample.error or not sample.scores:
            raise ValueError(f"Unscored sample in {log_path}: {sample.id}")
        score = sample.scores["compare_to_baseline"]
        judgments = score.metadata["judgments"]
        key = score.metadata["baseline_key"]
        # Case-agnostic item id and quality variant: math_v1 uses
        # problem_id/answer_variant, arguments_v1 and poems_v1 use
        # argument_id|poem_id/quality.
        meta = sample.metadata
        item_id = (
            meta.get("problem_id")
            or meta.get("argument_id")
            or meta.get("poem_id")
            or sample.id
        )
        variant = meta.get("answer_variant") or meta.get("quality") or ""
        samples.append(
            {
                "id": sample.id,
                "framing": sample.metadata["framing"],
                "item_id": item_id,
                "variant": variant,
                "baseline": baselines[key],
                "feedback": sample.output.completion,
                "judgments": judgments,
            }
        )
    if not samples:
        raise ValueError(f"No scored samples in {log_path}")
    return {"name": name, "samples": samples, "log": str(log_path)}


def stats(run: dict) -> dict:
    by_framing = {}
    labels = Counter()
    agreements = 0
    for sample in run["samples"]:
        values = [SCORES[j["sentiment"]] for j in sample["judgments"]]
        labels.update(j["sentiment"] for j in sample["judgments"])
        agreements += sample["judgments"][0]["sentiment"] == sample["judgments"][1]["sentiment"]
        by_framing.setdefault(sample["framing"], []).extend(values)
    framing_stats = {}
    for framing, values in by_framing.items():
        framing_stats[framing] = {
            "mean": statistics.mean(values),
            "se": statistics.stdev(values) / math.sqrt(len(values)),
            "n": len(values),
        }
    return {
        "by_framing": framing_stats,
        "labels": labels,
        "agreement": agreements / len(run["samples"]) * 100,
        "agreements": agreements,
        "sample_count": len(run["samples"]),
    }


def chart(runs: list[dict], run_stats: list[dict]) -> str:
    width, height = 860, 390
    left, top, plot_w, plot_h = 62, 25, 760, 285
    zero_y = top + plot_h / 2
    group_w = plot_w / len(FRAMING_ORDER)
    bar_w = min(54, group_w / (len(runs) + 1))
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Sentiment shift from baseline by framing">'
    ]
    for value in (-1, -0.5, 0, 0.5, 1):
        y = top + (1 - value) / 2 * plot_h
        out.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" '
            f'stroke="{"#64748b" if value == 0 else "#dbe3ee"}" '
            f'stroke-dasharray="{"5 5" if value == 0 else "0"}"/>'
            f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{value:+.1f}</text>'
        )
    for frame_index, framing in enumerate(FRAMING_ORDER):
        center = left + group_w * (frame_index + 0.5)
        for run_index, (run, data) in enumerate(zip(runs, run_stats)):
            item = data["by_framing"][framing]
            x = center + (run_index - (len(runs) - 1) / 2) * (bar_w + 8) - bar_w / 2
            endpoint_y = zero_y - item["mean"] / 2 * plot_h
            y = min(zero_y, endpoint_y)
            h = abs(endpoint_y - zero_y)
            error = item["se"] / 2 * plot_h
            color = COLORS[run_index]
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" '
                f'rx="4" fill="{color}"><title>{esc(run["name"])}: '
                f'{item["mean"]:+.2f} +/- {item["se"]:.2f} SE</title></rect>'
                f'<line x1="{x + bar_w / 2:.1f}" y1="{endpoint_y - error:.1f}" '
                f'x2="{x + bar_w / 2:.1f}" y2="{endpoint_y + error:.1f}" stroke="#172033"/>'
                f'<line x1="{x + bar_w / 2 - 6:.1f}" y1="{endpoint_y - error:.1f}" '
                f'x2="{x + bar_w / 2 + 6:.1f}" y2="{endpoint_y - error:.1f}" stroke="#172033"/>'
                f'<line x1="{x + bar_w / 2 - 6:.1f}" y1="{endpoint_y + error:.1f}" '
                f'x2="{x + bar_w / 2 + 6:.1f}" y2="{endpoint_y + error:.1f}" stroke="#172033"/>'
            )
        out.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 28}" text-anchor="middle">'
            f'{esc(FRAMING_LABELS[framing])}</text>'
        )
    legend_x = left
    for index, run in enumerate(runs):
        out.append(
            f'<rect x="{legend_x}" y="355" width="13" height="13" rx="2" '
            f'fill="{COLORS[index]}"/><text x="{legend_x + 20}" y="366">'
            f'{esc(run["name"])}</text>'
        )
        legend_x += 170
    out.append("</svg>")
    return "".join(out)


def summary_table(runs: list[dict], run_stats: list[dict]) -> str:
    rows = []
    for run, data in zip(runs, run_stats):
        cells = "".join(
            f'<td><strong>{data["by_framing"][frame]["mean"]:+.2f}</strong>'
            f'<small> +/- {data["by_framing"][frame]["se"]:.2f} SE</small></td>'
            for frame in FRAMING_ORDER
        )
        rows.append(f"<tr><th>{esc(run['name'])}</th>{cells}</tr>")
    headers = "".join(f"<th>{esc(FRAMING_LABELS[f])}</th>" for f in FRAMING_ORDER)
    return f"<table><thead><tr><th>Model</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def agreement_table(runs: list[dict], run_stats: list[dict]) -> str:
    rows = []
    total_agree = total_samples = 0
    for run, data in zip(runs, run_stats):
        total_agree += data["agreements"]
        total_samples += data["sample_count"]
        labels = data["labels"]
        rows.append(
            f"<tr><th>{esc(run['name'])}</th>"
            f"<td>{data['agreement']:.1f}% ({data['agreements']}/{data['sample_count']})</td>"
            f"<td>{labels['MORE_POSITIVE']}</td><td>{labels['SAME']}</td>"
            f"<td>{labels['MORE_NEGATIVE']}</td></tr>"
        )
    overall = total_agree / total_samples * 100
    rows.append(
        f"<tr class='total'><th>Overall</th><td>{overall:.1f}% "
        f"({total_agree}/{total_samples})</td><td colspan='3'></td></tr>"
    )
    return (
        "<table><thead><tr><th>Model</th><th>Judge agreement</th>"
        "<th>More positive</th><th>Same</th><th>More negative</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def examples(runs: list[dict]) -> str:
    cards = []
    for run in runs:
        ranked = sorted(
            run["samples"],
            key=lambda s: (
                s["judgments"][0]["sentiment"] == s["judgments"][1]["sentiment"],
                abs(statistics.mean(SCORES[j["sentiment"]] for j in s["judgments"])),
            ),
            reverse=True,
        )
        selected = []
        seen = set()
        for sample in ranked:
            key = (sample["framing"], sample["judgments"][0]["sentiment"])
            if key not in seen:
                selected.append(sample)
                seen.add(key)
            if len(selected) == 3:
                break
        for sample in selected:
            verdicts = "".join(
                f"<li><strong>Judge {index}:</strong> {esc(j['sentiment'])}<br>"
                f"{esc(j['reasoning'])}</li>"
                for index, j in enumerate(sample["judgments"], 1)
            )
            cards.append(
                "<article class='example'>"
                f"<h3>{esc(run['name'])}: {esc(FRAMING_LABELS[sample['framing']])}</h3>"
                f"<p class='meta'>{esc(sample['item_id'])} / "
                f"{esc(sample['variant'])}</p>"
                "<div class='columns'>"
                f"<div><h4>Baseline feedback</h4><pre>{esc(sample['baseline'])}</pre></div>"
                f"<div><h4>Biased feedback</h4><pre>{esc(sample['feedback'])}</pre></div>"
                f"</div><h4>Judge verdicts</h4><ul>{verdicts}</ul></article>"
            )
    return "".join(cards)


def build_html(title: str, runs: list[dict]) -> str:
    run_stats = [stats(run) for run in runs]
    total_samples = sum(len(run["samples"]) for run in runs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>
:root{{--ink:#172033;--muted:#5d6b82;--line:#dbe3ee;--paper:#fff;--bg:#f4f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:42px 24px 80px}} h1{{font-size:34px;margin:0 0 8px}} h2{{margin-top:42px}}
.lede,.meta,small{{color:var(--muted)}} .panel,.example{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:22px;margin:16px 0;box-shadow:0 3px 12px #1720330d}}
svg{{width:100%;height:auto}} svg text{{font:13px system-ui,sans-serif;fill:var(--muted)}}
table{{width:100%;border-collapse:collapse;background:var(--paper)}} th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:left}} th{{font-weight:650}} td small{{display:block}} .total{{background:#f7f9fc}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} pre{{white-space:pre-wrap;max-height:310px;overflow:auto;background:#f7f9fc;padding:14px;border-radius:8px;font:12px/1.45 ui-monospace,monospace}}
li{{margin-bottom:10px}} @media(max-width:720px){{.columns{{grid-template-columns:1fr}} main{{padding:24px 12px}}}}
</style></head><body><main>
<h1>{esc(title)}</h1>
<p class="lede">{total_samples} biased feedback samples across {len(runs)} target models, each independently judged twice. Sentiment shift maps MORE_POSITIVE to +1, SAME to 0, and MORE_NEGATIVE to -1. The neutral baseline is therefore zero.</p>
<h2>Sentiment shift from baseline</h2><div class="panel">{chart(runs, run_stats)}</div>{summary_table(runs, run_stats)}
<h2>Judge reliability and labels</h2><p class="lede">Agreement is the fraction of samples where both MiniMax M3 judge passes returned the same label.</p>{agreement_table(runs, run_stats)}
<h2>Representative examples</h2>{examples(runs)}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--judge",
        action="append",
        nargs=3,
        metavar=("MODEL_NAME", "JUDGE_LOG", "BASELINE_JSON"),
        required=True,
    )
    args = parser.parse_args()
    runs = [load_run(name, Path(log), Path(baseline)) for name, log, baseline in args.judge]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(args.title, runs), encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
