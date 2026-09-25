#!/usr/bin/env python3
"""Build the offline explorer and SVG launch figures from public result summaries."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK = "#111111"
MUTED = "#616161"
BG = "#ffffff"
BLUE = "#2563eb"


def svg_start(width, height, title, description):
    slug = "-".join(
        "".join(
            character.lower() if character.isascii() and character.isalnum() else " "
            for character in title
        ).split()
    )
    figure_id = f"{slug}-{width}-{height}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{figure_id}-title {figure_id}-desc">'
        f'<title id="{figure_id}-title">{html.escape(title)}</title>'
        f'<desc id="{figure_id}-desc">{html.escape(description)}</desc>'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
        "<style>text{font-family:Arial,Helvetica,sans-serif}</style>"
        '<path d="M40 30h22M40 30v22" fill="none" stroke="#111111"/>'
    )


def text(x, y, value, size=20, color=INK, weight=400):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{html.escape(str(value))}</text>'
    )


def build_results(results):
    benchmark = results["benchmarks"]["fresh"]
    methods = benchmark["methods"]
    displayed = ["water_fill", "coarse", "offset", "offset_certified"]
    labels = {
        "water_fill": "Fynd reference",
        "coarse": "Coarse + refinement",
        "offset": "Learned + refinement",
        "offset_certified": "Learned + checked skip",
    }
    bars = svg_start(
        1200,
        740,
        "Complete-solve time relative to Fynd",
        "Aggregate complete-solve time ratios on the fresh benchmark. "
        "Lower is faster. This is timing after the model freeze, separate from the held-out quality test.",
    )
    bars += text(64, 66, "SWAP ALLOCATION RESEARCH", 15, MUTED, 700)
    bars += text(64, 122, "Comparing initialization methods", 39, INK, 700)
    bars += text(64, 162, "Complete-solve time / Fynd reference · lower is faster", 22, MUTED)
    x0, scale = 385, 570
    bars += '<path d="M64 190H1136" stroke="#111111"/>'
    for index, method in enumerate(displayed):
        ratio = methods[method]["all"]["time_ratio_sum"]
        y = 222 + index * 79
        color = ["#b3b3b3", "#888888", "#7594cb", BLUE][index]
        bars += text(64, y + 27, labels[method], 21, INK, 600)
        bars += f'<rect x="{x0}" y="{y}" width="{scale}" height="38" fill="#f0f0f0"/>'
        bars += f'<rect x="{x0}" y="{y}" width="{scale * ratio:.3f}" height="38" fill="{color}"/>'
        bars += text(x0 + scale + 24, y + 28, f"{ratio:.3f}×", 26, INK, 700)
    bars += '<path d="M64 554H1136" stroke="#111111"/>'
    coarse_time = methods["coarse"]["all"]["time_ratio_sum"]
    learned_gain = 1 - methods["offset"]["all"]["time_ratio_sum"] / coarse_time
    checked_gain = 1 - methods["offset_certified"]["all"]["time_ratio_sum"] / coarse_time
    bars += text(
        64,
        596,
        f"Learned initialization cuts total solve time by {learned_gain:.1%} vs coarse.",
        22,
        INK,
        600,
    )
    bars += text(
        64,
        632,
        f"With checked refinement skipping, the reduction reaches {checked_gain:.1%}.",
        21,
        MUTED,
    )
    bars += text(
        64,
        687,
        f"Fresh benchmark · {benchmark['solves']:,} solves · "
        f"{benchmark['unique_orders']:,} unique orders · {len(benchmark['repeats'])} repeats",
        17,
        MUTED,
    )
    return bars + "</svg>"


def performance_metrics(results):
    method = results["benchmarks"]["fresh"]["methods"]["offset_certified"]
    return 1 / method["two_pool"]["time_ratio_median"], 1 - method["all"]["time_ratio_sum"]


def build_performance(results):
    speedup, reduction = performance_metrics(results)
    benchmark = results["benchmarks"]["fresh"]
    picture = svg_start(
        1200,
        680,
        "Learned warm starts: complete-solve performance",
        f"About {speedup:.1f} times faster median complete solves on eligible two-pool orders; "
        f"{reduction:.1%} less total time across the full fresh benchmark. "
        "Both compare learned warm starts with checked refinement skipping against the Fynd reference.",
    )
    picture += text(58, 65, "SWAP ALLOCATION RESEARCH", 15, MUTED, 700)
    picture += text(58, 119, "Less work for the numerical solver", 39, INK, 700)
    picture += text(58, 160, "Learned warm starts with checked refinement skipping", 24, MUTED)
    picture += '<rect x="58" y="202" width="542" height="302" fill="#111111"/>'
    picture += '<path d="M600 202H1142V504H600" fill="none" stroke="#111111"/>'
    picture += text(86, 315, f"≈{speedup:.1f}×", 96, "#ffffff", 700)
    picture += text(86, 364, "faster at the median", 26, "#ffffff", 600)
    picture += text(86, 427, "Complete solves on eligible", 23, "#c9c9c9")
    picture += text(86, 461, "two-pool orders", 23, "#c9c9c9")
    picture += text(632, 315, f"{reduction:.1%}", 90, INK, 700)
    picture += text(632, 364, "less total solve time", 26, INK, 600)
    picture += text(632, 427, "Across the full benchmark,", 23, MUTED)
    picture += text(632, 461, "including declined orders", 23, MUTED)
    picture += text(58, 555, "Both measured against the Fynd reference", 24, INK, 600)
    picture += text(
        58,
        597,
        f"Fresh benchmark · {benchmark['unique_orders']:,} orders · "
        f"{len(benchmark['repeats'])} repeats · frozen weights",
        20,
        MUTED,
    )
    picture += text(
        58,
        638,
        "Complete-solve timing includes inference, checks and remaining refinement.",
        19,
        MUTED,
    )
    return picture + "</svg>"


def build_architecture():
    picture = svg_start(
        1200,
        900,
        "Learned warm starts and a separate Codex experiment harness",
        "Within Fynd, a neural allocation initializer proposes a starting split. "
        "An allocation-regret estimator screens it for a numerical refinement skip check. "
        "Fynd refines when needed and selects the final result. Separately, Codex proposes "
        "bounded recipes for supervised training and validation. The frozen release and final test remain separate.",
    )
    picture += text(58, 60, "SWAP ALLOCATION RESEARCH", 15, MUTED, 700)
    picture += text(58, 111, "Learned warm starts inside Fynd", 39, INK, 700)
    picture += text(
        58,
        153,
        "The networks propose. Numerical checks decide when to skip refinement.",
        23,
        MUTED,
    )
    picture += '<path d="M58 190H1142" stroke="#111111"/>'
    picture += text(58, 226, "FROZEN METHOD · COMPLETE-SOLVE RUNTIME", 16, INK, 700)
    boxes = [
        (
            58,
            "Neural allocation",
            "initializer",
            ["Coarse split + learned", "correction"],
        ),
        (
            338,
            "Allocation-regret",
            "estimator",
            ["Screens the proposal", "for the skip check"],
        ),
        (618, "Refinement", "skip check", ["Neighboring quotes", "bound gross output"]),
        (
            898,
            "Fynd",
            "refinement",
            ["Refines when needed;", "selects the final result"],
        ),
    ]
    for index, (x, title, subtitle, lines) in enumerate(boxes):
        dark = index < 2
        fill, ink, muted = (INK, "#ffffff", "#cccccc") if dark else ("#ffffff", INK, MUTED)
        picture += (
            f'<rect x="{x}" y="250" width="244" height="194" fill="{fill}" stroke="#111111"/>'
        )
        picture += text(x + 16, 286, title, 21, ink, 700)
        picture += text(x + 16, 315, subtitle, 21, ink, 700)
        picture += f'<path d="M{x + 16} 337h212" stroke="{muted}" stroke-width="0.5"/>'
        for line_index, line in enumerate(lines):
            picture += text(x + 16, 374 + line_index * 28, line, 18, muted)
    for x in (308, 588, 868):
        picture += (
            f'<path d="M{x} 346h24m-8-7 8 7-8 7" fill="none" stroke="#111111" stroke-width="2"/>'
        )
    picture += text(
        58,
        489,
        "If initialization or skip screening declines, the solve continues through Fynd.",
        20,
        MUTED,
    )
    picture += '<path d="M58 538H1142" stroke="#111111" stroke-dasharray="4 5"/>'
    picture += text(58, 584, "CODEX EXPERIMENT HARNESS · SEPARATE TRAINING TRIAL", 16, INK, 700)
    stages = [
        (58, "Codex proposes", "A bounded training recipe"),
        (437, "Local code trains", "User-supplied prepared data"),
        (816, "Local code evaluates", "Validation results + timings"),
    ]
    for x, title, subtitle in stages:
        picture += f'<rect x="{x}" y="610" width="326" height="113" fill="#f4f4f4"/>'
        picture += f'<path d="M{x} 610h326" stroke="#111111" stroke-width="2"/>'
        picture += text(x + 20, 650, title, 24, INK, 600)
        picture += text(x + 20, 690, subtitle, 18, MUTED)
    for x in (395, 774):
        picture += (
            f'<path d="M{x} 667h30m-8-7 8 7-8 7" fill="none" stroke="#111111" stroke-width="2"/>'
        )
    picture += text(
        58,
        778,
        "Candidate trials do not replace the frozen method or use its final test.",
        22,
        INK,
        600,
    )
    picture += text(
        58,
        821,
        "The connected trial came after the original model development.",
        21,
        MUTED,
    )
    picture += text(58, 859, "Reusable autonomous orchestration remains in progress.", 19, MUTED)
    return picture + "</svg>"


DEMO = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swap allocation — recorded results explorer</title>
<style>
:root{--bg:#f4f4f4;--paper:#fff;--ink:#111;--muted:#606060;--line:#c9c9c9;--soft:#ededed;--blue:#2563eb;--amber:#b45309}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 Arial,Helvetica,sans-serif}
main{max-width:1240px;margin:auto;padding:40px 32px 26px}.eyebrow{font:600 11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.14em;text-transform:uppercase}.topline{display:flex;justify-content:space-between;gap:16px;align-items:center;border-top:2px solid var(--ink);border-bottom:1px solid var(--line);padding:14px 0}.badge{font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;color:var(--muted)}
h1{font-size:clamp(38px,5.6vw,70px);line-height:1.02;letter-spacing:-.055em;margin:34px 0 22px;max-width:1010px;font-weight:600}h2{font-size:24px;line-height:1.2;letter-spacing:-.025em;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}p{margin:0 0 14px}.intro{max-width:880px;color:var(--muted);font-size:18px;line-height:1.6}
.facts{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--ink);margin:32px 0 40px;background:var(--paper)}.fact{padding:23px 22px;border-right:1px solid var(--line)}.fact:last-child{border:0}.fact:first-child{background:var(--ink);color:#fff;border-color:var(--ink)}.fact strong{font-size:31px;letter-spacing:-.04em;display:block;font-weight:600;margin-bottom:8px;font-variant-numeric:tabular-nums}.fact span{font-size:13px;line-height:1.45;color:var(--muted)}.fact:first-child span{color:#ccc}
.explorer{display:grid;grid-template-columns:245px 1fr;gap:24px;margin-top:32px}.sidebar{padding-top:3px}.sidebar h2{font-size:16px;margin-bottom:17px}.cases{counter-reset:example}button{font:inherit;text-align:left;width:100%;background:var(--paper);border:1px solid var(--line);padding:17px 15px;margin:0 0 10px;color:var(--ink);cursor:pointer;position:relative;counter-increment:example}button::before{content:"0" counter(example);display:block;font:11px/1 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);margin-bottom:10px}button.active{border-color:var(--ink);background:var(--ink);color:#fff}button.active small,button.active::before{color:#ccc}button:hover{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}button:focus-visible{outline:2px solid var(--blue);outline-offset:4px}button strong{display:block;font-size:15px;line-height:1.35}button small{color:var(--muted);display:block;margin-top:7px;font-size:12px}
.panel{background:var(--paper);border:1px solid var(--line);border-top:2px solid var(--ink);padding:26px;min-width:0}.orderline{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}.orderline h2{margin-bottom:8px;font-variant-numeric:tabular-nums}.orderline .units{font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted)}.description{font-size:14px;color:var(--muted);max-width:740px;margin-bottom:24px}.legend{display:flex;gap:20px;font-size:12px;margin:8px 0 16px}.dot{display:inline-block;width:11px;height:11px;margin-right:7px;border:1px solid var(--ink)}.a{background:var(--blue)}.b{background:repeating-linear-gradient(135deg,var(--amber) 0,var(--amber) 4px,#d39b70 4px,#d39b70 5px)}.allocation{display:grid;grid-template-columns:repeat(3,1fr);gap:17px}.allocation h3{font-size:12px;min-height:32px}.bar{display:flex;height:25px;width:100%;overflow:hidden;border:1px solid var(--ink);background:var(--soft)}.bar div{min-width:0}.bar-labels{display:flex;justify-content:space-between;color:var(--muted);font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:8px}.status{margin:25px 0 18px;padding:14px 16px;background:var(--soft);border-left:3px solid var(--ink);font-size:13px}.status strong{display:block;margin-bottom:3px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;padding:11px 9px;border-bottom:1px solid var(--ink);white-space:nowrap}td{padding:13px 9px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}td:first-child{font-weight:600}th:first-child,td:first-child{padding-left:0}tr.chosen{background:#f4f4f4;color:var(--blue)}tr.chosen td{border-bottom:2px solid var(--blue)}.footnote{font-size:12px;color:var(--muted);line-height:1.6;margin-top:14px}.aggregate{display:grid;grid-template-columns:1.1fr 1fr;gap:30px;margin-top:28px}.aggregate h2{font-size:22px}.aggregate p{font-size:14px;color:var(--muted)}.chartrow{display:grid;grid-template-columns:165px 1fr 58px;gap:12px;align-items:center;font-size:12px;margin:18px 0}.track{height:17px;background:var(--soft);overflow:hidden}.track>div{height:100%;background:var(--blue)}.chartrow:first-child .track>div{background:#b3b3b3}.chartrow:nth-child(2) .track>div{background:#888}.chartrow:nth-child(3) .track>div{background:#7594cb}.ratio{text-align:right;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.scope{display:grid;grid-template-columns:1fr 1fr;gap:28px;border-top:1px solid var(--ink);padding-top:25px;margin-top:32px;font-size:13px;color:var(--muted)}.scope strong{color:var(--ink)}footer{display:flex;justify-content:space-between;gap:20px;font-size:11px;color:var(--muted);border-top:1px solid var(--line);padding:20px 0 0;margin-top:10px}a{color:var(--ink);text-underline-offset:3px}
@media(max-width:900px){.explorer{grid-template-columns:210px 1fr;gap:18px}.panel{padding:20px}.allocation{gap:10px}.aggregate{grid-template-columns:1fr;gap:12px}.facts .fact{padding:18px}.fact strong{font-size:27px}}
@media(max-width:800px){main{padding:24px 18px}.explorer{grid-template-columns:1fr;gap:16px}.cases{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}button{padding:13px;margin-bottom:0}button strong{font-size:13px}button small{font-size:11px}.sidebar h2{margin-bottom:12px}.facts{margin:25px 0 30px}.fact span{font-size:12px}.fact strong{font-size:24px}}
@media(max-width:480px){main{padding:18px 16px}.facts{grid-template-columns:1fr}.fact{border-right:0;border-bottom:1px solid var(--line)}.facts .fact{padding:17px 18px}.fact strong{font-size:29px;margin-bottom:4px}.fact span{font-size:12px}.topline{align-items:flex-start}.eyebrow{max-width:165px;font-size:10px}.badge{font-size:9px;max-width:132px;text-align:right;white-space:normal}.intro{font-size:16px}h1{margin-top:27px}.cases{grid-template-columns:1fr}button{padding:12px 14px}button::before{float:left;margin:3px 12px 24px 0}button small{margin-top:3px}.allocation{grid-template-columns:1fr;gap:14px}.allocation h3{margin-bottom:6px;min-height:0}.bar{height:20px}.bar-labels{margin-top:4px}.panel{padding:18px}.chartrow{grid-template-columns:133px 1fr 46px;gap:8px;font-size:11px}.ratio{font-size:11px}.scope{grid-template-columns:1fr;gap:10px}footer{display:block;font-size:11px}footer span{display:block;margin-bottom:8px}thead{display:none}table,tbody{display:block}tr{display:grid;grid-template-columns:1fr;gap:2px;padding:11px 0;border-bottom:1px solid var(--line)}td{display:flex;justify-content:space-between;border:0;padding:3px 0;font-size:12px}td:first-child{margin-bottom:3px}td[data-label]::before{content:attr(data-label);color:var(--muted);font-weight:400;margin-right:10px}tr.chosen td{border:0}tr.chosen{border-bottom:2px solid var(--blue)}}

</style></head><body><main>
<div class="topline"><div class="eyebrow">Swap allocation research</div><span class="badge">Recorded results · works offline</span></div>
<h1>Learned warm starts.<br>Less work for the solver.</h1>
<p class="intro">Neural allocation initialization with checked refinement skipping made eligible two-pool solves about __SPEEDUP__× faster at the median than the Fynd reference. Explore the recorded allocations, output and complete-solve timing below.</p>
<div class="facts"><div class="fact"><strong>≈__SPEEDUP__× faster</strong><span>Median complete solves on eligible two-pool orders vs Fynd</span></div><div class="fact"><strong>__REDUCTION__ less time</strong><span>Total solve time across the full fresh benchmark vs Fynd</span></div><div class="fact"><strong id="pairs"></strong><span>Orders in the separate held-out quality test; <span id="failures"></span> failures above 0.1 bp loss</span></div></div>
<div class="explorer"><aside class="sidebar"><h2>Choose a recorded example</h2><div class="cases" id="cases" aria-label="Recorded examples"></div><p class="footnote">Pool 1 and Pool 2 use the same order throughout. Allocation bars show input-token amounts.</p></aside>
<section class="panel" id="example" aria-live="polite"><div class="orderline"><h2 id="order"></h2><span class="units" id="direction"></span></div><p class="description" id="description"></p>
<div class="legend"><span><i class="dot a"></i>Pool 1</span><span><i class="dot b"></i>Pool 2</span></div><div class="allocation" id="allocations"></div>
<div class="status" id="status"></div><div class="table-wrap"><table><thead><tr><th>Method</th><th>Net output <span id="outunit"></span></th><th>vs Fynd</th><th>Total time</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="footnote">Net output includes modeled gas. One basis point (bp) is 0.01%. Initial bars show the proposal for the disjoint candidate; the final bar shows Fynd’s selected result. A row shows one measured solve; use the aggregate below for timing comparisons. The gross-output check does not guarantee net output after gas.</p></section></div>
<section class="panel aggregate"><div><div class="eyebrow">Beyond a single example</div><h2 style="margin-top:8px">Comparing initialization methods</h2><p>Learned initialization reduced total solve time by __LEARNED_GAIN__ compared with coarse initialization, with refinement enabled in both. Adding the checked refinement skip brought the reduction to __CHECKED_GAIN__.</p><p id="aggregate-note"></p><p>Bars show summed complete-solve time divided by the Fynd reference time. Lower is faster.</p></div><div id="chart" role="img" aria-label="Complete-solve time ratios, lower is faster"></div></section>
<div class="scope"><p><strong>How the method works</strong><br>The neural allocation initializer proposes a starting split. An allocation-regret estimator screens it for the numerical refinement skip check. Fynd supplies the reference solver and refines when needed.</p><p><strong>Evaluation scope</strong><br>These recorded results cover two WETH/USDC Uniswap v3 pools on Base, sampled order sizes and fixed gas assumptions. The explorer does not execute Fynd or fetch market data.</p></div>
<footer><span>Recorded examples from the benchmark. Explore your own historical inputs through the experiment harness.</span><span>Inspired by dSolver work at Dewiz. Fynd by Propeller Heads.</span></footer></main>
<script type="application/json" id="results-data">__RESULTS__</script><script type="application/json" id="examples-data">__EXAMPLES__</script>
<script>
'use strict';
const results=JSON.parse(document.getElementById('results-data').textContent);
const examples=JSON.parse(document.getElementById('examples-data').textContent).examples;
const labels={water_fill:'Fynd reference',coarse:'Coarse + refinement',offset:'Learned + refinement',offset_certified:'Learned + checked skip'};
const byId=id=>document.getElementById(id);
function units(raw, decimals, digits=6){const value=BigInt(raw);const sign=value<0n?'-':'';const absolute=value<0n?-value:value;const scale=10n**BigInt(decimals);const whole=(absolute/scale).toLocaleString('en-US');const fraction=(absolute%scale).toString().padStart(decimals,'0').slice(0,digits).replace(/0+$/,'');return sign+whole+(fraction?'.'+fraction:'');}
function split(values){const amounts=values.map(BigInt);const total=amounts.reduce((a,b)=>a+b,0n);return amounts.map(value=>total?Number(value*1000000n/total)/10000:0);}
function node(tag, cls, content){const element=document.createElement(tag);if(cls)element.className=cls;if(content!==undefined)element.textContent=content;return element;}
function allocation(title, amounts, example){const card=node('div');card.append(node('h3','',title));if(!amounts){card.append(node('p','footnote','No seed (one active pool)'));return card;}const percentages=split(amounts);const bar=node('div','bar');bar.setAttribute('role','img');bar.setAttribute('aria-label',amounts.map((amount,i)=>`Pool ${i?'2':'1'}: ${units(amount,example.input_decimals)} ${example.input_symbol}`).join('; '));amounts.forEach((amount,i)=>{const part=node('div',i?'b':'a');part.style.width=percentages[i]+'%';bar.append(part);});const values=node('div','bar-labels');percentages.forEach(percent=>values.append(node('span','',percent.toFixed(2)+'%')));card.append(bar,values);return card;}
function choose(index){const example=examples[index];document.querySelectorAll('#cases button').forEach((button,i)=>{button.classList.toggle('active',index===i);button.setAttribute('aria-pressed',String(index===i));});byId('order').textContent=units(example.amount_in,example.input_decimals)+' '+example.input_symbol;byId('direction').textContent=example.input_symbol+' → '+example.output_symbol;byId('description').textContent=example.description;byId('outunit').textContent='('+example.output_symbol+')';const methods=example.methods;const proposal=methods.offset_certified;byId('allocations').replaceChildren(allocation('Coarse starting split',methods.coarse.initial_pool_amounts,example),allocation('Neural allocation proposal',proposal.initial_pool_amounts,example),allocation('Final checked method',proposal.final_pool_amounts,example));const skipped=proposal.refinement_skipped;const declined=proposal.seed_status&&String(proposal.seed_status).toLowerCase().includes('declin');let headline=skipped?'Output check passed; refinement skipped.':declined?'Initializer declined; Fynd continued.':'Refinement remained necessary.';byId('status').replaceChildren(node('strong','',headline),document.createTextNode(skipped?'The gross-output check accepted this starting allocation.':declined?'The solver retained its numerical path for this order.':'The checked method completed its numerical refinement.'));byId('rows').replaceChildren();Object.keys(labels).forEach(key=>{const method=methods[key];if(!method)return;const row=node('tr',key==='offset_certified'?'chosen':'');row.append(node('td','',labels[key]),node('td','',units(method.net_out,example.output_decimals,8)),node('td','',(method.delta_bp>0?'+':'')+(Math.abs(method.delta_bp)>0&&Math.abs(method.delta_bp)<0.00001?Number(method.delta_bp).toExponential(2):Number(method.delta_bp).toFixed(5))+' bp'),node('td','',Number(method.elapsed_us).toLocaleString('en-US',{maximumFractionDigits:1})+' µs'));row.children[1].dataset.label='Net output ('+example.output_symbol+')';row.children[2].dataset.label='vs Fynd';row.children[3].dataset.label='Total time';byId('rows').append(row);});}
examples.forEach((example,index)=>{const button=node('button');button.type='button';button.append(node('strong','',example.title),node('small','',units(example.amount_in,example.input_decimals)+' '+example.input_symbol+' → '+example.output_symbol));button.onclick=()=>choose(index);byId('cases').append(button);});
byId('pairs').textContent=Number(results.held_out_runtime.successful_pairs).toLocaleString('en-US');byId('failures').textContent=results.held_out_runtime.gate_failures;const benchmark=results.benchmarks.fresh;byId('aggregate-note').textContent='Fresh benchmark: '+benchmark.solves.toLocaleString('en-US')+' solves across '+benchmark.unique_orders.toLocaleString('en-US')+' orders, '+benchmark.repeats.length+' repeats. This is separate from the held-out quality test.';
Object.keys(labels).forEach(key=>{const ratio=benchmark.methods[key].all.time_ratio_sum;const row=node('div','chartrow');const track=node('div','track');const fill=node('div');fill.style.width=(ratio*100)+'%';track.append(fill);row.append(node('span','',labels[key]),track,node('span','ratio',ratio.toFixed(3)+'×'));byId('chart').append(row);});choose(0);
</script></body></html>"""


def main():
    results = json.loads((ROOT / "reports/results.json").read_text())
    examples = json.loads((ROOT / "reports/examples.json").read_text())
    assets = ROOT / "assets"
    demo = ROOT / "demo"
    assets.mkdir(exist_ok=True)
    demo.mkdir(exist_ok=True)
    for name, body in [
        ("results", build_results(results)),
        ("architecture", build_architecture()),
        ("performance", build_performance(results)),
    ]:
        (assets / f"{name}.svg").write_text(body + "\n")
    methods = results["benchmarks"]["fresh"]["methods"]
    coarse_time = methods["coarse"]["all"]["time_ratio_sum"]
    speedup, reduction = performance_metrics(results)
    page = DEMO.replace("__SPEEDUP__", f"{speedup:.1f}").replace(
        "__REDUCTION__", f"{reduction:.1%}"
    )
    page = page.replace(
        "__LEARNED_GAIN__",
        f"{1 - methods['offset']['all']['time_ratio_sum'] / coarse_time:.1%}",
    )
    page = page.replace(
        "__CHECKED_GAIN__",
        f"{1 - methods['offset_certified']['all']['time_ratio_sum'] / coarse_time:.1%}",
    )
    page = page.replace("__RESULTS__", json.dumps(results).replace("<", "\\u003c"))
    page = page.replace("__EXAMPLES__", json.dumps(examples).replace("<", "\\u003c"))
    (demo / "index.html").write_text(page)
    print("Built demo/index.html and assets/{architecture,results,performance}.svg")


if __name__ == "__main__":
    main()
