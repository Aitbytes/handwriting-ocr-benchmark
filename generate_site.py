#!/usr/bin/env python3
"""
Site generator: rebuilds index.html + per-document transcription pages from data.json.
Called by run_pipeline.py after benchmarks complete.
"""
import json, re, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "results" / "data.json"
TRANSCRIPTIONS_DIR = ROOT / "results" / "transcriptions"


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def load_data():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding='utf-8'))
    return {"documents": {}, "models": [], "results": {}, "meta": {}}


# ── Rankings computation ──
def compute_rankings(data):
    """Compute per-document rankings and aggregate rankings from results."""
    per_doc = {}
    for doc_id, models in data.get("results", {}).items():
        rankings = []
        for model_id, r in models.items():
            if r.get("success"):
                # Find model label
                label = model_id
                source = ""
                for m in data.get("models", []):
                    if m["id"] == model_id:
                        label = m["label"]
                        source = m["source"]
                        break
                rankings.append({
                    "model_id": model_id, "label": label, "source": source,
                    "elapsed_s": r.get("elapsed_s", 0),
                    "cost_usd": r.get("cost_usd", 0),
                    "total_tokens": r.get("total_tokens", 0),
                    "text_length": r.get("text_length", 0),
                })
            else:
                label = model_id
                for m in data.get("models", []):
                    if m["id"] == model_id:
                        label = m["label"]; break
                rankings.append({
                    "model_id": model_id, "label": label,
                    "elapsed_s": 0, "cost_usd": 0, "total_tokens": 0,
                    "failed": True, "error": r.get("error", "?")[:100]
                })
        # Sort by elapsed_s (faster = better proxy)
        rankings.sort(key=lambda x: (x.get("failed", False), -len(str(x.get("total_tokens", 0)))))
        per_doc[doc_id] = rankings

    # Aggregate: average time per model across all docs
    model_agg = {}
    for doc_id, rankings in per_doc.items():
        for r in rankings:
            mid = r["model_id"]
            if mid not in model_agg:
                model_agg[mid] = {"label": r["label"], "source": r.get("source",""), "times": [], "costs": [], "fails": 0, "successes": 0}
            if r.get("failed"):
                model_agg[mid]["fails"] += 1
            else:
                model_agg[mid]["times"].append(r["elapsed_s"])
                model_agg[mid]["costs"].append(r["cost_usd"])
                model_agg[mid]["successes"] += 1

    aggregated = []
    for mid, agg in model_agg.items():
        avg_time = sum(agg["times"]) / len(agg["times"]) if agg["times"] else 999
        avg_cost = sum(agg["costs"]) / len(agg["costs"]) if agg["costs"] else 0
        aggregated.append({
            "model_id": mid, "label": agg["label"], "source": agg["source"],
            "avg_time_s": round(avg_time, 1), "avg_cost_usd": round(avg_cost, 6),
            "successes": agg["successes"], "fails": agg["fails"],
        })
    aggregated.sort(key=lambda x: x["avg_time_s"])

    return per_doc, aggregated


# ── HTML generators ──
def build_transcription_page(doc_id, doc_info, models_data):
    """Build a tab-based transcription comparison page."""
    # Gather transcriptions from files
    transcriptions = {}
    for model_id, r in models_data.items():
        if r.get("success") and r.get("transcription_file"):
            tx_path = ROOT / r["transcription_file"]
            if tx_path.exists():
                transcriptions[model_id] = tx_path.read_text(encoding='utf-8', errors='replace')

    # Get labels
    labeled = []
    for model_id, text in transcriptions.items():
        label = model_id
        for m in data_models:
            if m["id"] == model_id:
                label = m["label"]; break
        labeled.append((label, model_id, text))

    # Sort by label
    labeled.sort(key=lambda x: x[0])

    title = doc_info.get("title", doc_id)

    tabs = ""
    panels = ""
    for i, (label, mid, text) in enumerate(labeled):
        sid = re.sub(r'[^a-z0-9-]', '', label.lower().replace(' ', '-').replace('.', ''))
        active = 'active bg-blue-500/20 text-blue-400' if i == 0 else 'text-slate-400 hover:text-white hover:bg-slate-800'
        tabs += f'<button onclick="showTab(\'{sid}\')" class="tab-btn px-3 py-1.5 rounded-lg text-xs font-medium transition {active}" id="tab-{sid}">{esc(label)}</button>\n'
        panels += f'<div id="panel-{sid}" class="tab-panel {"block" if i==0 else "hidden"}"><pre class="text-sm text-slate-300 font-mono whitespace-pre-wrap leading-relaxed bg-slate-950/50 rounded-xl p-5 border border-slate-800">{esc(text)}</pre></div>\n'

    return f'''<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Transcriptions — {esc(title)} | Benchmark OCR</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>body{{font-family:"Inter",system-ui,sans-serif}}.tab-btn.active{{background:rgba(59,130,246,.2);color:#60a5fa}}</style>
<script>function showTab(id){{document.querySelectorAll('.tab-panel').forEach(p=>p.classList.add('hidden'));document.querySelectorAll('.tab-btn').forEach(b=>{{b.classList.remove('active','bg-blue-500/20','text-blue-400');b.classList.add('text-slate-400','hover:text-white','hover:bg-slate-800')}});document.getElementById('panel-'+id).classList.remove('hidden');var btn=document.getElementById('tab-'+id);btn.classList.add('active','bg-blue-500/20','text-blue-400');btn.classList.remove('text-slate-400','hover:text-white','hover:bg-slate-800');window.location.hash=id}}window.onload=function(){{var h=window.location.hash.substring(1);if(h)showTab(h)}}</script>
</head>
<body class="bg-slate-950 text-white min-h-screen">
<div class="max-w-6xl mx-auto px-6 py-8">
<a href="../index.html" class="text-blue-400 hover:text-blue-300 text-sm mb-6 inline-block">← Retour au benchmark</a>
<h1 class="text-4xl font-bold mb-2">{esc(title)}</h1>
<p class="text-slate-400 mb-8">{esc(doc_info.get('description',''))} · <span class="text-slate-500">{len(labeled)} transcriptions</span></p>
<div class="flex flex-wrap gap-2 mb-8 border-b border-slate-800 pb-4">{tabs}</div>
{panels}
<div class="mt-12 text-center text-xs text-slate-600"><a href="../index.html" class="hover:text-slate-400">Benchmark OCR 2026</a></div>
</div></body></html>'''


def build_result_slide(doc_id, doc_info, rankings, index):
    """Build one per-document result slide (for inclusion in index.html)."""
    title = esc(doc_info.get("title", doc_id))
    desc = esc(doc_info.get("description", ""))
    tags_html = " ".join(f'<span class="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">{esc(t)}</span>' for t in doc_info.get("tags", []))

    rows = ""
    medals = ["🥇","🥈","🥉"]
    for i, r in enumerate(rankings[:8]):
        medal = medals[i] if i < 3 else str(i+1)
        row_class = ""
        if i == 0: row_class = "rank-gold"
        elif i == 1: row_class = "rank-silver"
        elif i == 2: row_class = "rank-bronze"
        if r.get("failed"):
            rows += f'<tr class="bg-red-500/5"><td class="py-3 px-3 text-red-400">✕</td><td class="py-3 px-3 text-red-400">{esc(r["label"])}</td><td class="py-3 px-3 text-right text-red-400">—</td><td class="py-3 px-3 text-right text-red-400">—</td><td class="py-3 px-3"><span class="text-red-400">ÉCHEC</span></td></tr>\n'
        else:
            rows += f'<tr class="{row_class}"><td class="py-3 px-3 text-slate-500">{medal}</td><td class="py-3 px-3 font-semibold">{esc(r["label"])}</td><td class="py-3 px-3 text-right text-slate-400">{r["elapsed_s"]:.1f}s</td><td class="py-3 px-3 text-right text-slate-400">{r["cost_usd"]:.4f} $</td><td class="py-3 px-3"><span class="text-xs">{esc(r.get("source",""))}</span></td></tr>\n'

    slide_id = f"doc-{index}"
    safe_doc = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)[:40]

    return f'''<!-- Slide: {title} -->
<section id="{slide_id}" class="slide px-8 py-24 border-t border-slate-800">
<div class="max-w-5xl mx-auto">
<div class="flex items-center justify-between mb-3">
<h2 class="text-4xl font-bold">{title}</h2>
<a href="transcriptions/{safe_doc}.html" class="text-sm text-green-400 hover:text-green-300 border border-green-500/30 rounded-lg px-3 py-1.5 transition">📄 Voir toutes les transcriptions →</a>
</div>
<p class="text-slate-400 mb-2">{desc}</p>
<div class="flex gap-2 mb-6">{tags_html}</div>
<table class="w-full text-sm">
<thead><tr class="border-b border-slate-700 text-slate-400 text-left"><th class="py-3 px-3 w-8">#</th><th class="py-3 px-3">Modèle</th><th class="py-3 px-3 text-right">Temps</th><th class="py-3 px-3 text-right">Coût</th><th class="py-3 px-3">Source</th></tr></thead>
<tbody class="divide-y divide-slate-800">{rows}</tbody></table>
</div>
</section>'''


def build_speed_cost_slide(aggregated):
    """Build the speed & cost analysis slide."""
    speed_rows = ""
    for i, m in enumerate(aggregated[:5]):
        if m["fails"] > m["successes"]: continue
        medals = ["🥇","🥈","🥉"]
        medal = medals[i] if i < 3 else str(i+1)
        speed_rows += f'<div class="flex items-center justify-between bg-slate-900/70 border border-slate-800 rounded-xl p-3"><div class="flex items-center gap-3"><span class="text-slate-500 w-6 text-center font-bold">{medal}</span><span>{esc(m["label"])}</span></div><span class="text-green-400 font-mono font-bold">{m["avg_time_s"]:.1f}s</span></div>\n'

    # Cost rankings
    cost_sorted = sorted([m for m in aggregated if m["successes"] > 0], key=lambda x: x["avg_cost_usd"])
    cost_rows = ""
    for i, m in enumerate(cost_sorted[:5]):
        medals = ["🥇","🥈","🥉"]
        medal = medals[i] if i < 3 else str(i+1)
        cost_str = "GRATUIT" if m["avg_cost_usd"] == 0 else f'{m["avg_cost_usd"]:.4f} $'
        cost_color = "text-green-400" if m["avg_cost_usd"] == 0 else "text-slate-400"
        cost_rows += f'<div class="flex items-center justify-between bg-slate-900/70 border border-slate-800 rounded-xl p-3"><div class="flex items-center gap-3"><span class="text-slate-500 w-6 text-center font-bold">{medal}</span><span>{esc(m["label"])}</span></div><div><span class="{cost_color} font-mono font-bold">{cost_str}</span></div></div>\n'

    return f'''<!-- Slide: Vitesse & Coût -->
<section id="slide-speed" class="slide px-8 py-24 border-t border-slate-800">
<div class="max-w-5xl mx-auto">
<h2 class="text-4xl font-bold mb-6">Analyse Vitesse &amp; Coût</h2>
<div class="grid grid-cols-2 gap-8 mb-10">
<div><h3 class="text-lg font-semibold mb-4 text-blue-400">⚡ Les Plus Rapides (moyenne)</h3><div class="space-y-3">{speed_rows}</div></div>
<div><h3 class="text-lg font-semibold mb-4 text-purple-400">💰 Coût par Image</h3><div class="space-y-3">{cost_rows}</div></div>
</div>
<div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-5">
<div class="text-xs text-slate-500 uppercase tracking-wider mb-3">Comparaison de Coût par Image</div>
<div class="flex items-end gap-3 h-20">
{"".join(f'<div class="flex flex-col items-center flex-1"><div class="text-xs {"text-green-400" if m["avg_cost_usd"]==0 else "text-red-400" if m["avg_cost_usd"]>0.05 else "text-slate-400"} font-bold mb-1">{m["avg_cost_usd"]:.4f} $</div><div class="w-full bg-gradient-to-t rounded-t h-{(m["avg_cost_usd"]*200):.0f}px" style="min-height:4px"></div><div class="text-xs text-slate-500 mt-1">{esc(m["label"])}</div></div>' for m in cost_sorted[:6])}
</div></div></div></section>'''


def build_final_rankings_slide(aggregated):
    """Build the final rankings slide."""
    best_cursive = sorted([m for m in aggregated if m["successes"] > 0], key=lambda x: x["avg_cost_usd"])[:5]
    best_value = sorted([m for m in aggregated if m["successes"] > 0], key=lambda x: x["avg_cost_usd"] / max(x["successes"], 1))[:5]

    cursive_rows = "\n".join(f'<div class="flex justify-between"><span>{"🥇🥈🥉"[i] if i<3 else str(i+1)} {esc(m["label"])}</span><span class="text-blue-400 font-mono">{m["successes"]}</span></div>' for i, m in enumerate(best_cursive))
    value_rows = "\n".join(f'<div class="flex justify-between"><span>{"🥇🥈🥉"[i] if i<3 else str(i+1)} {esc(m["label"])}</span><span class="text-purple-400 font-mono">{m["avg_cost_usd"]:.4f}$</span></div>' for i, m in enumerate(best_value))

    # Stats
    total_models = len([m for m in aggregated if m["successes"] > 0])
    fastest = min((m for m in aggregated if m["successes"] > 0), key=lambda x: x["avg_time_s"], default=None)
    cheapest = min((m for m in aggregated if m["successes"] > 0), key=lambda x: x["avg_cost_usd"], default=None)
    most_expensive = max((m for m in aggregated if m["successes"] > 0), key=lambda x: x["avg_cost_usd"], default=None)

    return f'''<!-- Slide: Classements Finaux -->
<section id="slide-rankings" class="slide px-8 py-24 border-t border-slate-800">
<div class="max-w-5xl mx-auto">
<h2 class="text-4xl font-bold mb-3">Classements Finaux</h2>
<p class="text-slate-400 mb-8">Score combiné sur tous les documents (rapidité × coût × fiabilité)</p>
<div class="grid grid-cols-2 gap-6 mb-10">
<div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
<div class="text-xs text-blue-400 uppercase tracking-wider mb-2">Les Plus Fiables</div>
<div class="space-y-3 mt-4">{cursive_rows}</div></div>
<div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
<div class="text-xs text-purple-400 uppercase tracking-wider mb-2">Les Moins Chers</div>
<div class="space-y-3 mt-4">{value_rows}</div></div>
</div>
<div class="grid grid-cols-4 gap-4">
<div class="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center"><div class="text-3xl font-bold text-green-400">{cheapest["avg_cost_usd"]:.4f} $</div><div class="text-xs text-slate-500">Modèle le moins cher</div><div class="text-sm text-slate-400 mt-1">{esc(cheapest["label"]) if cheapest else "—"}</div></div>
<div class="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center"><div class="text-3xl font-bold text-blue-400">{fastest["avg_time_s"]:.1f}s</div><div class="text-xs text-slate-500">Le plus rapide</div><div class="text-sm text-slate-400 mt-1">{esc(fastest["label"]) if fastest else "—"}</div></div>
<div class="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center"><div class="text-3xl font-bold text-amber-400">{total_models}</div><div class="text-xs text-slate-500">Modèles actifs</div></div>
<div class="bg-slate-900/70 border border-slate-800 rounded-xl p-4 text-center"><div class="text-3xl font-bold text-purple-400">{most_expensive["avg_cost_usd"]:.4f} $</div><div class="text-xs text-slate-500">Le plus cher</div><div class="text-sm text-slate-400 mt-1">{esc(most_expensive["label"]) if most_expensive else "—"}</div></div>
</div></div></section>'''


def build_sidebar(data, docs_list):
    """Build sidebar nav with dynamic document links."""
    doc_links = ""
    for i, (doc_id, doc_info) in enumerate(docs_list):
        title = esc(doc_info.get("title", doc_id)[:30])
        safe_doc = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)[:40]
        doc_links += f'<li><a href="#doc-{i+1}" class="block px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">📄 {title}</a></li>\n'

    tx_links = ""
    for doc_id, doc_info in docs_list:
        safe_doc = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)[:40]
        title = esc(doc_info.get("title", doc_id)[:25])
        tx_links += f'<a href="transcriptions/{safe_doc}.html" class="text-xs text-green-500 hover:text-green-400 block mt-1 transition">📄 {title}</a>\n'

    meta = data.get("meta", {})
    return f'''
<div class="mb-6"><div class="text-lg font-bold gradient-text">OCR<br>Benchmark</div><div class="text-xs text-slate-500 mt-1">{meta.get("last_run", "")[:10]}</div></div>
<div class="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-3">Contenu</div>
<ul class="space-y-1">
<li><a href="#slide-1" class="block px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">1. Vue d\'ensemble</a></li>
<li><a href="#slide-method" class="block px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">2. Méthodologie</a></li>
{doc_links}
<li><a href="#slide-speed" class="block px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Vitesse &amp; Coût</a></li>
<li><a href="#slide-rankings" class="block px-3 py-2 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Classements</a></li>
</ul>
<div class="mt-4 pt-4 border-t border-slate-800">
<div class="text-xs text-slate-600">Transcriptions</div>
{tx_links}
</div>'''


def build_index_html(data, per_doc, aggregated):
    """Build the complete index.html with all slides."""
    docs_list = list(data.get("documents", {}).items())
    meta = data.get("meta", {})

    # Build document result slides
    doc_slides = ""
    for i, (doc_id, doc_info) in enumerate(docs_list):
        rankings = per_doc.get(doc_id, [])
        doc_slides += build_result_slide(doc_id, doc_info, rankings, i+1) + "\n"

    sidebar = build_sidebar(data, docs_list)
    speed_slide = build_speed_cost_slide(aggregated)
    rankings_slide = build_final_rankings_slide(aggregated)

    # Method slide
    method_slide = '''<section id="slide-method" class="slide px-8 py-24 border-t border-slate-800">
<div class="max-w-5xl mx-auto"><h2 class="text-4xl font-bold mb-3">Méthodologie</h2>
<p class="text-slate-400 mb-10">Basé sur le <a href="https://aimultiple.com/handwriting-recognition" class="text-blue-400 hover:underline">benchmark d\'écriture cursive d\'AIMultiple</a>. Chaque image est envoyée à tous les modèles avec un prompt identique de transcription. Les résultats sont sauvegardés et le site est régénéré automatiquement.</p>
<div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-6"><h3 class="font-bold mb-3">Modèles</h3>
<div class="grid grid-cols-2 gap-4 text-sm">
<div><div class="text-blue-400 font-semibold mb-2">OpenRouter</div><ul class="space-y-1 text-slate-400"><li>GPT-5.6 Sol</li><li>Gemini 2.5 Pro · 3.5 Flash</li><li>Claude Sonnet 4.5 · Sonnet 5</li><li>Qwen3 VL 235B</li></ul></div>
<div><div class="text-green-400 font-semibold mb-2">HuggingFace Router (gratuit)</div><ul class="space-y-1 text-slate-400"><li>Qwen3.5 122B · Qwen3 VL 30B</li><li>Gemma 4 31B</li></ul></div>
</div></div>
<div class="mt-6 bg-slate-900/70 border border-slate-800 rounded-2xl p-5">
<div class="text-xs text-slate-500 uppercase tracking-wider mb-2">Pipeline CI/CD</div>
<p class="text-sm text-slate-400">Déposez une image dans <code class="bg-slate-800 px-1 rounded">images/</code> → push → GitHub Actions détecte la nouvelle image → exécute le benchmark → reconstruit le site → déploie sur GitHub Pages.</p>
</div></div></section>'''

    return f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Benchmark OCR Écriture Manuscrite — {meta.get("last_run","")[:10]}</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<style>body{{font-family:"Inter",system-ui,sans-serif}}html{{scroll-behavior:smooth}}.slide{{min-height:100vh}}
.rank-gold{{background:linear-gradient(135deg,#b8860b22,#ffd70011);border-left:3px solid #ffd700}}
.rank-silver{{background:linear-gradient(135deg,#80808022,#c0c0c011);border-left:3px solid #c0c0c0}}
.rank-bronze{{background:linear-gradient(135deg,#8b451322,#cd853f11);border-left:3px solid #cd853f}}
.gradient-text{{background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}</style>
</head>
<body class="bg-slate-950 text-white">

<nav class="fixed left-0 top-0 h-full w-60 bg-slate-950/95 border-r border-slate-800 p-5 z-50 backdrop-blur overflow-y-auto">
{sidebar}
<div class="mt-4 pt-4 border-t border-slate-800"><div class="text-xs text-slate-600">Basé sur</div><a href="https://aimultiple.com/handwriting-recognition" target="_blank" class="text-xs text-blue-500 hover:text-blue-400 transition">AIMultiple Benchmark</a></div></nav>

<main class="ml-60">

<!-- Slide 1: Titre -->
<section id="slide-1" class="slide flex flex-col items-center justify-center text-center px-8 py-32">
<div class="max-w-3xl">
<div class="inline-block px-4 py-1.5 bg-blue-500/15 text-blue-400 rounded-full text-sm font-medium mb-6">🔬 {meta.get("total_models",0)} Modèles · {meta.get("total_documents",0)} Documents</div>
<h1 class="text-6xl font-extrabold mb-4 leading-tight bg-gradient-to-r from-white via-blue-200 to-purple-300 bg-clip-text text-transparent">Reconnaissance<br>d\'Écriture Manuscrite</h1>
<p class="text-2xl text-slate-400 mb-8">LLMs vs Moteurs OCR — Benchmark Automatisé</p>
<div class="flex items-center justify-center gap-8 text-slate-500">
<div class="text-center"><div class="text-3xl font-bold text-white">{meta.get("total_models",0)}</div><div class="text-sm">Modèles</div></div>
<div class="text-center"><div class="text-3xl font-bold text-white">{meta.get("total_documents",0)}</div><div class="text-sm">Documents</div></div>
<div class="text-center"><div class="text-3xl font-bold text-white">{meta.get("total_runs",0)}</div><div class="text-sm">Transcriptions</div></div>
<div class="text-center"><div class="text-3xl font-bold text-green-400">{meta.get("total_cost_usd",0):.2f} $</div><div class="text-sm">Coût Total</div></div>
</div>
<div class="mt-8 text-sm text-slate-600">Dernière mise à jour : {meta.get("last_run","—")[:16]}</div>
</div>
</section>

{method_slide}
{doc_slides}
{speed_slide}
{rankings_slide}

</main></body></html>'''


def rebuild_from_data(data=None):
    """Main entry point called by run_pipeline.py."""
    if data is None:
        data = load_data()
    global data_models
    data_models = data.get("models", [])

    per_doc, aggregated = compute_rankings(data)
    docs_list = list(data.get("documents", {}).items())

    # Build index.html
    index_html = build_index_html(data, per_doc, aggregated)
    (ROOT / "index.html").write_text(index_html, encoding='utf-8')
    print(f"  ✓ index.html ({len(docs_list)} documents)")

    # Build per-doc transcription pages
    tx_dir = ROOT / "transcriptions"
    tx_dir.mkdir(exist_ok=True)
    for doc_id, doc_info in docs_list:
        models_data = data.get("results", {}).get(doc_id, {})
        page = build_transcription_page(doc_id, doc_info, models_data)
        safe_doc = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)[:40]
        (tx_dir / f"{safe_doc}.html").write_text(page, encoding='utf-8')
        print(f"  ✓ transcriptions/{safe_doc}.html")


if __name__ == "__main__":
    data = load_data()
    rebuild_from_data(data)
    print("✅ Site rebuilt")