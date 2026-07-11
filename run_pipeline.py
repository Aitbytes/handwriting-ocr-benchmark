#!/usr/bin/env python3
"""
Pipeline: detect new images → run benchmark → rebuild site.
Designed for GitHub Actions but also works locally.

Usage:
  python3 run_pipeline.py          # detect + benchmark + rebuild
  python3 run_pipeline.py --rebuild-only  # just rebuild site from existing data
  python3 run_pipeline.py --check    # only detect new images, don't run
"""
import json, hashlib, os, sys, time, base64, subprocess, tempfile, ssl
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"
DATA_FILE = RESULTS_DIR / "data.json"
TRACKER_FILE = RESULTS_DIR / "tracker.json"
TRANSCRIPTIONS_DIR = RESULTS_DIR / "transcriptions"

PROMPT = """Transcribe all handwritten text in this image verbatim. Include all text, numbers, code, punctuation, and special characters exactly as written. Preserve line breaks and formatting. For any text you cannot read confidently, mark it as [?]. Output ONLY the transcribed text, nothing else."""

# ── AI Analysis prompt template ──
PROMPT_TEMPLATE = """Tu es un analyste expert en OCR et IA. Analyse ces résultats de benchmark d'écriture manuscrite et produis TROIS sections HTML détaillées en français. Chaque carte doit faire 3-5 phrases riches en données, pas juste un titre.

## Section 1 : Surprises & Découvertes (id="slide-ai-findings")
4 cartes (grid-cols-2). Chaque carte doit citer des chiffres précis et expliquer POURQUOI c'est surprenant.

EXEMPLE de carte bien rédigée :
<div class="bg-slate-900/70 border border-red-500/20 rounded-2xl p-5">
  <div class="flex items-center gap-2 mb-2"><span class="text-xl">\u274c</span><h3 class="font-bold text-red-400">GPT-5 Bloqué par le Filtre de Contenu</h3></div>
  <p class="text-sm text-slate-300">GPT-5 a totalement refusé de transcrire le document I2C — réponse vide. Le modèle a probablement interprété le code C et les schémas techniques comme du contenu sensible, déclenchant le filtre de sécurité d'OpenAI. C'est un échec critique pour un usage académique ou ingénierie.</p>
</div>

## Section 2 : Recommandations (id="slide-ai-recommendations")
4 cartes (grid-cols-2) couvrant : précision maximale, meilleur rapport qualité/prix, pipeline de production, choix équilibré. Nommer les modèles, leurs métriques et le scénario d'usage.

EXEMPLE de carte bien rédigée :
<div class="bg-gradient-to-br from-green-500/10 to-transparent border border-green-500/20 rounded-2xl p-5">
  <div class="text-xs text-green-400 uppercase tracking-wider mb-2">Meilleur Rapport Qualité/Prix</div>
  <h3 class="text-lg font-bold mb-2">Qwen3.5 122B + Gemma 4 31B</h3>
  <p class="text-sm text-slate-400">Ces deux modèles sont GRATUITS via HuggingFace Router. Qwen3.5 122B excelle sur les documents structurés (9,5/10 sur le formulaire BTS), tandis que Gemma 4 31B est le plus rapide (3,1s en moyenne). Pour 0$, vous couvrez 90% des cas d'usage.</p>
</div>

## Section 3 : Modèles Non Testés (id="slide-ai-untested")
Liste TOUS les modèles du benchmark AIMultiple non testés, avec pour chacun : pourquoi pas testé, comment le tester concrètement.

EXEMPLE d'élément bien rédigé :
<div class="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 flex gap-4">
  <div class="flex-shrink-0 w-12 h-12 bg-amber-500/20 rounded-xl flex items-center justify-center text-2xl">\U0001f52c</div>
  <div class="flex-1">
    <div class="flex items-center gap-2 mb-1"><h3 class="font-bold">olmOCR-2-7B-1025-FP8</h3><span class="text-xs px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded">Top 4 du benchmark AIMultiple</span></div>
    <p class="text-sm text-slate-400">Modèle OCR spécialisé d'Allen AI. Non disponible via API serverless — nécessite un HF Inference Endpoint (environ 1-3$/h de GPU) ou une installation locale avec conda + Python 3.11 + vLLM. Alternative : utiliser le playground gratuit sur olmocr.allenai.org.</p>
  </div>
</div>

FORMAT IMPÉRATIF — chaque section doit suivre EXACTEMENT ce squelette :

<div id="slide-ai-findings" class="slide px-8 py-24 border-t border-slate-800"><div class="max-w-5xl mx-auto"><h2 class="text-4xl font-bold mb-8">Surprises et Découvertes</h2><div class="grid grid-cols-2 gap-6">...4 CARTES DÉTAILLÉES...</div></div></div>
<div id="slide-ai-recommendations" class="slide px-8 py-24 border-t border-slate-800"><div class="max-w-5xl mx-auto"><h2 class="text-4xl font-bold mb-8">Recommandations</h2><div class="grid grid-cols-2 gap-6">...4 CARTES DÉTAILLÉES...</div></div></div>
<div id="slide-ai-untested" class="slide px-8 py-24 border-t border-slate-800"><div class="max-w-5xl mx-auto"><h2 class="text-4xl font-bold mb-3">Modèles Non Testés</h2><p class="text-slate-400 mb-8">Ces modèles du benchmark AIMultiple nécessitent des APIs ou infrastructures dédiées</p><div class="space-y-4">...UN ÉLÉMENT PAR MODÈLE...</div></div></div>

RÉSULTATS DU BENCHMARK :
{summary}"""

# ── Models to benchmark ──
BENCHMARK_MODELS = [
    # Direct AIMultiple matches (OpenRouter)
    {"id": "openai/gpt-5.6-sol",        "label": "GPT-5.6 Sol",         "source": "OpenRouter", "enabled": True},
    {"id": "google/gemini-2.5-pro",      "label": "Gemini 2.5 Pro",      "source": "OpenRouter", "enabled": True},
    {"id": "anthropic/claude-sonnet-4.5","label": "Claude Sonnet 4.5",   "source": "OpenRouter", "enabled": True},
    {"id": "google/gemini-3.5-flash",    "label": "Gemini 3.5 Flash",    "source": "OpenRouter", "enabled": True},
    {"id": "anthropic/claude-sonnet-5",  "label": "Claude Sonnet 5",     "source": "OpenRouter", "enabled": True},
    {"id": "qwen/qwen3-vl-235b-a22b-instruct", "label": "Qwen3 VL 235B","source": "OpenRouter", "enabled": True},
    # HF Router (free tier)
    {"id": "Qwen/Qwen3.5-122B-A10B",    "label": "Qwen3.5 122B",       "source": "HF Router",  "enabled": True},
    {"id": "google/gemma-4-31B-it",     "label": "Gemma 4 31B",         "source": "HF Router",  "enabled": True},
    {"id": "Qwen/Qwen3-VL-30B-A3B-Instruct", "label": "Qwen3 VL 30B",  "source": "HF Router",  "enabled": True},
]

CERT_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
SSL_CTX = None
if os.path.exists(CERT_BUNDLE):
    os.environ["SSL_CERT_FILE"] = CERT_BUNDLE
    SSL_CTX = ssl.create_default_context(cafile=CERT_BUNDLE)


# ── Data management ──
def load_data():
    """Load or initialize the benchmark data."""
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding='utf-8'))
    return {"documents": {}, "models": BENCHMARK_MODELS, "results": {}, "meta": {}}

def save_data(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def load_tracker():
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text(encoding='utf-8'))
    return {}

def save_tracker(tracker):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding='utf-8')


# ── Image fingerprinting ──
def get_image_fingerprint(path):
    """Compute SHA256 of image for dedup tracking."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_image_mime(path):
    ext = path.suffix.lower()
    return "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"

def discover_images():
    """Find all images in the images/ directory."""
    if not IMAGES_DIR.exists():
        return []
    return sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])


# ── Benchmark runners ──
def openrouter_ocr(model_id, image_path):
    """Send image to OpenRouter model."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"success": False, "error": "OPENROUTER_API_KEY not set"}

    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    mime = get_image_mime(image_path)
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
        ]}],
        "max_tokens": 4000, "temperature": 0.0,
    })

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmpfile.write(payload); tmpfile.close()

    start = time.time()
    try:
        r = subprocess.run([
            'curl', '-s', '--cacert', CERT_BUNDLE,
            'https://openrouter.ai/api/v1/chat/completions',
            '-H', f'Authorization: Bearer {api_key}',
            '-H', 'Content-Type: application/json',
            '-H', 'HTTP-Referer: https://github.com/Aitbytes/handwriting-ocr-benchmark',
            '-d', f'@{tmpfile.name}', '--max-time', '180'
        ], capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start
        os.unlink(tmpfile.name)

        resp = json.loads(r.stdout)
        if "error" in resp and not resp.get("choices"):
            return {"success": False, "error": str(resp["error"])[:200], "elapsed_s": round(elapsed, 2)}

        choices = resp.get("choices", [])
        if not choices:
            return {"success": False, "error": "No choices", "elapsed_s": round(elapsed, 2)}

        msg = choices[0].get("message", {})
        text = msg.get("content", "")
        if not text:
            return {"success": False, "error": f"Empty content (reason: {choices[0].get('finish_reason')})", "elapsed_s": round(elapsed, 2)}

        usage = resp.get("usage", {})
        return {
            "success": True, "text": text, "elapsed_s": round(elapsed, 2),
            "cost_usd": round(usage.get("cost", 0), 6),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - start
        try: os.unlink(tmpfile.name)
        except: pass
        return {"success": False, "error": str(e)[:200], "elapsed_s": round(elapsed, 2)}


def hf_router_ocr(model_id, image_path):
    """Send image to HuggingFace Router model."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return {"success": False, "error": "HF_TOKEN not set"}

    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    mime = get_image_mime(image_path)
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
        ]}],
        "max_tokens": 4000, "temperature": 0.0,
    })

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmpfile.write(payload); tmpfile.close()

    start = time.time()
    try:
        r = subprocess.run([
            'curl', '-s', '--cacert', CERT_BUNDLE,
            'https://router.huggingface.co/v1/chat/completions',
            '-H', f'Authorization: Bearer {hf_token}',
            '-H', 'Content-Type: application/json',
            '-d', f'@{tmpfile.name}', '--max-time', '180'
        ], capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start
        os.unlink(tmpfile.name)

        resp = json.loads(r.stdout)
        if "error" in resp and not resp.get("choices"):
            return {"success": False, "error": str(resp["error"])[:200], "elapsed_s": round(elapsed, 2)}

        choices = resp.get("choices", [])
        if not choices:
            return {"success": False, "error": "No choices", "elapsed_s": round(elapsed, 2)}

        msg = choices[0].get("message", {})
        text = msg.get("content", "")
        if not text:
            return {"success": False, "error": f"Empty content", "elapsed_s": round(elapsed, 2)}

        usage = resp.get("usage", {})
        return {
            "success": True, "text": text, "elapsed_s": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost_usd": 0,  # HF Router is free for supported models
        }
    except Exception as e:
        elapsed = time.time() - start
        try: os.unlink(tmpfile.name)
        except: pass
        return {"success": False, "error": str(e)[:200], "elapsed_s": round(elapsed, 2)}


# ── Pipeline steps ──
def run_benchmark_on_image(doc_id, image_path, models):
    """Run all enabled models on a single image, save transcriptions."""
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model in models:
        if not model.get("enabled", True):
            continue

        model_id = model["id"]
        source = model["source"]
        label = model["label"]
        print(f"  [{label}] ({source}) ...")

        if source == "OpenRouter":
            r = openrouter_ocr(model_id, image_path)
        elif source == "HF Router":
            r = hf_router_ocr(model_id, image_path)
        else:
            r = {"success": False, "error": f"Unknown source: {source}"}

        results[model_id] = r

        # Save transcription to file
        if r.get("success") and r.get("text"):
            safe_id = model_id.replace("/", "_").replace("\\", "_")[:80]
            tx_file = TRANSCRIPTIONS_DIR / f"{doc_id}_{safe_id}.txt"
            tx_file.write_text(r["text"], encoding='utf-8')
            # Store just the file reference in results, not the full text
            r["transcription_file"] = str(tx_file.relative_to(ROOT))
            del r["text"]  # keep data.json lean

        if r.get("success"):
            print(f"     ✅ {r['elapsed_s']}s | ${r.get('cost_usd',0):.6f} | {r.get('total_tokens',0)} tok")
        else:
            print(f"     ❌ {r.get('error','?')[:80]}")

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Only detect new images")
    ap.add_argument("--rebuild-only", action="store_true", help="Only rebuild site from existing data")
    ap.add_argument("--image", type=str, help="Process a specific image path")
    args = ap.parse_args()

    # Load existing data
    data = load_data()
    tracker = load_tracker()

    # Ensure images directory
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only:
        print("♻️  Rebuild-only mode — regenerating site from existing data")
        rebuild_site(data)
        return

    if args.image:
        # Process a single image
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        images = [img_path]
    else:
        images = discover_images()

    if not images:
        print("📭 No images found in images/")
        if not data["results"]:
            rebuild_site(data)
        return

    # Compute model fingerprint (which models + prompt = what version of benchmark)
    models_fp = hashlib.sha256(
        json.dumps([m["id"] for m in BENCHMARK_MODELS if m.get("enabled", True)], sort_keys=True).encode()
    ).hexdigest()[:12]

    new_images = []
    for img_path in images:
        fp = get_image_fingerprint(img_path)
        doc_id = img_path.stem.replace(" ", "_").replace("/", "_")[:60]

        # Check if already processed with current models
        if doc_id in tracker and tracker[doc_id].get("models_fp") == models_fp:
            print(f"⏭️  {doc_id}: already processed (fingerprint {fp[:10]})")
            continue

        new_images.append((img_path, doc_id, fp))
        print(f"🆕 {doc_id}: new image (fingerprint {fp[:10]})")

    if args.check:
        print(f"\n📊 {len(new_images)} new image(s) detected, {len(images)} total")
        return

    if not new_images:
        print("✅ All images already processed — rebuilding site for consistency")
        rebuild_site(data)
        return

    # Run benchmarks on new images
    enabled_models = [m for m in BENCHMARK_MODELS if m.get("enabled", True)]
    print(f"\n🚀 Running benchmark: {len(new_images)} images × {len(enabled_models)} models\n")

    for img_path, doc_id, fp in new_images:
        print(f"\n📄 {doc_id} ({fp[:10]})")
        print("=" * 50)

        results = run_benchmark_on_image(doc_id, img_path, enabled_models)

        # Store in data
        data["documents"][doc_id] = {
            "title": doc_id.replace("_", " "),
            "description": "",
            "tags": [],
            "fingerprint": fp,
            "added_at": datetime.now().isoformat(),
        }
        data["results"][doc_id] = results

        # Update tracker
        tracker[doc_id] = {"fingerprint": fp, "models_fp": models_fp, "processed_at": datetime.now().isoformat()}

    # Update meta
    total_cost = 0
    total_runs = 0
    for doc_id, models in data["results"].items():
        for model_id, r in models.items():
            if r.get("success"):
                total_cost += r.get("cost_usd", 0)
                total_runs += 1

    data["meta"] = {
        "last_run": datetime.now().isoformat(),
        "total_documents": len(data["documents"]),
        "total_models": len(enabled_models),
        "total_runs": total_runs,
        "total_cost_usd": round(total_cost, 4),
    }

    save_data(data)
    save_tracker(tracker)

    print(f"\n📊 Saved: {len(data['documents'])} docs, {total_runs} runs, ${total_cost:.4f} total")

    # Rebuild site
    rebuild_site(data)


def generate_ai_analysis(data):
    """Use DeepSeek V4 Pro to generate analysis slides from benchmark data."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("⚠️  No OPENROUTER_API_KEY — skipping AI analysis")
        return None

    # Build a compact summary of results for the LLM
    summary_lines = ["Voici les résultats d'un benchmark de reconnaissance d'écriture manuscrite:"]
    summary_lines.append(f"- {data['meta']['total_documents']} documents testés, {data['meta']['total_models']} modèles, {data['meta']['total_runs']} transcriptions")
    summary_lines.append(f"- Coût total: ${data['meta']['total_cost_usd']:.4f}")

    for doc_id, doc_info in data.get("documents", {}).items():
        summary_lines.append(f"\nDocument: {doc_info.get('title', doc_id)} — {doc_info.get('description', '')}")
        results = data.get("results", {}).get(doc_id, {})
        for model_id, r in sorted(results.items(), key=lambda x: x[1].get('elapsed_s', 999)):
            if not r.get("success"):
                summary_lines.append(f"  ❌ {model_id}: ÉCHEC — {r.get('error', '?')[:80]}")
            else:
                label = model_id
                for m in data.get("models", []):
                    if m["id"] == model_id: label = m["label"]; break
                summary_lines.append(f"  ✅ {label}: {r['elapsed_s']:.1f}s, ${r.get('cost_usd',0):.6f}, {r.get('total_tokens',0)} tokens")

    # Untested models
    untested = ["olmOCR-2-7B-1025-FP8", "Moondream OCR", "DeepSeek OCR", "PaddleOCR-VL",
                "Azure Cognitive Service", "Google Vision API", "Amazon Textract", "Mistral OCR"]
    summary_lines.append(f"\nModèles non testés (nécessitent des APIs/clés séparées): {', '.join(untested)}")

    summary = "\n".join(summary_lines)

    prompt = PROMPT_TEMPLATE.format(summary=summary)

    payload = json.dumps({
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 12000,
        "temperature": 0.7,
    })

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmpfile.write(payload); tmpfile.close()

    print("🤖 Génération de l'analyse par DeepSeek V4 Flash...")
    start = time.time()
    try:
        r = subprocess.run([
            'curl', '-s', '--cacert', CERT_BUNDLE,
            'https://openrouter.ai/api/v1/chat/completions',
            '-H', f'Authorization: Bearer {api_key}',
            '-H', 'Content-Type: application/json',
            '-H', 'HTTP-Referer: https://github.com/Aitbytes/handwriting-ocr-benchmark',
            '-d', f'@{tmpfile.name}', '--max-time', '600'
        ], capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        os.unlink(tmpfile.name)

        resp = json.loads(r.stdout)
        text = resp["choices"][0]["message"]["content"]
        
        # Extract the three div sections by splitting at known boundaries
        import re
        # Split at each slide boundary
        parts = re.split(r'(?=<div id="slide-ai-(?:findings|recommendations|untested)")', text)
        analysis = {"findings": None, "recommendations": None, "untested": None}
        for part in parts:
            part = part.strip()
            for key in ["findings", "recommendations", "untested"]:
                if part.startswith(f'<div id="slide-ai-{key}"'):
                    # Count div depth to find the closing tag of the outer wrapper
                    depth = 0
                    end_pos = 0
                    for m in re.finditer(r'</div>|<div\b', part):
                        if m.group() == '</div>':
                            depth -= 1
                            if depth == 0:
                                end_pos = m.end()
                                break
                        else:
                            depth += 1
                    analysis[key] = part[:end_pos] if end_pos > 0 else part
                    break

        if any(analysis.values()):
            print(f"  ✅ Analyse générée: findings={'✓' if analysis['findings'] else '✗'}, recos={'✓' if analysis['recommendations'] else '✗'}, untested={'✓' if analysis['untested'] else '✗'}")
            return analysis
        else:
            print(f"  ⚠️  Impossible d'extraire les sections. Réponse brute ({len(text)} chars):")
            print(f"  {text[:300]}...")
            # Store raw text as fallback
            return {"raw": text[:8000]}

    except Exception as e:
        print(f"  ❌ Échec génération IA: {e}")
        try: os.unlink(tmpfile.name)
        except: pass
        return None


def rebuild_site(data):
    """Rebuild all dynamic HTML from data.json."""
    print("\n🔨 Rebuilding site...")

    # Generate AI analysis (only if data changed or first run)
    ai_analysis = None
    if data.get("results"):
        ai_analysis = generate_ai_analysis(data)
        # Store in data for caching
        if ai_analysis:
            data["ai_analysis"] = ai_analysis
            save_data(data)
    elif data.get("ai_analysis"):
        ai_analysis = data["ai_analysis"]

    # Import and run the site generator
    sys.path.insert(0, str(ROOT))
    try:
        from generate_site import rebuild_from_data
        rebuild_from_data(data, ai_analysis)
        print("✅ Site rebuilt successfully")
    except ImportError:
        print("⚠️  generate_site.py not found or missing rebuild_from_data()")
    except Exception as e:
        print(f"❌ Site rebuild failed: {e}")
        raise


if __name__ == "__main__":
    main()