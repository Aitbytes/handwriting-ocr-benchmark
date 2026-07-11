#!/usr/bin/env python3
"""
Handwriting Recognition Benchmark
Based on AIMultiple's benchmark: https://aimultiple.com/handwriting-recognition
Compares top models on OpenRouter + GLM OCR via ZAI.
"""

import base64, json, urllib.request, time, os, sys, re, ssl, subprocess, tempfile
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────
IMAGES = [
    "/home/a8taleb/Downloads/WhatsApp Image 2026-07-11 at 00.49.29.jpg",
    "/home/a8taleb/Downloads/WhatsApp Image 2026-07-11 at 00.57.39.jpeg",
]

OUTPUT_DIR = Path("/tmp/OCR/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models with their AIMultiple benchmark mapping
BENCHMARK_MODELS = {
    # ── Direct benchmark matches ──
    "openai/gpt-5":                  {"label": "GPT-5",                         "match": "✅ Direct"},
    "google/gemini-2.5-pro":         {"label": "Gemini 2.5 Pro",               "match": "✅ Direct"},
    "anthropic/claude-sonnet-4.5":   {"label": "Claude Sonnet 4.5",            "match": "✅ Direct"},
    # ── Close equivalents ──
    "google/gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro Preview",       "match": "≈ Gemini 3 Pro Preview"},
    # ── Newer/better versions ──
    "openai/gpt-5.6-sol":            {"label": "GPT-5.6 Sol",                   "match": "⬆️ Newer GPT"},
    "google/gemini-3.5-flash":       {"label": "Gemini 3.5 Flash",             "match": "⬆️ Newer Gemini"},
    "anthropic/claude-sonnet-5":     {"label": "Claude Sonnet 5",              "match": "⬆️ Newer Claude"},
    # ── Strong VL for comparison ──
    "qwen/qwen3-vl-235b-a22b-instruct": {"label": "Qwen3 VL 235B",            "match": "🔬 VL comparison"},
}

PROMPT = """Transcribe all handwritten text in this image verbatim. Include all text, numbers, code, punctuation, and special characters exactly as written. Preserve line breaks and formatting as much as possible. For any text you cannot read confidently, mark it as [?]. Output ONLY the transcribed text, nothing else — no commentary, no introductions."""

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ZAI_API_KEY = os.environ.get("ZAI_API_KEY", "")

# Fix SSL on NixOS
CERT_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
if os.path.exists(CERT_BUNDLE):
    os.environ["SSL_CERT_FILE"] = CERT_BUNDLE
    os.environ["REQUESTS_CA_BUNDLE"] = CERT_BUNDLE
    os.environ["CURL_CA_BUNDLE"] = CERT_BUNDLE

SSL_CONTEXT = ssl.create_default_context(cafile=CERT_BUNDLE) if os.path.exists(CERT_BUNDLE) else None


# ── OpenRouter API ─────────────────────────────────────────────
def openrouter_ocr(model_id, image_path, max_tokens=4000):
    """Send image to OpenRouter model for handwriting transcription via curl."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    # Use curl for reliability (NixOS SSL works with curl --cacert)
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })

    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tmpfile.write(payload)
    tmpfile.close()

    start = time.time()
    try:
        result = subprocess.run([
            "curl", "-s", "--cacert", CERT_BUNDLE,
            "https://openrouter.ai/api/v1/chat/completions",
            "-H", f"Authorization: Bearer {OPENROUTER_API_KEY}",
            "-H", "Content-Type: application/json",
            "-H", "HTTP-Referer: https://github.com/a8taleb/handwriting-benchmark",
            "-H", "X-Title: Handwriting Benchmark",
            "-d", f"@{tmpfile.name}",
            "--max-time", "180"
        ], capture_output=True, text=True, timeout=200)
        elapsed = time.time() - start
        os.unlink(tmpfile.name)

        if result.returncode != 0:
            return {"success": False, "error": f"curl exit {result.returncode}: {result.stderr[:200]}", "elapsed_s": round(elapsed, 2)}

        resp = json.loads(result.stdout)

        if "error" in resp and not resp.get("choices"):
            err = resp["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"success": False, "error": err_msg[:200], "elapsed_s": round(elapsed, 2)}

        choices = resp.get("choices", [])
        if not choices:
            return {"success": False, "error": f"No choices: {json.dumps(resp)[:300]}", "elapsed_s": round(elapsed, 2)}

        msg = choices[0].get("message", {})
        text = msg.get("content") or ""
        if not text:
            return {"success": False, "error": f"Empty content. Finish: {choices[0].get('finish_reason')}", "elapsed_s": round(elapsed, 2)}

        usage = resp.get("usage", {})
        return {
            "success": True,
            "text": text,
            "elapsed_s": round(elapsed, 2),
            "cost_usd": round(usage.get("cost", 0), 6),
            "model_used": resp.get("model", model_id),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - start
        try: os.unlink(tmpfile.name)
        except: pass
        return {"success": False, "error": str(e)[:300], "elapsed_s": round(elapsed, 2)}


# ── GLM OCR via ZAI ────────────────────────────────────────────
def glm_ocr_zai(image_path):
    """Use GLM-OCR via ZAI SDK for handwriting OCR."""
    try:
        import base64 as b64
        from zai import ZaiClient

        client = ZaiClient(api_key=ZAI_API_KEY)

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        img_b64 = b64.b64encode(img_bytes).decode()
        ext = Path(image_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        start = time.time()
        resp = client.layout_parsing.create(
            model="glm-ocr",
            file=f"data:{mime};base64,{img_b64}"
        )
        elapsed = time.time() - start

        return {
            "success": True,
            "text": resp.md_results,
            "elapsed_s": round(elapsed, 2),
            "cost_usd": round(resp.usage.total_tokens * 0.03 / 1_000_000, 6) if hasattr(resp, 'usage') and hasattr(resp.usage, 'total_tokens') else 0,
            "model_used": "glm-ocr",
            "prompt_tokens": getattr(resp.usage, 'prompt_tokens', 0) if hasattr(resp, 'usage') else 0,
            "completion_tokens": getattr(resp.usage, 'completion_tokens', 0) if hasattr(resp, 'usage') else 0,
            "total_tokens": getattr(resp.usage, 'total_tokens', 0) if hasattr(resp, 'usage') else 0,
            "note": "ZAI API / GLM-OCR",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_s": 0,
        }


# ── Main Benchmark ─────────────────────────────────────────────
def run_benchmark():
    results = {}

    for img_path in IMAGES:
        img_name = Path(img_path).stem
        print(f"\n{'='*70}")
        print(f"📄 IMAGE: {img_name}")
        print(f"{'='*70}")

        results[img_name] = {}

        # OpenRouter models
        for model_id, meta in BENCHMARK_MODELS.items():
            label = meta["label"]
            match_info = meta["match"]
            print(f"\n🔍 [{label}] ({match_info}) — sending to {model_id}...")

            result = openrouter_ocr(model_id, img_path)
            results[img_name][model_id] = result

            if result.get("success") and result.get("text"):
                preview = result["text"][:120].replace("\n", "\\n")
                print(f"   ✅ {result['elapsed_s']}s | ${result['cost_usd']:.6f} | {result['total_tokens']} tokens")
                print(f"   📝 {preview}...")
            else:
                print(f"   ❌ FAILED: {result.get('error', 'unknown')[:120]}")

        # GLM OCR via ZAI
        print(f"\n🔍 [GLM OCR] (ZAI API) — sending...")
        glm_result = glm_ocr_zai(img_path)
        results[img_name]["zai/glm-ocr"] = glm_result

        if glm_result.get("success"):
            preview = glm_result["text"][:120].replace("\n", "\\n")
            print(f"   ✅ {glm_result['elapsed_s']}s | ${glm_result.get('cost_usd', 0):.6f}")
            print(f"   📝 {preview}...")
        else:
            print(f"   ❌ FAILED: {glm_result.get('error', 'unknown')[:120]}")

    return results


# ── Save & Report ──────────────────────────────────────────────
def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw JSON
    json_path = OUTPUT_DIR / f"benchmark_{timestamp}.json"
    # Make results JSON-serializable
    serializable = {}
    for img_name, models in results.items():
        serializable[img_name] = {}
        for model_id, data in models.items():
            serializable[img_name][model_id] = {
                k: v for k, v in data.items()
                if k != "text"  # keep text in separate files
            }

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    # Save individual transcriptions
    txt_dir = OUTPUT_DIR / "transcriptions"
    txt_dir.mkdir(exist_ok=True)

    for img_name, models in results.items():
        for model_id, data in models.items():
            if not data.get("success"):
                continue
            safe_model = re.sub(r"[^\w-]", "_", model_id)
            fname = f"{timestamp}_{img_name[:40]}_{safe_model[:40]}.txt"
            with open(txt_dir / fname, "w") as f:
                f.write(data["text"])

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"📊 BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    for img_name, models in results.items():
        print(f"\n📄 {img_name}")
        print(f"{'Model':<35} {'Status':<8} {'Time':>8} {'Cost':>12} {'Tokens':>8}")
        print("-" * 75)
        for model_id, data in models.items():
            label = BENCHMARK_MODELS.get(model_id, {}).get("label", model_id)
            if data.get("success"):
                print(f"{label:<35} {'✅':<8} {data['elapsed_s']:>6.1f}s ${data.get('cost_usd', 0):>10.6f} {data.get('total_tokens', 0):>8}")
            else:
                print(f"{label:<35} {'❌':<8} {'—':>8} {'—':>12} {'—':>8}")

    print(f"\n📁 Full results: {json_path}")
    print(f"📁 Transcriptions: {txt_dir}/")

    return json_path


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not set. Run: source ~/scripts/vault-env ai-keys")
        sys.exit(1)

    print("🚀 Starting Handwriting Recognition Benchmark")
    print(f"   Images: {len(IMAGES)}")
    print(f"   Models: {len(BENCHMARK_MODELS)} (OpenRouter) + GLM OCR (ZAI)")
    print(f"   Based on: https://aimultiple.com/handwriting-recognition")

    results = run_benchmark()
    save_results(results)
    print("\n✅ Benchmark complete!")
