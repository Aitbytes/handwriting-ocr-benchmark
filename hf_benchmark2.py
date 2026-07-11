#!/usr/bin/env python3
"""
HF Router benchmark — test additional VL models not on OpenRouter
"""
import base64, json, subprocess, time, os, tempfile, ssl, urllib.request
from pathlib import Path

HF_TOKEN = os.environ.get('HF_TOKEN', '')
CERT = '/etc/ssl/certs/ca-certificates.crt'
SSL_CTX = ssl.create_default_context(cafile=CERT)
IMAGES = [
    "/home/a8taleb/Downloads/WhatsApp Image 2026-07-11 at 00.49.29.jpg",
    "/home/a8taleb/Downloads/WhatsApp Image 2026-07-11 at 00.57.39.jpeg",
]

PROMPT = "Transcribe all handwritten text in this image verbatim. Include all text, numbers, code, punctuation, and special characters exactly as written. Preserve line breaks and formatting. Output ONLY the transcribed text."

# HF Router models to test (not already tested via OpenRouter)
HF_MODELS = {
    "Qwen/Qwen2.5-VL-72B-Instruct":       {"label": "Qwen2.5 VL 72B",      "match": "🔬 VL (HF Router)"},
    "Qwen/Qwen3-VL-30B-A3B-Instruct":      {"label": "Qwen3 VL 30B",        "match": "🔬 VL (HF Router)"},
    "google/gemma-4-31B-it":              {"label": "Gemma 4 31B",          "match": "🔬 VL (HF Router)"},
    "zai-org/GLM-4.6V":                    {"label": "GLM 4.6V",             "match": "🔬 VL (HF Router)"},
    "Qwen/Qwen3.5-122B-A10B":             {"label": "Qwen3.5 122B",         "match": "🔬 VL (HF Router)"},
    "moonshotai/Kimi-K2.6":               {"label": "Kimi K2.6",            "match": "🔬 VL (HF Router)"},
}


def hf_chat_ocr(model_id, image_path):
    """Use HF router chat API for image transcription."""
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"

    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
        ]}],
        "max_tokens": 4000,
        "temperature": 0.0,
    })

    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmpfile.write(payload)
    tmpfile.close()

    start = time.time()
    try:
        result = subprocess.run([
            'curl', '-s', '--cacert', CERT,
            'https://router.huggingface.co/v1/chat/completions',
            '-H', f'Authorization: Bearer {HF_TOKEN}',
            '-H', 'Content-Type: application/json',
            '-d', f'@{tmpfile.name}',
            '--max-time', '300'
        ], capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        os.unlink(tmpfile.name)

        resp = json.loads(result.stdout)
        if 'error' in resp:
            return {"success": False, "error": str(resp['error'])[:300], "elapsed_s": round(elapsed, 2)}
        if not resp.get('choices'):
            return {"success": False, "error": f"No choices: {json.dumps(resp)[:300]}", "elapsed_s": round(elapsed, 2)}
        
        text = resp['choices'][0]['message']['content']
        usage = resp.get('usage', {})
        return {
            "success": True,
            "text": text,
            "elapsed_s": round(elapsed, 2),
            "model_used": model_id,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "provider": "HF Router",
        }
    except Exception as e:
        elapsed = time.time() - start
        try: os.unlink(tmpfile.name)
        except: pass
        return {"success": False, "error": str(e)[:300], "elapsed_s": round(elapsed, 2)}


# Run benchmark
results = {}
for img_path in IMAGES:
    name = Path(img_path).stem
    print(f"\n{'='*60}")
    print(f"📄 {name}")
    print(f"{'='*60}")
    results[name] = {}

    for model_id, meta in HF_MODELS.items():
        label = meta["label"]
        print(f"  🔍 [{label}] ...")
        r = hf_chat_ocr(model_id, img_path)
        results[name][model_id] = r
        if r["success"]:
            preview = r["text"][:100].replace('\n', '\\n')
            print(f"     ✅ {r['elapsed_s']}s | {r.get('total_tokens','?')} tok | {preview}...")
        else:
            print(f"     ❌ {r.get('error','?')[:100]}")

# Save
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path("/tmp/OCR/results/hf_benchmark_" + ts + ".json")
serializable = {}
for img, models in results.items():
    serializable[img] = {}
    for mid, data in models.items():
        serializable[img][mid] = {k: v for k, v in data.items() if k != 'text'}
with open(out, 'w') as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)

# Save transcriptions
txt_dir = Path("/tmp/OCR/results/transcriptions/")
for img, models in results.items():
    for mid, data in models.items():
        if data.get('success') and data.get('text'):
            safe = mid.replace('/', '_').replace('\\', '_')
            with open(txt_dir / f"{ts}_{img[:30]}_{safe}.txt", 'w') as f:
                f.write(data['text'])

# Summary
print(f"\n\n{'='*80}")
print("📊 HF ROUTER BENCHMARK RESULTS")
print(f"{'='*80}")
for img, models in results.items():
    print(f"\n📄 {img}")
    print(f"{'Model':<30} {'Status':<8} {'Time':>8} {'Tokens':>8}")
    print("-" * 60)
    for mid, data in models.items():
        label = HF_MODELS[mid]["label"]
        if data.get('success'):
            print(f"{label:<30} {'✅':<8} {data['elapsed_s']:>6.1f}s {data.get('total_tokens','?'):>8}")
        else:
            print(f"{label:<30} {'❌':<8} —")

print(f"\n✅ HF router benchmark complete! Saved to {out}")