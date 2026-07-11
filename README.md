# Benchmark OCR d'Écriture Manuscrite 2026

[🇫🇷 Voir le rapport complet](https://aitbytes.github.io/handwriting-ocr-benchmark/)

Benchmark comparant 14 modèles (LLMs et OCR) sur deux documents manuscrits français :
1. **Protocole I²C** — écriture cursive + code C
2. **Examen BTS** — formulaire structuré avec diagrammes

Basé sur le [benchmark AIMultiple](https://aimultiple.com/handwriting-recognition).

## Modèles testés
- **OpenRouter** : GPT-5, GPT-5.6 Sol, Gemini 2.5 Pro, Gemini 3.1 Pro, Gemini 3.5 Flash, Claude Sonnet 4.5, Claude Sonnet 5, Qwen3 VL 235B
- **HuggingFace Router** : Qwen3.5 122B, Qwen3 VL 30B, Qwen2.5 VL 72B, Gemma 4 31B, GLM 4.6V
- **ZAI API** : GLM OCR

## Résultats clés
- 🥇 **Meilleure écriture cursive** : GPT-5.6 Sol (9,5/10)
- 🥇 **Meilleurs documents structurés** : Qwen3.5 122B (9,5/10) — **GRATUIT**
- ⚡ **Plus rapide** : Gemma 4 31B (1,1s)
- 💰 **Coût total du benchmark** : 0,46 $

## Structure
```
├── index.html              # Rapport principal (diapositives en français)
├── transcriptions/
│   ├── doc1-i2c.html       # Toutes les transcriptions du Document 1
│   └── doc2-bts.html       # Toutes les transcriptions du Document 2
├── benchmark.py            # Script OpenRouter + ZAI
├── hf_benchmark2.py        # Script HuggingFace Router
└── results/                # Résultats JSON
```
