# 🔬 Benchmark OCR d'Écriture Manuscrite

[🇫🇷 **Voir le rapport**](https://aitbytes.github.io/handwriting-ocr-benchmark/)

Benchmark automatisé comparant 14 modèles (LLMs et OCR) sur des documents manuscrits français via **GitHub Actions CI/CD**.

## 🚀 Pipeline

```
Déposez une image → git push → GitHub Actions → benchmark → site mis à jour
```

1. Ajoutez une image dans `images/`
2. `git push`
3. La pipeline détecte la nouvelle image (fingerprint SHA256)
4. Exécute le benchmark sur tous les modèles activés
5. Reconstruit `index.html` + pages de transcription
6. Déploie sur GitHub Pages

## 🔧 Configuration

### Secrets GitHub requis

| Secret | Description |
|--------|-------------|
| `OPENROUTER_API_KEY` | Clé API OpenRouter pour GPT, Gemini, Claude |
| `HF_TOKEN` | Token HuggingFace pour les modèles gratuits (Qwen, Gemma) |

```bash
gh secret set OPENROUTER_API_KEY --body "sk-or-..."
gh secret set HF_TOKEN --body "hf_..."
```

### Ajouter/supprimer des modèles

Modifier `results/data.json` → `models` → `enabled: true/false`

## 📁 Structure

```
├── images/                    # ← Déposez vos images ici
├── .github/workflows/benchmark.yml  # Pipeline CI/CD
├── run_pipeline.py            # Détection + benchmark
├── generate_site.py           # Générateur de site
├── results/
│   ├── data.json              # Source unique de vérité
│   ├── tracker.json           # Suivi des images traitées
│   └── transcriptions/        # Fichiers texte des résultats
├── transcriptions/            # Pages HTML de comparaison
└── index.html                 # Rapport principal (généré)
```

## ⚡ Local

```bash
# Benchmark
python3 run_pipeline.py

# Seulement reconstruire le site
python3 run_pipeline.py --rebuild-only

# Détecter sans exécuter
python3 run_pipeline.py --check
```

Variables d'environnement : `OPENROUTER_API_KEY`, `HF_TOKEN`