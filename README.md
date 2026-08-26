# ExtractLoRA

Pipeline d'extraction documentaire structurée par fine-tuning LoRA :
`image -> OCR -> LLM + adaptateur LoRA -> JSON validé`.

Cahier des charges complet : voir l'artifact publié séparément.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation rapide

1. **Générer le dataset synthétique** (factures factices, rendues en image + JSON de référence) :

   ```bash
   python -m extractlora.data.generate_synthetic
   ```

2. **Test de bout en bout** (OCR + extraction baseline, entraîne un adaptateur si absent) :

   ```bash
   python -m extractlora.scripts.smoke_test
   ```

3. **Entraîner l'adaptateur LoRA** sur le dataset synthétique :

   ```bash
   python -m extractlora.training.train_lora
   ```

   L'adaptateur est sauvegardé dans `adapters/facture/`.

4. **Lancer l'API de démonstration** :

   ```bash
   uvicorn extractlora.api.main:app --reload
   ```

   Puis tester avec :

   ```bash
   curl -F "file=@extractlora/data/synthetic/images/invoice_0000.png" http://127.0.0.1:8000/extract
   ```

## Structure

```
extractlora/
  schemas/       Schéma JSON cible (Pydantic)
  data/          Générateur de dataset synthétique
  ocr/           Extraction de texte brut (EasyOCR)
  extraction/    Prompting + pipeline génération -> validation JSON
  training/      Fine-tuning LoRA (peft)
  api/           Service FastAPI de démonstration
  scripts/       Test de bout en bout
adapters/        Adaptateurs LoRA entraînés (un dossier par type de document)
```

## Notes

- Modèle de base : `HuggingFaceTB/SmolLM2-360M-Instruct` (léger, tourne sur CPU) —
  changer `BASE_MODEL` dans `extractlora/extraction/config.py` pour un modèle plus
  puissant (ex. Qwen2.5-7B-Instruct) une fois un GPU disponible.
- L'entraînement fourni est un run de démonstration (dataset et nombre de pas
  réduits) : il prouve que le mécanisme LoRA fonctionne de bout en bout, pas
  une garantie de qualité de production — voir critères d'acceptation dans le
  cahier des charges pour la cible réelle.
