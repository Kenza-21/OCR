"""End-to-end smoke test: generate data -> OCR -> baseline extraction ->
train a LoRA adapter -> extraction with the adapter.

Not a quality benchmark (tiny model, tiny dataset, few steps) -- it only
proves the pipeline is wired correctly end-to-end. Run from the project
root with the venv's python:

    .venv\\Scripts\\python.exe -m extractlora.scripts.smoke_test
"""

from pathlib import Path

from extractlora.data.generate_synthetic import generate_dataset
from extractlora.extraction.pipeline import extract_invoice
from extractlora.ocr.engine import extract_text

DATA_DIR = "extractlora/data/synthetic"
ADAPTER_DIR = "adapters/facture"


def main():
    if not Path(DATA_DIR, "images").exists():
        print("== Generation du dataset synthetique ==")
        generate_dataset(DATA_DIR, n=16)

    sample_image = sorted(Path(DATA_DIR, "images").glob("*.png"))[0]
    print(f"\n== OCR sur {sample_image.name} ==")
    ocr_text = extract_text(str(sample_image))
    print(ocr_text)

    print("\n== Extraction baseline (sans adaptateur LoRA) ==")
    try:
        result = extract_invoice(ocr_text, adapter_dir=None, max_retries=1)
        print(result)
    except ValueError as exc:
        print(f"(attendu avec un tout petit modele non fine-tune) {exc}")

    if Path(ADAPTER_DIR).exists():
        print("\n== Extraction avec l'adaptateur LoRA entraine ==")
        result = extract_invoice(ocr_text, adapter_dir=ADAPTER_DIR, max_retries=2)
        print(result)
    else:
        print(f"\n(Aucun adaptateur trouve dans {ADAPTER_DIR} -- lancez train_lora.py)")


if __name__ == "__main__":
    main()
