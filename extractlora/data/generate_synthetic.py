"""Generates synthetic invoice images + reference JSON for training/testing.

Renders plain-text invoices as images (no external template assets needed),
so the whole dataset can be produced offline in seconds.
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FOURNISSEURS = [
    ("Atelier Textile SARL", "12 Rue des Artisans, Casablanca"),
    ("Menuiserie du Nord", "45 Avenue Hassan II, Tanger"),
    ("Electro Plus", "3 Boulevard Zerktouni, Rabat"),
    ("Papeterie Centrale", "8 Rue Ibn Sina, Fes"),
    ("Fournitures Info SARL", "21 Rue Allal Ben Abdellah, Marrakech"),
]

PRODUITS = [
    "Tissu coton",
    "Panneau bois",
    "Cable electrique",
    "Ramette papier A4",
    "Cartouche encre",
    "Clavier USB",
    "Ecran 24 pouces",
]


def _random_invoice() -> dict:
    nom, adresse = random.choice(FOURNISSEURS)
    n_lignes = random.randint(1, 3)
    lignes = []
    montant_ht = 0.0
    for _ in range(n_lignes):
        qte = random.randint(1, 20)
        prix = round(random.uniform(15, 300), 2)
        lignes.append(
            {
                "description": f"{random.choice(PRODUITS)} - {qte}u",
                "quantite": qte,
                "prix_unitaire": prix,
            }
        )
        montant_ht += qte * prix
    montant_ht = round(montant_ht, 2)
    tva = 0.20
    montant_ttc = round(montant_ht * (1 + tva), 2)

    return {
        "type_document": "facture",
        "numero_facture": f"F-2026-{random.randint(1000, 9999)}",
        "date_emission": f"2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "fournisseur": {"nom": nom, "adresse": adresse},
        "montant_ht": montant_ht,
        "taux_tva": tva,
        "montant_ttc": montant_ttc,
        "lignes": lignes,
    }


def _render_invoice_image(invoice: dict, path: Path) -> None:
    img = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    y = 30
    draw.text((30, y), "FACTURE", fill="black", font=font)
    y += 40
    draw.text((30, y), f"N. {invoice['numero_facture']}", fill="black", font=font_small)
    y += 28
    draw.text((30, y), f"Date : {invoice['date_emission']}", fill="black", font=font_small)
    y += 28
    draw.text((30, y), invoice["fournisseur"]["nom"], fill="black", font=font_small)
    y += 26
    draw.text((30, y), invoice["fournisseur"]["adresse"], fill="black", font=font_small)
    y += 40

    draw.text((30, y), "Description", fill="black", font=font_small)
    draw.text((550, y), "Qte", fill="black", font=font_small)
    draw.text((650, y), "P.U.", fill="black", font=font_small)
    y += 24
    draw.line((30, y, 850, y), fill="black")
    y += 12
    for ligne in invoice["lignes"]:
        draw.text((30, y), ligne["description"], fill="black", font=font_small)
        draw.text((550, y), str(ligne["quantite"]), fill="black", font=font_small)
        draw.text((650, y), f"{ligne['prix_unitaire']:.2f}", fill="black", font=font_small)
        y += 26

    y += 20
    draw.text((550, y), f"Montant HT : {invoice['montant_ht']:.2f}", fill="black", font=font_small)
    y += 26
    draw.text((550, y), f"TVA (20%) : {invoice['taux_tva']:.2f}", fill="black", font=font_small)
    y += 26
    draw.text((550, y), f"Montant TTC : {invoice['montant_ttc']:.2f}", fill="black", font=font_small)

    img.save(path)


def generate_dataset(out_dir: str, n: int = 40, seed: int = 42) -> None:
    random.seed(seed)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        invoice = _random_invoice()
        stem = f"invoice_{i:04d}"
        _render_invoice_image(invoice, out / "images" / f"{stem}.png")
        (out / "labels" / f"{stem}.json").write_text(
            json.dumps(invoice, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Genere {n} factures synthetiques dans {out.resolve()}")


if __name__ == "__main__":
    generate_dataset("extractlora/data/synthetic", n=40)
