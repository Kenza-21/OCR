import json

from extractlora.schemas.invoice import INVOICE_JSON_SCHEMA

SYSTEM_PROMPT = (
    "Tu es un moteur d'extraction de documents. On te donne le texte brut "
    "issu d'un OCR (souvent bruite, mal ordonne). Tu dois repondre "
    "UNIQUEMENT avec un objet JSON valide conforme au schema fourni, sans "
    "aucun texte avant ou apres."
)


def build_messages(ocr_text: str, previous_error: str | None = None) -> list[dict]:
    user_content = (
        f"Schema JSON attendu :\n{json.dumps(INVOICE_JSON_SCHEMA, ensure_ascii=False)}\n\n"
        f"Texte OCR :\n{ocr_text}\n\n"
        "Reponds uniquement avec le JSON."
    )
    if previous_error:
        user_content += (
            f"\n\nTa reponse precedente etait invalide : {previous_error}\n"
            "Corrige et renvoie uniquement un JSON valide."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
