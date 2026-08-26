import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from extractlora.extraction.config import BASE_MODEL, MAX_NEW_TOKENS
from extractlora.extraction.prompts import build_messages
from extractlora.schemas.invoice import Facture


@lru_cache(maxsize=4)
def load_model(adapter_dir: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    if adapter_dir and Path(adapter_dir).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return tokenizer, model


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _generate(tokenizer, model, messages: list[dict]) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_invoice(
    ocr_text: str, adapter_dir: str | None = None, max_retries: int = 2
) -> dict:
    """Runs the OCR text through the (optionally LoRA-adapted) LLM and returns
    a JSON dict validated against the Facture schema. Raises ValueError if no
    valid JSON could be produced after max_retries attempts."""
    tokenizer, model = load_model(adapter_dir)

    error: str | None = None
    for _ in range(max_retries + 1):
        messages = build_messages(ocr_text, previous_error=error)
        raw = _generate(tokenizer, model, messages)
        candidate = _extract_first_json_object(raw)

        if candidate is None:
            error = "aucun objet JSON trouve dans la reponse"
            continue

        try:
            data = json.loads(candidate)
            validated = Facture.model_validate(data)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as exc:
            error = str(exc)

    raise ValueError(f"Echec de generation d'un JSON valide apres {max_retries + 1} tentatives : {error}")
