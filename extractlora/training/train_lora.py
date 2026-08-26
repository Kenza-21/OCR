"""Fine-tunes a LoRA adapter on the synthetic invoice dataset.

CPU-friendly demo run: small base model, small dataset, few epochs. This
proves the LoRA fine-tuning mechanism end-to-end; scale up the dataset and
training steps for real quality (see cahier des charges, sections 6-7).
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from extractlora.extraction.config import BASE_MODEL, DEFAULT_ADAPTER_DIR
from extractlora.extraction.prompts import build_messages
from extractlora.ocr.engine import extract_text


class InvoiceDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer):
        self.tokenizer = tokenizer
        data_dir = Path(data_dir)
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        ocr_cache_dir = data_dir / "ocr_cache"
        ocr_cache_dir.mkdir(exist_ok=True)

        self.examples = []
        for label_path in sorted(labels_dir.glob("*.json")):
            stem = label_path.stem
            image_path = images_dir / f"{stem}.png"
            cache_path = ocr_cache_dir / f"{stem}.txt"
            if cache_path.exists():
                ocr_text = cache_path.read_text(encoding="utf-8")
            else:
                ocr_text = extract_text(str(image_path))
                cache_path.write_text(ocr_text, encoding="utf-8")
            label = json.loads(label_path.read_text(encoding="utf-8"))
            self.examples.append((ocr_text, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ocr_text, label = self.examples[idx]
        messages = build_messages(ocr_text)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        target = json.dumps(label, ensure_ascii=False) + self.tokenizer.eos_token
        full_text = prompt + target

        full_ids = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        input_ids = full_ids["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = prompt_ids["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_ids["attention_mask"][0],
            "labels": labels,
        }


def _collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        labels[i, :n] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(data_dir: str = "extractlora/data/synthetic", out_dir: str = DEFAULT_ADAPTER_DIR, epochs: int = 3) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = InvoiceDataset(data_dir, tokenizer)

    args = TrainingArguments(
        output_dir="extractlora/training/runs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Adapter LoRA sauvegarde dans {out_dir}")


if __name__ == "__main__":
    train()
