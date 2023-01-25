# WANDB_MODE=offline python3 downstream/jnlpba/train_ner.py

from pathlib import Path
from typing import Any, Dict, List

import pydantic
import torch
import torch.nn as nn
from data import build_data_from_multiple_files, fetch_unique_labels
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BigBirdForTokenClassification

import wandb

IGNORE_INDEX = -10


class TrainingArgs(pydantic.BaseModel):
    epochs: int = 10
    batch_size: int = 8
    num_workers: int = 0

    lr: float = 1.0e-5
    num_accumulation_steps: int = 1
    max_length: int = 4096

    save_dir: str = "checkpoints"

    project_name: str = "bigbird-downstream"


train_files = [
    "/Users/vasudevgupta/downloads/Genia4ERtraining/Genia4ERtask1.iob2",
    "/Users/vasudevgupta/downloads/Genia4ERtraining/Genia4ERtask2.iob2",
]

valid_files = [
    "/Users/vasudevgupta/downloads/Genia4ERtest/Genia4EReval1.iob2",
    "/Users/vasudevgupta/downloads/Genia4ERtest/Genia4EReval2.iob2",
]

train_data, train_labels = build_data_from_multiple_files(train_files)
valid_data, valid_labels = build_data_from_multiple_files(valid_files)

train_unique_labels, train_total_labels = fetch_unique_labels(train_labels)
valid_unique_labels, valid_total_labels = fetch_unique_labels(valid_labels)
print(
    "valid_unique_labels - train_unique_labels:",
    valid_unique_labels - train_unique_labels,
)

label2idx = {label: idx for idx, label in enumerate(sorted(train_unique_labels))}
idx2label = {idx: label for label, idx in label2idx.items()}
num_labels = len(label2idx)

args = TrainingArgs()
print(args)

logger = wandb.init(project=args.project_name, config=args.dict())

model_id = "ddp-iitm/biobigbird-base-uncased"
model = BigBirdForTokenClassification.from_pretrained(
    model_id, use_auth_token=True, num_labels=num_labels
)
model.config.label2id = label2idx
model.config.id2label = idx2label

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data = [
    {"sent": sent, "labels": labels} for sent, labels in zip(train_data, train_labels)
]
valid_data = [
    {"sent": sent, "labels": labels} for sent, labels in zip(valid_data, valid_labels)
]

print(train_data[0])
print(valid_data[0])


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_text = [sample["sent"] for sample in batch]
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
        is_split_into_words=True,
    )
    # batch_size, seqlen
    print(inputs)
    print([sample["labels"] for sample in batch])

    labels = []

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # batch_size, seqlen

    padding_lengths = (input_ids.shape[1] - attention_mask.sum(dim=1)).numpy().tolist()
    labels = [
        sample + [IGNORE_INDEX] * padding_lengths[i] for i, sample in enumerate(labels)
    ]
    labels = torch.tensor(labels, dtype=torch.long)
    # batch_size, seqlen

    exit()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    pin_memory=True,
    collate_fn=collate_fn,
    num_workers=args.num_workers,
    shuffle=True,
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=args.batch_size,
    pin_memory=True,
    collate_fn=collate_fn,
    num_workers=args.num_workers,
    shuffle=False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

checkpoint_dir = Path(args.save_dir)
checkpoint_dir.mkdir(exist_ok=True, parents=True)

batch_loss = torch.tensor(0.0, device=device)
for epoch in range(args.epochs):
    desc = f"Running epoch-{epoch+1}"
    for step, batch in tqdm(enumerate(train_dataloader), desc=desc):
        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits

        loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
        loss = loss / args.num_accumulation_steps
        batch_loss += loss.detach()
        loss.backward()

        if (step + 1) % args.num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.log({"train_loss": batch_loss.item() / args.num_accumulation_steps})
            batch_loss = torch.tensor(0.0, device=device)

    val_loss = torch.tensor(0.0, device=device)
    num_iters = 0
    for batch in tqdm(valid_dataloader, f"evaulating epoch-{epoch+1}"):
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        val_loss += loss_fn(logits.view(-1, num_labels), labels.view(-1))
        num_iters += 1
    logger.log({"validation_loss": val_loss.item() / num_iters, "epoch": epoch + 1})

    save_dir = checkpoint_dir / f"epoch-{epoch+1}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(optimizer.state_dict(), save_dir / "optimizer_state.bin")

logger.finish()
