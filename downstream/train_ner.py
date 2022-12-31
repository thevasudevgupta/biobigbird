from pathlib import Path
from typing import Any, Dict, List

import pydantic
import torch
import torch.nn as nn
from datasets import load_dataset
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


args = TrainingArgs()
print(args)

logger = wandb.init(project=args.project_name, config=args.dict())

# TODO: update based on your task
num_labels = 3

model_id = "ddp-iitm/biobigbird-base-uncased"
model = BigBirdForTokenClassification.from_pretrained(
    model_id, use_auth_token=True, num_labels=num_labels
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TODO: update your dataset here
# you can use `torch.utils.data.Dataset` or list like following
train_data = [
    {"input_text": "protein oxygen", "labels": [0, 2, 1, 0]},
    {"input_text": "oxygen", "labels": [0, 1, 0]},
]
validation_data = [{"input_text": "oxygen", "labels": [0, 1, 0]}]

# TODO: you need to adapt following collate function based on how your data looks
# output must contain input_ids, attention_mask, labels
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_text = [sample["input_text"] for sample in batch]
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    # batch_size, seqlen

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # batch_size, seqlen

    padding_lengths = (input_ids.shape[1] - attention_mask.sum(dim=1)).numpy().tolist()
    labels = [
        sample["labels"] + [IGNORE_INDEX] * padding_lengths[i]
        for i, sample in enumerate(batch)
    ]
    labels = torch.tensor(labels, dtype=torch.long)
    # batch_size, seqlen

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
validation_dataloader = torch.utils.data.DataLoader(
    validation_data,
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
        loss = (
            loss_fn(logits.view(-1, num_labels), labels.view(-1))
            / args.num_accumulation_steps
        )
        batch_loss += loss.detach()

        loss.backward()

        if (step + 1) % args.num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.log({"train_loss": batch_loss.item() / args.num_accumulation_steps})
            batch_loss = torch.tensor(0.0, device=device)

    val_loss = torch.tensor(0.0, device=device)
    num_iters = 0
    for batch in tqdm(validation_dataloader, f"evaulating epoch-{epoch+1}"):
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
