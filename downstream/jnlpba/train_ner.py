# WANDB_MODE=offline python3 downstream/jnlpba/train_ner.py

from pathlib import Path
from typing import Any, Dict, List

import evaluate
import pydantic
import torch
import torch.nn as nn
from data import build_data_from_multiple_files, fetch_unique_labels
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BigBirdForTokenClassification

import wandb

IGNORE_INDEX = -10
SEED = 0

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)


class TrainingArgs(pydantic.BaseModel):
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 0

    lr: float = 5.0e-5
    num_accumulation_steps: int = 1
    max_length: int = 4096

    save_dir: str = "ner_checkpoints"

    project_name: str = "bigbird-downstream"

    push_to_hub: bool = False
    repo_id: str = "ddp-iitm/ner_jnlpba"


train_files = [
    "train_data/Genia4ERtask1.iob2",
    "train_data/Genia4ERtask2.iob2",
]

valid_files = [
    "valid_data/Genia4EReval1.iob2",
    "valid_data/Genia4EReval2.iob2",
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

b_to_i_label = {}
for idx, label in idx2label.items():
    if label.startswith("B-") and label.replace("B-", "I-") in label2idx:
        b_to_i_label[label] = label.replace("B-", "I-")
    else:
        b_to_i_label[label] = label
print("b_to_i_label:", b_to_i_label)

args = TrainingArgs()
print(args.json(indent=2))

logger = wandb.init(project=args.project_name, config=args.dict())

model_id = "ddp-iitm/biobigbird-base-uncased"
model = BigBirdForTokenClassification.from_pretrained(
    model_id, use_auth_token=True, num_labels=num_labels
)
model.config.label2id = label2idx
model.config.id2label = idx2label

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
# print(model)

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


train_data = train_data[:10]


def tokenize_labels(batch_labels, inputs):
    global label2idx, b_to_i_label
    output_labels = []
    for i, labels in enumerate(batch_labels):
        tmp_labels = []
        previous_word_idx = None
        for word_idx in inputs.word_ids(batch_index=i):
            if word_idx is None:
                tmp_labels.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                # tmp_labels.append(labels[word_idx])
                tmp_labels.append(label2idx[labels[word_idx]])
            else:
                # tmp_labels.append(b_to_i_label[labels[word_idx]])
                tmp_labels.append(label2idx[b_to_i_label[labels[word_idx]]])
            previous_word_idx = word_idx
        output_labels.append(tmp_labels)

    return output_labels


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    global tokenizer
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

    labels = tokenize_labels([sample["labels"] for sample in batch], inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    # batch_size, seqlen

    labels[labels == label2idx["O"]] = IGNORE_INDEX

    # print(inputs["input_ids"].shape, inputs["attention_mask"].shape, labels.shape)

    # exit()
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
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


def get_predictions_and_references(logits, labels):
    predictions = logits.argmax(-1)
    assert predictions.shape == labels.shape
    y_pred = predictions.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    # Remove ignored index (special tokens)
    predictions = [
        [idx2label[p] for (p, l) in zip(pred, gold_label) if l != IGNORE_INDEX]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    references = [
        [idx2label[l] for (p, l) in zip(pred, gold_label) if l != IGNORE_INDEX]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return predictions, references


batch_loss = torch.tensor(0.0, device=device)
for epoch in range(args.epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader),
        desc=f"Running epoch-{epoch+1}",
        total=len(train_dataloader),
    ):
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

    metric = evaluate.load("seqeval")
    val_loss = torch.tensor(0.0, device=device)
    num_iters = 0
    for batch in tqdm(
        valid_dataloader,
        desc=f"evaulating epoch-{epoch+1}",
        total=len(valid_dataloader),
    ):
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        val_loss += loss_fn(logits.view(-1, num_labels), labels.view(-1))
        num_iters += 1

        predictions, references = get_predictions_and_references(logits, labels)
        metric.add_batch(predictions=predictions, references=references)

    eval_metric = metric.compute()
    print(eval_metric)

    logger.log(
        {
            "validation_loss": val_loss.item() / num_iters,
            "epoch": epoch + 1,
            **eval_metric,
        }
    )

    save_dir = checkpoint_dir / f"epoch-{epoch+1}"
    commit_message = str(save_dir)
    torch.save(optimizer.state_dict(), save_dir / "optimizer_state.bin")
    model.save_pretrained(
        str(save_dir),
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        commit_message=commit_message,
    )
    tokenizer.save_pretrained(
        str(save_dir),
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        commit_message=commit_message,
    )
    print("model saved in", save_dir)

logger.finish()
