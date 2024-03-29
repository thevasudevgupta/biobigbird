import math
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from flax.training import train_state
from transformers import AutoTokenizer, BigBirdConfig, FlaxBigBirdForMaskedLM

from biobigbird.constants import HF_TOKEN, IGNORE_INDEX
from biobigbird.training import (BaseConfig, Trainer, TrainerConfig,
                                 TrainingStepOutput, ValidationStepOutput)
from biobigbird.utils import (create_tx, hf_save_fn,
                              linear_scheduler_with_warmup, read_yaml)

seed = 42


def build_datasets(dataset_ids):
    datasets = []
    for dataset_id in dataset_ids:
        ds = load_dataset(
            dataset_id, streaming=True, split="train", use_auth_token=True
        )
        datasets.append(ds)
    return interleave_datasets(datasets, stopping_strategy="all_exhausted")


def cross_entropy(logits, labels, ignore_index=IGNORE_INDEX):
    """
    Args:
        logits: bsz, seqlen, vocab_size
        labels: bsz, seqlen
    """
    loss_mask = labels != ignore_index

    vocab_size = logits.shape[-1]
    labels = (labels[..., None] == jnp.arange(vocab_size)[None]).astype("f4")
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * logits, axis=-1)

    loss = jnp.where(loss_mask, loss, 0).sum()
    return loss / jnp.sum(loss_mask)


def training_step(
    state: train_state.TrainState,
    dropout_rng: jnp.DeviceArray,
    batch: Dict[str, jnp.DeviceArray],
) -> TrainingStepOutput:
    new_drp_rng, drp_rng = jax.random.split(dropout_rng, num=2)

    def loss_fn(params):
        labels = batch.pop("labels")

        outputs = state.apply_fn(
            **batch,
            params=params,
            dropout_rng=drp_rng,
            train=True,
        )

        # taking mean is fine as long as batches are equally distributed
        return state.loss_fn(outputs.logits, labels)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    return TrainingStepOutput(
        state=new_state,
        dropout_rng=new_drp_rng,
        loss=jax.lax.pmean(loss, axis_name="batch"),
        lr=state.lr_scheduler(state.step),
    )


def validation_step(
    state: train_state.TrainState, batch: Dict[str, jnp.DeviceArray]
) -> ValidationStepOutput:

    labels = batch.pop("labels")
    outputs = state.apply_fn(**batch, params=state.params, train=False)

    loss = state.loss_fn(outputs.logits, labels)
    loss = jax.lax.pmean(loss, axis_name="batch")

    return ValidationStepOutput(loss=loss)


class DataCollatorForMLMConfig(BaseConfig):
    max_length: int
    mlm_probability: float

    column_name: str = "text"


class DataCollatorForMLM:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]):
        # abstracts = [sample["abstract"] for sample in batch]
        articles = [
            " ".join(sample[self.config.column_name].split()) for sample in batch
        ]
        # _ = [print(article) for article in articles]
        # print()

        inputs = self.tokenizer(
            # abstracts,
            articles,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
            return_special_tokens_mask=True,
        )

        special_tokens_mask = inputs.pop("special_tokens_mask")
        input_ids, labels = self.mask_tokens(inputs["input_ids"], special_tokens_mask)

        return {**inputs, "input_ids": input_ids, "labels": labels}

    def mask_tokens(
        self, input_ids: np.ndarray, special_tokens_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens input_ids/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = input_ids.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.config.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = IGNORE_INDEX  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool")
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(
            "bool"
        )
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype="i4"
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels


class TrainState(train_state.TrainState):
    loss_fn: Callable = flax.struct.field(pytree_node=False)
    lr_scheduler: Callable = flax.struct.field(pytree_node=False)


configs_dict = read_yaml(sys.argv[1])
print(configs_dict)
print(jax.devices())

model_config = configs_dict["model"]
model_id = model_config.pop("model_id")
tokenizer_id = model_config["tokenizer_id"]
revision = model_config.pop("revision", None)
if model_id is None:
    assert tokenizer_id is not None
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, use_auth_token=True, revision=revision
    )
    config = BigBirdConfig(**model_config, vocab_size=tokenizer.vocab_size)
    model = FlaxBigBirdForMaskedLM(config)
else:
    model = FlaxBigBirdForMaskedLM.from_pretrained(
        model_id, **model_config, use_auth_token=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_auth_token=True, revision=revision
    )

print(model.config)
print(tokenizer)

datacollator_config = DataCollatorForMLMConfig.from_dict(configs_dict["data_collator"])
collate_fn = DataCollatorForMLM(datacollator_config, tokenizer)

save_fn = partial(
    hf_save_fn,
    model_save_fn=model.save_pretrained,
    tokenizer_save_fn=tokenizer.save_pretrained,
    push_to_hub=configs_dict["script"]["push_to_hub"],
    repo_id=configs_dict["script"]["repo_id"],
)

trainer_config = TrainerConfig.from_dict(configs_dict["trainer"])
trainer = Trainer(
    trainer_config,
    training_step,
    validation_step,
    train_pmap_kwargs={"axis_name": "batch", "donate_argnums": (0, 1)},
    val_pmap_kwargs={"axis_name": "batch"},
    collate_fn=collate_fn,
    model_save_fn=save_fn,
)

data_config = configs_dict["data"]
streaming = data_config["streaming"]
should_load_from_disk = data_config["should_load_from_disk"]
dataset_id = data_config["dataset_id"]
num_examples = data_config["num_examples"]

assert streaming
# if streaming:
#     assert not should_load_from_disk
#     dataset = load_dataset(dataset_id, streaming=True, use_auth_token=True)
# elif should_load_from_disk:
#     dataset = load_from_disk(dataset_id)
# else:
#     dataset = load_dataset(dataset_id, use_auth_token=True)

dataset: Dataset = build_datasets(configs_dict["data"]["dataset_id"])
dataset = dataset.shuffle(seed=42)
# dataset = dataset.train_test_split(test_size=40000, seed=42)
# train_data, val_data = dataset["train"], dataset["validation"]
val_data = dataset.take(40000)
train_data = dataset.skip(40000)

print("train_data:", train_data)
print("val_data:", val_data)

num_examples = num_examples if num_examples is not None else len(train_data)

# we are dropping the last batch for now
batch_size = trainer_config.batch_size_per_device * jax.device_count()
total_num_steps_per_epoch = math.ceil(num_examples // batch_size)
num_steps = total_num_steps_per_epoch * trainer_config.max_epochs

lr_scheduler = linear_scheduler_with_warmup(
    configs_dict["optax"]["lr"],
    configs_dict["optax"]["init_lr"],
    configs_dict["optax"]["warmup_steps"],
    num_steps,
)
tx = create_tx(lr_scheduler, configs_dict["optax"]["weight_decay"])

state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=tx,
    loss_fn=cross_entropy,
    lr_scheduler=lr_scheduler,
)

new_state = trainer.train(
    state,
    train_data,
    val_data,
    wandb_configs=configs_dict,
    total_num_steps_per_epoch=total_num_steps_per_epoch,
)
