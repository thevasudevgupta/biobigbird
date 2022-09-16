
from biobigbird.training import Trainer, TrainerConfig
from biobigbird.utils import read_yaml, hf_save_fn, linear_scheduler_with_warmup, create_tx
from biobigbird.constants import HF_TOKEN

import optax
from typing import Dict, Callable
import jax.numpy as jnp
import jax
import flax
from flax.training import train_state, TrainingStepOutput, ValidationStepOutput
from transformers import FlaxBigBirdForMaskedLM, AutoTokenizer
from functools import partial

import math
from datasets import load_dataset


def training_step(
    state: train_state.TrainState,
    dropout_rng: jnp.DeviceArray,
    batch: Dict[str, jnp.DeviceArray],
) -> TrainingStepOutput:
    new_drp_rng, drp_rng = jax.random.split(dropout_rng, num=2)

    def loss_fn(params):
        labels = batch.pop("labels")
        label_paddings = batch.pop("label_paddings")

        outputs = state.apply_fn(
            **batch,
            params=params,
            dropout_rng=drp_rng,
            train=True,
            freeze_feature_encoder=True
        )
        seqlen = outputs.logits.shape[1]

        input_lengths = jnp.sum(batch["attention_mask"], axis=1)
        input_lengths = state.get_feat_extract_output_lengths(input_lengths)
        logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

        # taking mean is fine as long as batches are equally distributed
        return state.loss_fn(
            outputs.logits, logit_paddings, labels, label_paddings
        ).mean()

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
    label_paddings = batch.pop("label_paddings")
    batch.pop("mask_time_indices", None)

    input_lengths = jnp.sum(batch["attention_mask"], axis=1)
    input_lengths = state.get_feat_extract_output_lengths(input_lengths)

    outputs = state.apply_fn(**batch, params=state.params, train=False)

    seqlen = outputs.logits.shape[1]
    logit_paddings = input_lengths[..., None] <= jnp.arange(seqlen)

    loss = state.loss_fn(outputs.logits, logit_paddings, labels, label_paddings).mean()
    loss = jax.lax.pmean(loss, axis_name="batch")

    return ValidationStepOutput(loss=loss)


class DataCollatorForMLM:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        return


class TrainState(train_state.TrainState):
    loss_fn: Callable = flax.struct.field(pytree_node=False)
    lr_scheduler: Callable = flax.struct.field(pytree_node=False)


configs_dict = read_yaml("config.yaml")

collate_fn = DataCollatorForMLM(**configs_dict["data_collator"])

model_config = configs_dict["model"]
model_id = model_config.pop("model_id")
model = FlaxBigBirdForMaskedLM.from_pretrained(model_id, **model_config, use_auth_token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

save_fn = partial(
    hf_save_fn,
    model_save_fn=model.save_pretrained,
    tokenizer_save_fn=tokenizer.save_pretrained,
    push_to_hub=False,
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

dataset = load_dataset("scientific_papers", "pubmed")
train_data, val_data = dataset["train"], dataset["validation"]
print(train_data, val_data)

# we are dropping the last batch for now
batch_size = trainer_config.batch_size_per_device * jax.device_count()
num_steps = math.ceil(len(train_data) // batch_size)

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
    loss_fn=optax.softmax_cross_entropy,
    lr_scheduler=lr_scheduler,
)

new_state = trainer.train(state, train_data, val_data, wandb_configs=configs_dict)
