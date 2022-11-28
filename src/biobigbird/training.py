from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import pydantic
from flax import jax_utils, struct
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import shard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

PathType = Union[Path, str]
OPTIMIZER_STATE_PATH = "optim_state.msgpack"
MODEL_PATH = "flax_model.msgpack"
TRAINING_STATE_PATH = "training_state.yaml"


@struct.dataclass
class TrainingStepOutput:
    state: train_state.TrainState
    dropout_rng: jnp.DeviceArray

    # following are used only for logging purposes
    loss: jnp.DeviceArray
    lr: Optional[jnp.DeviceArray] = None


@struct.dataclass
class ValidationStepOutput:
    loss: jnp.DeviceArray


class BaseConfig(pydantic.BaseModel):
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        return cls(**config_dict)

    def to_dict(self):
        return self.dict()


from typing import Tuple

import datasets


class IterableDataLoader:
    def __init__(
        self,
        dataset: datasets.IterableDataset,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(
        self,
    ) -> Union[Tuple[jnp.DeviceArray], Dict[str, jnp.DeviceArray], jnp.DeviceArray]:
        batch = []
        for i, sample in enumerate(self.dataset):
            batch.append(sample)

            if (i + 1) % self.batch_size == 0:
                if self.collate_fn is not None:
                    batch = self.collate_fn(batch)

                yield batch
                batch = []

    def shuffle(self, seed: int):
        self.dataset.set_epoch(seed)


class TrainerConfig(BaseConfig):
    max_epochs: int
    batch_size_per_device: int
    wandb_project_name: str = "biobigbird"
    epochs_save_dir: Optional[str] = None
    num_save_steps: int = 10000
    num_eval_steps: int = 10000
    logging_steps: int = 1
    max_steps_per_epoch: int = -1

    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> "TrainerConfig":
        return cls(**dictionary)


@dataclass
class Trainer:
    config: TrainerConfig
    training_step: Callable
    validation_step: Callable
    train_pmap_kwargs: Dict[str, Any] = field(default_factory=dict)
    val_pmap_kwargs: Dict[str, Any] = field(default_factory=dict)
    collate_fn: Optional[Callable] = None

    # input signature has `save_dir` & `params`
    model_save_fn: Optional[Callable] = None

    def train(
        self,
        state: train_state.TrainState,
        train_data,
        val_data,
        wandb_configs: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        total_num_steps_per_epoch: Optional[int] = None,
    ):
        wandb_configs = wandb_configs or self.config.to_dict()
        logger = wandb.init(
            project=self.config.wandb_project_name, config=wandb_configs
        )

        # jax.profiler.start_trace("./tensorboard")

        batch_size = self.config.batch_size_per_device * jax.device_count()

        pin_memory = self.config.dataloader_pin_memory
        num_workers = self.config.dataloader_num_workers
        prefetch_factor = self.config.dataloader_prefetch_factor

        train_data = IterableDataLoader(
            train_data.shuffle(seed=seed),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            # shuffle=False,
            # pin_memory=pin_memory,
            # num_workers=num_workers,
            # drop_last=True,
            # prefetch_factor=prefetch_factor,
        )

        val_data = IterableDataLoader(
            val_data,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            # shuffle=False,
            # pin_memory=pin_memory,
            # num_workers=num_workers,
            # drop_last=True,
            # prefetch_factor=prefetch_factor,
        )

        state = jax_utils.replicate(state)
        training_step = jax.pmap(self.training_step, **self.train_pmap_kwargs)
        validation_step = jax.pmap(self.validation_step, **self.val_pmap_kwargs)

        rng = jax.random.PRNGKey(seed)
        dropout_rng = jax.random.split(rng, jax.device_count())

        for epoch in range(self.config.max_epochs):
            tr_loss, avg_tr_loss = jnp.array(0), jnp.array(0)

            pbar = tqdm(
                enumerate(train_data),
                desc=f"Running epoch-{epoch}",
                total=total_num_steps_per_epoch,
            )
            if seed is not None:
                train_data.shuffle(epoch + seed)
            for step, batch in pbar:
                batch = shard(batch)

                outputs = training_step(state, dropout_rng, batch)
                state, dropout_rng = outputs.state, outputs.dropout_rng

                loss = jax_utils.unreplicate(outputs.loss)
                tr_loss += loss
                avg_tr_loss += loss

                if (step + 1) % self.config.logging_steps == 0:
                    logs = {
                        "tr_loss": tr_loss.item() / self.config.logging_steps,
                        "avg_tr_loss": avg_tr_loss.item() / (step + 1),
                    }
                    if outputs.lr is not None:
                        logs["lr"] = jax_utils.unreplicate(outputs.lr).item()

                    pbar.set_postfix(**logs)
                    logger.log(logs)
                    tr_loss = jnp.array(0)

                if (step + 1) == self.config.max_steps_per_epoch:
                    break

                if (step + 1) % self.config.num_save_steps == 0 and self.config.num_save_steps > 0:
                    if self.config.epochs_save_dir is not None:
                        print("saving step -", step + 1)
                        self.save_checkpoint(
                            jax_utils.unreplicate(state),
                            Path(self.config.epochs_save_dir, f"epoch-{epoch+1}", f'step-{step+1}'),
                        )

                if (step + 1) % self.config.num_eval_steps == 0 and self.config.num_eval_steps > 0:
                    val_steps, val_loss = 0, jnp.array(0)
                    for batch in tqdm(val_data, desc="evaluating ..."):
                        batch = shard(batch)
                        outputs = validation_step(state, batch)
                        val_loss += jax_utils.unreplicate(outputs.loss)
                        val_steps += 1

                        if val_steps % self.config.max_steps_per_epoch == 0:
                            break

                    loggings = {"val_loss": val_loss.item() / val_steps, "epoch": epoch}
                    print(loggings)
                    logger.log(loggings)

            if self.config.epochs_save_dir is not None:
                print('saving epoch -', epoch + 1)
                self.save_checkpoint(
                    jax_utils.unreplicate(state),
                    Path(self.config.epochs_save_dir, f"epoch-{epoch + 1}"),
                )

        # jax.profiler.stop_trace()

        return jax_utils.unreplicate(state)

    def save_checkpoint(
        self,
        state: train_state.TrainState,
        ckpt_dir: PathType,
    ) -> Path:
        # state must be unreplicated before passing it

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        if self.model_save_fn is not None:
            self.model_save_fn(ckpt_dir, state.params, commit_message=str(ckpt_dir))
        else:
            with open(ckpt_dir / MODEL_PATH, "wb") as f:
                f.write(to_bytes(state.params))
        with open(ckpt_dir / OPTIMIZER_STATE_PATH, "wb") as f:
            f.write(to_bytes(state.opt_state))

        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: PathType):
        ...
