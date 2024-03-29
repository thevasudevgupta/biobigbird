import optax
import yaml
from flax import traverse_util


def read_yaml(path):
    return yaml.safe_load(open(path, "r"))


def hf_save_fn(
    save_dir,
    params,
    model_save_fn,
    tokenizer_save_fn,
    push_to_hub=False,
    repo_id=None,
    commit_message=None,
):
    save_dir = str(save_dir)
    if push_to_hub:
        assert repo_id is not None
    model_save_fn(
        save_dir,
        params=params,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    tokenizer_save_fn(
        save_dir,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        commit_message=commit_message,
    )


def linear_scheduler_with_warmup(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=1e-7, transition_steps=decay_steps
    )
    lr = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )
    return lr


def create_tx(lr, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {
            k: (k[-1] != "bias" and k[-2:] != ("LayerNorm", "scale"))
            for k in params.keys()
        }
        return traverse_util.unflatten_dict(mask)

    tx = optax.adamw(
        learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask
    )
    return tx
