from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

ds = load_dataset("bigbio/ebm_pico")
print(ds)

ds = ds.map(
    lambda x: {
        "entity": [s["text"] for s in x["entities"]],
        "annotation_type": [s["annotation_type"] for s in x["entities"]],
    }
)
print(ds)


def prepare_labels(ds):
    all_labels = []
    for sample in tqdm(ds):
        entities = sample["entity"]
        e2a = {e: a for e, a in zip(entities, sample["annotation_type"])}
        hased_entities = set(entities)
        labels = []
        for word in sample["text"].split():
            if word in hased_entities:
                labels.append(e2a[word])
            else:
                labels.append("<ignore>")
        all_labels.append(labels)
    ds = ds.map(lambda x, i: {"labels": all_labels[i]}, with_indices=True)
    ds = ds.map(lambda x: {"tokens": x["text"].split()})

    # for sample in tqdm(ds):
    #     assert len(ds["tokens"]) == len(ds["labels"])
    return ds


ds = DatasetDict(
    {
        "train": prepare_labels(ds["train"]),
        "test": prepare_labels(ds["test"]),
    }
)
print(ds)

push_to_hub = True

if push_to_hub:
    ds.push_to_hub("ddp-iitm/ebm_pico", private=True)

# from datasets import load_dataset
# ds = load_dataset("ddp-iitm/ebm_pico")
