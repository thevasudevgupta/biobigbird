import json

from tqdm import tqdm


def build_and_save_data(
    abstract_file="ChemProt_Corpus/chemprot_training/chemprot_training_abstracts.tsv",
    entity_file="ChemProt_Corpus/chemprot_training/chemprot_training_entities.tsv",
    relation_file="ChemProt_Corpus/chemprot_training/chemprot_training_relations.tsv",
    target_file="chemprot_train.json",
):
    data = open(abstract_file).read().split("\n")
    id2text = {
        sent.split("\t", 1)[0]: sent.split("\t", 1)[1] for sent in data if len(sent) > 0
    }

    data = open(entity_file).read().split("\n")
    entities = {}
    for line in tqdm(data):
        items = line.split("\t", 5)
        if len(items) < 6:
            print(items)
            continue
        entities[f"{items[0]}.{items[1]}"] = items[-1]
    print(list(entities.items())[:10])

    ds = []

    data = open(relation_file).read().split("\n")
    for line in tqdm(data):
        items = line.split("\t", 5)
        if len(items) < 6:
            print(items)
            continue

        id = items[0]
        e1 = items[4].split(":", 1)[1]
        e2 = items[5].split(":", 1)[1]

        e1 = entities[f"{id}.{e1}"]
        e2 = entities[f"{id}.{e2}"]

        text = id2text[id]
        relation = items[3]

        sample = f"find relation between entities - {e1} and {e2} in {text}"
        target = relation

        ds.append({"sample": sample, "target": target})

    with open(target_file, "w") as f:
        json.dump(ds, f)


build_and_save_data(
    abstract_file="ChemProt_Corpus/chemprot_training/chemprot_training_abstracts.tsv",
    entity_file="ChemProt_Corpus/chemprot_training/chemprot_training_entities.tsv",
    relation_file="ChemProt_Corpus/chemprot_training/chemprot_training_relations.tsv",
    target_file="chemprot_train.json",
)
build_and_save_data(
    abstract_file="ChemProt_Corpus/chemprot_development/chemprot_development_abstracts.tsv",
    entity_file="ChemProt_Corpus/chemprot_development/chemprot_development_entities.tsv",
    relation_file="ChemProt_Corpus/chemprot_development/chemprot_development_relations.tsv",
    target_file="chemprot_valid.json",
)

from datasets import DatasetDict, load_dataset

ds = DatasetDict(
    {
        "train": load_dataset("json", data_files="chemprot_train.json", split="train"),
        "valid": load_dataset("json", data_files="chemprot_valid.json", split="train"),
    }
)

print(ds)

push_to_hub = False

if push_to_hub:
    ds.push_to_hub("ddp-iitm/chemprot", private=True)
