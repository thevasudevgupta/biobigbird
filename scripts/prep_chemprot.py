from tqdm import tqdm

data = open("ChemProt_Corpus/chemprot_training/chemprot_training_abstracts.tsv").read().split("\n")
id2text = {sent.split("\t", 1)[0]: sent.split("\t", 1)[1] for sent in data if len(sent) > 0}

data = open("ChemProt_Corpus/chemprot_training/chemprot_training_entities.tsv").read().split("\n")
entities = {}
for line in tqdm(data):
    items = line.split("\t", 5)
    if len(items) < 6:
        print(items)
        continue
    entities[f"{items[0]}.{items[1]}"] = items[-1]
print(list(entities.items())[:10])

ds = []

data = open("ChemProt_Corpus/chemprot_training/chemprot_training_relations.tsv").read().split("\n")
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

import json
with open("chemprot.json", "w") as f:
    json.dump(ds, f)

from datasets import load_dataset
ds = load_dataset("json", data_files="chemprot.json", split="train")
print(ds)

print(ds[0])

push_to_hub = False

if push_to_hub:
    ds.push_to_hub("ddp-iitm/chemprot", private=True)
