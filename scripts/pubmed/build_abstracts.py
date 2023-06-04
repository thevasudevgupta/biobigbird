# python3 scripts/pubmed/build_abstracts.py
push_to_hub = False

from datasets import load_dataset


def extract_abstracts(sample):
    title = sample["MedlineCitation"]["Article"]["ArticleTitle"]
    abstract = sample["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
    return {"text": " ".join((title + " " + abstract).split())}


def build_dataset():
    ds = load_dataset("pubmed")
    print(ds)
    remove_columns = ["MedlineCitation", "PubmedData"]
    ds = ds.map(lambda x: extract_abstracts(x), remove_columns=remove_columns)
    print(ds)

    ok = iter(ds)
    for _ in range(2):
        print(next(ok))
        print("\n\n")

    return ds


ds = build_dataset()
ds = ds.train_test_split(test_size=40000, seed=42)
print(ds)

if push_to_hub:
    ds.push_to_hub("ddp-iitm/pubmed_abstracts", private=True)
