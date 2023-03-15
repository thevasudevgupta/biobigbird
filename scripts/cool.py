from datasets import load_dataset, interleave_datasets

def build_datasets():
    pubmed_ds = load_dataset("ddp-iitm/pubmed_abstracts", streaming=True, split="train")
    print(pubmed_ds)

    ok = iter(pubmed_ds)
    for _ in range(2):
        print(next(ok))
        print("\n\n")

    mimic3_ds = load_dataset("ddp-iitm/mimic3_raw_v2", streaming=True, split="train")
    print(mimic3_ds)

    ok = iter(mimic3_ds)
    for _ in range(2):
        print(next(ok))
        print("\n\n")

    ds = interleave_datasets([pubmed_ds, mimic3_ds], stopping_strategy="all_exhausted")
    print(ds)

    ok = iter(ds)
    for _ in range(10):
        print(next(ok))
        print("\n\n")

    return ds
