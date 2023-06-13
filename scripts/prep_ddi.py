from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm


def extract_data(filename):
    with open(filename) as f:
        soup = BeautifulSoup(f, "html.parser")
    data = []
    document = soup.find_all("document")
    for doc in document:
        sentence = doc.find_all("sentence")
        for sent in sentence:
            text = sent["text"]
            pairs = sent.find_all("pair")
            for pair in pairs:
                if pair["ddi"] == "true" and "type" in pair.attrs:
                    e1 = sent.find_all("entity", id=pair["e1"])
                    e2 = sent.find_all("entity", id=pair["e2"])
                    assert len(e1) == 1
                    assert len(e2) == 1
                    data.append((text, e1[0]["text"], e2[0]["text"], pair["type"]))
    return data


def build_data(data_dir):
    data = []
    files = Path(data_dir).glob("**/*.xml")
    for filename in tqdm(files):
        data.extend(extract_data(filename))
    df = pd.DataFrame(data, columns=["text", "e1", "e2", "relation"])
    return df


def create_important_cols(df):
    df["sample"] = df.apply(
        lambda x: f"find relation between entities - {x['e1']} and {x['e2']} in {x['text']}",
        axis=1,
    )
    df["target"] = df["relation"]
    return df


data_dir = "/Users/vasudevgupta/downloads/DDICorpus/Train"
df = create_important_cols(build_data(data_dir))
df.to_csv("ddi_train.csv", index=False)
print(df)

data_dir = "/Users/vasudevgupta/downloads/DDICorpus/Test"
df = create_important_cols(build_data(data_dir))
df.to_csv("ddi_test.csv", index=False)
print(df)

ds = DatasetDict(
    {
        "train": load_dataset("csv", data_files="ddi_train.csv", split="train"),
        "valid": load_dataset("csv", data_files="ddi_test.csv", split="train"),
    }
)
print(ds)
push_to_hub = False
if push_to_hub:
    ds.push_to_hub("bisectgroup/ddi", private=True)
