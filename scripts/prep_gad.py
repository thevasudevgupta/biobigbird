import pandas as pd

df1 = pd.read_csv("GAD_Corpus_IBIgroup/GAD_Y_N.csv", sep="\t")
df2 = pd.read_csv("GAD_Corpus_IBIgroup/GAD_F.csv", sep="\t")
df = pd.concat([df1, df2])
df = df[["GAD_ASSOC", "NER_GENE_ENTITY", "NER_DISEASE_ENTITY", "GAD_CONCLUSION"]]
print(df.shape)
print(df.columns)

def build_and_save_data():
    df["sample"] = df.apply(lambda x: f"find relation between entities - {x['NER_GENE_ENTITY']} and {x['NER_DISEASE_ENTITY']} in {x['GAD_CONCLUSION']}", axis=1)
    df["target"] = df["GAD_ASSOC"]
    df.to_csv("gad.csv", index=False)

build_and_save_data()

from datasets import DatasetDict, load_dataset

ds = load_dataset("csv", data_files="gad.csv", split="train")
ds = ds.train_test_split(test_size=0.1, seed=1337)

ds = DatasetDict(
    {
        "train": ds["train"],
        "valid": ds["test"],
    }
)

print(ds)

push_to_hub = False

if push_to_hub:
    ds.push_to_hub("ddp-iitm/gad", private=True)
