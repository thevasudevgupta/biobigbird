from datasets import load_from_disk
from datasets import DatasetDict

print("loading ...")
ds = load_from_disk('pubmed_raw_text_v3')
print(ds)
print(len(ds[0]['article']))
print(len(ds[0]['preprocessed']))

ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
test_data = ds['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)

ds = DatasetDict({"train": ds['train'], "validation": test_data['train'], "test": test_data['test']})
print(ds)
print("final preprocessing ...")
ds = ds.remove_columns('article')
ds = ds.rename_column('preprocessed', 'article')
print("done bro!")
print(ds)
ds.push_to_hub('ddp-iitm/pubmed_raw_text_v3', private=True)

# print("uploading boss")
# data.push_to_hub("ddp-iitm/pubmed_raw_text_v2", private=True)
# print("done baby!")
