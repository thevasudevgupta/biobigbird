from datasets import load_dataset

data = load_dataset("pubmed", cache_dir="/outputs/pubmed-hf")
print(data)

print(data["train"][0])
