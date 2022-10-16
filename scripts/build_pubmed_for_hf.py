# python3 scripts/build_pubmed_for_hf.py

from pathlib import Path

import jsonlines
from datasets import load_dataset
from tqdm.auto import tqdm

target_filename = "pubmed.jsonl"
data_dir = "downloads/pubmed_extracted/"
push_to_hub = False

print("collecting data files from", data_dir, "...")
data_files = [str(f) for f in Path(data_dir).glob("**/*") if f.is_file()]
print(f"total number of data files: {len(data_files)}")
# data_files = data_files[:2]

# we are unable to load some of the files with HF datasets due to some utf-8 error
# hence we are first reading the files and putting all the content in single file
with jsonlines.open(target_filename, "a") as writer:
    for filename in tqdm(data_files):
        content = open(filename, "r", errors="ignore", encoding="utf-8").read()
        writer.write({"article": content})

# let's just test if we are able to load data using huggingface datasets
data = load_dataset("json", data_files=target_filename, split="train")
print(data)

if push_to_hub:
    data.push_to_hub("ddp-iitm/pubmed_raw_text", private=True)
