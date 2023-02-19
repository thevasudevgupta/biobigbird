# python3 scripts/mimic3/upload_hf.py

from datasets import load_dataset

push_to_hub = True

ds = load_dataset("text", data_files="data.txt", split="train")
print(ds)

if push_to_hub:
    ds.push_to_hub("ddp-iitm/mimic3_raw_v2", private=True)
