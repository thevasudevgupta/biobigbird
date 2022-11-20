from datasets import load_dataset
from tqdm.auto import tqdm

download_pubmed = False

# books_data = load_dataset("ddp-iitm/biobooks_raw_text", split="train", use_auth_token=True, cache_dir="data/biobooks_raw_text")
# print(books_data)

if download_pubmed:
    pubmed_data = load_dataset(
        "ddp-iitm/pubmed_raw_text",
        split="train",
        use_auth_token=True,
        cache_dir="tmp/pubmed_raw_text",
    )
    pubmed_data.save_to_disk("data/pubmed_raw_text")

# rm -rf tmp
