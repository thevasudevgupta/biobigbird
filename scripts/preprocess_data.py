from datasets import load_dataset

books_data = load_dataset("ddp-iitm/biobooks_raw_text", split="train")
print(books_data)

pubmed_data = load_dataset("ddp-iitm/pubmed_raw_text", split="train")
print(pubmed_data)
