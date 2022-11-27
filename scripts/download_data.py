from datasets import load_dataset

download_books = False
download_pubmed = True

if download_books:
    books_data = load_dataset(
        "ddp-iitm/biobooks_raw_text",
        split="train",
        use_auth_token=True,
        cache_dir="data/biobooks_raw_text",
    )
    print(books_data)
    books_data.save_to_disk("data/biobooks_raw_text")


if download_pubmed:
    pubmed_data = load_dataset(
        "ddp-iitm/pubmed_raw_text",
        split="train",
        use_auth_token=True,
        cache_dir="tmp/pubmed_raw_text",
    )
    print(pubmed_data)
    pubmed_data.save_to_disk("data/pubmed_raw_text")
