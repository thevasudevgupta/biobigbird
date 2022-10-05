# python3 scripts/extract_text_from_books.py

from pathlib import Path
from typing import List

import jsonlines
import PyPDF2
from tqdm.auto import tqdm

target_filename = "books.jsonl"
books_dir = Path("books")
print("total no of books:", len(list(books_dir.iterdir())))


def extract_text_from_pdf(filename: str, verbose=True) -> List[str]:
    obj = open(filename, "rb")
    pdf = PyPDF2.PdfFileReader(obj)
    pages = []
    iterator = tqdm(range(pdf.numPages)) if verbose else range(pdf.numPages)
    for i in iterator:
        page = pdf.getPage(i).extractText()
        if page == "":
            continue
        pages.append(page)
    obj.close()
    return pages


with jsonlines.open(target_filename, "a") as writer:
    for filename in tqdm(list(books_dir.iterdir())):
        try:
            content = {"pages": extract_text_from_pdf(filename, verbose=False)}
            writer.write(content)
        except Exception as e:
            print("skipping:", filename)
            print(e)

# just checking if we are able to load this data properly
from datasets import load_dataset

data = load_dataset("json", data_files=target_filename, split="train")
print(data)
