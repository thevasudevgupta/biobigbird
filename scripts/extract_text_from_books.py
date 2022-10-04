# python3 src/extract_text_from_books.py

from pathlib import Path
from typing import List

import jsonlines
import PyPDF2
from tqdm.auto import tqdm

target_filename = "books.jsonl"
books_dir = Path("books")
print(list(books_dir.iterdir()))


def extract_text_from_pdf(filename: str) -> List[str]:
    obj = open(filename, "rb")
    pdf = PyPDF2.PdfFileReader(obj)
    pages = []
    for i in tqdm(range(pdf.numPages)):
        pages.append(pdf.getPage(i).extractText())
    obj.close()
    return pages


with jsonlines.open(target_filename, "a") as writer:
    for filename in books_dir.iterdir():
        content = {"pages": extract_text_from_pdf(filename)}
        writer.write(content)


# just checking if we are able to load this data properly
from datasets import load_dataset

data = load_dataset("json", data_files=target_filename, split="train")
print(data)
