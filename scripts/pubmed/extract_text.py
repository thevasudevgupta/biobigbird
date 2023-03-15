# gzip -d -r downloads/pubmed_extracted/

import glob
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm.auto import tqdm

push_to_hub = False


def extract_text(filename):
    with open(filename, "r") as f:
        data = f.read()
    data = BeautifulSoup(data, "xml")
    data = data.find_all("AbstractText")
    data = [" ".join(tag.text.split()) for tag in data]
    return data


data_dir = Path("downloads/pubmed_extracted/")
files = glob.glob(str(data_dir / "*.xml"))

target_filename = Path("pubmed_abstracts.txt")
assert not target_filename.exists()
with open(target_filename, "w") as f:
    for filename in tqdm(files):
        f.write("\n".join(extract_text(filename)))

ds = load_dataset("text", data_files=str(target_filename))
print(ds)

if push_to_hub:
    ds.push_to_hub("ddp-iitm/pubmed_abstracts", private=True)
