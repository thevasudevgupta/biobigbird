from datasets import load_from_disk
from tqdm.auto import tqdm


def count_tokens(data, column_name="pages"):
    num_tokens = 0
    for sample in tqdm(data, desc="counting # tokens"):
        for page in sample[column_name]:
            num_tokens += len(page.split())
    return num_tokens


data = load_from_disk("data/biobooks_raw_text")

print("# tokens", count_tokens(data))
