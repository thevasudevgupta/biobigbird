from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ddp-iitm/pubmed_raw_text_v3_tokenizer", use_auth_token=True)

def count_tokens(data, column_name="text"):
    num_tokens = 0
    for sample in tqdm(data, desc="counting # tokens"):
        # num_tokens += len(sample[column_name].split())
        num_tokens += len(tokenizer.tokenize(sample[column_name]))
    return num_tokens

ds = load_dataset("ddp-iitm/pubmed_raw_text_v3", streaming=True, use_auth_token=True)
print("num tokens in train", count_tokens(ds["train"]))
print("num tokens in validation", count_tokens(ds["validation"]))
