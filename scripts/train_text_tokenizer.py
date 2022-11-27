from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

push_to_hub = True
vocab_size = 32000
batch_size = 1000
num_samples = 100000

tokenizer_save_dir = "pubmed_raw_text_v3_tokenizer"
tokenizer_id = "ddp-iitm/pubmed_raw_text_v3_tokenizer"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ds = load_from_disk("pubmed_raw_text_v3")
ds = load_dataset(
    "ddp-iitm/pubmed_raw_text_v3", split="train", streaming=True, use_auth_token=True
)
ds = ds.shuffle(seed=42).select(range(num_samples))

iterator = (ds[i : i + batch_size] for i in range(0, len(ds), batch_size))

tokenizer = tokenizer.train_new_from_iterator(iterator, vocab_size=vocab_size)
print(tokenizer)

tokenizer.save_pretrained(tokenizer_save_dir)
if push_to_hub:
    tokenizer.push_to_hub(tokenizer_id, private=True)
