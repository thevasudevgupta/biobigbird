from datasets import load_from_disk
from transformers import AutoTokenizer

push_to_hub = True
vocab_size = 32000
batch_size = 1000
num_samples = 1000000

tokenizer_save_dir = "pubmed_raw_text_v2_tokenizer"
tokenizer_id = "ddp-iitm/pubmed_raw_text_v2_tokenizer"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data = load_from_disk("pubmed_raw_text_v2")
data = data.shuffle(seed=0).select(range(num_samples))

iterator = (data[i : i + batch_size] for i in range(0, len(data), batch_size))

tokenizer = tokenizer.train_new_from_iterator(iterator, vocab_size=vocab_size)
print(tokenizer)

tokenizer.save_pretrained(tokenizer_save_dir)
if push_to_hub:
    tokenizer.push_to_hub(tokenizer_id, private=True)
