from transformers import AutoTokenizer

vocab_size = 32000
batch_size = 64
use_auth_token = True
tokenizer_save_dir = "pubmed_preprocessed_v1_tokenizer"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data = load_dataset("ddp-iitm/pubmed_preprocessed_v1", use_auth_token=use_auth_token)
iterator = iter(data[i : i + batch_size] for i in range(0, len(data), batch_size))

tokenizer = tokenizer.train_new_from_iterator(iterator, vocab_size=vocab_size)
print(tokenizer)
print(tokenizer.save_pretrained(tokenizer_save_dir))
