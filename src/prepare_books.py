from datasets import load_dataset
filename = "books.jsonl"

data = load_dataset("json", data_files=filename, split="train")
print(data)

def preprocess(sample):
    text = []
    for page in sample["pages"]:
        if page == "":
            continue
        text.append(page)       
    return {"text": "\n".join(text)}

data = data.map(preprocess)
print(data)
