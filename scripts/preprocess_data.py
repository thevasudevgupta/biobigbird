# nothing is useful after conclusion?
# we need to introduce <sep> between topic changes => abstract <sep> topic1 <sep> conclusion
# sequences > 4096 should be truncated? or should we split into multiple samples
# remove sequences whose length < 512
# remove samples whoch donot have introduction
# lower case all the text

# steps:
# remove exact duplicates
# remove all the text after Conclusion/Conclusions
# start with Introduction
# Introduction - - Conclusion
# checkout how many sequences have length > 4096
import re

from datasets import load_dataset
from tqdm.auto import tqdm

data = load_dataset(
    "parquet", data_files="train-00000-of-00389-a3285e04b5e3defa.parquet", split="train"
)
print(data)


def count_tokens(data, column_name="article"):
    num_tokens = 0
    for sample in tqdm(data, desc="counting # tokens"):
        num_tokens += len(sample[column_name].split())
    return num_tokens


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


numbers_pattern = re.compile("[0-9]")
other_pattern = re.compile("[^0-9]")


def preprocess(text):
    # print(text)
    paragraphs = text.lower().split("\n")

    start_idx = end_idx = None
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        # we look for the 1st occurence of `introduction`
        if paragraph == "introduction":
            if start_idx is None:
                start_idx = i
        # we look for the last occurence of `conclusion`
        elif paragraph == "conclusion":
            # +1 because we want to include the conclusion paragraph
            # +1 because of list indexing
            end_idx = i + 1 + 1

    if start_idx is None:
        start_idx = 0

    if end_idx is not None and end_idx <= start_idx:
        end_idx = None

    new_paragraphs = [paragraph.strip() for paragraph in paragraphs[start_idx:end_idx]]
    new_paragraphs = [
        " ".join(paragraph.split()) for paragraph in new_paragraphs if paragraph != ""
    ]
    new_paragraphs = [
        paragraph
        for paragraph in new_paragraphs
        if not (paragraph.startswith("fig.") or paragraph.startswith("table"))
    ]
    new_paragraphs = [
        paragraph
        for paragraph in new_paragraphs
        if safe_divide(
            len(numbers_pattern.findall(paragraph)),
            len(other_pattern.findall(paragraph)),
        )
        < 0.06
    ]

    text = "\n".join(new_paragraphs)
    # print(text)
    # exit()
    return text


def filter_samples_with_useful_content(text):
    text = text.lower()

    # these samples probably do not have any useful content
    if text.find("introduction") == -1 and text.find("conclusion") == -1:
        return False

    # we probably don't need to keep super short samples
    # as bigbird won't work very well on them
    if len(text.split()) < 512:
        return False

    return True


print("before preprocessing:", count_tokens(data, column_name="article"))
print(len(data))
data = data.filter(lambda x: filter_samples_with_useful_content(x["article"]))
print(len(data))
data = data.map(
    lambda x: {"preprocessed": preprocess(x["article"])}, load_from_cache_file=False
)
print("after preprocessing:", count_tokens(data, column_name="preprocessed"))


def filter_duplicates(sample, index, last_index, column_name):
    if index == last_index:
        return True
    return data[index + 1][column_name] != sample[column_name]


column_name = "preprocessed"
last_index = len(data) - 1
data = data.sort(column_name)
data = data.filter(
    lambda x, index: filter_duplicates(x, index, last_index, column_name),
    with_indices=True,
)

print(data)

# num_proc = 8
# load_from_cache_file = False
# streaming = True

# data = load_dataset("ddp-iitm/pubmed_raw_text", split="train", streaming=streaming, use_auth_token=True, cache_dir="data/pubmed_raw_text")
# # data = data.select(range(10))
