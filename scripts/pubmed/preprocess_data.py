# nothing is useful after conclusion?
# we need to introduce <sep> between topic changes => abstract <sep> topic1 <sep> conclusion
# sequences > 4096 should be truncated? or should we split into multiple samples
# remove sequences whose length < 512
# remove samples whoch donot have introduction
# lower case all the text

# python3 scripts/pubmed/preprocess_data.py

# steps:
# remove exact duplicates
# remove all the text after Conclusion/Conclusions
# start with Introduction
# Introduction - - Conclusion
# checkout how many sequences have length > 4096
import re

from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm

push_to_hub = False
num_proc = 8
numbers_pattern = re.compile("\d+")


article_end = set(
    [
        "disclosures",
        "disclosure",
        "acknowledgements",
        "acknowledgment",
        "reference",
        "references",
        "refs",
        "funding",
        "abbreviations",
        "additional file",
    ]
)


def count_tokens(data, column_name="article"):
    num_tokens = 0
    for sample in tqdm(data, desc="counting # tokens"):
        num_tokens += len(sample[column_name].split())
    return num_tokens


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


def startswith_or_endswith(text, patterns):
    for pattern in patterns:
        if text.startswith(pattern) or text.endswith(pattern):
            return True
    return False


def preprocess(text):
    # print(text)
    paragraphs = text.lower().split("\n")

    start_idx = end_idx = None
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        # we look for the 1st occurence of `introduction`
        if startswith_or_endswith(paragraph, ["introduction"]):
            if start_idx is None:
                start_idx = i
                if paragraph.endswith("introduction"):
                    start_idx += 1
        # we look for the first occurence of `conclusion`
        elif start_idx is not None:
            if startswith_or_endswith(paragraph, ["conclusions", "conclusion"]):
                # +1 because we want to include the conclusion paragraph
                # +1 because of list indexing
                end_idx = i + 1 + 1
                break
            elif startswith_or_endswith(paragraph, article_end):
                end_idx = i
                break

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
            len(paragraph.split()),
        )
        < 0.3
    ]

    text = "\n".join(new_paragraphs)

    text = text.replace("==== front", "").strip()
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


def filter_duplicates(sample, index, last_index, column_name):
    if index == last_index:
        return True
    return data[index + 1][column_name] != sample[column_name]


# data = load_dataset(
#     "parquet", data_files="train-00041-of-00389-ad86292e94260987.parquet", split="train", num_proc=num_proc
# )
# data = load_from_disk("data/pubmed_raw_text")
# print(data)
# # print("# tokens (before preprocessing) :", count_tokens(data, column_name="article"))

# print("let's go again")
# data = data.map(
#     lambda x: {"preprocessed": preprocess(x["article"])},
#     load_from_cache_file=False,
#     num_proc=num_proc,
# )

# print("saving")
# data.save_to_disk("pubmed_raw_text_v2")
# print("saved")
# exit()

# print("loading ... ")
# data = load_from_disk("pubmed_raw_text_v2")
# print("loading done")

# print("starting step-1")
# data = data.filter(
#     lambda x: filter_samples_with_useful_content(x["preprocessed"]),
#     load_from_cache_file=False,
#     num_proc=num_proc,
# )
# print("step-1 finished")
# print("saving")
# data.save_to_disk("pubmed_raw_text_v3")
# print("saved")
# exit()

data = load_from_disk("pubmed_raw_text_v3")
print(data)

print("starting step-2")
column_name = "preprocessed"
last_index = len(data) - 1
data = data.sort(column_name)
data = data.filter(
    lambda x, index: filter_duplicates(x, index, last_index, column_name),
    with_indices=True,
    load_from_cache_file=False,
)
print(data)
print("step-2 finished")

print("saving")
data.save_to_disk("pubmed_raw_text_v4")
print("saved")

print(
    "# tokens (after preprocessing) :", count_tokens(data, column_name="preprocessed")
)
print(data)

if push_to_hub:
    data.push_to_hub("ddp-iitm/pubmed_raw_text_v2", private=True)
