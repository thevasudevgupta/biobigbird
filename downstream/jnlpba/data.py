# python3 downstream/jnlpba/data.py
from typing import List


def build_data(filepath: str):
    with open(filepath) as f:
        data = f.read().split("\n")
        sents, tmp_sent, labels, tmp_labels = [], [], [], []
        for line in data:
            if line.startswith("###MEDLINE"):
                continue
            if line == "" and len(tmp_sent) == 0 and len(tmp_labels) == 0:
                continue

            if line == "":
                assert len(tmp_labels) == len(tmp_sent)
                sents.append(tmp_sent)
                labels.append(tmp_labels)
                tmp_sent, tmp_labels = [], []
            else:
                token, label = line.split("\t")
                tmp_sent.append(token)
                tmp_labels.append(label)

        assert len(labels) == len(sents)
    return sents, labels


def build_data_from_multiple_files(files: List[str]):
    all_data, all_labels = [], []
    for filepath in files:
        sents, labels = build_data(filepath)
        all_data.extend(sents)
        all_labels.extend(labels)
    return all_data, all_labels


def fetch_unique_labels(labels: List[List[str]]):
    unique_labels, total_labels = set(), []
    for sample in labels:
        for label in sample:
            unique_labels.add(label)
            total_labels.append(label)
    return unique_labels, total_labels
