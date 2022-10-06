# BioBigBird

BigBird for bio-medical domain

| Description | Link |
|-------------|------|
| Microsoft's PubMedBERT checkpoint is made compatible with BigBird | [🤗](https://huggingface.co/ddp-iitm/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract) |
| Continual learning converted PubMedBERT in bigbird-style (not trained until convergence) | [🤗](https://huggingface.co/ddp-iitm/biobigbird_pubmed_scientific_papers) |
| Downloaded PubMed articles from [official webpage](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/) | [🤗](https://huggingface.co/datasets/ddp-iitm/pubmed-articles) |
| Extracted raw text from bio-books | [🤗](https://huggingface.co/datasets/ddp-iitm/biobooks) |

## Installation

```bash
# clone the repository and run following from the root
pip3 install -e .
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Setting Up

### PubMed Data

```bash
# following takes nearly 2 hrs and would require around 200 GB disk space
python3 scripts/download_pubmed.py

# following takes nearly 6 hrs
python3 scripts/build_pubmed_for_hf.py

# clone HuggingFace repo
git clone https://huggingface.co/datasets/ddp-iitm/pubmed-articles

# shard single jsonl file into multiple small files (takes nearly 1 hr)
python3 shard.py pubmed-articles pubmed.jsonl --num_shards=20

# push dataset to huggingface hub
cd pubmed-articles
git lfs install && git lfs track "*.jsonl"
git add . && git commit -m "add pubmed shards" && git push
```

### Books

```bash
# extract text from books PDF
python3 scripts/extract_text_from_books.py
```
