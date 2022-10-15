# BioBigBird

BigBird for bio-medical domain

| Description | Link |
|-------------|------|
| Microsoft's PubMedBERT checkpoint is made compatible with BigBird | [`HuggingFace`](https://huggingface.co/ddp-iitm/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract) |
| Continual learning converted PubMedBERT in bigbird-style (not trained until convergence) | [`HuggingFace`](https://huggingface.co/ddp-iitm/biobigbird_pubmed_scientific_papers) |
| Downloaded PubMed articles from [official webpage](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/) | [`HuggingFace`](https://huggingface.co/datasets/ddp-iitm/pubmed-articles) |
| Extracted raw text from bio-books | [`HuggingFace`](https://huggingface.co/datasets/ddp-iitm/biobooks) |

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

# following will load pubmed in huggingface format and will push data to hub
python3 scripts/build_pubmed_for_hf.py
# it takes nearly 6 hrs
```

### Books

```bash
export HF_TOKEN='<something>'

# we have stored biobooks in huggingface hub
wget https://vasudevgupta:{$HF_TOKEN}@huggingface.co/datasets/ddp-iitm/biobooks-pdf/resolve/main/books.zip
unzip books.zip

# extract text from books PDF (takes nearly 1 hr)
python3 scripts/extract_text_from_books.py
```
