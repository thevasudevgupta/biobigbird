# BioBigBird

BigBird for bio-medical domain

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

# following takes nearlly 6 hrs
python3 scripts/build_pubmed_for_hf.py
```
