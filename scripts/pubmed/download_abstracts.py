"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n0002.xml.gz"
"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n1163.xml.gz"


def get_url(x):
    x = str(x).zfill(4)
    return f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n{x}.xml.gz"


import subprocess


def download_url(url, download_path):
    download_path = str(download_path)
    cmd = f"wget {url} -O {download_path}".split()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return download_path


if __name__ == "__main__":
    from pathlib import Path

    from tqdm.auto import tqdm

    urls = [get_url(x) for x in range(1, 1167)]

    download_dir = Path("downloads/pubmed_downloaded/")
    output_dir = Path("downloads/pubmed_extracted/")
    download_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    for url in tqdm(urls):
        try:
            download_path = download_dir / Path(url).name
            if not download_path.exists():
                filepath = download_url(url, download_path)
            else:
                print("skipping", url)
        except Exception as e:
            print("unable tp", url)
            print(e)
