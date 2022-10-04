# python3 src/download_pubmed.py

# https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/
oa_comm_urls = [
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC000xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC001xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC002xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC003xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC004xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC005xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC006xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC007xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC008xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.PMC009xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-09.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-10.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-11.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-12.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-13.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-14.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-15.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-16.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-17.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-18.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-19.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-20.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-21.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-22.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-23.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-24.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-25.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-26.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-27.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-28.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-29.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-09-30.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-10-01.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-10-02.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-10-03.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/oa_comm_txt.incr.2022-10-04.tar.gz",
]


# https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/
oa_noncomm_urls = [
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC001xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC002xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC003xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC004xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC005xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC006xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC007xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC008xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.PMC009xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-09.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-10.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-11.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-12.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-13.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-14.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-15.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-16.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-17.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-18.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-19.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-20.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-21.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-22.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-23.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-24.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-25.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-26.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-27.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-28.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-29.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-09-30.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-10-01.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-10-02.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-10-03.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/oa_noncomm_txt.incr.2022-10-04.tar.gz",
]


# https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/
oa_other_urls = [
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC000xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC001xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC002xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC003xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC004xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC005xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC006xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC007xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC008xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.PMC009xxxxxx.baseline.2022-09-08.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-09.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-10.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-11.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-12.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-13.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-14.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-15.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-16.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-17.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-18.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-19.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-20.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-21.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-22.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-23.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-24.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-25.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-26.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-27.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-28.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-29.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-09-30.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-10-01.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-10-02.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-10-03.tar.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/txt/oa_other_txt.incr.2022-10-04.tar.gz",
]


import subprocess


def download_url(url, download_path):
    download_path = str(download_path)
    cmd = f"wget {url} -O {download_path}".split()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return download_path


def unzip_file(filepath, output_dir):
    filepath, output_dir = str(filepath), str(output_dir)
    cmd = f"tar -xf {filepath} -C {output_dir}".split()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    from pathlib import Path

    from tqdm.auto import tqdm

    urls = oa_comm_urls + oa_noncomm_urls + oa_other_urls

    download_dir = Path("downloads/pubmed_downloaded/")
    output_dir = Path("downloads/pubmed_extracted/")
    download_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    for url in tqdm(urls):
        filepath = download_url(url, download_dir / Path(url).name)
        unzip_file(filepath, output_dir)
