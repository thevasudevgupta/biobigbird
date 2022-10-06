import json
from pathlib import Path

import fire
import jsonlines
from tqdm.auto import tqdm

# python3 scripts/shard.py dummy /outputs/books.jsonl


def shard(target_dir: str, filepath: str, num_shards: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=False)

    extension = Path(filepath).suffix
    assert extension in [".json", ".jsonl"]
    shard_name = lambda i: str(target_dir / f"shard_{i}{extension}")
    files = [jsonlines.open(shard_name(i), "a") for i in range(num_shards)]

    with open(filepath) as f:
        num_lines = sum(1 for _ in tqdm(f, desc="counting number of lines"))
    num_lines_per_shard = num_lines // num_shards

    print("num_shard:", num_shards)
    print("num_lines:", num_lines)
    print("num_lines_per_shard:", num_lines_per_shard)

    last_shard_writer = None
    shard_no = 0
    with open(filepath) as f:
        pbar = tqdm(enumerate(f), desc=shard_name(shard_no), total=num_lines)
        for i, line in pbar:
            line = json.loads(line)

            if shard_no < len(files):
                files[shard_no].write(line)

            if (i + 1) % num_lines_per_shard == 0:
                files[shard_no].close()
                shard_no += 1
                pbar.set_description(shard_name(shard_no))

            if shard_no == len(files) and last_shard_writer is None:
                last_shard_writer = jsonlines.open(shard_name(shard_no - 1), "a")
            elif shard_no == len(files):
                last_shard_writer.write(line)

    if last_shard_writer is not None:
        last_shard_writer.close()


if __name__ == "__main__":
    fire.Fire(shard)
