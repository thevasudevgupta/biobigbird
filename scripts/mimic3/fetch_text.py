# python3 scripts/mimic3/fetch_text.py

import pandas as pd
from tqdm.auto import tqdm

data = pd.read_csv("NOTEEVENTS.csv", dtype=str)
data = data["TEXT"].tolist()

data = [" ".join(l.split()) for l in tqdm(data)]
print(data[0])

with open("data.txt", "w") as f:
    f.write("\n".join(data))

print("data saved in data.txt")
