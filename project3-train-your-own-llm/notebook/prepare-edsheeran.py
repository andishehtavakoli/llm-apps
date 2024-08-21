import os
import tiktoken
import numpy as np
import pandas as pd



df = pd.read_csv("/Users/andishehtavakoli/Documents/github-project/llm-apps/project3-train-your-own-llm/notebook/ed_sheeran.csv")
data = df["Lyrics"].str.cat(sep="\n")

n = len(data)

train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train has 433,585 tokens
# val has 48,662 tokens



