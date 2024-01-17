# pip install transformers
# this loads the embeddings into the format accepted by stage 2

import embed
import yaml
import torch
import pandas as pd

config = {}
with open('./config.yaml') as f:
    config = yaml.load(f)['3-dedup']

dataset = config['dataset']
if dataset == "R1":
    inputpath = './data/anli_v1.0/R1/train.jsonl'
elif dataset == "R2":
    inputpath = './data/anli_v1.0/R2/train.jsonl'
elif dataset == "R3":
    inputpath = './data/anli_v1.0/R3/train.jsonl'

train = pd.read_json(path_or_buf=inputpath, lines=True)
print(train.shape)
# ignore uid, reason, tag, genre, emturk, model_label
train = train.drop_duplicates(subset=['context', 'hypothesis', 'label'])
print('filtered', train.shape)
with open(inputpath+'_dedup_filtered.jsonl', 'w+') as f:
    f.write(train.to_json(orient='records', lines=True))