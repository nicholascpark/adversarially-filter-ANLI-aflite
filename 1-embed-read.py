# pip install transformers
# this loads the embeddings into the format accepted by stage 2

import embed
import yaml
import torch
import pandas as pd

config = {}
with open('./config.yaml') as f:
    config = yaml.load(f)['1-embed']

dataset = config['dataset']
batch_size = config['batch_size']

e = embed.load_embeddings(dataset=dataset, batch=batch_size)
training_x = torch.Tensor(e)

if dataset == "R1":
    inputpath = './data/anli_v1.0/R1/train.jsonl'
    outputpath = "./embedding_files/R1/batch-"
elif dataset == "R2":
    inputpath = './data/anli_v1.0/R2/train.jsonl'
    outputpath = "./embedding_files/R2/batch-"
elif dataset == "R3":
    inputpath = './data/anli_v1.0/R3/train.jsonl'
    outputpath = "./embedding_files/R3/batch-"

train = pd.read_json(path_or_buf=inputpath, lines=True)
ids = {
        "e": 0,
        "n": 1,
        "c": 2,
    }
labels = []
for row in train['label']:
    labels.append(ids[row])

labels = torch.Tensor(labels).unsqueeze(1)
# we attach the labels at the last index for ease of saving
torch.save(torch.cat((training_x, labels), dim=1), config['output'])