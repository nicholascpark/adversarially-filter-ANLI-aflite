from os import remove
import torch
import yaml
import pandas as pd

config = {}
with open('config.yaml') as f:
    config = yaml.load(f)['3-train']

input_file = config['input']
filter = torch.load(input_file)
print(filter.shape, filter)

dataset = config['dataset']
if dataset == "R1":
    inputpath = './data/anli_v1.0/R1/train.jsonl'
    outputpath = "./embedding_files/R1/batch-"
elif dataset == "R2":
    inputpath = './data/anli_v1.0/R2/train.jsonl'
    outputpath = "./embedding_files/R2/batch-"
elif dataset == "R3":
    inputpath = './data/anli_v1.0/R3/train.jsonl'
    outputpath = "./embedding_files/R3/batch-"

result = []
removed = []
with open(inputpath) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if filter[i]:
            result.append(line)
        else:
            removed.append(line)
print(len(result))
with open(inputpath + '_filtered.jsonl', 'w+') as f:
    f.writelines(result)

with open(inputpath + '_removed.jsonl', 'w+') as f:
    f.writelines(removed)