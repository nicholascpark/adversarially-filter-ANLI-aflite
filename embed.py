from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import pandas as pd
import time
import numpy as np
import math

def save_embeddings(dataset, batch = 100):

    # input:
    #       dataset = "R1" or "R2" or "R3"
    #       batch = recommend no bigger than 300 at a time - crashes o.w.
    # output: None, saves the output embeddings in the ./embedding_files/Rx directory

    start = time.time()

    assert dataset == "R1" or dataset == "R2" or dataset == "R3", "only R1 or R2 or R3 allowed"

    if dataset == "R1":
        inputpath = './data/anli_v1.0/R1/train.jsonl'
        outputpath = "./embedding_files/R1/batch-"
    elif dataset == "R2":
        inputpath = './data/anli_v1.0/R2/train.jsonl'
        outputpath = "./embedding_files/R2/batch-"
    elif dataset == "R3":
        inputpath = './data/anli_v1.0/R3/train.jsonl'
        outputpath = "./embedding_files/R3/batch-"

    print("Using gpu?", torch.cuda.is_available())

    max_length = 256

    train = pd.read_json(path_or_buf=inputpath, lines=True)
    context = train["context"]
    hypothesis = train["hypothesis"]

    hg_model_hub_name = "roberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    config = AutoConfig.from_pretrained(hg_model_hub_name, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name, config = config)

    # print(model.config.output_hidden_states)
    m = math.ceil(len(train.index)/batch)
    print("Number of batches of embeddings to produce for this dataset:", m)

    for i in np.arange(m):
        embedding = []
        for j in np.arange(i*batch, min(i*batch + batch, len(train.index))):
            tokenized_input_seq_pair = tokenizer.encode_plus(context[j], hypothesis[j],
                                                         max_length=max_length,
                                                         return_token_type_ids=True, truncation=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

            outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
            # print((outputs[1][-1][:,0,:].shape))
            hidden_states = outputs[1][-1]
            # last_layer_state = hidden_states[-1]
            pooled_hidden_state = model.roberta.pooler.forward(hidden_states)
            # print(pooled_hidden_state.shape)
            pooled_hidden_state = pooled_hidden_state[0] # remove outer brackets (1, 1024) into (1024,)
            # print(pooled_hidden_state.shape)
            embedding.append(pooled_hidden_state.detach().numpy())
            # print(pooled_last_layer)
        np.save(outputpath+str(i), embedding) ## error!! dimension doesn't match due to varying seq_len
        print("Saved batch", i, "of size", len(embedding), "in the ./embedding_files directory. So far", min(i * batch + batch, len(train.index)), "examples saved...")
        # print(embedding)

    end = time.time()
    print("Total embedding time for the dataset:", end - start, "seconds")

def load_embeddings(dataset, batch = 100):

    # to be ran with all the batch-i.npy files populated in the embeddingfiles/Rx/ directory.
    # input:
    #       dataset: "R1" or "R2" or "R3"
    #                 if R1 there should be 170 .npy files, i.e., 0 ~ 169 inclusive
    #                 if R2 there should be 455 .npy files
    #                 if R3 there should be 1005 .npy files
    #       batch: the number of batch size used for save embeddings, using 100 for this project by def.
    # output: returns a list of 1d np.arrays

    assert dataset == "R1" or "R2" or "R3", "only R1 or R2 or R3 allowed"

    if dataset == "R1":
        samplesize = 16946
        loadpath = "./embedding_files/R1/batch-"
    elif dataset == "R2":
        samplesize = 45460
        loadpath = "./embedding_files/R2/batch-"
    elif dataset == "R3":
        samplesize = 100459
        loadpath = "./embedding_files/R3/batch-"

    m = math.ceil(samplesize/batch)

    embeddings = []

    print("loading ", m, " batches of embedding into a single list")

    for i in np.arange(m):
        batch_embeddings = np.load(loadpath+str(i)+".npy")
        embeddings.append(batch_embeddings)
        print("Batch", i, "loaded..")
        # print(len(batch_embeddings))

    embeddings = [item for batch_embeddings in embeddings for item in batch_embeddings]
    # Converts 3d list into 2d list in order
    print("...")
    print("Total", len(embeddings), "embeddings loaded.")

    return embeddings


if __name__ == '__main__':

    save_embeddings("R1")
    save_embeddings("R2")
    save_embeddings("R3")
    # x = load_embeddings("R1")
    # print(x)
    # print(len(x))
    # print(len(x[0]))


