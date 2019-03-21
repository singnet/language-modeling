import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
from pytorch_pretrained_bert.tokenization import load_vocab, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig, BertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import math
import argparse
from tqdm import tqdm, trange
import os
import re
from tqdm import tqdm

base_path = os.path.dirname(os.path.abspath(__file__))


tokenizer = BertTokenizer(vocab_file='{}/data/GCB_vocab.txt'.format(base_path), do_lower_case=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained('checkpoint/')
model.to(device)
model.eval()

vocab = load_vocab('{}/data/GCB_vocab.txt'.format(base_path))
inv_vocab = {v: k for k, v in vocab.items()}


def getMI(tokens, tailnumber, filename):
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    tokens_length = len(tokens)
    result = []
    for i in range(tokens_length) :
        result.append([None]*tokens_length)

    filename = filename.replace(".txt_headless_split_U", "")

    f = open("{}/data/ull/source/{}.ull".format(base_path, filename), "a")
    fu = open("{}/data/ull/direct/{}.ull".format(base_path, filename), "a")
    fs = open("{}/data/ull/merged/{}.ull".format(base_path, filename), "a")

    for i, token1 in enumerate(tokens):
        # tokens preprocessing
        tokens[i] = '[MASK]'
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # processing
        if (len(ids) > 128): 
            ids = ids[0:128]

        ids_mask = [1] * len(ids)
        ids_segm = [0] * len(ids)
        while len(ids) < 128:
            ids.append(0)
            ids_mask.append(0)
            ids_segm.append(0)

        input_ids = torch.tensor([ids], dtype=torch.long)
        input_mask = torch.tensor([ids_mask], dtype=torch.long)
        segment_ids = torch.tensor([ids_segm], dtype=torch.long)

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)

        prediction_scores = model(
            input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # normalization from pt
        scores = torch.nn.functional.softmax(
            prediction_scores.squeeze(), dim=1)
        token1_id = vocab[token1]
        
        result[i] = [scores[i][vocab[token2]].item() for token2 in tokens]

        # tokens postprocessing
        tokens[i] = token1

    for i, token1 in enumerate(tokens):
        # tokens preprocessing
        for j, token2 in enumerate(tokens):
            if i != j :
                if j > i :
                    tokens[i] = '[MASK]'
                    tokens[j] = '[MASK]'
                    ids = tokenizer.convert_tokens_to_ids(tokens)

                    # processing
                    if (len(ids) > 128): 
                        ids = ids[0:128]

                    ids_mask = [1] * len(ids)
                    ids_segm = [0] * len(ids)
                    while len(ids) < 128:
                        ids.append(0)
                        ids_mask.append(0)
                        ids_segm.append(0)

                    input_ids = torch.tensor([ids], dtype=torch.long)
                    input_mask = torch.tensor([ids_mask], dtype=torch.long)
                    segment_ids = torch.tensor([ids_segm], dtype=torch.long)

                    input_ids = input_ids.to(device)
                    segment_ids = segment_ids.to(device)
                    input_mask = input_mask.to(device)

                    prediction_scores = model(
                        input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                    # normalization from pt
                    scores = torch.nn.functional.softmax(
                        prediction_scores.squeeze(), dim=1)
                    token2_id = vocab[token2]
                    token1_id = vocab[token1]

                    result[i][j] = math.log(result[i][j] / scores[i][token2_id].item())
                    result[j][i] = math.log(result[j][i] / scores[j][token1_id].item())

                    # tokens postprocessing
                    tokens[i] = token1
                    tokens[j] = token2
            else:
                result[i][j] = 0
    tokens[0] = "###LEFT-WALL###"
    del tokens[-1]
    f.write(" ".join(tokens))
    fu.write(" ".join(tokens))
    fs.write(" ".join(tokens))

    f.write("\n")
    fu.write("\n")
    fs.write("\n")         
       
    [f.write("{} {} {} {} {}\n".format(i, tid1, j, tid2, result[i][j])) for i, tid1 in enumerate(tokens[0:tokens_length]) for j, tid2 in enumerate(tokens[0:tokens_length]) if i != j]
    [fu.write("{} {} {} {} {}\n".format(i, tid1, j, tid2, result[i][j])) for i, tid1 in enumerate(tokens[0:tokens_length]) for j, tid2 in enumerate(tokens[0:tokens_length]) if i < j]
    [fs.write("{} {} {} {} {}\n".format(i, tid1, j, tid2, result[i][j]+result[j][i])) for i, tid1 in enumerate(tokens[0:tokens_length]) for j, tid2 in enumerate(tokens[0:tokens_length]) if i < j]

    f.write("\n")
    fu.write("\n")
    fs.write("\n")

    f.close()
    fu.close()
    fs.close()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--tailnumber', type=int, default=1, help='Tail number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Workers number')
    parser.add_argument('-f', '--filename', type=str, default="11-0.txt_headless_split_U", help='Target file name')

    args = parser.parse_args()

    lines = [line.rstrip('\n') for line in open('{}/data/capital/{}'.format(base_path,args.filename)) if len(line) > 1]
    lines = [tokenizer.tokenize(line.lower()) for line in lines]
    # lines = lines[0:100]

    t = math.ceil(len(lines)/args.workers)

    for line in lines[(args.tailnumber-1)*t:args.tailnumber*t]:
        getMI(line, args.tailnumber, args.filename) 

    # print(offline.plot([go.Heatmap(z=mi, x=tokens_x, y=tokens)], show_link=True, link_text='Export to plot.ly', validate=True, output_type='file',
    #                     include_plotlyjs=True, filename='pt_norm_base.html', auto_open=False, image=None, image_filename='raw_base', image_width=800, image_height=600))
    # return result, tokens, tokens, miout
