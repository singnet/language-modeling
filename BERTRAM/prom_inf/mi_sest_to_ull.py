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
import pickle

base_path = os.path.dirname(os.path.abspath(__file__))

tokenizer = BertTokenizer(vocab_file='{}/data/vocab.txt'.format(base_path), do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)
model.eval() 

def getSentenceEmbedding(tokens, model) :
    ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(ids) > 128 :
        ids = ids[0:128] 

    ids_mask = [1] * len(ids)
    ids_segm = [0] * len(ids)
   
    if len(ids) < 128:
        ids += [0]*(128-len(ids))
        ids_mask += [0]*(128-len(ids_mask))
        ids_segm += [0]*(128-len(ids_segm))

    input_ids = torch.tensor([ids], dtype=torch.long)
    input_mask = torch.tensor([ids_mask], dtype=torch.long)
    segment_ids = torch.tensor([ids_segm], dtype=torch.long)

    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    input_mask = input_mask.to(device)

    # get output from layer norm, before classifier
    global out
    def lm_LayerNorm(module, input_, output):
        global out
        out = output
    
    model.cls.predictions.transform.LayerNorm.register_forward_hook(lm_LayerNorm)
    model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
 
    return out.squeeze().mean(0).cpu().detach().numpy()

def getFMI(query, sentences_all, embs_all, pos1, pos2, sid) :
    embs_dist = []
    idx = 0
    for sentence in sentences_all:
        score = np.dot(embs_all[sid], embs_all[idx]) / np.linalg.norm(embs_all[sid]) / np.linalg.norm(embs_all[idx])
        embs_dist += [score]
        idx += 1

    X = query[pos1]
    Y = query[pos2]
    Nss = NXs = NsY = NXY = 0
    L = len(sentences_all)
    # REM: Nss can be calculated once per sentence if we want to calculate MI for each pair of words
    # NXs, NsY, NXX can then be calculated using corresponding subsets of sentences
    for idx in range(L):
        score = embs_dist[idx]
        sentence = sentences_all[idx]
        Nss += score
        if X in sentence:
            #print(sentence)
            NXs += score
        if Y in sentence:
            NsY += score
        if (X in sentence) and (Y in sentence):
            if sentence.index(X) < sentence.index(Y):
                NXY += score
    try:
        FMIe = math.log(NXY * Nss / (NXs * NsY)) if NXY > 0 else -1000
        return FMIe
    except Exception as e: 
        print(e)
        print("idx: {} Nss: {} NXY: {} NXs: {}\n".format(idx, Nss, NXY, NXs))
        print("Sent: {}\n".format(query))
        return -500

def writeMI(args):
    f = open('{}/data/ull/SESTv2/{}'.format(base_path, args.filename), "a")

    lines = [line.rstrip('\n') for line in open('{}/data/subsets/{}'.format(base_path, args.filename)) if len(line) > 1]
    lines = [tokenizer.tokenize(line.lower()) for line in lines]

    lines_ext = lines[:]
    sentences_all = []
    for line in lines_ext:
        sentence = line[:]
        sentence.insert(0,"[CLS]")
        sentence += ["[SEP]"]
        sentences_all += [sentence]
      
    embs_all = []
    for sentence in sentences_all:
        embs_all.append(getSentenceEmbedding(sentence, model))
    
    lines = [line.rstrip('\n') for line in open('{}/data/subsets/{}'.format(base_path, args.filename)) if len(line) > 1]
    lines = [line.lower().split() for line in lines]

    lines_ext = lines[:]
    sentences_all = []
    for line in lines_ext:
        sentence = line[:]
        sentence.insert(0,"[CLS]")
        sentence += ["[SEP]"]
        sentences_all += [sentence]

    for sid, sentence in enumerate(sentences_all):
        del sentence[-1]
        sent = sentence[:]
        sent[0] = "###LEFT-WALL###" 
        f.write(" ".join(sent))
        f.write("\n")
        for i in range(len(sentence)) :
            for j in range(len(sentence)) :
                if j > i :
                    FMI = getFMI(sentence, sentences_all, embs_all, i, j, sid)

                    f.write("{} {} {} {} {}\n".format(i, sent[i], j, sent[j], FMI))
    
        f.write("\n")

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--tailnumber', type=int, default=1, help='Tail number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Workers number')
    parser.add_argument('-f', '--filename', type=str, default="11-0.txt_headless_split_U.ull", help='Target file name')

    args = parser.parse_args()

    writeMI(args)