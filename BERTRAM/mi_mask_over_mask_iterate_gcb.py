import asyncio
import websockets
import json

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

def getMI(sentence, event):
    tokens = tokenizer.tokenize(sentence)
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    tokens_length = len(tokens)
    result = []
    for i in range(tokens_length) :
        result.append([None]*tokens_length)

    for i, token1 in enumerate(tokens):
        # tokens preprocessing
        for j, token2 in enumerate(tokens):
            if i != j :
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
                token2_id = vocab[token2]
                token1_id = vocab[token1]

                result[i][j] = scores[i][token2_id].item()

                # tokens postprocessing
                tokens[i] = token1
            else:
                result[i][j] = 0

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

                    if event == 'momi_direct' :
                        result[j][i] = 0
                    
                    if event == 'momi_source' :
                        pass

                    if event == 'momi_merged' :
                        result[i][j] += result[j][i]
                        result[j][i] = 0
                    
                    if event == 'momi_max' :
                        result[i][j] = max(result[i][j], result[j][i])
                        result[j][i] = 0
                    
                    if event == 'momi_mod' :
                        result[i][j] = math.fabs(result[i][j]) + math.fabs(result[j][i])
                        result[j][i] = 0
                    
                    if event == 'momi_modmax' :
                        result[i][j] = max(math.fabs(result[i][j]), math.fabs(result[j][i]))
                        result[j][i] = 0

                    # tokens postprocessing
                    tokens[i] = token1
                    tokens[j] = token2
            else:
                result[i][j] = 0
    tokens[0] = "###LEFT-WALL###"
        
    miout = []
    if event == 'momi_source' :
        [miout.append("{} {} {}\n".format(tid1, tid2, result[i][j])) for i, tid1 in enumerate(tokens[0:tokens_length]) for j, tid2 in enumerate(tokens[0:tokens_length]) if i != j]
    else :
        [miout.append("{} {} {}\n".format(tid1, tid2, result[i][j])) for i, tid1 in enumerate(tokens[0:tokens_length]) for j, tid2 in enumerate(tokens[0:tokens_length]) if i < j]

    
    return result, tokens, tokens, miout  

if __name__ == "__main__":
    async def ws(websocket, path):
        async for data in websocket:
            data = json.loads(data)
            if data['event'] :
                try:
                    mi, tokens, tokens_x, miout = getMI(data['sentence'], data['event'])
                    response = json.dumps({'event': 'success', 'mi': mi, 'tokens': tokens,
                                          'tokens_x': tokens, 'miout': miout, 'label_x': 'word1', 'label_y': 'word2'})
                    await websocket.send(response)
                except KeyError as e:
                    print("Error: {}".format(e))
                    response = json.dumps(
                        {'event': 'error', 'msg': 'Running error!\n{}'.format(e)})
                    await websocket.send(response)

    print('WS Server started.\n')
    asyncio.get_event_loop().run_until_complete(
        websockets.serve(ws, '0.0.0.0', 8167))
    asyncio.get_event_loop().run_forever()

