from createModel import *
import torch.nn as nn
import torch
import numpy as np

def get_perplex_score(cover_text, model, tokenizer, startingSecret=". "):
    probas = []
    token_secret = tokenizer.encode(cover_text)
    token_start = tokenizer.encode(startingSecret)
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([token_start])
    m = nn.Softmax(dim=0)
  # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')
    pred = []
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        tab = m(predictions[:, -1, :][0])
        pred.append(tab[token_secret[0]].item())
        for i in range(1, len(token_secret)):
            tokens_tensor = torch.cat((tokens_tensor.to('cpu').view(-1), torch.Tensor([token_secret[i]])), dim=-1).view(1, -1)
            outputs = model(tokens_tensor.type(torch.long).to("cuda"))
            predictions = outputs[0]
            tab = m(predictions[:, -1, :][0])
            pred.append(tab[token_secret[i]].item())            
    s = 0
    for p in pred:
        s += np.log2(p)
    score = 2**((-1/len(pred))*s)
    return score