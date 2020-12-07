import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig

def get_ranks_from_cover(mod, tok, cover_text, startOfText):
  inputs_cover = tok.encode(cover_text, return_tensors="pt", add_special_tokens=False)

  input_start = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  ranks = []

  for s in range(0, len(inputs_cover[0])-len(input_start[0])):
    tab = input_start.numpy()
    pred = mod(input_start)[0]
    #print(len(inputs_cover[0])-len(input_start[0]))
    #print(inputs_cover[0][len(input_start[0]):])
    #break
    index = torch.where(torch.argsort(pred[0, -1, :], descending=True) == inputs_cover[0][len(input_start[0]+s)])[0].item()
    if (index==3):
        break
    if(index>2):
        index-=1
    ranks.append(index)
    tab = [np.append(tab[0],inputs_cover[0][len(input_start[0]+s)])]
    input_start = torch.Tensor(tab).type(torch.long)
  #print("ranks after:",ranks)
  
  return ranks

def get_secret_from_ranks(mod, tok, precondSec, ranks):
  inputs = tok.encode(precondSec, return_tensors="pt", add_special_tokens=False)
  #print(inputs)
  #secret_token = tok(secret, return_tensors="pt")['input_ids'][0]
  #ranks = []
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)
    #print(inputs)

  retrived_text = tok.decode(inputs[0])
  return retrived_text

def decryptRoBERTa (mod, tok, cover_text, precondSec, startOfText):
  ranks=get_ranks_from_cover(mod, tok, cover_text, startOfText)
  secretText=get_secret_from_ranks(mod, tok, precondSec, ranks)
  return secretText