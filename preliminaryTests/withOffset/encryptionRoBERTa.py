import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig

def get_ranks(mod, tok, precondSec, secret):
  ## Encoding
  inputs = tok.encode(precondSec, return_tensors="pt", add_special_tokens=False)
  secret_token = tok.encode(secret, return_tensors="pt", add_special_tokens=False)[0]
  ranks = []
  for s in secret_token:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.where(torch.argsort(pred[0, -1, :], descending=True) == s)[0].item()
    ranks.append(index)
    tab = [np.append(tab[0],s)]
    inputs = torch.Tensor(tab).type(torch.long)
  ranks=np.array(ranks)
  ranks[ranks>2]+=1
  ranks=np.append(ranks, 3)
  #print("ranks before:", ranks)
  return ranks

def completeMessage(mod, tok, ind, max_length=50):
  tokens_tensor = torch.tensor([ind])
  outInd = mod.generate(tokens_tensor, max_length=50)
  outText=tok.decode(outInd[0].tolist())
  newText=outText[len(tok.decode(ind)):]
  newText=newText.split(sep=".", maxsplit=1)[0]
  newText="".join((newText, "."))
  outInd=ind+tok.encode(newText)
  #outText=tok.decode(outInd)
  #return outText, outInd
  return outInd

def generate_cover_text(mod, tok, startOfText, ranks):
  inputs = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)
  inputs=completeMessage(mod, tok, inputs.tolist()[0])
  cover_text = tok.decode(inputs)
  return cover_text

def encryptMessageRoBERTa(mod, tok, secret, precondSec, startOfText):
  ranks = get_ranks(mod, tok, precondSec, secret)
  cover_text = generate_cover_text(mod, tok, startOfText, ranks)
  return cover_text
