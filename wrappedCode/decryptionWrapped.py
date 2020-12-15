import numpy as np
import torch
from evaluationModelHelpers import *
## Part for GPT
def recoverSecretRanks_GPT(mod_rec, tok_rec, startingText, outInd, finishSentence=True):
    #mod_rec, tok_rec=buildModelGPT()
    startingInd=tok_rec.encode(startingText)
    endingInd=outInd[len(startingInd):]
    #secretTokensRec=np.zeros(len(endingInd), dtype=int)
    secretTokensRec=[]
    for i in range(len(endingInd)):
      token=getSecretTokens_GPT(mod_rec, tok_rec, startingInd, endingInd[i])
      if (finishSentence):
        if (token==3):
          break
        if(token>2):
          token-=1
      startingInd.append(endingInd[i])
      secretTokensRec.append(token[0].tolist())
    return secretTokensRec

def getTextFromText_GPT(mod, tok, publicText, startingSecret="Secret: ", startingText="This year's Shakespeare Festival", finishSentence=True):
    #mod, tok=buildModelGPT()
    #ranks=getSecretRanks(publicText, startingText)
    indizes=tok.encode(publicText)
    ranks=recoverSecretRanks_GPT(mod, tok, startingText, indizes, finishSentence)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId_GPT(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd)    
    return outText
    
def getTextFromInd_GPT(mod, tok, publicInd, startingSecret="Secret: ", startingText="This year's Shakespeare Festival", finishSentence=True):
    #mod, tok=buildModelGPT()
    #ranks=getSecretRanks(publicText, startingText)
    indizes=publicInd
    ranks=recoverSecretRanks_GPT(mod, tok, startingText, indizes, finishSentence)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId_GPT(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd)    
    return outText

## Part BERT


## Part for RoBERTa
def getRanksFromCover_RoBERTa(mod, tok, cover_text, startOfText, completeMessage=True):
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
    if (completeMessage):
      if (index==3):
          break
      if(index>2):
          index-=1
    ranks.append(index)
    tab = [np.append(tab[0],inputs_cover[0][len(input_start[0]+s)])]
    input_start = torch.Tensor(tab).type(torch.long)
  #print("ranks after:",ranks)
  
  return ranks

def getSecretFromRanks_RoBERTa(mod, tok, precondSec, ranks):
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
  ranks=getRanksFromCover_RoBERTa(mod, tok, cover_text, startOfText)
  secretText=getSecretFromRanks_RoBERTa(mod, tok, precondSec, ranks)
  return secretText