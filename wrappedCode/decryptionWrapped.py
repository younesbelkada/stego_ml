import numpy as np
import torch
from .evaluationModelHelpers import *
from .createModel import getModelType
from .encryptionWrapped import *
## Part for GPT
def recoverSecretRanks_GPT(mod_rec, tok_rec, startingText, outInd, finishSentence=True):
    """
    Function to calculate the secret ranks of GPT2 LM of a cover text given the cover text
    """
    startingInd=tok_rec.encode(startingText)
    endingInd=outInd[len(startingInd):]
    secretTokensRec=[]
    for i in range(len(endingInd)):
      token=getTokens_GPT(mod_rec, tok_rec, startingInd, endingInd[i])
      if (finishSentence):
        if (token==3):
          break
        if(token>2):
          token-=1
      startingInd.append(endingInd[i])
      secretTokensRec.append(token[0].tolist())
    return secretTokensRec



def getTextFromText_GPT(mod, tok, publicText, startingSecret="Secret: ", startingText="This year's Shakespeare Festival", finishSentence=True):
    """
    Function to recover the secret text given the cover text with GPT2 LM
    """
    indizes=tok.encode(publicText)
    ranks=recoverSecretRanks_GPT(mod, tok, startingText, indizes, finishSentence)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId_GPT(mod, outInd, ranks[i]) 
    outText=tok.decode(outInd)    
    return outText
    
def getTextFromInd_GPT(mod, tok, publicInd, startingSecret="Secret: ", startingText="This year's Shakespeare Festival", finishSentence=True):
    """
    Function to recover the secret text given the tokens of the cover text with GPT2 LM
    """
    indizes=publicInd
    ranks=recoverSecretRanks_GPT(mod, tok, startingText, indizes, finishSentence)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId_GPT(mod, outInd, ranks[i]) 
    outText=tok.decode(outInd)    
    return outText


## Part for RoBERTa/ BERT
def getRanksFromCover_RoBERTa(mod, tok, cover_text, startOfText, completeMessage=True):
  """
  Function to calculate the ranks of the cover text for RoBERTa and BERT LM
  """
  inputs_cover = tok.encode(cover_text, return_tensors="pt", add_special_tokens=False)

  input_start = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  ranks = []
  for s in range(0, len(inputs_cover[0])-len(input_start[0])):
    tab = input_start.numpy()
    pred = mod(input_start)[0]
    index = torch.where(torch.argsort(pred[0, -1, :], descending=True) == inputs_cover[0][len(input_start[0]+s)])[0].item()
    if (completeMessage):
      if (index==3):
          break
      if(index>2):
          index-=1
    ranks.append(index)
    tab = [np.append(tab[0],inputs_cover[0][len(input_start[0]+s)])]
    input_start = torch.Tensor(tab).type(torch.long)
  
  return ranks

def getSecretFromRanks_RoBERTa(mod, tok, precondSec, ranks):
  """
  Function to calculate the secret given the ranks of the cover text
  """
  inputs = tok.encode(precondSec, return_tensors="pt", add_special_tokens=False)
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)

  retrived_text = tok.decode(inputs[0])
  return retrived_text

def decryptRoBERTa (mod, tok, cover_text, precondSec, startOfText, completeMessage):
  """
  function to decrypt a cover text given RoBERTa or BERT LM was used 
  """
  ranks=getRanksFromCover_RoBERTa(mod, tok, cover_text, startOfText, completeMessage)
  secretText=getSecretFromRanks_RoBERTa(mod, tok, precondSec, ranks)
  return secretText


def decryptMessage(mod, tok, coverText, precondSec, startOfText, completeMessage=True):
    """
    Function to decrypt a message 
    """
    modelType=getModelType(mod)
    if (modelType=="gpt2"):
        text = getTextFromText_GPT(mod, tok, coverText, precondSec, startOfText, finishSentence=completeMessage)
    elif (modelType=="bert"):
        text=decryptRoBERTa(mod, tok, coverText, precondSec, startOfText, completeMessage)
    elif (modelType=="roBERTa"):
        text=decryptRoBERTa(mod, tok, coverText, precondSec, startOfText, completeMessage)
    else:
        print("ERRROR")
        return 0
    text = text[len(precondSec):]
    return text