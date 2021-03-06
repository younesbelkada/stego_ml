## Part for GPT

""" File for the encryption side of things, the sender! """
from .createModel import *
from .evaluationModelHelpers import *
import numpy as np
import torch
#from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig

def getRanks_GPT(mod, tok, secretText="This is too secret for Joe Biden!", startingSecret="Secret: ", finishSentence=True):
    """
    Code to calculate the Ranks given the secret text
    """
    secretTextEnc=[]
    arrayNumber=int(np.ceil(len(secretText)/900))
    start=0
    for n in range(arrayNumber):
      end=secretText.find(" ",(n+1)*900, (n+1)*900+100)
      if (end==-1):
        end=min(len(secretText),(n+1)*1000)
      encode=secretText[start:end]
      secretTextEnc.extend(tok.encode(encode))
      start=end
    totalEnc=tok.encode(startingSecret)
    totalEnc.extend(secretTextEnc)
    ranksSecret=[]
    for i in range(len(tok.encode(startingSecret)), len(totalEnc)):
        start=max(i-1000, 0)
        ranksSecret.append(getTokens_GPT(mod, tok, totalEnc[start:i], totalEnc[i]).item())
    ranksSecret=np.array(ranksSecret)
    if (finishSentence):
        ranksSecret[ranksSecret>2]+=1 #Create offset
        ranksSecret=np.append(ranksSecret, 3)
    return ranksSecret

def completeMessage_GPT(mod, tok, ind, max_length=50):
    """
    Function to complete the final sentence of the text created by GPT2 model
    """
    tokens_tensor = torch.tensor([ind[-1000:]])
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    mod.to('cuda')
    outInd = mod.generate(tokens_tensor, max_length=50)
    outText=tok.decode(outInd[0].tolist())
    newText=outText[len(tok.decode(ind)):]
    newText=newText.split(sep=".", maxsplit=1)[0]
    newText="".join((newText, "."))
    outInd=ind+tok.encode(newText)
    outText=tok.decode(outInd)
    return outText, outInd

def encryptMessage_GPT(mod, tok, secretText="This is too secret for Joe Biden!", startingSecret="Secret: ", startingText="This year's Shakespeare Festival", finishSentence=True):
    """
    Encryption function for GPT2 Model
    """
    ranks=getRanks_GPT(mod, tok, secretText, startingSecret, finishSentence)
    outInd=tok.encode(startingText)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId_GPT(mod, outInd, ranks[i]) 
    if (finishSentence):    
        outText, outInd=completeMessage_GPT(mod, tok, outInd, max_length=50)
    else:
        outText=tok.decode(outInd) 
    return outText, outInd

## Part for Bert

def completeMessage_BERT(mod, tok, ind, max_length=50):
  """
  Sentence Completion of the secret text from BERT
  """ 
  tokens_tensor = torch.tensor([ind])
  outInd = mod.generate(tokens_tensor, max_length=50)
  outText=tok.decode(outInd[0].tolist())
  newText=outText[len(tok.decode(ind)):]
  newText=newText.split(sep=".", maxsplit=1)[0]
  newText="".join((newText, "."))
  outInd=ind+tok.encode(newText)
  return outInd

def getRanks_BERT(mod, tok, precondSec, secret, completeMessage):
  """
  Function to calculate the ranks of the secret text
  """
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
  if (completeMessage):
    ranks[ranks>2]+=1
    ranks=np.append(ranks, 3)
  return ranks

def generateCoverText_BERT(mod, tok, startOfText, ranks, completeMessage):
  """
  Function to get the cover text that is sent from Alice to Bob based on the ranks of the secret text
  """
  inputs = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)
  inputs=inputs.tolist()[0]
  if (completeMessage):
    inputs=completeMessage_BERT(mod, tok, inputs)
  cover_text = tok.decode(inputs)
  return cover_text, inputs

def encryptMessage_BERT(mod, tok, secret, precondSec, startOfText, completeMessage=True):
  """
  Function to encrypt the message using BERT
  """
  ranks = getRanks_BERT(mod, tok, precondSec, secret, completeMessage)
  cover_text, ind = generateCoverText_BERT(mod, tok, startOfText, ranks, completeMessage)
  return cover_text, ind



## Part for RoBERTa
def getRanks_RoBERTa(mod, tok, precondSec, secret, completeMessage=True):
  """
  Function to calculate the Ranks of the secret text for RoBERTa language model
  """
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
  if (completeMessage):
    ranks[ranks>2]+=1
    ranks=np.append(ranks, 3)
  return ranks

def completeMessage_RoBERTa(mod, tok, ind, max_length=50):
  """
  Function to complete the message based on RoBERTa LM
  """
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

def generateCoverText_RoBERTa(mod, tok, startOfText, ranks, completeMessage=True):
  """
  Function to generate the cover text given the ranks of the secret text to be hidden 
  """
  inputs = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)
  inputs=inputs.tolist()[0]
  if (completeMessage):
    inputs=completeMessage_RoBERTa(mod, tok, inputs)
  cover_text = tok.decode(inputs)
  return cover_text, inputs

def encryptMessage_RoBERTa(mod, tok, secret, precondSec, startOfText, completeMessage=True):
  """
  Function to encrypt the message using RoBERTa
  """
  ranks = getRanks_RoBERTa(mod, tok, precondSec, secret, completeMessage)
  cover_text, indices = generateCoverText_RoBERTa(mod, tok, startOfText, ranks, completeMessage)
  return cover_text, indices



## Overall together
def encryptMessage(mod, tok, secret, precondSec, startOfText, completeMessage=True):
  """
  Wrapper function to choose the correct model to encrypt with. 
  """
  
  modelType=getModelType(mod)
  ind="Null"
  enryptionProblem=True
  if (modelType=="gpt2"):
      text, ind=encryptMessage_GPT(mod, tok, secret, precondSec, startOfText, finishSentence=completeMessage)
      encryptionProblem= ~np.all(tok.encode(text)== ind)
  elif (modelType=="bert"):
      text, ind=encryptMessage_BERT(mod, tok, secret, precondSec, startOfText, completeMessage=completeMessage)
      encryptionProblem= ~np.all(tok.encode(text)[1:-1]== ind)
  elif (modelType=="roBERTa"):
      text, ind=encryptMessage_RoBERTa(mod, tok, secret, precondSec, startOfText, completeMessage=completeMessage)
      encryptionProblem= ~np.all(tok.encode(text)[1:-1]== ind)
  else:
      print("ERRROR")
  if not (np.all(tok.encode(text)== ind)):
    print("WARNING: They may be some issues during the decryption, you may want to change the preconditioning of the cover text to have an accurate result")
  return text, ind
