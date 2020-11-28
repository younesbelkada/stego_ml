import torch
import numpy as np
from transformers import BertTokenizer, BertLMHeadModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
#logging.basicConfig(level=logging.INFO)

def buildModelBERT():    
  """ 
  This function builds the model of the function und returns it based on BERT
  """
  ## Create Model
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  # Load pre-trained model (weights)
  model = BertLMHeadModel.from_pretrained('bert-base-uncased', return_dict=True)

  # Set the model in evaluation mode to deactivate the DropOut modules
  # This is IMPORTANT to have reproducible results during evaluation!
  model.eval()
  return model, tokenizer


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
  #print("ranks before:", ranks)
  return ranks

def generate_cover_text(mod, tok, startOfText, ranks):
  inputs = tok.encode(startOfText, return_tensors="pt", add_special_tokens=False)
  for s in ranks:
    tab = inputs.numpy()
    pred = mod(inputs)[0]
    index = torch.argsort(pred[0, -1, :], descending=True)[s]
    tab = [np.append(tab[0],index)]
    inputs = torch.Tensor(tab).type(torch.long)
  cover_text = tok.decode(inputs[0])
  return cover_text

## Decoding

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


startOfText="Hello world, today we are going to present you a tutorial about how to cook a turkey. Please follow this tutorial carefully, "
precondSec=". "
secret="this message is super secret please do not share it"
mod, tok = buildModelBERT()


## Enconding

ranks = get_ranks(mod, tok, precondSec, secret)
cover_text = generate_cover_text(mod, tok, startOfText, ranks)

print("ranks before :", ranks)
print("cover_text: ", cover_text)

print()
## Decoding

ranks_after = get_ranks_from_cover(mod, tok, cover_text, startOfText)
retrived_secret = get_secret_from_ranks(mod, tok, precondSec, ranks_after)
print('ranks after:', ranks_after)
print(retrived_secret)