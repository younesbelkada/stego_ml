## Important Imports:   A lot from https://huggingface.co/transformers/quickstart.html
#!pip install -q git+https://github.com/huggingface/transformers.git
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaTokenizer, RobertaForCausalLM, RobertaConfig
torch.manual_seed(0)
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
#logging.basicConfig(level=logging.INFO)

def buildModelGPT(modelType='gpt2-medium'):    
  """ 
  This function builds the model of the function und returns it based on GPT
  """
  ## Create Model
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = GPT2Tokenizer.from_pretrained(modelType)
  # Load pre-trained model (weights)
  model = GPT2LMHeadModel.from_pretrained(modelType)

  # Set the model in evaluation mode to deactivate the DropOut modules
  # This is IMPORTANT to have reproducible results during evaluation!
  model.eval()
  return model, tokenizer

def buildModelRoBERTa():    
  """ 
  This function builds the model of the function und returns it based on RoBERTa
  """
  ## Create Model
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  config = RobertaConfig.from_pretrained("roberta-base") 
  config.is_decoder = True
  model = RobertaForCausalLM.from_pretrained('roberta-base', config=config)

  # Set the model in evaluation mode to deactivate the DropOut modules
  # This is IMPORTANT to have reproducible results during evaluation!
  model.eval()
  return model, tokenizer

def evaluateWithInputText(model=buildModelGPT()[0], tokenizer=buildModelGPT()[1], text="We need to start working harder.", nLikely=0):
  indexed_tokens = tokenizer.encode(text)

  # Convert indexed tokens in a PyTorch tensor
  tokens_tensor = torch.tensor([indexed_tokens[-1000:]])

  # If you have a GPU, put everything on cuda
  tokens_tensor = tokens_tensor.to('cuda')
  model.to('cuda')
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

  # get the predicted next sub-word (in our case, the word 'man')
  #predicted_index = torch.argmax(predictions[0, -1, :]).item()
  #print(predicted_index)
  predicted_index = torch.argsort(predictions[0, -1, :], descending=True)[nLikely].item()
  predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
  #assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
  #print(predicted_text)
  predicted_indizes=indexed_tokens + [predicted_index]
  return predicted_text, predicted_indizes

def evaluateWithInputId(model, indexed_tokens, nLikely=0):
  # Convert indexed tokens in a PyTorch tensor
  tokens_tensor = torch.tensor([indexed_tokens[-1000:]])

  # If you have a GPU, put everything on cuda
  tokens_tensor = tokens_tensor.to('cuda')
  model.to('cuda')
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

  # get the predicted next sub-word (in our case, the word 'man')
  #predicted_index = torch.argmax(predictions[0, -1, :]).item()
  #print(predicted_index)
  predicted_index = torch.argsort(predictions[0, -1, :], descending=True)[nLikely].item()
  #predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
  
  #assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
  predicted_indizes=indexed_tokens + [predicted_index]
  #return predicted_text, predicted_indizes
  return predicted_indizes

def getSecretTokens(model, tokenizer, startingInd, addedInd, k=0):
  """Return the secret message tokens stepwise"""
  tokens_tensor = torch.tensor([startingInd])

  # If you have a GPU, put everything on cuda
  tokens_tensor = tokens_tensor.to('cuda')
  model.to('cuda')
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

  ordered_index = torch.where(torch.argsort(predictions[0, -1, :], descending=True)==addedInd)[0]
  return ordered_index
