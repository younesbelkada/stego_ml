### Here all the important functions for the later encryption and decryption are saved
## Part for GPT
import torch

def evaluateWithInputText_GPT(model, tokenizer, text="We need to start working harder.", nLikely=0):
  """
  Function to find the next word given the rank with GPT2 LM and the previous text
  """
  indexed_tokens = tokenizer.encode(text)

  # Convert indexed tokens in a PyTorch tensor
  tokens_tensor = torch.tensor([indexed_tokens[-1000:]])

  tokens_tensor = tokens_tensor.to('cuda')
  model.to('cuda')
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

  predicted_index = torch.argsort(predictions[0, -1, :], descending=True)[nLikely].item()
  predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
  predicted_indizes=indexed_tokens + [predicted_index]
  return predicted_text, predicted_indizes

def evaluateWithInputId_GPT(model, indexed_tokens, nLikely=0):
  """
  Function to find the next word given the rank with GPT2 LM and the previous tokens
  """
  # Convert indexed tokens in a PyTorch tensor
  tokens_tensor = torch.tensor([indexed_tokens[-1000:]])

  tokens_tensor = tokens_tensor.to('cuda')
  model.to('cuda')
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

  predicted_index = torch.argsort(predictions[0, -1, :], descending=True)[nLikely].item()
  predicted_indizes=indexed_tokens + [predicted_index]
  return predicted_indizes

def getTokens_GPT(model, tokenizer, startingInd, addedInd, k=0):
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


