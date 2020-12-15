import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaTokenizer, RobertaForCausalLM, RobertaConfig, BertTokenizer, BertLMHeadModel

torch.manual_seed(0)

def getModelType(input):
    """
    Helper function to return the model type that is used for the problem
    """
    modType=""
    if(input.name_or_path) in ["gpt2-medium", "gpt2-large", "gpt2", "gpt2-xl"]:
        modType="gpt2"
    elif(input.name_or_path) in ['roberta-base']:
        modType="roBERTa"
    elif(input.name_or_path) in ["bert-base-uncased"]:
        modType="bert"
    return modType

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

def buildModel(modelType="gpt2-medium"):
    """
    Helper Function to build the model based on the user input
    """

    if (modelType in  ["gpt2-medium", "gpt2-large", "gpt2", "gpt2-xl"]):
        mod, tok=buildModelGPT(modelType)
    elif (modelType=='roberta-base'):
        mod, tok=buildModelRoBERTa()
    elif (modelType=='bert-base-uncased'):
        mod, tok=buildModelBERT()
    else:
        print("ERROR: No correct model input!")
    return mod, tok