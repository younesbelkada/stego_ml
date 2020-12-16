from .createModel import *
from .evaluationModelHelpers import *
from .encryptionWrapped import *
import matplotlib.pyplot as plt
import torch.nn as nn
import ipywidgets as widgets
from IPython.display import display

w = widgets.Dropdown(
    options=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', "bert-base-uncased", 'roberta-base'],
    value='gpt2',
    description='Model Type:',
    disabled=False,
)
w2 = widgets.SelectMultiple(
    options=['gpt2', 'bert', 'roberta'],
    value=['gpt2'],
    #rows=10,
    description='model type',
    disabled=False
)
w3 = widgets.Select(
    options=['CompleteSentence', 'NoCompleteSentence'],
    value='CompleteSentence',
    # rows=10,
    description='Option:',
    disabled=False
)
w4 = widgets.Text(
    value="I can't image how much I had fun to program this notebook at 03:00 AM",
    placeholder='Type something',
    description='Secret :',
    disabled=False
)
w5 = widgets.Text(
    value="This message is super secret ! ",
    placeholder='Type the starting of the cover text',
    description='Cover text start :',
    disabled=False
)

def plot_ranks(secret, precondSecret, tab_list, bert=False, roberta=False):
	if "bert" in tab_list:
		bert=True
	if 'roberta' in tab_list:
		roberta=True
	mod, tok=buildModelGPT("gpt2")
	ranks_GPT=getRanks_GPT(mod, tok, secret, precondSecret, False)
	x=np.arange(len(ranks_GPT))
	if roberta:
		mod, tok=buildModelRoBERTa()
		ranks_RoBERTa=getRanks_GPT(mod, tok, secret, precondSecret, False)
	if bert:
		mod, tok=buildModelBERT()
		ranks_BERT=getRanks_GPT(mod, tok, secret, precondSecret, False)
	fig, ax=plt.subplots()
	ax.plot(x, ranks_GPT, '--', label='GPT2')
	if roberta:
		ax.plot(x, ranks_RoBERTa[:len(x)], '--', label='RoBERTa')
	if bert:
		ax.plot(x, ranks_BERT, '--', label='BERT')
	leg = fig.legend()
	fig.show()

def get_score(mod, tok, text, typ="gpt2"):
	if "gpt2" in typ:
		return get_perplex_score_GTP2(mod, tok, text)
	else:
		return get_perplex_score_BERT(text, mod, tok)

def get_perplex_score_GTP2(mod, tok, txt):
    """
    Returns the perplexity score of the text given the language model and the tokenizer. This code is adapted only for GPT2 models
    and is inspired from hugging face website. 
    """
    encodings = tok('\n\n'.join(txt), return_tensors='pt')
    max_length = mod.config.n_positions
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100
    with torch.no_grad():
        outputs = mod(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

def get_perplex_score_BERT(cover_text, model, tokenizer, startingSecret=". "):
    """
    Returns the perplexity score for BERT and RoBERTa models given a text
    """
    probas = []
    model.to("cuda")
    model.eval()
    token_secret = tokenizer.encode(cover_text)
    token_start = tokenizer.encode(startingSecret)
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([token_start]).to("cuda")
    m = nn.Softmax(dim=-1)
  # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor
    pred = []
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        tab = m(predictions[:, -1, :][0])
        pred.append(tab[token_secret[0]].item())
        for i in range(1, len(token_secret)):
            tokens_tensor = torch.cat((tokens_tensor.to('cpu').view(-1), torch.Tensor([token_secret[i]])), dim=-1).view(1, -1).to("cuda")
            outputs = model(tokens_tensor.type(torch.long))
            predictions = outputs[0]
            tab = m(predictions[:, -1, :][0])
            pred.append(tab[token_secret[i]].item())
            
    s = 0
    for p in pred:
        s += np.log(p)
    score = np.exp((-1/len(pred))*s)
    return score