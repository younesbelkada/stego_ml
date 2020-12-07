""" File for the encryption side of things, the sender! """
from createModel import *
import numpy as np

def getSecretRanks(mod, tok, secretText="This is too secret for Joe Biden!", startingSecret="Secret: "):
    #mod, tok=buildModelGPT()
    secretTextEnc=tok.encode(secretText)
    totalEnc=tok.encode(startingSecret)
    totalEnc.extend(secretTextEnc)
    ranksSecret=[]
    for i in range(len(tok.encode(startingSecret)), len(totalEnc)):
        ranksSecret.append(getSecretTokens(mod, tok, totalEnc[:i], totalEnc[i]).item())
    ranksSecret=np.array(ranksSecret)
    ranksSecret[ranksSecret>2]+=1 #Create offset
    ranksSecret=np.append(ranksSecret, 3)
    return ranksSecret

def completeMessage(mod, tok, ind, max_length=50):
    tokens_tensor = torch.tensor([ind])
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

def encryptMessage(mod, tok, secretText="This is too secret for Joe Biden!", startingSecret="Secret: ", startingText="This year's Shakespeare Festival"):
    #mod, tok=buildModelGPT()
    ranks=getSecretRanks(mod, tok, secretText, startingSecret)
    #print(ranks)
    outInd=tok.encode(startingText)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    #outText=tok.decode(outInd) ## This will be passed forward
    outText, outInd=completeMessage(mod, tok, outInd, max_length=50)
    return outText, outInd

