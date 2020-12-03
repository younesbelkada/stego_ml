""" File for the decryption side of things, the receiver! """
from createModel import *
import numpy as np

def recoverSecretRanks(mod_rec, tok_rec, startingText, outInd):
    #mod_rec, tok_rec=buildModelGPT()
    startingInd=tok_rec.encode(startingText)
    endingInd=outInd[len(startingInd):]
    #secretTokensRec=np.zeros(len(endingInd), dtype=int)
    secretTokensRec=[]
    for i in range(len(endingInd)):
      token=getSecretTokens(mod_rec, tok_rec, startingInd, endingInd[i])
      if (token==3):
        break
      if(token>2):
        token-=1
      startingInd.append(endingInd[i])
      secretTokensRec.append(token[0].tolist())
    return secretTokensRec

def getTextFromText(mod, tok, publicText, startingSecret="Secret: ", startingText="This year's Shakespeare Festival"):
    #mod, tok=buildModelGPT()
    #ranks=getSecretRanks(publicText, startingText)
    indizes=tok.encode(publicText)
    ranks=recoverSecretRanks(mod, tok, startingText, indizes)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd)    
    return outText
    
def getTextFromInd(mod, tok, publicInd, startingSecret="Secret: ", startingText="This year's Shakespeare Festival"):
    #mod, tok=buildModelGPT()
    #ranks=getSecretRanks(publicText, startingText)
    indizes=publicInd
    ranks=recoverSecretRanks(mod, tok, startingText, indizes)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd)    
    return outText
