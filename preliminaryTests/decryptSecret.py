""" File for the decryption side of things, the receiver! """
from createModel import *
import numpy as np

def recoverSecretRanks(startingText, outInd):
    mod_rec, tok_rec=buildModelGPT()
    startingInd=tok_rec.encode(startingText)
    endingInd=outInd[len(startingInd):]
    secretTokensRec=np.zeros(len(endingInd), dtype=int)
    for i in range(len(endingInd)):
      token=getSecretTokens(mod_rec, tok_rec, startingInd, endingInd[i])
      startingInd.append(endingInd[i])
      secretTokensRec[i]=token
    return secretTokensRec

def getTextFromText(publicText, startingSecret="Secret: ", startingText="This year's Shakespeare Festival"):
    mod, tok=buildModelGPT()
    #ranks=getSecretRanks(publicText, startingText)
    indizes=tok.encode(publicText)
    ranks=recoverSecretRanks(startingText, indizes)
    #print(ranks)
    outText=startingSecret
    outInd=tok.encode(startingSecret)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd)    
    return outText