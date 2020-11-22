""" File for the encryption side of things, the sender! """
from createModel import *

def getSecretRanks(secretText="This is too secret for Joe Biden!", startingSecret="Secret: "):
    mod, tok=buildModelGPT()
    secretTextEnc=tok.encode(secretText)
    totalEnc=tok.encode(startingSecret)
    totalEnc.extend(secretTextEnc)
    ranksSecret=[]
    for i in range(len(tok.encode(startingSecret)), len(totalEnc)):
        ranksSecret.append(getSecretTokens(mod, tok, totalEnc[:i], totalEnc[i]).item())
    return ranksSecret

def encryptMessage(secretText="This is too secret for Joe Biden!", startingSecret="Secret: ", startingText="This year's Shakespeare Festival"):
    mod, tok=buildModelGPT()
    ranks=getSecretRanks(secretText, startingSecret)
    #print(ranks)
    outInd=tok.encode(startingText)
    for i in range(len(ranks)):
        outInd=evaluateWithInputId(mod, outInd, ranks[i]) # We can find be faster, with past for example or so
    outText=tok.decode(outInd) ## This will be passed forward
    return outText, outInd

