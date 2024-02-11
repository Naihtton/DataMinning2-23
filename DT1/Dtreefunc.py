import numpy as np
def entropy(*args):
    total = np.sum(args)
    ent = 0
    for arg in args:
        if arg != 0:
            ent -= (arg / total) * np.log2(arg / total)
    return ent

def inforD(m, feature_entropy):
    total = np.sum(m)
    inD = 0
    for i, count in enumerate(m):
        inD += (count / total) * feature_entropy[i]
    return inD
