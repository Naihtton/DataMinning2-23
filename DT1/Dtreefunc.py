import numpy as np
import math

def entropy(*counts):
    """
    Expected information 
    Info(D) = -sum(pi)log2(pi)
    """
    total_count = sum(counts)
    if total_count == 0:
        return 0
    else:
        entropy_val = 0
        for count in counts:
            if count != 0:
                pi = count / total_count
                entropy_val -= pi * math.log(pi, 2)
        return entropy_val



def inforD(m,n): 
    """
    m is array of class in each attb group by domain
    n is array of entropy
    """
    c=len(m)
    #print(" size m is ",c)
    out=0
    i=0
    for i in range (c):
       # print("m[i] is", m[i])
        #print("sum(m) is ", sum(m))
        out +=(m[i]/sum(m))*n[i]
        #print("out is ",out)
    return(out)