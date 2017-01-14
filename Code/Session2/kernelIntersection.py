import numpy as np

def histogramIntersection(M,N):
    K_int=np.zeros([len(M),len(N)])

    for Mi in range(len(M)):
        for Ni in range(len(N)):
            K_int[Mi,Ni]= intersection(M[Mi,:],N[Ni,:])
    return K_int

def intersection(u,v):
    intersect=0
    if len(u)!=len(v):
        raise RuntimeError('u and v should have the same length')
    
    for i in range(len(u)):
        intersect=intersect+min(u[i],v[i])
    
    return intersect