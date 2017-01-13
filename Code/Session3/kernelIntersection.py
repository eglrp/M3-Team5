import numpy as np

def histogramIntersection(M,N):
    K_int=np.zeros(len(M))

    for x in range(int(N.shape[0])):
        for y in range(int(N.shape[1])):
            K_int[x]= K_int[x] + min(M[x][y],N[x][y])
    return K_int