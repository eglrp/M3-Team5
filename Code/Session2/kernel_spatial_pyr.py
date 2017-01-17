# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 18:07:33 2017

@author: Roque
"""
import numpy as np
import kernelIntersection


def spatialPyramidKernel(M,N,levels_pyramid):

    K_int=np.zeros([len(M),len(N)])
    

    for Mi in range(len(M)):
        for Ni in range(len(N)):
            K_int[Mi,Ni]= pyramidkernel(M[Mi,:],N[Ni,:],M.size[1])
    return K_int
    
    
def pyramidkernel(hist1,hist2,levels_pyramid,length):
    
    weight_hist1 = add_weights(hist1,levels_pyramid,length)
    weight_hist2 = add_weights(hist2,levels_pyramid,length)
    intersection_Pyramid = kernelIntersection.intersection( weight_hist1, weight_hist2 );
    return intersection_Pyramid
    
def add_weights(fullHistogram,levels_pyramid,length):
    num_subim = list(np.append([1], [levels_pyramid[i][0]*levels_pyramid[i][1] for i in range(len(levels_pyramid))]))
    acc_grid = list(np.cumsum(num_subim))
    k = length/acc_grid[len(acc_grid)] 
    L = len(levels_pyramid)
    fullHistogram[0:k] = np.float32( fullHistogram[0:k])*(1.0/2**L)
    for i in range(1, len(num_subim)):
        fullHistogram[k*(acc_grid[i - 1]):k*(acc_grid[i])] = np.float32( fullHistogram[ k*(acc_grid[i - 1]):k*(acc_grid[i])])*(1.0/(2**(L - i + 1)))
    
    