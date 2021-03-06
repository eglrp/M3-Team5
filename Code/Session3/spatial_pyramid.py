
import numpy as np
import itertools
from yael import ynumpy
#Given an image and its descriptors, compute the spatial pyramid dividing 
#the image in 4 parts

def spatial_pyramid(size_image, descriptors, coordinates_keypoints, codebook, k, levels_pyramid):
    num_subim = list(np.append([1], [levels_pyramid[i][0]*levels_pyramid[i][1] for i in range(len(levels_pyramid))]))
    num_grids = sum(num_subim)
    acc_grid = list(np.cumsum(num_subim))
    
    visual_words_full = np.zeros((1, num_grids*k), dtype = np.float32)
    
    
    #First, we compute the histogram for the whole image
    words = codebook.predict(descriptors)

    visual_words_full[0, 0:k] = np.bincount(words, minlength = k)
    
    for i in range(1, len(num_subim)):
        
    #For each level of the pyramid, we divide the image in the specified parts    
        grid = levels_pyramid[i - 1]
            
        X = np.floor(np.linspace(0, size_image[0] - 1, num = grid[0] + 1))
        Y = np.floor(np.linspace(0, size_image[1] - 1, num = grid[1] + 1))
            
       #COmpute the corners of each subimage
        up_corner = list(itertools.product(X[:-1], Y[:-1]))
        down_corner = list(itertools.product(X[1:], Y[1:]))
        
        descriptors_subimages = [[] for j in range(num_subim[i])]
                                     
        for l in range(len(coordinates_keypoints)):
                #For each descriptor, determine the subimage it belongs to                      
            for j in range(num_subim[i]):
                x = coordinates_keypoints[l][0]
                y = coordinates_keypoints[l][1]
            
                if x > up_corner[j][0] and y > up_corner[j][1] and x < down_corner[j][0] and y < down_corner[j][1]:
                    descriptors_subimages[j].append(descriptors[l])
                    break
      
                    
            #for each subimage, we compute the visual words and we concatenate all
        for j in range(num_subim[i]):            
            if len(descriptors_subimages[j]) != 0:
                words = codebook.predict(np.array(descriptors_subimages[j]))
                visual_words_full[0, k*(acc_grid[i - 1] + j):k*(acc_grid[i - 1] + j + 1)] = np.bincount(words, minlength = k)        
                          
    return visual_words_full 

def spatial_pyramid_fisher(size_image, descriptors, coordinates_keypoints, k, gmm, levels_pyramid):
        
    num_subim = list(np.append([1], [levels_pyramid[i][0]*levels_pyramid[i][1] for i in range(len(levels_pyramid))]))
    num_grids = sum(num_subim)
    acc_grid = list(np.cumsum(num_subim))
    d = int(descriptors.shape[1])
    dim_vec = 2*d*k
     
        
    fisher_vector = np.zeros((1, num_grids*dim_vec), dtype = np.float32)
    
    #First, we compute the Fisher Vector for the whole image

    fisher_vector[0, 0:dim_vec] = ynumpy.fisher(gmm, descriptors, include = ['mu','sigma'])
    
    for i in range(1, len(num_subim)):
        
    #For each level of the pyramid, we divide the image in the specified parts    
        grid = levels_pyramid[i - 1]
            
        X = np.floor(np.linspace(0, size_image[0] - 1, num = grid[0] + 1))
        Y = np.floor(np.linspace(0, size_image[1] - 1, num = grid[1] + 1))
            
        #Compute the corners of each subimage
        up_corner = list(itertools.product(X[:-1], Y[:-1]))
        down_corner = list(itertools.product(X[1:], Y[1:]))
        
        descriptors_subimages = [[] for j in range(num_subim[i])]
                                     
        for l in range(len(coordinates_keypoints)):
            #For each descriptor, determine the subimage it belongs to                      
            for j in range(num_subim[i]):
                x = coordinates_keypoints[l][0]
                y = coordinates_keypoints[l][1]
            
                if x > up_corner[j][0] and y > up_corner[j][1] and x < down_corner[j][0] and y < down_corner[j][1]:
                    descriptors_subimages[j].append(descriptors[l])
                    break
      
                    
        #For each subimage, we compute the visual words and we concatenate all
        for j in range(num_subim[i]):            
            if len(descriptors_subimages[j]) != 0:
                
                vector = ynumpy.fisher(gmm, np.array(descriptors_subimages[j], dtype = np.float32), include = ['mu','sigma'])
                fisher_vector[0, dim_vec*(acc_grid[i - 1] + j):dim_vec*(acc_grid[i - 1] + j + 1)] = vector       
                          
    return fisher_vector

def add_weights():
    num_subim = list(np.append([1], [levels_pyramid[i][0]*levels_pyramid[i][1] for i in range(len(levels_pyramid))]))
    num_grids = sum(num_subim)
    acc_grid = list(np.cumsum(num_subim))
    d = int(descriptors.shape[1])
    dim_vec = 2*d*k
    L = len(levels_pyramid)
    fisher_vector[:, 0:dim_vec] = np.float32(fisher_vector[:, 0:dim_vec])*(1.0/2**L)
    for i in range(1, len(num_subim)):
       fisher_vector[:, dim_vec*(acc_grid[i - 1]):dim_vec*(acc_grid[i - 1] + num_subim[i])] = np.float32(fisher_vector[:, dim_vec*(acc_grid[i - 1]):dim_vec*(acc_grid[i - 1] + num_subim[i])])*(1.0/(2**(L - i + 1)))
    
    
