
import numpy as np
import itertools
#Given an image and its descriptors, compute the spatial pyramid dividing 
#the image in 4 parts
def spatial_pyramid(size_image, descriptors, coordinates_keypoints, codebook, k, levels_pyramid):
    
        
#    limits_whole_image = [[0, 0], [size_image[0] - 1, size_image[1] - 1]]
    num_grids = sum([levels_pyramid[i][0]*levels_pyramid[i][1] for i in range(len(levels_pyramid))]) + 1
    visual_words_full = np.zeros((1, num_grids*k), dtype = np.float32)
    
    
    #First, we compute the histogram for the whole image
    words = codebook.predict(descriptors)

    visual_words_full[0, 0:k] = np.bincount(words, minlength = k)
    
    for i in range(len(levels_pyramid)):
        #For each level of the pyramid, we divide the image in the specified parts    
        grid = levels_pyramid[i]
        X = np.floor(np.linspace(0, size_image[0] - 1, num = grid[0]))
        Y = np.floor(np.linspace(0, size_image[1] - 1, num = grid[1]))
        
        up_corner = list(itertools.product(X[:-1], Y[:-1]))
        down_corner = list(itertools.product(X[1:], Y[1:]))
        
        descriptors_subimages = [[] for j in range(len(up_corner))]
                                 
        for k in range(len(coordinates_keypoints)):
            #For each descrptor, determine the subimage it belongs to                      
            for j in range(len(up_corner)):
                x = coordinates_keypoints[k][0]
                y = coordinates_keypoints[k][1]
                
                if x > up_corner[j][0] and y > up_corner[j][1] and x < down_corner[j][0] and y < down_corner[j][1]:
                    descriptors_subimages[j].append(descriptors[k])
                    break
      ##Continue here 
        words = codebook.predict(np.array(descriptors_subimage00))
        visual_words[0, 0:k] =  np.bincount(words, minlength = k)
    
    visual_words_sub = words_subimages_4x4(limits_whole_image, descriptors, keypoints, codebook, k)
    
    visual_words_full[0, k:5*k ] = visual_words_4x4
    
    #Third, we take each subimage of 4x4 and divide it again
    subimages = np.zeros([4, 2, 2], dtype = np.float32)
    subimages[0] = ([[limits_whole_image[0][0], limits_whole_image[0][1]], [limits_whole_image[1][0]/2, limits_whole_image[1][1]/2]])
    subimages[1] =([[limits_whole_image[0][0], limits_whole_image[1][1]/2], [limits_whole_image[1][0]/2, limits_whole_image[1][1]]])
    subimages[2] =([[limits_whole_image[1][0]/2, limits_whole_image[0][1]], [limits_whole_image[1][0], limits_whole_image[1][1]/2]])
    subimages[3] =([[limits_whole_image[1][0]/2, limits_whole_image[1][1]/2], [limits_whole_image[1][0], limits_whole_image[1][1]]]) 
    
    for i in range(4):
        visual_words_16x16 = words_subimages_4x4(subimages[i], descriptors, keypoints, codebook, k, alpha_0)
        visual_words_full[0, k*(5 + i*4):k*(5 + (i + 1)*4) ] = visual_words_16x16
                          
    return visual_words_full  



    
    
def words_subimages_4x4(limits_image, descriptors, coordinates_keypoints, codebook, k, alpha):   
    #limits_image contains the upper left and the down right corner of the image

    subimages = np.zeros([4, 2, 2], dtype = np.float32)  
    
    #Define limits of each subimage
    subimages[0] = ([[limits_image[0][0], limits_image[0][1]], [limits_image[1][0]/2, limits_image[1][1]/2]])
    subimages[1] =([[limits_image[0][0], limits_image[1][1]/2], [limits_image[1][0]/2, limits_image[1][1]]])
    subimages[2] =([[limits_image[1][0]/2, limits_image[0][1]], [limits_image[1][0], limits_image[1][1]/2]])
    subimages[3] =([[limits_image[1][0]/2, limits_image[1][1]/2], [limits_image[1][0], limits_image[1][1]]])  
    
    subimages = np.floor(subimages)
    
    
    descriptors_subimage00 = []
    descriptors_subimage01 = []
    descriptors_subimage10 = []
    descriptors_subimage11 = []

    
    #Determine which subimage each keypoint belongs
    for i in range(len(coordinates_keypoints)):
        
        x = coordinates_keypoints[i][0]
        y = coordinates_keypoints[i][1]
        
        
        if x > subimages[0, 0, 0] and y > subimages[0, 0, 1] and x < subimages[0, 1, 0] and y < subimages[0, 1, 1]:
            
            #Belongs to upperleft subimage
            descriptors_subimage00.append(descriptors[i])
            
        elif x > subimages[1, 0, 0] and y > subimages[1, 0, 1] and x < subimages[1, 1, 0] and y < subimages[1, 1, 1]:
            
            #Belongs to upperright subimage
            descriptors_subimage01.append(descriptors[i])

        elif x > subimages[2, 0, 0] and y > subimages[2, 0, 1] and x < subimages[2, 1, 0] and y < subimages[2, 1, 1]:    
            
            #Belongs to downleft subimage
            descriptors_subimage10.append(descriptors[i])
        else:
            
            #Belongs to downright subimage
            descriptors_subimage11.append(descriptors[i])
    #Compute histogram for all points of each subimage
    visual_words = np.zeros((1, 4*k), dtype = np.float32)
    
    if len(descriptors_subimage00) != 0:
        words00 = codebook.predict(np.array(descriptors_subimage00))
        visual_words[0, 0:k] =  np.bincount(words00, minlength = k)*alpha
                
    if len(descriptors_subimage01) != 0:    
        words01 = codebook.predict(np.array(descriptors_subimage01))
        visual_words[0, k:2*k] =  np.bincount(words01, minlength = k)*alpha 
                
    if len(descriptors_subimage10) != 0:
        words10 = codebook.predict(np.array(descriptors_subimage10))
        visual_words[0, 2*k:3*k] =  np.bincount(words10, minlength = k)*alpha
                
    if len(descriptors_subimage11) != 0:
        words11 = codebook.predict(np.array(descriptors_subimage11))
        visual_words[0, 3*k:4*k] =  np.bincount(words11, minlength = k)*alpha
    

    return visual_words