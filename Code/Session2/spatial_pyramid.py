
import numpy as np

#Given an image and its descriptors, compute the spatial pyramid dividing 
#the image in 4 parts
def spatial_pyramid(size_image, descriptors, keypoints, codebook, k):
    
    alpha_0 = 1/2
    alpha_1 = 1/4
    alpha_2 = 1/4
        
    limits_whole_image = [[0, 0],[size_image[0], size_image[1]]]
    
    visual_words_full = np.zeros((1, 21*k), dtype = np.int64)
    
    #First, we compute the histogram for the whole image
    words = codebook.predict(descriptors)

    visual_words_full[0, 0:k] = np.bincount(words, minlength = k)*alpha_2
    
    #Second, we compute the histogram for the 4x4 subimages
    
    visual_words_4x4 = words_subimages_4x4(limits_whole_image, descriptors, keypoints, codebook, k, alpha_1)
    
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