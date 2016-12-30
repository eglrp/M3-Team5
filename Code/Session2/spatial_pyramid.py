#import cv2
import numpy as np

#Given an image and its descriptors, compute the spatial pyramid dividing 
#the image in 4 parts
def spatial_pyramid(image, descriptors, keypoints):
    size = np.float32(image.shape)
    subimages = []
    
    #Transform keypoint objects to 2D coordinates
    coordinates_keypoints = []
    for i in range(len(keypoints)):
        coordinates_keypoints.append(np.float32(keypoints[i].pt))
        
    #Define limits of each subimage
    subimages.append([[0, 0], [size[0]/2, size[1]/2]])
    subimages.append([[0, size[1]/2], [size[0]/2, size[1]]])
    subimages.append([[size[0]/2, 0], [size[0], size[1]/2]])
    subimages.append([[size[0]/2, size[1]/2], [size[0], size[1]]])  
    #Determine which subimage each keypoint belongs
    Keypoint_location = []
    for i in range(len(keypoints)):
        
        x = coordinates_keypoints[i][0]
        y = coordinates_keypoints[i][1]
        
        if x > subimages[0][0][0] and y > subimages[0][0][1] and x < subimages[0][1][0] and y < subimages[0][1][1]:
            #Belongs to upperleft subimage
            Keypoint_location.append([0])
        elif x > subimages[1][0][0] and y > subimages[1][0][1] and x < subimages[1][1][0] and y < subimages[1][1][1]:
            #Belongs to upperright subimage
            Keypoint_location.append([1])
        elif x > subimages[2][0][0] and y > subimages[2][0][1] and x < subimages[2][1][0] and y < subimages[2][1][1]:    
            #Belongs to downleft subimage
            Keypoint_location.append([2])
        else:
            #Belongs to downright subimage
            Keypoint_location.append([3])
            
    #Compute histogram for all points of each subimage        