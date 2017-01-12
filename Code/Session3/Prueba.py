# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:13:45 2017

@author: onofre
"""

import matplotlib.pyplot as plt
import cv2
import cPickle
import spatial_pyramid as spt_py
import numpy as np
SIFTdetector = cv2.SIFT(nfeatures=100)
test_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
filename = test_images_filenames[784]
print 'Reading image ' + filename
ima = cv2.imread(filename)
gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
kpt, des = SIFTdetector.detectAndCompute(gray,None)

img=cv2.drawKeypoints(ima,kpt,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(  img)
cv2.imwrite('../Other files/Results week 02/Spatialpyramis_ppt.png', img)
#img = cv2.drawKeypoints(ima, kpt, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(  img)
size_image = np.float32(gray.shape)
coordinates_keypoints = [kp.pt for kp in kpt]
k=512
levels_pyramid = [[2,2],[4,4]]
codebook = cPickle.load(open('codebook.dat', 'r'))
descriptors = des
vw = spt_py.spatial_pyramid(size_image, descriptors, coordinates_keypoints, codebook, k, levels_pyramid)