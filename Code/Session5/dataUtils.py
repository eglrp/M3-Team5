import numpy as np
import os
from shutil import copyfile, copytree

def createDataPaths(useServer,train_percentage):
    if useServer:
        data_dir='/data/MIT/train'
        test_dir='/data/MIT/test'
    else:
        data_dir='../../Databases/MIT_split/train'
        test_dir='../../Databases/MIT_split/test'
    
    dst_dir='./Databases/MIT'
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        copyData(data_dir,test_dir,dst_dir,train_percentage)

def copyData(data_dir,test_dir,dst_dir,train_percentage):
    classFolders=os.listdir(data_dir)
    
    for folder in classFolders:
        images=os.listdir(data_dir+'/'+folder)
        numlabels=len(images)
        max_class_train_images=int(np.round(numlabels*train_percentage))
        
        train_images=images[:max_class_train_images-1]
        val_images=images[max_class_train_images:]
        
        train_dir=dst_dir+'/train/'+folder
        os.makedirs(train_dir)
        val_dir=dst_dir+'/validation/'+folder
        os.makedirs(val_dir)
        
        for trainim in train_images:
            copyfile(data_dir+'/'+folder+'/'+trainim, train_dir+'/'+trainim)
        
        for valim in val_images:
            copyfile(data_dir+'/'+folder+'/'+valim, val_dir+'/'+valim)
    
    copytree(test_dir, dst_dir+'/test')