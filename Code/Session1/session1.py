import time
import inputOutputUtils
import Evaluation
from descriptors import SIFTDescriptor #add the rest when created
from SVMClassifiers import SVMClassifier

start = time.time()

# Read the train and test files
train_images_filenames,test_images_filenames,train_labels,test_labels=inputOutputUtils.readData()

# Create Descriptors (SIFT, SURF, etc)
mySIFTDescriptor=SIFTDescriptor();#SIFT Descriptor

# Obtain descriptors and labels for the training set
max_class_train_images=30
D,L=mySIFTDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)

# Train a linear SVM classifier
mySVMClassifier=SVMClassifier(C=1,kernel_type='linear')#SVMClassifier with C=1 and linear Kernel
#TODO: Try other parameters for SVM
clf,stdSlr=mySVMClassifier.train(D,L)


# Get all the test data and predict their labels
predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySIFTDescriptor,clf,stdSlr)

# Performance evaluation
accuracy=Evaluation.computeAccuracy(predictedClasses,test_labels)


end=time.time()
print 'Done in '+str(end-start)+' secs.'

## 38.78% in 797 secs.