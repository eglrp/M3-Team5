import time
import inputOutputUtils
import Evaluation
import PCA_computing
from descriptors import SIFTDescriptor #add the rest when created
from SVMClassifiers import SVMClassifier
from sklearn.externals import joblib

start = time.time()

# Read the train and test files
train_images_filenames,test_images_filenames,train_labels,test_labels=inputOutputUtils.readData()

# Create Descriptors (SIFT, SURF, etc)
mySIFTDescriptor=SIFTDescriptor();#SIFT Descriptor

# Obtain descriptors and labels for the training set
max_class_train_images=30
D,L=mySIFTDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)

#Apply PCA to descriptors
print 'Applying PCA'
number_components = 110
D, pca = PCA_computing.PCA_to_data(D, number_components)

# Train a linear SVM classifier
mySVMClassifier=SVMClassifier(C=1,kernel_type='linear')#SVMClassifier with C=1 and linear Kernel


#TODO: Try other parameters for SVM
clf, stdSlr=mySVMClassifier.train(D,L)

#Save the trained SVM
#joblib.dump(clf, 'SVMClassifier.pkl')

# Get all the test data and predict their labels
predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySIFTDescriptor,clf,stdSlr, pca)

# Performance evaluation
accuracy=Evaluation.computeAccuracy(predictedClasses,test_labels)


end=time.time()
print 'Done in '+str(end-start)+' secs.'

## PCA:128 38.78% in 797 secs.
## PCA:120 39.52% in 885 secs.
## PCA:110 38.66% in 996 secs.
## PCA:100 39.03% in 646 secs.
## PCA:90 40.64% in 708 secs.
## PCA:80 39.15% in 636 secs.
## PCA:70 39.65% in 821 secs.



