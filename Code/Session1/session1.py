import time
import inputOutputUtils
import Evaluation
import PCA_computing
from descriptors import SIFTDescriptor
from descriptors import SURFDescriptor
from descriptors import ORBDescriptor
from SVMClassifiers import SVMClassifier
from sklearn.externals import joblib

start = time.time()

# Read the train and test files
train_images_filenames,test_images_filenames,train_labels,test_labels=inputOutputUtils.readData()

# Create Descriptors (SIFT, SURF, etc)
#mySIFTDescriptor=SIFTDescriptor();# SIFT Descriptor
mySURFDescriptor = SURFDescriptor(); # SURF Descriptor
#myORBDescriptor = ORBDescriptor(); # ORB Descriptor


# Obtain descriptors and labels for the training set
max_class_train_images=30

#D,L=mySIFTDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)
D,L=mySURFDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)
# D,L=myORBDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)

#Apply PCA to descriptors
print 'Applying PCA'
number_components = 100
D, pca = PCA_computing.PCA_to_data(D, number_components)

# Train a linear SVM classifier
mySVMClassifier=SVMClassifier(C=1,kernel_type='linear')#SVMClassifier with C=1 and linear Kernel


#TODO: Try other parameters for SVM
clf, stdSlr=mySVMClassifier.train(D,L)

#Save the trained SVM
#joblib.dump(clf, 'SVMClassifier.pkl')

# Get all the test data and predict their labels
#predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySIFTDescriptor,clf,stdSlr, pca)
predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySURFDescriptor,clf,stdSlr, pca)
#predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,myORBDescriptor,clf,stdSlr, pca)



# Performance evaluation
accuracy=Evaluation.computeAccuracy(predictedClasses,test_labels)


end=time.time()
print 'Done in '+str(end-start)+' secs.'

## 38.78% in 797 secs.