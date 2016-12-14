import numpy as np

def computeAccuracy(predictedClasses,test_labels):
    numcorrect=np.sum(predictedClasses==test_labels)
    accuracy=numcorrect*100.0/len(test_labels)
    print 'Final accuracy: ' + str(accuracy)
    
    return accuracy