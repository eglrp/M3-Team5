import numpy as np

def computeAccuracy(predictedClasses,test_labels):
    numcorrect=np.sum([x[0]==x[1] for x in zip(predictedClasses,test_labels)])
    accuracy=numcorrect*100.0/len(test_labels)
    print 'Final accuracy: ' + str(accuracy)
    
    return accuracy