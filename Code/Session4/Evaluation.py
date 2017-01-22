import numpy as np

def computeAccuracyOld(predictedClasses,test_labels):
    numcorrect=np.sum([x[0]==x[1] for x in zip(predictedClasses,test_labels)])
    accuracy=numcorrect*100.0/len(test_labels)
    
    return accuracy
    
def getMeanAccuracy(clf,predictedLabels,test_labels):
    accuracy = 100*clf.score(predictedLabels, test_labels)

    return accuracy