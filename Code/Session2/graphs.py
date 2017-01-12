# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 01:57:51 2017

@author: Roque
"""

import itertools
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def rcurve(predictedLabels,validation_labels,clf):
    
    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
 
    probas_ = clf.predict_proba(predictedLabels)

    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(validation_labels, probas_[:,1], classes[i])

        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label = classes[i] % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.show()
    
    
def plot_confusion_matrix(clf,validation_labels,predictedLabels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
    predicted = clf.predict(predictedLabels)
    cm = confusion_matrix(validation_labels,predicted)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    plt.show()
