#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# trying a smaller data_set
'''
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
'''

print('creating SVC classifier..')
clf = SVC(C=10000.0, kernel='rbf')

print('training our classifier..')
t0 = time()
clf.fit(features_train, labels_train)
print('training time:', round(time()-t0, 3), 'sec')

print('running predictions')
t1 = time()
pred = clf.predict(features_test)
print('prediction time:', round(time()-t1, 3), 'sec')

# pred 10, 26, 50 assumin 0 indexed already
'''
print('10th pred:', pred[10])
print('26th pred:', pred[26])
print('50th pred:', pred[50])
'''

print('accuracy..')
accuracy = accuracy_score(pred, labels_test)

print('accuracy:', accuracy)

# classification_report (wasn't in mini-batch)
print('classification report:')
print(classification_report(labels_test, pred))


# predicted to be chris..
chris_pred = 0
for p in pred:
    if p == 1: chris_pred += 1

print('predicted to be chris:', chris_pred)

#########################################################
