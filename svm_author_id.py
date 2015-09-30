#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("/Users/dadda/Dropbox (MIT)/Online Courses/Intro to ML/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


#########################################################
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', C = 10000.0)

t0 = time()
clf.fit(features_train, labels_train)
print "fitting time:", round(time()-t0, 3), "s"


pred = clf.predict(features_test)

accuracy = clf.score(features_test, labels_test)

print(accuracy)
print(sum(pred))
print(len(pred))

#########################################################


