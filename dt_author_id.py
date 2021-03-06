#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("/Users/dadda/Dropbox (MIT)/Online Courses/Intro to ML/ud120-projects/tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

print(len(features_train[0]))

#########################################################
### your code goes here ###

clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()

clf = clf.fit(features_train, labels_train)
clf.predict(features_test)

print('Time is: ',  round(time()-t0, 3))

print(clf.score(features_test, labels_test))


#########################################################


