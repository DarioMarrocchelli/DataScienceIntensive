#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("/Users/dadda/Dropbox (MIT)/Online Courses/Intro to ML/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("/Users/dadda/Dropbox (MIT)/Online Courses/Intro to ML/ud120-projects-master/final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print(len(pred))

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
