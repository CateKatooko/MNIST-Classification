# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:45:21 2023

@author: DeLL
"""

import pandas as pd
from sklearn.datasets import fetch_openml
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
print(X.shape)
y
y.shape

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28) 
    plt.imshow(image, cmap="binary") 
    plt.axis("off")

# some_digit = X[1]
some_digit = X[0]
plot_digit(some_digit)
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y[0]
#y[1]

# Training a Binary Classifier
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')

# count of False and True values/lables for digit 5
import math
y=0
z=0
for x in y_train:
    if x == '5':
        y+= 1
    else:
        z+=1

print('number of true values:', y,'\nnumber of false values:',z)
print('true values (instances of digit 5) %:', math.ceil((y/z)*100))
 
# Training the SGD Scikit model   
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# detect images of the number 5:
sgd_clf.predict([some_digit]) #variable some_digit initialized at line 28

# 1 Evaluate model’s performance, p4mc measure - CROSS VALIDATION with k-folds
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# classify every single image in the most frequent class i.e the false class
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train))) # prints False: no 5s detected

# Cross Val on above dummy model, using (3)kfolds. Return the evaluation scores
cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Implelment Cross-Validation with Control
# StratifiedKFold class produces folds that contain a representative 
# ratio of each class
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3) # add shuffle=True if the dataset is
# not already shuffled

# At each iteration create a clone of the classifier, train it on the 
# training folds, and make predictions on the test fold.
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf) 
    X_train_folds = X_train[train_index] 
    y_train_folds = y_train_5[train_index] 
    X_test_fold = X_train[test_index] 
    y_test_fold = y_train_5[test_index] 
    #print(y_test_fold)
    clone_clf.fit(X_train_folds, y_train_folds) 
    y_pred = clone_clf.predict(X_test_fold)
    #print(y_pred)
    #print(X_test_fold)
# count no of correct predictions and output the ratio of correct predictions
    n_correct = sum(y_pred == y_test_fold) 
    print(n_correct / len(y_pred)) 

# 2. Evaluating the model: CONFUSSION MATRIX
# cross_val_predict() performs k-fold cross-validation and returns the 
# predictions made on each test fold
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
# pass the target classes (y_train_5) and the predicted classes (y_train_pred)
cm = confusion_matrix(y_train_5, y_train_pred)
cm
#****irrelevant - example just
y_train_perfect_predictions = y_train_5 # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)
#*****

# checking for model accuracy
# PRECISION and RECALL
from sklearn.metrics import precision_score, recall_score
# when sdg classifier claims an image reps a 5, its correct only 83.7% of
# the time and only detects 65.1% of the 5s.
print('Precision Score: ', precision_score(y_train_5, y_train_pred))# == 3530 / (687 + 3530)
print('Recall Score: ', recall_score(y_train_5, y_train_pred)) # == 3530 / (1891 + 3530)

# (precision + recall) metric = F1 scrore (harmonic mean of precision and recall)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred) 

# precision and recall trade-off
# Scikit-Learn does not let you directly set the threshold, but gives you access to 
# the decision scores that it uses to make predictions. The decision_function() method, 
# returns a score for each instance. You then use any threshold to make predictions based on those scores
y_scores = sgd_clf.decision_function([some_digit])
y_scores
# set threshold
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

# Increase threshold
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

# SELECTING A GOOD precision/recall trade-off
# 1. deciding which threshold to use
# specify to return decision scores instead of predictions
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# then use precision_recall_curve() function to compute precision and 
# recall for all possible thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# use Matplotlib to plot precision and recall as functions of the threshold value
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.show()

# 2. Plot precision directly against recall
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.show()

# finding the threshold you need to use(eg that gives 90% precision)
# use first plot, n search for the lowest threshold that gives you at least 90% precision
# use the NumPy array’s argmax() method. Returns first index of the max value 
# (in this case, the first True value)
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

# make predictions (on the training set), without having to call the classifier’s 
# predict() method
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
# check these predictions’ precision and recall
print('Precision Score:', precision_score(y_train_5, y_train_pred_90))
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print('Recall Score:', recall_at_90_precision)

# THE ROC (Receiver Operating Chartacteristic)
from sklearn.metrics import roc_curve
# compute the TPR and FPR for various thresholds
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
# plot the FPR against the TPR
# look for the index of the desired threshold that gives 90% precision. 
# Thresholds are listed in decreasing order, hence the <=
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.show()

# Creat a RandomForestClassifier, whose PR curve and Fscore can be compared 
# to those of the SGDClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
method="predict_proba")

# COMPARE classifiers by measuring the area under the curve (AUC).
# ROC AUC for perfect classifier = 1. Purely random classifier will've a ROC AUC = 0.5. 
# Scikit-Learn's ROC AUC estimator
from sklearn.metrics import roc_auc_score
# this score is not accurate because there are few positives (5s) compared to the 
# negatives (non-5s). In contrast, the PR curve makes it clear that the classifier
# has room for improvement:
roc_auc_score(y_train_5, y_scores)

# Lets create a RandomForestClassifier, whose PR curve and F score we
# can compare to those of the SGDClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)

# precision_recall_curve() function expects labels and scores for each
# instance, so we need to train the random forest classifier and make it assign a
# score to each instance. RandomForestClassifier class does not have a
# decision_function() method but has a predict_proba() method that returns class
# probabilities for each instance. we can just use the probability of the positive class as a score, so it will work fine





