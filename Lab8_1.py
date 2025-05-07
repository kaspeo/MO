#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importing necessary libraries and modules:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generating the dataset:
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# Splitting the dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Creating individual classifiers:
# log_clf (Logistic Regression)
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf (Random Forest)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf (Support Vector Machine)
svm_clf = SVC(gamma="scale", random_state=42)

# Creating a VotingClassifier with hard voting:
# Evaluating classifiers with hard voting:
print("VotingClassifier with hard voting:")

# Each classifier (log_clf, rnd_clf, svm_clf) is fitted on the training data.
for clf in (log_clf, rnd_clf, svm_clf):
    clf.fit(X_train, y_train)

# The predict method is used to make predictions on the testing data.
pred_log = log_clf.predict(X_test)
pred_rnd = rnd_clf.predict(X_test)
pred_svm = svm_clf.predict(X_test)

# Here we can chear a little bit
# Our answers in this classification problem
# is either 0 or 1
# Therefore we can perform majority voting by taking
# the average of our predictions
ensemble_pred_taking_average = []
for i in range(len(X_test)):
    votes = np.array([pred_log[i], pred_rnd[i], pred_svm[i]])
    average = np.average(votes)
    if average >= 0.5:
        result = 1.
    else:
        result = 0.
    ensemble_pred_taking_average.append(result)

# Calculate and print the accuracy score
accuracy_cheat = accuracy_score(y_test, ensemble_pred_taking_average)
print("Accuracy (taking the average):", accuracy_cheat)


ensemble_pred_voting = []
for i in range(len(X_test)):
    votes = [pred_log[i], pred_rnd[i], pred_svm[i]]
    majority_vote = max(set(votes), key=votes.count)
    ensemble_pred_voting.append(majority_vote)

# Calculate and print the accuracy score
accuracy_voting = accuracy_score(y_test, ensemble_pred_voting)
print("Accuracy (correct voting):", accuracy_voting)
