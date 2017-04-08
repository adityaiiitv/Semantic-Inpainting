from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC()
clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.support_vectors_array([[ 0.,  0.],[ 1.,  1.]])
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y) 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
dec = lin_clf.decision_function([[1]])
dec.shape[1]
