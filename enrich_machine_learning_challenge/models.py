#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:18:15 2021

@author: ngvozdenovic
"""


# Import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV



def logistic_regression(X_train, y_train, cv_type='grid'):
    """Tune Logistic Regression model.
    
       Input:
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           cv_type: string,  values: 'grid' and 'rand'
    
       Output:
           search: sklern object (fitted model)
           
    """
    
    # 5-fold cross validation with 3 repeats (15-fold cross validation)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    
    param_grid = {"C":np.logspace(-3,3,7), 
                  "penalty":["l2"], # ridge regularization
                  "solver":["lbfgs"]}
    
    # Randomized hyperparameters searching
    if cv_type == 'rand': 
        search = RandomizedSearchCV(LogisticRegression(),
                                    param_distributions=param_grid,
                                    cv=cv,
                                    n_iter=10,
                                    scoring='f1_weighted',
                                    verbose=1)
    # Exhaustive hyperparameters searching
    else:
        search = GridSearchCV(LogisticRegression(), 
                              param_grid=param_grid, 
                              cv=cv, 
                              scoring="f1_weighted", 
                              verbose=1)
        
    search.fit(X_train, y_train)
    
    return search


def naive_bayes(X_train, y_train, cv_type=None):
    """Fit Naive Bayes algorithm on train dataset.
       Note: Naive Bayes doesn't have hyperparameters to tune.
       
       Input:
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           cv_type: None, not used
           
       Output:
           model: sklearn object (fitted model)
    """
    
    model = ComplementNB()
        
    model.fit(X_train, y_train)
    
    return model


def svm(X_train, y_train, cv_type='grid'):
    """Tune SVM model.
    
       Input:
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           cv_type: string,  values: 'grid' and 'rand'
    
       Output:
           search: sklern object (fitted model)
           
    """
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    
    param_grid = {#'C': [.0001, .001, .01],
                  'C': [1, 10],
                  #'gamma': [.0001, .001, .01, .1, 1, 10, 100],
                  #'degree': [1, 2, 3, 4, 5],
                  #'kernel': ['linear', 'poly', 'rbf'],
                  'probability': [True]}
    
    if cv_type == 'rand': 
        search = RandomizedSearchCV(SVC(),
                                    param_distributions=param_grid,
                                    cv=cv,
                                    n_iter=5,
                                    scoring='f1_weighted',
                                    verbose=1)
    else:
        search = GridSearchCV(SVC(), 
                              param_grid=param_grid, 
                              cv=cv, 
                              scoring="f1_weighted", 
                              verbose=1)
        
    search.fit(X_train, y_train)
    
    return search


def random_forest(X_train, y_train, cv_type='grid'):
    """Tune Random Forest model.
    
       Input:
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           cv_type: string,  values: 'grid' and 'rand'
    
       Output:
           search: sklern object (fitted model)
           
    """
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    
    param_grid = {'n_estimators': [10, 50, 100],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth' : [4,5,6],
                  'criterion' :['gini', 'entropy']}
    
    if cv_type == 'rand': 
        search = RandomizedSearchCV(RandomForestClassifier(),
                                    param_distributions=param_grid,
                                    cv=cv,
                                    n_iter=5,
                                    scoring='f1_weighted',
                                    verbose=1)
    else:
        search = GridSearchCV(RandomForestClassifier(), 
                              param_grid=param_grid, 
                              cv=cv, 
                              scoring="f1_weighted", 
                              verbose=1)
        
    search.fit(X_train, y_train)
    
    return search


def gradient_boosting(X_train_vect, y_train, cv_type='grid'):
    """Tune Gradient Boosting model.
    
       Input:
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           cv_type: string,  values: 'grid' and 'rand'
    
       Output:
           search: sklern object (fitted model)
           
    """
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    
    param_grid = {"loss":["deviance"],
                  "learning_rate": [0.05, 0.1, 0.2],
                  "min_samples_split": np.linspace(0.1, 0.5, 5),
                  "min_samples_leaf": np.linspace(0.1, 0.5, 5),
                  "max_depth":[3, 5],
                  "max_features":["log2","sqrt"],
                  "criterion": ["friedman_mse",  "mae"],
                  "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                  "n_estimators":[10]}
    
    if cv_type == 'rand': 
        search = RandomizedSearchCV(GradientBoostingClassifier(),
                                    param_distributions=param_grid,
                                    cv=cv,
                                    n_iter=5,
                                    scoring='f1_weighted',
                                    verbose=1)
    else:
        search = GridSearchCV(GradientBoostingClassifier(), 
                              param_grid=param_grid, 
                              cv=cv, 
                              scoring="f1_weighted", 
                              verbose=1)
        
    search.fit(X_train_vect, y_train)
    
    return search



