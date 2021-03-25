#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:39:07 2021

@author: ngvozdenovic
"""

# Inport libraries and functions
import numpy as np
from predicting import model_predict, get_results
from model_training import load_data
import pickle

# Load the scorecard data 
scorecard_df = load_data("data/scorecard.csv")

# Save scorecard cleaned data
scorecard_df.to_csv("data/scorecard_cleaned.csv", index=False) 

# Make predictions with all models
lr_predictions = model_predict('models/logistic_regression.pickle', scorecard_df)
nb_predictions = model_predict('models/naive_bayes.pickle', scorecard_df)
svm_predictions = model_predict('models/svm.pickle', scorecard_df)
rf_predictions = model_predict('models/random_forest.pickle', scorecard_df)
gb_predictions = model_predict('models/gradient_boosting.pickle', scorecard_df)
    
# Back to original columns
with open('models/train_label.pickle', 'rb') as file_content:
    label_encoder = pickle.load(file_content)
    
lr_predictions.columns = list(label_encoder.inverse_transform(lr_predictions.columns))
nb_predictions.columns = list(label_encoder.inverse_transform(nb_predictions.columns))
svm_predictions.columns = list(label_encoder.inverse_transform(svm_predictions.columns))
rf_predictions.columns = list(label_encoder.inverse_transform(rf_predictions.columns))
gb_predictions.columns = list(label_encoder.inverse_transform(gb_predictions.columns))

# Add columns 'transaction_class' and 'confidence'
scorecard_df['transaction_class'] = ""
scorecard_df['confidence'] = np.nan

# Make copies of dataframe (one for each model) 
lr_results = scorecard_df.copy()
nb_results = scorecard_df.copy()
svm_results = scorecard_df.copy()
rf_results = scorecard_df.copy() 
gb_results = scorecard_df.copy()

# Final results (predicted classes and confidences)
lr_results = get_results(lr_predictions, lr_results)
nb_results = get_results(nb_predictions, nb_results)
svm_results = get_results(svm_predictions, svm_results)
rf_results = get_results(rf_predictions, rf_results)
gb_results = get_results(gb_predictions, gb_results)

# Save results to csv
lr_results.to_csv("results/scorecard_lr.csv", index=False)
nb_results.to_csv("results/scorecard_nb.csv", index=False)
svm_results.to_csv("results/scorecard_svm.csv", index=False)
rf_results.to_csv("results/scorecard_rf.csv", index=False)
gb_results.to_csv("results/scorecard_gb.csv", index=False)

# Number of sure misclassifications
print(lr_results['error'].sum().astype(int))
print(nb_results['error'].sum().astype(int))
print(svm_results['error'].sum().astype(int))
print(rf_results['error'].sum().astype(int))
print(gb_results['error'].sum().astype(int))

# Class distribution
lr_results['transaction_class'].value_counts()
nb_results['transaction_class'].value_counts()
svm_results['transaction_class'].value_counts()
rf_results['transaction_class'].value_counts()
gb_results['transaction_class'].value_counts()
