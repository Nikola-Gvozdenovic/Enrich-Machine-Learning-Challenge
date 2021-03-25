#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:52:57 2021

@author: ngvozdenovic
"""


# Import libraries and functions
import pickle
import pandas as pd
from feature_transformation import transform_input, encode


def model_predict(model_path, data_set):
    """Make predictions (probabilities for each class) and map them on confidence score
    
       Input: 
           model_path: string
           data_set: numpy.ndarray(2D)
       Output:
           predictions_df: pd.DataFrame
    """
    
    # Load model
    with open(model_path, 'rb') as file_content:
        model, transformation_type = pickle.load(file_content)
    
    # encode categorical features
    categorical_features = encode(data_set)
    
    # Transform dataset
    X_vect = transform_input(data_set, categorical_features, transformer='train_' + transformation_type)
    
    # Make predictions
    probabilities = model.predict_proba(X_vect)
    classes = model.classes_
    
    # Prapare confidences and classes
    predictions_df = pd.DataFrame(probabilities, columns=classes)
    
    return predictions_df    
    


def get_results(df1, df2):
    """Insert predicted class and confidence for each transaction.
       Make evidence of possible errors with new column.
       
       Input:
           df1: pd.DataFrame
           df2: pd.DataFrame
           
       Output:
           df: pd.DataFrame
    """
    
    # Class with maximum probability is predicted class
    # This probability represents confidence
    for index, row in df1.iterrows():
        df2.loc[index, 'transaction_class'] = row.idxmax()
        df2.loc[index, 'confidence'] = row.max()
        if (row.idxmax() in ['payment', 'transfer', 'cash-withdrawal', 'bank-fee'] and df2.loc[index, 'transaction_type'] == 'credit') or \
            (row.idxmax() in ['interest', 'refund', 'transfer'] and df2.loc[index, 'transaction_type'] == 'debit'): 
            df2.loc[index, 'error'] = 0
        else:
            df2.loc[index, 'error'] = 1
    return df2




