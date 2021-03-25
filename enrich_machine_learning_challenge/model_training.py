#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:53:21 2021

@author: ngvozdenovic
"""


# Import libraries and functions
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from nlp_feature_engineering import nlp_preprocessing


def convert_transaction_amount(transaction_amount):
    """Credit: transaction amount lower than 0
       Debit: transaction amount higher than 0
       
       Input:
           transaction_amount: pd.Series
           
       Output:
           transaction_amount: pd.Series
    """
    
    return np.where(transaction_amount > 0, 'debit', 'credit')



def load_data(data_path):
    """Load and preprocess columns for futher work.
    
       Input:
           data_path: string
        
       Output:
           df: pd.DataFrame
    """
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Perform feature transformations
    df["description_processed"] = nlp_preprocessing(df["transaction_description"])
    df['transaction_type'] = convert_transaction_amount(df['transaction_amount'])
    
    df.loc[~df['transaction_account_type'].isin(('transaction', 'savings', 'credit-card')), 'transaction_account_type'] = "other"
    
    return df    



def split_data(df, test_size):
    """Split data into training and test datasets.
    
       Input:
           df: pd.DataFrame
           test_size: float (fraction)
           
       Output:
           X_train: pd.DataFrame
           X_test: pd.DataFrame
           y_train: pd.Series
           y_test: pd.Series
    """    

    X_train, X_test, y_train, y_test = train_test_split(df[["description_processed", "transaction_type", "transaction_account_type"]],
                                                        df['transaction_class'],
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test



def train_model(algorithm, X_train, y_train, X_test, y_test, cv_type='rand', transformation_type='tf'):
    """Train ML model and evaluate on test set.
    
       Input:
           algorithm: def (sklearn object)
           X_train: numpy.ndarray(2D)
           y_train: pd.Series
           X_test: numpy.ndarray(2D)
           y_test: pd.Series
           cv_type: string,  values: 'rand' and 'grid'
           transformation_type: string,  values: 'tf' and 'tf-idf'
       
       Output:
           model: sklearn object
           model_score: float (fraction, f1_score metric)
           transformation_type: string,  values: 'tf' and 'tf-idf'
    """
    
    model = algorithm(X_train, y_train, cv_type=cv_type)
    model_preds = model.predict(X_test)
    model_score = f1_score(y_test, model_preds, average='weighted')
    
    return model, model_score, transformation_type



# Output (void) function
def write_save_better_model(models, model_name):
    """Write better model score to csv and save that trained model.
    
       Input:
           models: list of tuples with trained models
           model_name: name of the model (ML algorithm)                    
    """
    
    first_model_score = models[0][1]
    second_model_score = models[1][1]
    
    model_scores = (first_model_score, second_model_score)
    
    max_index = model_scores.index(max(model_scores))
    
    model_df = pd.DataFrame([[model_name, models[max_index][1], models[max_index][2]]], 
                            columns=['model', 'f1_score', 'transformation_type'])
    model_df.to_csv('validation_results/' + model_name + '.csv')
    
    with open('models/' + model_name + '.pickle', 'wb') as output:
        pickle.dump((models[max_index][0], models[max_index][2]), output)
        

        
# Output (void) function
def save_transformer(transformer, name):
    """Save transformed input data.
    
       Input:
           transformer: sklearn object
           name:  string 
    """
    
    with open('models/' + name + '.pickle', 'wb') as output:
        pickle.dump(transformer, output)
    
    
