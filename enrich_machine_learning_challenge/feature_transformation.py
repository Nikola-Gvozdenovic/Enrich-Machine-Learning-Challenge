#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:34:17 2021

@author: ngvozdenovic
"""


# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd


def vectorize(transformation_type='tfidf'):
    """Convert text file into numerical feature representations (vectors).
       These transformations use Bag Of Words model. 
       Two options:
       1. Convert text to a matrix of token counts. Term Frequency (TF) or Count Vectorizer is applied transformation.
       2. Convert text to a matrix of normalized token counts. Term Frequency - Inverse Document Frequency (TF-IDF) is applied transformation.
       
       Input: 
           transformation_type: string,  values: 'tf' and 'tfidf' 
    
       Output:
           transformer: sklearn object 
    """
    
    try:
        if transformation_type == "tf":    
            # CountVectorizer (TF)
            transformer = CountVectorizer(encoding='utf-8',
                                     ngram_range=(1, 3),  # we want to create unigrams, bigrams and trigrams
                                     stop_words=None,  # already applied
                                     lowercase=False,  # already applied
                                     max_df=0.95,
                                     # remove all terms that have document frequency higher than 95th percentile
                                     min_df=0.025,
                                     # remove all terms that have document frequency lower than 2.5th percentile
                                     max_features=100)
        elif transformation_type == "tfidf":        
            # TF-IDF
            transformer = TfidfVectorizer(encoding='utf-8',
                                     ngram_range=(1, 3),  # we want to create unigrams, bigrams and trigrams
                                     stop_words=None,  # already applied
                                     lowercase=False,  # already applied
                                     max_df=0.95,
                                     # remove all terms that have document frequency higher than 95th percentile
                                     min_df=0.025,
                                     # remove all terms that have document frequency lower than 2.5th percentile
                                     max_features=100)
            
        return transformer
    except Exception as exception:
        return f"See again function definition: {exception}"
    


def encode(input_data, columns=['transaction_type', 'transaction_account_type'], prefixes=['tt', 'tat']):
    """Convert categorical features into numeric representation. 
       All categories will be 0 except one that contains value. And that value will be 1.
       
       Input:
           input_data: pd.DataFrame
           columns: list of strings that represent columns to be transformed
           prefixes: list of strings
         
       Output:
           onehot_features: pd.DataFrame          
    """  
    
    feature_1 = pd.get_dummies(input_data[columns[0]].reset_index(drop=True), prefix=prefixes[0])
    feature_2 = pd.get_dummies(input_data[columns[1]].reset_index(drop=True), prefix=prefixes[1])
    onehot_features = pd.concat([feature_1, feature_2], axis=1) # append two dataframes horizontaly
    
    return onehot_features
    


def transform_input(input_data, one_hot_features, transformer='train_tf', column='description_processed'):
    """Apply transformations on a input data (vectorizatation on text column and one-hot-encoding on categorical columns).
    
       Input:
           input_data: pd.DataFrame
           one_hot_features: pd.DataFrame
           transformer: string,  values: 'train_tf' and 'train_tfidf'
           column: string 
           
       Output:
           X_vect: numpy.ndarray(2D)
    """    

    # Load transformer
    with open('models/' + transformer + '.pickle', 'rb') as file_content:
        transformer = pickle.load(file_content)

    X_train = transformer.transform(input_data[column]).toarray()
    description_feature = pd.DataFrame(X_train, columns=transformer.get_feature_names())
    
    X_vect = pd.concat([description_feature, one_hot_features], axis=1) # append two dataframes horizontaly
    
    return X_vect
    

    
    
    
    