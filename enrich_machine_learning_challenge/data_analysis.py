#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:48:38 2021

@author: ngvozdenovic
"""

import pandas as pd
from pandas_profiling import ProfileReport
from nlp_feature_engineering import convert_transaction_amount

# Load train dataset
transactions_df = pd.read_csv('data/train-dataset.csv')

# Get dataset report
train_report = ProfileReport(transactions_df, title='Train Data Report')

# Save train report
train_report.to_file("train_data.html")

# Load test dataset
scoredata_df = pd.read_csv('data/scorecard.csv')

# Get dataset report
test_report = ProfileReport(scoredata_df, title='Test Data Report')

# Save test report
test_report.to_file("test_data.html")

#######################################################################
# TRAIN DATASET

# Train dataset columns
transactions_df.columns

# Convert transaction amount into transaction type(categorical values)
transactions_df['transaction_type'] = convert_transaction_amount(transactions_df['transaction_amount'])

# Categorical columns that neeed to be one-hot-encoded:

# transaction_type 
transactions_df['transaction_type'].value_counts()
# => credit is 2 times more frequent than debit, but we have enough of both values

# transaction_account_type
transactions_df['transaction_account_type'].value_counts()
# => we have next classes:
# transaction (the most frequent)
# savings
# credit-card
# term-deposit
# mortgage
# loan
# investment

######################################################################
# TEST DATASET

# Test dataset columns
scoredata_df.columns

# Convert transaction amount into transaction type(categorical values)
scoredata_df['transaction_type'] = convert_transaction_amount(scoredata_df['transaction_amount'])

# Categorical columns that neeed to be one-hot-encoded:

# transaction_type 
scoredata_df['transaction_type'].value_counts()
# => we have only 9 debit data

# transaction_account_type
scoredata_df['transaction_account_type'].value_counts()
# => we have next classes:
# transaction (the most frequent)
# savings
# credit-card
# mortgage
# loan
# => we don't have 'term-deposit' and 'investment' categories like in train dataset
# => WE NEED TO BE CAREFULL WHEN DOING ONE-HOT-ENCODING!



