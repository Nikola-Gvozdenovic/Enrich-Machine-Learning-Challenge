#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:16:25 2021

@author: ngvozdenovic
"""


# Import libraries and functions 
from models import logistic_regression, naive_bayes, svm, random_forest, gradient_boosting
from feature_transformation import transform_input
from model_training import load_data, split_data, train_model, write_save_better_model, save_transformer
from sklearn.preprocessing import LabelEncoder
from feature_transformation import vectorize, encode


# Load transactions data
transactions_df = load_data('data/train-dataset.csv')

# Split data into training and test datasets
X_train, X_test, y_train, y_test = split_data(transactions_df, 0.2)

# Save data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Convert classes into numerical representation, not necessary
output_encoder = LabelEncoder()
output_encoder.fit(y_train)
y_train = output_encoder.transform(y_train)
y_test = output_encoder.transform(y_test)

# Save output_encoder
save_transformer(output_encoder, 'train_label')

# Prepare transformations to apply on features (TF)
transformer_tf = vectorize(transformation_type='tf')

# Prepare transformations to apply on features (TF-IDF)
transformer_tfidf = vectorize('tfidf')

# Fit all transformers
transformer_tf.fit(X_train['description_processed'])
transformer_tfidf.fit(X_train['description_processed'])

# Save all transformers
save_transformer(transformer_tf, 'train_tf')
save_transformer(transformer_tfidf, 'train_tfidf')

# one-hot-encoding categorical features
onehot_features_train = encode(X_train)
onehot_features_test = encode(X_test)

# Apply transformations on input dataset
X_train_tf = transform_input(X_train, onehot_features_train, 'train_tf', 'description_processed')
X_train_tfidf = transform_input(X_train, onehot_features_train, 'train_tfidf', 'description_processed')
X_test_tf = transform_input(X_test, onehot_features_test, 'train_tf', 'description_processed')
X_test_tfidf = transform_input(X_test, onehot_features_test, 'train_tfidf', 'description_processed')

# Training ML algorithms:
# 1. Logistic regression
# 2. Naive Bayes
# 3. Support Vector Machine (SVM)
# 4. Random Forest
# 5. Gradient Boosting

# LOGISTIC REGRESSION
model_1_params = train_model(logistic_regression, X_train_tf, y_train, X_test_tf, y_test)
model_2_params = train_model(logistic_regression, X_train_tfidf, y_train, X_test_tfidf, y_test)
models = [model_1_params, model_2_params]
write_save_better_model(models, 'logistic_regression')

# NAIVE BAYES
model_1_params = train_model(naive_bayes, X_train_tf, y_train, X_test_tf, y_test)
model_2_params = train_model(naive_bayes, X_train_tfidf, y_train, X_test_tfidf, y_test)
models = [model_1_params, model_2_params]
write_save_better_model(models, 'naive_bayes')

# SVM
model_1_params = train_model(svm, X_train_tf, y_train, X_test_tf, y_test)
model_2_params = train_model(svm, X_train_tfidf, y_train, X_test_tfidf, y_test)
models = [model_1_params, model_2_params]
write_save_better_model(models, 'svm')

# RANDOM FOREST
model_1_params = train_model(random_forest, X_train_tf, y_train, X_test_tf, y_test)
model_2_params = train_model(random_forest, X_train_tfidf, y_train, X_test_tfidf, y_test)
models = [model_1_params, model_2_params]
write_save_better_model(models, 'random_forest')

# GRADIENT BOOSTING
model_1_params = train_model(gradient_boosting, X_train_tf, y_train, X_test_tf, y_test)
model_2_params = train_model(gradient_boosting, X_train_tfidf, y_train, X_test_tfidf, y_test)
models = [model_1_params, model_2_params]
write_save_better_model(models, 'gradient_boosting')






