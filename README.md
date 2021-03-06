# Enrich-Machine-Learning-Challenge


## Context

This dataset contains 100k transactions. A transaction is consisted of:
- transaction_description: string describing a transaction
- transaction_amount: the amount spent/received
- transaction_account_type: the type of account used in the transaction
The goal is to train the model that will be used to predict transaction_class for a given transaction.

## Project

The solution is comprised of two general approaches Machine Learning and Deep Learning.

### Machine Learning
Code is divided into several scripts:
- Brief data analysis: data_analysis.py
- NLP feature processing: nlp_feature_engineering.py
- Feature transformation: feature_transformation.py
- Model training and validation: model_training.py, models.py and training_validation.py
- Predicting and applying models to unseen data: predicting.py and testing.py

### Deep Learning
- Multilayer Perceptron Multi-Classification: ANN_basiq.ipynb
- Convolutional Neural Networks: CNN_basiq.ipynb

## Summary

Used Machine Learning models:
- Logistic Regression
- Naive Bayes
- SVM
- Random Forest
- Gradient Boosting

BOW + (TF, TF-IDF) is used for word representation

Used Deep Learning models:
- MLP 
- CNN 
 
Word embedding is used for word representation

As input, models use textual data and categorical data. Categorical data is transformed using One-Hot-Encoding technique or merged together with textual data.

## Created with
- Python 3.7
- Spyder
- Google Colab (Jupyter Notebook)
