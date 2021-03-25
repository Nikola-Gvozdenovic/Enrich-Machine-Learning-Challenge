#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 09:41:51 2021

@author: ngvozdenovic
"""

# set working directory
import os

os.chdir('/home/ngvozdenovic/NIKOLA/BASIQ/enrich_machine_learning_challenge')

# Import libraries
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from appos.appos import appos_dict
from slangs.slangs import slangs_dict
from autocorrect import Speller
from unicodedata import normalize


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# NLP preprocessing #
#####################

def to_lower(text):
    """Convert text into lowercase.
       
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    return text.str.lower()



def tokenize_date(text):
    """Replace text that represents date with DATE token.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    # Patterns for date are:
    # 1. date: dd/mm/yyyy
    # 2. date dd month yyyy
    date_pattern_1 = "(date(:)?(\\s)?)?[\\d+]{1,2}[- /.][\\d+]{1,2}[- /.][\\d+]{2,4}"
    date_pattern_2 = "(date(:)?(\\s)?)?\\d{1,2}\\s+(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\\s+\\d{2,4}"
    
    text = text.str.replace(date_pattern_1, "date")
    text = text.str.replace(date_pattern_2, "date")

    return text



def tokenize_time(text):
    """Replace text that represents time with TIME token.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    # Patterns for time are:
    time_pattern = "(time\\s)?(\\d{2}:\\d{2})(am|pm)?"
    
    text = text.str.replace(time_pattern, "time")
    
    return text



def replace_abbreviations(text):
    """Replace frequent abbreviations with original word.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """

    abbreviations_dict = {
        "w/d": "withdrawal",
        "trf": "transfer",
        "tfr": "transfer",
        "tfer": "transfer",
        "pymt": "payment",
        "o/s": "outstandingbalance",
        "r/w": "residentwithholding"
    }
    return text.replace(abbreviations_dict, regex=True)



def remove_repeated_characters(text):
    """Remove repeated characters (>2) in words to max limit of 2.
    
       Example: I am verrry happpyyy today => I am verry happyy today
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    regex_pattern = r'(.)\1+'
    text = text.str.replace(regex_pattern, r'\1\1')
    
    return text



def separate_digit_text(text):
    """Separate digit and words with space in text.
       
       Example: I will be booking tickets for 2adults => I will be booking tickets for 2 adults
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    pattern_1 = r'([\d]+)([a-zA-Z]+)([\d]+)?'
    pattern_2 = r'([a-zA-Z]+)([\d]+)([a-zA-Z]+)?'
    text = text.str.replace(pattern_1, r'\1 \2 \3')
    text = text.str.replace(pattern_2, r'\1 \2 \3')
    
    return text



def clean_words(text):
    """Remove unnecessary characters from words.
       
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    pattern_1 = r'([\\*-/]+)?([a-zA-Z]+)([\\*-/]+)?'
    pattern_2 = r'([a-zA-Z]+)([\\*-/]+)([a-zA-Z]+)?'
    text = text.str.replace(pattern_1, r'\2')
    text = text.str.replace(pattern_2, r'\1 \3')
    
    return text



def remove_punctuations(text):
    """Remove special characters from text
    
       Input: pd.Series
       
       Output: pd.Series
    """
      
    regex_pattern = r'[\,+\:\?\!\"\(\)!\'\.\%\[\]]+'
    text = text.str.replace(regex_pattern, r' ')
    text = text.str.replace('-', '')
    
    return text



def remove_stopwords(text):
    """Remove stopwords (from the NTLK package).
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """

    # Load the stop words in english
    stop_words = list(stopwords.words('english'))

    # remove stopwords
    return text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))



def remove_numbers(text):
    """Remove stand-alone numbers.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    return text.str.replace("[0-9]{1,}", " ")



def remove_short_words(text):
    """Remove words with one or two letters.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    return text.str.replace("\\b[\\w]{1,2}\\b", " ")



def remove_extra_whitespace(text):
    """Remove unnecessary whitespace.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    return text.str.replace("\\s+", " ")



def remove_special_character(text, special_characters=None):
    """Remove special characters.
    
       Input:
           text: pd.Series
           special_characters: string
           
       Output:
           text: pd.Series
    """
    
    if special_characters is None:
        special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'
    text = text.apply(lambda x: x.translate(str.maketrans('', '', special_characters)))
    return text



def remove_card(text):
    """Remove word card from text. Card numbers are removed with other functions.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    pattern_1 = "(card)?(\\s+)?[0-9]{1,10}(\\w+)?(\*+)?[0-9]{1,10}"
    pattern_2 = "card"
    text = text.str.replace(pattern_1, " ")
    text = text.str.replace(pattern_2, " ")
    
    return text



def appos_look_up(text):
    """Convert apostrophes word to original form.
    
       Example: I don't know what is going on?  => I do not know what is going on? 
    
       Input: 
           text: pd.Series
        
       Output:
           text: pd.Series (text with converted apostrophes)
    """
    
    return text.apply(lambda row: " ".join([appos_dict[word] if word in appos_dict else word for word in row.split()]))



def slang_look_up(text):
    """Replace slang word in text to their original form.
    
       Example: hi, thanq so mch => hi, thank you so much
    
       Input: 
           text: pd.Series
    
       Output:
           text: pd.Series (cleaned text with replaced slang)
    """
    
    return text.apply(lambda row: " ".join([slangs_dict[word] if word in slangs_dict else word for word in row.split()]))



def normalize_unicode(text):
    """Normalize unicode data to remove umlauts, and accents, etc. 
       
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    return text.apply(lambda row: normalize('NFKD', row).encode('ASCII', 'ignore').decode('utf8'))



def remove_email(text):
    """Remove email in the input text. 
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    regex_pattern = '[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'
    
    return text.str.replace(regex_pattern, ' ')



def remove_phone_number(text):
    """Remove phone number in the input text.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    regex_pattern = '(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?'
    
    return text.str.replace(regex_pattern, ' ')


def remove_url(text):
    """Remove urls from text.
       
       Input: pd.Series
       
       Output: pd.Series
    """

    pattern_1 = "(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*"
    pattern_2 = "\w{1,}[.][com|html]"
    
    text = text.str.replace(pattern_1, " ")
    text = text.str.replace(pattern_2, " ")
    
    return text


def autospell(text):
    """Correct the spelling of the word.
    
       Input:
           text: pd.Series
        
       Output:
           text: pd.Series
    """
    
    spell = Speller()
    word_tokenizer = WhitespaceTokenizer()
    return text.apply(lambda row: " ".join([spell(w) for w in word_tokenizer.tokenize(row)]))



def remove_same_letter_words(text):
    """Remove words consisting of the same letter.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """
    
    word_tokenizer = WhitespaceTokenizer()
    return text.apply(lambda row: " ".join([word for word in word_tokenizer.tokenize(row) if len(set(word)) > 1]))



def lemmatization(text: pd.Series):
    """Perform lematization over the text.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
    """

    word_tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()

    return text.apply(lambda row: " ".join([lemmatizer.lemmatize(token) for token in word_tokenizer.tokenize(row)]))



def stemming(text, stemmer=None):
    """Stem each token in a text.
    
       Input:
           text: pd.Series
           stemmer: nltk.stem object
    """
    
    if stemmer is None:
        stemmer = PorterStemmer()
        
    word_tokenizer = WhitespaceTokenizer()
        
    return text.apply(lambda row: " ".join([stemmer.stem(token) for token in word_tokenizer.tokenize(row)]))


def nlp_preprocessing(text):
    """Apply preprocessing steps on textual data.
    
       Input:
           text: pd.Series
           
       Output:
           text: pd.Series
           
    """

    return text \
    .pipe(to_lower) \
    .pipe(remove_url) \
    .pipe(remove_email) \
    .pipe(remove_punctuations) \
    .pipe(tokenize_date) \
    .pipe(tokenize_time) \
    .pipe(remove_card) \
    .pipe(separate_digit_text) \
    .pipe(remove_numbers) \
    .pipe(replace_abbreviations) \
    .pipe(clean_words) \
    .pipe(remove_repeated_characters) \
    .pipe(remove_phone_number) \
    .pipe(remove_short_words) \
    .pipe(remove_same_letter_words) \
    .pipe(remove_special_character) \
    .pipe(appos_look_up) \
    .pipe(slang_look_up) \
    .pipe(normalize_unicode) \
    .pipe(lemmatization) \
    .pipe(remove_stopwords) \
    .pipe(remove_extra_whitespace)

