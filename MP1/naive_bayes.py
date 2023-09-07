# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
import numpy as npy
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    print(f"File Path: {trainingdir}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.01, pos_prior=0.80, silently=False):
    yhats = []
    neg_prior = 1 - pos_prior

    # trains the naive bayes algorithm by going through the training set data
    negative_known_prob, negative_unknown_prob = trainingPhase(train_set, train_labels, laplace, 0)
    positive_known_prob, positive_unknown_prob = trainingPhase(train_set, train_labels, laplace, 1)
    
    # goes through the development set to calculate whether the reviews are positive or negative
    for individual_set in dev_set:
        positive_words_prob = 0
        negative_words_prob = 0

        positive_words_prob += npy.log(pos_prior)
        negative_words_prob += npy.log(neg_prior)

        # calculates the probability of each review being positive or negative
        # takes the log function to prevent underflow
        for words in individual_set:
            if words in negative_known_prob:
                negative_words_prob += npy.log(negative_known_prob[words])
            else:
                negative_words_prob += npy.log(negative_unknown_prob)
            
            if words in positive_known_prob:
                positive_words_prob += npy.log(positive_known_prob[words])
            else:
                positive_words_prob += npy.log(positive_unknown_prob)

        # appends a 1 for positive reviews, 0 for negative reviews
        if (positive_words_prob >= negative_words_prob):
            yhats.append(1)
        else:
            yhats.append(0)
    
    return yhats


def trainingPhase(train_set, train_labels, laplace_smoothing, type):
    review_num = len(train_labels)
    words = {}
    known_prob = {}
    unknown_prob = 0
    total_words = 0
    total_types = 0

    # calculates the number of words for each word in a positive or negative review - stored in list "words"
    # example -> like: 15, hate: 17, sad: 3
    for i in range(review_num):
        if (train_labels[i] == type):
            for train_word in train_set[i]:
                if train_word in words:
                    words[train_word] += 1
                else:
                    words[train_word] = 1

    # calculates the probability of a word while applying laplace smoothing equation
    # doing this for unknown probability, which are all the unseen words, and for the known probability, which are all the seen words
    total_types = len(words)
    for word_counter in words:
        total_words += words[word_counter]

    for word_counter_new in words:
        known_prob[word_counter_new] = (words[word_counter_new] + laplace_smoothing) / (total_words + (laplace_smoothing * (total_types + 1)))

    unknown_prob = laplace_smoothing / (total_words + (laplace_smoothing * (total_types + 1)))

    return known_prob, unknown_prob

