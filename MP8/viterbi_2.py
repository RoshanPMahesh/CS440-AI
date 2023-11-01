"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

# def viterbi_2(train, test):
#     '''
#     input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#             test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
#     output: list of sentences, each sentence is a list of (word,tag) pairs.
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''
#     return []









"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)

    count = 0
    tag_count = 0
    num_h_tags = 0
    allTags = {}
    curr_next_tag = {}
    allWords = {}
    h_tags = {}

    for each_sentence in sentences:
        sentence_num = -1

        for each_word in each_sentence:
            if sentence_num != len(each_sentence) - 2:
                curr_tag = each_sentence[sentence_num][1]
                next_tag = each_sentence[sentence_num + 1][1]
            
                if curr_tag not in curr_next_tag:
                    curr_next_tag[curr_tag] = {}

                if next_tag not in curr_next_tag[curr_tag]:
                    curr_next_tag[curr_tag][next_tag] = 1
                else:
                    curr_next_tag[curr_tag][next_tag] += 1
                
            word, tag = each_word
            if word not in allWords:
                allWords[word] = 1
            else:
                allWords[word] += 1

            if tag not in allTags:
                allTags[tag] = {}
            
            if word not in allTags[tag]:
                allTags[tag][word] = 1
            else:
                allTags[tag][word] += 1

            sentence_num += 1

    for each_sentence in sentences:
        for each_word, each_tag in each_sentence:
            if allWords[each_word] == 1:
                if each_tag not in h_tags:
                    h_tags[each_tag] = 1
                    num_h_tags += 1
                else:
                    h_tags[each_tag] += 1
                    num_h_tags += 1

    h_prob = {}
    for each_tag in allTags:
        h_prob[each_tag] = emit_epsilon * (h_tags.get(each_tag, 0) + emit_epsilon) / (num_h_tags + emit_epsilon * (len(allTags) - 2))

    for each_tag in allTags:
        unique_words = 0
        unique_words_tags = 0
        total_words = 0
        total_words_tags = 0

        init_prob[each_tag] = epsilon_for_pt

        for tags in curr_next_tag[each_tag]:
            unique_words_tags += 1
            total_words_tags += curr_next_tag[each_tag][tags]
            if tag_count < 2:
                print("TAGS: ", curr_next_tag[each_tag][tags])
                tag_count += 1

        for each_word in allTags[each_tag]:
            unique_words += 1
            total_words += allTags[each_tag][each_word]
            if count < 2:
                print("BRO PLZ WORK: ", allTags[each_tag][each_word])
                count += 1

        if h_prob[each_tag] == (emit_epsilon) / (num_h_tags + emit_epsilon * len(allTags)):
            scale = h_prob[each_tag]
        else:
            scale = emit_epsilon * h_prob[each_tag]
        emit_prob[each_tag]["UNK"] = scale / (total_words + (emit_epsilon * unique_words))

        for word in allTags[each_tag]:
            Pe = (allTags[each_tag][word] + emit_epsilon) / (total_words + (emit_epsilon * unique_words))
            if Pe == 0:
                emit_prob[each_tag][word] = 0.0001
            else:
                emit_prob[each_tag][word] = Pe

        for unknown_tag in allTags:
            if unknown_tag not in curr_next_tag[each_tag]:
                trans_prob[each_tag][unknown_tag] = epsilon_for_pt / (total_words_tags + (epsilon_for_pt * unique_words_tags))

        for tag_iterate in curr_next_tag[each_tag]:
            Pt = (curr_next_tag[each_tag][tag_iterate] + epsilon_for_pt) / (total_words_tags + (epsilon_for_pt * unique_words_tags))
            if Pt == 0:
                trans_prob[each_tag][tag_iterate] = 0.0001
            else:
                trans_prob[each_tag][tag_iterate] = (curr_next_tag[each_tag][tag_iterate] + epsilon_for_pt) / (total_words_tags + (epsilon_for_pt * unique_words_tags))

    return init_prob, emit_prob, trans_prob



def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    
    # DONT CARE FOR i = 0 CASE SINCE I SKIP START
    if i >= 0:
        count = 0
        best_prob = -np.inf
        best_tag = None
        prob = 0

        # iterating through the tags in the emission probabilities and the tags from the previous column of lattice
        for each_tag in emit_prob:
            for tags in prev_prob:
                prob = 0

                # adds the probability from the tag in the previous column of lattice
                prob += prev_prob[tags]

                # checks if the inputted word is a known tag
                # if not, then we just give the unknown tag probability
                if word not in emit_prob[each_tag]:
                    prob += log(emit_prob[each_tag]["UNK"])
                else:
                    prob += log(emit_prob[each_tag][word])

                # checks if the tag in the previous column of lattice exists in the transition probabilities list
                # if it does, then find the probability to transition to a word given that tag
                if tags not in trans_prob:
                    prob += log(0.0001)
                elif each_tag not in trans_prob[tags]:
                    prob += log(0.0001)
                else:
                    prob += log(trans_prob[tags][each_tag])

                # take the tag and the probability that are highest
                if prob > best_prob:
                    best_prob = prob
                    best_tag = tags
            
            log_prob[each_tag] = best_prob
            predict_tag_seq[each_tag] = prev_predict_tag_seq[best_tag] + each_tag.split()
            best_prob = -np.inf
            best_tag = None

    return log_prob, predict_tag_seq

def viterbi_2(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    prediction = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        sentence_range = range(len(sentence))
        best_tag = None
        best_prob = -np.inf
        for tags in log_prob:
            if log_prob[tags] > best_prob:
                best_tag = tags
                best_prob = log_prob[tags]

        # why dont u work
        #for num in sentence_range:
           # prediction.insert(len(prediction), (sentence[num], predict_tag_seq[best_tag][num]))
            #predicts.insert(len(predicts), prediction)

        # realized fix through this website: https://www.w3schools.com/python/python_lists_comprehension.asp
        prediction = [tuple((sentence[num], predict_tag_seq[best_tag][num])) for num in sentence_range]
        predicts.insert(len(predicts), prediction)

    return predicts

