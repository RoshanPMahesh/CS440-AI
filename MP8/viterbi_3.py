"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

# def viterbi_3(train, test):
#     '''
#     input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#             test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
#     output: list of sentences, each sentence is a list of (word,tag) pairs.
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''
#     return []



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
    #emit_prob = {}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)

    count = 0
    tag_count = 0
    num_h_tags = 0
    num_h_ing_tags = 0
    num_h_ly_tags = 0
    num_h_ed_tags = 0
    num_h_ity_tags = 0
    num_h_s_tags = 0
    num_h_ment_tags = 0
    num_h_hyph_tags = 0
    num_h_un_tags = 0
    num_h_ful_tags = 0
    num_h_er_tags = 0
    num_h_plu_tags = 0
    num_h_tion_tags = 0
    num_h_ive_tags = 0
    num_h_less_tags = 0
    num_h_al_tags = 0
    num_h_able_tags = 0
    num_h_ance_tags = 0
    num_h_or_tags = 0
    num_h_ate_tags = 0
    num_h_en_tags = 0
    num_h_fy_tags = 0
    num_h_es_tags = 0
    num_h_send_tags = 0
    num_h_num_tags = 0
    num_h_ist_tags = 0
    num_h_us_tags = 0
    num_h_dol_tags = 0
    allTags = {}
    curr_next_tag = {}
    allWords = {}   
    h_tags = {}
    h_ing_tags = {}
    h_ly_tags = {}
    h_ed_tags = {}
    h_ity_tags = {}
    h_s_tags = {}
    h_ment_tags = {}
    h_hyph_tags = {}
    h_un_tags = {}
    h_ful_tags = {}
    h_er_tags = {}
    h_plu_tags = {}
    h_tion_tags = {}
    h_ive_tags = {}
    h_less_tags = {}
    h_al_tags = {}
    h_able_tags = {}
    h_ance_tags = {}
    h_or_tags = {}
    h_ate_tags = {}
    h_en_tags = {}
    h_fy_tags = {}
    h_es_tags = {}
    h_send_tags = {}
    h_num_tags = {}
    h_ist_tags = {}
    h_us_tags = {}
    h_dol_tags = {}


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

    # nested for loop is for hapax stuff, keeping track of tags that are associated with words that occur once
    # also keep track of total number of hapax words
    for each_sentence in sentences:
        for each_word, each_tag in each_sentence:
            if allWords[each_word] == 1:

                if each_word.endswith("ing"):
                    num_h_ing_tags += 1
                    if each_tag not in h_ing_tags:
                        h_ing_tags[each_tag] = 1
                    else:
                        h_ing_tags[each_tag] += 1

                if each_word.endswith("ly"):
                    num_h_ly_tags += 1
                    if each_tag not in h_ly_tags:
                        h_ly_tags[each_tag] = 1
                    else:
                        h_ly_tags[each_tag] += 1

                if each_word.endswith("ed"):
                    num_h_ed_tags += 1
                    if each_tag not in h_ed_tags:
                        h_ed_tags[each_tag] = 1
                    else:
                        h_ed_tags[each_tag] += 1

                if each_word.endswith("ity"):
                    num_h_ity_tags += 1
                    if each_tag not in h_ity_tags:
                        h_ity_tags[each_tag] = 1
                    else:
                        h_ity_tags[each_tag] += 1

                if each_word.endswith("'s"):
                    num_h_s_tags += 1
                    if each_tag not in h_s_tags:
                        h_s_tags[each_tag] = 1
                    else:
                        h_s_tags[each_tag] += 1

                if each_word.endswith("ment"):
                    num_h_ment_tags += 1
                    if each_tag not in h_ment_tags:
                        h_ment_tags[each_tag] = 1
                    else:
                        h_ment_tags[each_tag] += 1

                if "-" in each_word:
                    num_h_hyph_tags += 1
                    if each_tag not in h_hyph_tags:
                        h_hyph_tags[each_tag] = 1
                    else:
                        h_hyph_tags[each_tag] += 1

                if each_word.startswith("un"):
                    num_h_un_tags += 1
                    if each_tag not in h_un_tags:
                        h_un_tags[each_tag] = 1
                    else:
                        h_un_tags[each_tag] += 1

                if each_word.endswith("ful"):
                    num_h_ful_tags += 1
                    if each_tag not in h_ful_tags:
                        h_ful_tags[each_tag] = 1
                    else:
                        h_ful_tags[each_tag] += 1

                if each_word.endswith("er"):
                    num_h_er_tags += 1
                    if each_tag not in h_er_tags:
                        h_er_tags[each_tag] = 1
                    else:
                        h_er_tags[each_tag] += 1

                if each_word.endswith("s'"):
                    num_h_plu_tags += 1
                    if each_tag not in h_plu_tags:
                        h_plu_tags[each_tag] = 1
                    else:
                        h_plu_tags[each_tag] += 1

                if each_word.endswith("ion"):
                    num_h_tion_tags += 1
                    if each_tag not in h_tion_tags:
                        h_tion_tags[each_tag] = 1
                    else:
                        h_tion_tags[each_tag] += 1

                if each_word.endswith("ive"):
                    num_h_ive_tags += 1
                    if each_tag not in h_ive_tags:
                        h_ive_tags[each_tag] = 1
                    else:
                        h_ive_tags[each_tag] += 1

                if each_word.endswith("less"):
                    num_h_less_tags += 1
                    if each_tag not in h_less_tags:
                        h_less_tags[each_tag] = 1
                    else:
                        h_less_tags[each_tag] += 1

                if each_word.endswith("al"):
                    num_h_al_tags += 1
                    if each_tag not in h_al_tags:
                        h_al_tags[each_tag] = 1
                    else:
                        h_al_tags[each_tag] += 1

                if each_word.endswith("able"):
                    num_h_able_tags += 1
                    if each_tag not in h_able_tags:
                        h_able_tags[each_tag] = 1
                    else:
                        h_able_tags[each_tag] += 1

                if each_word.endswith("ance"):
                    num_h_ance_tags += 1
                    if each_tag not in h_ance_tags:
                        h_ance_tags[each_tag] = 1
                    else:
                        h_ance_tags[each_tag] += 1

                if each_word.endswith("or"):
                    num_h_or_tags += 1
                    if each_tag not in h_or_tags:
                        h_or_tags[each_tag] = 1
                    else:
                        h_or_tags[each_tag] += 1

                if each_word.endswith("ate"):
                    num_h_ate_tags += 1
                    if each_tag not in h_ate_tags:
                        h_ate_tags[each_tag] = 1
                    else:
                        h_ate_tags[each_tag] += 1

                if each_word.endswith("en"):
                    num_h_en_tags += 1
                    if each_tag not in h_en_tags:
                        h_en_tags[each_tag] = 1
                    else:
                        h_en_tags[each_tag] += 1

                if each_word.endswith("fy"):
                    num_h_fy_tags += 1
                    if each_tag not in h_fy_tags:
                        h_fy_tags[each_tag] = 1
                    else:
                        h_fy_tags[each_tag] += 1

                if each_word.endswith("es"):
                    num_h_es_tags += 1
                    if each_tag not in h_es_tags:
                        h_es_tags[each_tag] = 1
                    else:
                        h_es_tags[each_tag] += 1

                if each_word.endswith("s"):
                    num_h_send_tags += 1
                    if each_tag not in h_send_tags:
                        h_send_tags[each_tag] = 1
                    else:
                        h_send_tags[each_tag] += 1

                if each_word[0].isnumeric():
                    num_h_num_tags += 1
                    if each_tag not in h_num_tags:
                        h_num_tags[each_tag] = 1
                    else:
                        h_num_tags[each_tag] += 1

                if each_word.endswith("ist"):
                    num_h_ist_tags += 1
                    if each_tag not in h_ist_tags:
                        h_ist_tags[each_tag] = 1
                    else:
                        h_ist_tags[each_tag] += 1

                if each_word.endswith("us"):
                    num_h_us_tags += 1
                    if each_tag not in h_us_tags:
                        h_us_tags[each_tag] = 1
                    else:
                        h_us_tags[each_tag] += 1

                if each_word.startswith("$"):
                    num_h_dol_tags += 1
                    if each_tag not in h_dol_tags:
                        h_dol_tags[each_tag] = 1
                    else:
                        h_dol_tags[each_tag] += 1

                if each_tag not in h_tags:
                    h_tags[each_tag] = 1
                    num_h_tags += 1
                else:
                    h_tags[each_tag] += 1
                    num_h_tags += 1

    # hapax probability calculation
    for each_tag in allTags:
        if each_tag in h_tags:
            h_tags[each_tag] = (h_tags[each_tag] + emit_epsilon) / (num_h_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_tags[each_tag] = emit_epsilon

        if each_tag in h_ing_tags:
            h_ing_tags[each_tag] = (h_ing_tags[each_tag] + emit_epsilon) / (num_h_ing_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ing_tags[each_tag] = emit_epsilon

        if each_tag in h_ly_tags:
            h_ly_tags[each_tag] = (h_ly_tags[each_tag] + emit_epsilon) / (num_h_ly_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ly_tags[each_tag] = emit_epsilon

        if each_tag in h_ed_tags:
            h_ed_tags[each_tag] = (h_ed_tags[each_tag] + emit_epsilon) / (num_h_ed_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ed_tags[each_tag] = emit_epsilon

        if each_tag in h_ity_tags:
            h_ity_tags[each_tag] = (h_ity_tags[each_tag] + emit_epsilon) / (num_h_ity_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ity_tags[each_tag] = emit_epsilon

        if each_tag in h_s_tags:
            h_s_tags[each_tag] = (h_s_tags[each_tag] + emit_epsilon) / (num_h_s_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_s_tags[each_tag] = emit_epsilon

        if each_tag in h_ment_tags:
            h_ment_tags[each_tag] = (h_ment_tags[each_tag] + emit_epsilon) / (num_h_ment_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ment_tags[each_tag] = emit_epsilon

        if each_tag in h_hyph_tags:
            h_hyph_tags[each_tag] = (h_hyph_tags[each_tag] + emit_epsilon) / (num_h_hyph_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_hyph_tags[each_tag] = emit_epsilon

        if each_tag in h_un_tags:
             h_un_tags[each_tag] = (h_un_tags[each_tag] + emit_epsilon) / (num_h_un_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_un_tags[each_tag] = emit_epsilon

        if each_tag in h_ful_tags:
            h_ful_tags[each_tag] = (h_ful_tags[each_tag] + emit_epsilon) / (num_h_ful_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ful_tags[each_tag] = emit_epsilon

        if each_tag in h_er_tags:
            h_er_tags[each_tag] = (h_er_tags[each_tag] + emit_epsilon) / (num_h_er_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_er_tags[each_tag] = emit_epsilon

        if each_tag in h_plu_tags:
            h_plu_tags[each_tag] = (h_plu_tags[each_tag] + emit_epsilon) / (num_h_plu_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_plu_tags[each_tag] = emit_epsilon

        if each_tag in h_tion_tags:
            h_tion_tags[each_tag] = (h_tion_tags[each_tag] + emit_epsilon) / (num_h_tion_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_tion_tags[each_tag] = emit_epsilon

        if each_tag in h_ive_tags:
            h_ive_tags[each_tag] = (h_ive_tags[each_tag] + emit_epsilon) / (num_h_ive_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ive_tags[each_tag] = emit_epsilon

        if each_tag in h_less_tags:
            h_less_tags[each_tag] = (h_less_tags[each_tag] + emit_epsilon) / (num_h_less_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_less_tags[each_tag] = emit_epsilon 

        if each_tag in h_al_tags:
            h_al_tags[each_tag] = (h_al_tags[each_tag] + emit_epsilon) / (num_h_al_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_al_tags[each_tag] = emit_epsilon    

        if each_tag in h_able_tags:
            h_able_tags[each_tag] = (h_able_tags[each_tag] + emit_epsilon) / (num_h_able_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_able_tags[each_tag] = emit_epsilon 

        if each_tag in h_ance_tags:
            h_ance_tags[each_tag] = (h_ance_tags[each_tag] + emit_epsilon) / (num_h_ance_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ance_tags[each_tag] = emit_epsilon

        if each_tag in h_or_tags:
            h_or_tags[each_tag] = (h_or_tags[each_tag] + emit_epsilon) / (num_h_or_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_or_tags[each_tag] = emit_epsilon  

        if each_tag in h_ate_tags:
            h_ate_tags[each_tag] = (h_ate_tags[each_tag] + emit_epsilon) / (num_h_ate_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ate_tags[each_tag] = emit_epsilon

        if each_tag in h_en_tags:
            h_en_tags[each_tag] = (h_en_tags[each_tag] + emit_epsilon) / (num_h_en_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_en_tags[each_tag] = emit_epsilon  

        if each_tag in h_fy_tags:
            h_fy_tags[each_tag] = (h_fy_tags[each_tag] + emit_epsilon) / (num_h_fy_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_fy_tags[each_tag] = emit_epsilon

        if each_tag in h_es_tags:
            h_es_tags[each_tag] = (h_es_tags[each_tag] + emit_epsilon) / (num_h_es_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_es_tags[each_tag] = emit_epsilon

        if each_tag in h_send_tags:
            h_send_tags[each_tag] = (h_send_tags[each_tag] + emit_epsilon) / (num_h_send_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_send_tags[each_tag] = emit_epsilon

        if each_tag in h_num_tags:
            h_num_tags[each_tag] = (h_num_tags[each_tag] + emit_epsilon) / (num_h_num_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_num_tags[each_tag] = emit_epsilon 

        if each_tag in h_ist_tags:
            h_ist_tags[each_tag] = (h_ist_tags[each_tag] + emit_epsilon) / (num_h_ist_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_ist_tags[each_tag] = emit_epsilon

        if each_tag in h_us_tags:
            h_us_tags[each_tag] = (h_us_tags[each_tag] + emit_epsilon) / (num_h_us_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_us_tags[each_tag] = emit_epsilon 

        if each_tag in h_dol_tags:
            h_dol_tags[each_tag] = (h_dol_tags[each_tag] + emit_epsilon) / (num_h_dol_tags + emit_epsilon * (len(allTags) - 2))
        else:
            h_dol_tags[each_tag] = emit_epsilon   


    for each_tag in allTags:
        unique_words = 0
        unique_words_tags = 0
        total_words = 0
        total_words_tags = 0

        init_prob[each_tag] = epsilon_for_pt

        for tags in curr_next_tag[each_tag]:
            unique_words_tags += 1
            total_words_tags += curr_next_tag[each_tag][tags]

        for each_word in allTags[each_tag]:
            unique_words += 1
            total_words += allTags[each_tag][each_word]

        for word in allTags[each_tag]:
            emit_prob[each_tag][word] = (allTags[each_tag][word] + emit_epsilon) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ING"] = (emit_epsilon * h_ing_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-LY"] = (emit_epsilon * h_ly_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ED"] = (emit_epsilon * h_ed_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ITY"] = (emit_epsilon * h_ity_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-'S"] = (emit_epsilon * h_s_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-MENT"] = (emit_epsilon * h_ment_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["-"] = (emit_epsilon * h_hyph_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["UN-X"] = (emit_epsilon * h_un_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-FUL"] = (emit_epsilon * h_ful_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ER"] = (emit_epsilon * h_er_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-S'"] = (emit_epsilon * h_plu_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ION"] = (emit_epsilon * h_tion_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-IVE"] = (emit_epsilon * h_ive_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-LESS"] = (emit_epsilon * h_less_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-AL"] = (emit_epsilon * h_al_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ABLE"] = (emit_epsilon * h_able_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ANCE"] = (emit_epsilon * h_ance_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-OR"] = (emit_epsilon * h_or_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ATE"] = (emit_epsilon * h_ate_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-EN"] = (emit_epsilon * h_en_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-FY"] = (emit_epsilon * h_fy_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-ES"] = (emit_epsilon * h_es_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-S"] = (emit_epsilon * h_send_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["NUM"] = (emit_epsilon * h_num_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-IST"] = (emit_epsilon * h_ist_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["X-US"] = (emit_epsilon * h_us_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["$-X"] = (emit_epsilon * h_dol_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))
            emit_prob[each_tag]["UNK"] = (emit_epsilon * h_tags[each_tag]) / (total_words + (emit_epsilon * unique_words))

        for unknown_tag in allTags:
            if unknown_tag not in curr_next_tag[each_tag]:
                trans_prob[each_tag][unknown_tag] = epsilon_for_pt / (total_words_tags + (epsilon_for_pt * unique_words_tags))

        for tag_iterate in curr_next_tag[each_tag]:
            Pt = (curr_next_tag[each_tag][tag_iterate] + epsilon_for_pt) / (total_words_tags + (epsilon_for_pt * unique_words_tags))
            if Pt == 0:
                trans_prob[each_tag][tag_iterate] = epsilon_for_pt
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
                    if word.endswith("ing"):
                        prob += log(emit_prob[each_tag]["X-ING"])
                    elif word.endswith("ly"):
                        prob += log(emit_prob[each_tag]["X-LY"])
                    elif word.endswith("er"):
                        prob += log(emit_prob[each_tag]["X-ER"])
                    elif word.endswith("or"):
                        prob += log(emit_prob[each_tag]["X-OR"])
                    elif word.endswith("ed"):
                        prob += log(emit_prob[each_tag]["X-ED"])
                    elif word.endswith("s'"):
                        prob += log(emit_prob[each_tag]["X-S'"])
                    elif word.endswith("'s"):
                        prob += log(emit_prob[each_tag]["X-'S"])
                    elif word.endswith("es"):
                        prob += log(emit_prob[each_tag]["X-ES"])
                    elif word.endswith("us"):
                        prob += log(emit_prob[each_tag]["X-US"])
                    elif word.endswith("s"):
                        prob += log(emit_prob[each_tag]["X-S"])
                    elif word.endswith("ate"):
                        prob += log(emit_prob[each_tag]["X-ATE"]) 
                    elif word.endswith("able"):
                        prob += log(emit_prob[each_tag]["X-ABLE"])
                    elif word.endswith("al"):
                        prob += log(emit_prob[each_tag]["X-AL"])
                    elif word.endswith("fy"):
                        prob += log(emit_prob[each_tag]["X-FY"]) # just added
                    elif word.endswith("ance"):
                        prob += log(emit_prob[each_tag]["X-ANCE"])
                    elif word.endswith("ful"):
                        prob += log(emit_prob[each_tag]["X-FUL"])
                    elif word.endswith("ive"):
                        prob += log(emit_prob[each_tag]["X-IVE"])
                    elif word.endswith("ist"):
                        prob += log(emit_prob[each_tag]["X-IST"])
                    elif word.endswith("ion"):
                        prob += log(emit_prob[each_tag]["X-ION"])
                    elif word.endswith("en"):
                        prob += log(emit_prob[each_tag]["X-EN"]) #got screwed
                    elif word.endswith("less"):
                        prob += log(emit_prob[each_tag]["X-LESS"])
                    elif word.startswith("un"):
                        prob += log(emit_prob[each_tag]["UN-X"])
                    elif word.endswith("ity"):
                        prob += log(emit_prob[each_tag]["X-ITY"])
                    elif word.endswith("ment"):
                        prob += log(emit_prob[each_tag]["X-MENT"])
                    elif word[0].isnumeric():
                        prob += log(emit_prob[each_tag]["NUM"])
                    elif word.startswith("$"):
                        prob += log(emit_prob[each_tag]["$-X"])
                    elif "-" in word:
                        prob += log(emit_prob[each_tag]["-"])
                    else:
                        prob += log(emit_prob[each_tag]["UNK"])
                else:
                    prob += log(emit_prob[each_tag][word])

                # checks if the tag in the previous column of lattice exists in the transition probabilities list
                # if it does, then find the probability to transition to a word given that tag
                if tags not in trans_prob:
                    prob += log(epsilon_for_pt)
                elif each_tag not in trans_prob[tags]:
                    prob += log(epsilon_for_pt)
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

def viterbi_3(train, test, get_probs=training):
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

