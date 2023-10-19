"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_list = {}
    count_w_t = {}
    output = []
    greatest_tag = float('-inf')
    chosen_tag = None
    often_tag = None
    count = 0
    next_count = 0
    
    # goes through train set to figure out how often each tag occurs for each word
    for each_sentence in train:
        for each_word in each_sentence:
            word, tag = each_word

            # skips the START and END words since that's just there to help indicate a sentence
            if word == 'START' or word == 'END':
                continue

            # if the word is not in the dict, then add it and add the tag with it
            if word not in count_w_t:
               count_w_t[word] = {}
               count_w_t[word][tag] = 1
               if count < 5:
                   print("LETS SEE: ", count_w_t)
                   count += 1    

            # if the tag is not in the dict of the word, then add it
            if (word in count_w_t and tag not in count_w_t[word]):
                count_w_t[word][tag] = 1
            else:
                count_w_t[word][tag] += 1
            
            # list of tags part
            # counts the occurences of each tag
            if tag in tag_list:
                tag_list[tag] += 1
            else:
                tag_list[tag] = 1
                if next_count < 5:
                    print("NEXT: ", tag_list)
                    next_count += 1
    
    # used for unseen words - calculates the most common tag
    for tag in tag_list:
        if tag_list[tag] > greatest_tag:
            greatest_tag = tag_list[tag]
            often_tag = tag

    print("TAG: ", often_tag)

    # goes through test set to assign a tag for each word
    for sentences in test:
        word_tag_pair = []
        for words in sentences:
            if words in count_w_t:
                # seen words
                #for tagging in count_w_t[words]:
                  #  if count_w_t[words][tagging] > greatest_num:
                    #    greatest_num = count_w_t[words][tagging]
                    #    chosen_tag = tagging
                chosen_tag = max(count_w_t[words], key=count_w_t[words].get)
                inserting = tuple((words, chosen_tag)) 
                word_tag_pair.insert(len(word_tag_pair), inserting)
            else:
                # unseen words
                inserting = tuple((words, often_tag))
                word_tag_pair.insert(len(word_tag_pair), inserting) 

        # allows the output to be a list of sentences
        output.insert(len(output), word_tag_pair)
    
    #print("OUTPUT: ", output)
    return output
            
    # return []