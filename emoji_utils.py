import csv
import numpy as np
import emoji

emoji_dictionary = {"0": "\u2764\uFE0F",    # may print black heart also
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

# Converts a given label (int or string) to the  corresponding emoji code (string)
def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


# for loading the glove word embedding matrix values
def load_glove(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as file:
        # unique words
        words = set()
        word_to_vec = {}
        # each line starts with a word then the values for the different features
        for line in file:
            line = line.strip().split()
            # take the word 
            curr_word = line[0]
            words.add(curr_word)
            # rest of the features for the word
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
        
        # make a dict mapping 
        i = 1
        words_to_index = {}
        index_to_words = {}

        for word in sorted(words):
            words_to_index[word] = i
            index_to_words[i] = word
            i = i + 1

    return words_to_index, index_to_words, word_to_vec


# converts a given sentence to a vector of numerical indices
def sentence_to_indices(X, word_to_index, max_len):
    # number of training examples
    m = X.shape[0]                                   
    
    # Initialize a numpy matrix of zeros
    X_indices = np.zeros((m, max_len))
    
    # do this for all examples
    for i in range(m):                               
        # convert the ith sentence to a list of lower cased words
        words = X[i].lower().split()
        
        j = 0
        # Loop over the words of sentence_words
        for w in words:
            # set the ith word to its index 
            X_indices[i, j] = word_to_index[w]
            j = j + 1
            
    return X_indices


# for loading the csv data
def load_csv(filename):
    sentence = []
    emoji = []

    # open the csv and separate the text sentences and emoji
    with open(filename) as csv_file:
        csv_data = csv.reader(csv_file)

        for row in csv_data:
            sentence.append(row[0])
            emoji.append(row[1])

    X = np.asarray(sentence)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

# for converting to One hot Encoding
def convert_to_OHE(Y, C):
    return np.eye(C)[Y.reshape(-1)]
    
