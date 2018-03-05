import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
    



              
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
    
def predict(X, Y, W, b, word_to_vec):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average words' vectors
        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec[w]
        avg = avg/len(words)

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred
