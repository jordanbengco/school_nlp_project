"""
Contains a continuous bag of words implementation of word2vec
Order of operation:
    - Create a word2vec class (w/ argurments: filename of corpus)
    - Get the Weight vectors W1, W2 by running (W1, W2, loss = cbow.run())
    - Make a set of reviews to predict -- ([[review1], [review2], ..., [review n]]) 
    or by running reviews_list = cbow.corpus[0:10]
    - Run get_predictions with the arguements created from above. (reviews_list, cbow_model, W1, W2)
######## This program is copied and adapted from claudiobellei.com/2018/01/07/backprop-word2vec-python/ ##### 

Possible Issues: Does not deal with reviews that contain words that does not exist within its vocabulary
"""

import pandas as pd
import numpy as np
import string
import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing import sequence

class Word2vec:
    def __init__(self, window_size=1, n_hidden=2, epochs=1, corpus_name='', learning_rate=0.1, test_len=0):
        self.window = window_size
        self.N = n_hidden
        self.epochs = epochs
        self.corpus, self.rating = get_corpus(corpus_name) ## full text = corpus.reshape((1,))
        self.eta = learning_rate
        self.vocab, self.vocab_size, self.tokenizer = get_vocab(self.corpus.reshape((-1,)))
        self.test = test_len
        self.test_corpus = self.corpus[0:self.test-1]
        self.test_rating = self.rating[0:self.test-1]
        
    def cbow(self, context, label, W1, W2, loss):
        x = np.mean(context.reshape(-1, context.shape[-1]), axis=0)
        h = np.dot(W1.T, x.reshape(x.shape[0],1))
        u = np.dot(W2.T, h)
        y_pred = softmax(u)
        
        e = -label.reshape(-1,1) + y_pred
        dW2 = np.outer(h,e)
        dW1 = np.outer(x.reshape(x.shape[0],1), np.dot(W2, e))
        new_W1 = W1 - (self.eta * dW1)
        new_W2 = W2 - (self.eta * dW2)
        
        loss += -float(u[label.T == 1]) + np.log(np.sum(np.exp(u)))
        
        return new_W1, new_W2, loss
    
    def run(self):
        # initialize W1, W2, loss array
        np.random.seed(100)
        W1 = np.random.rand(self.vocab_size, self.N)
        W2 = np.random.rand(self.N, 2)
        loss_vs_epoch = []
        train_s = self.test
        train_e = len(self.corpus)-1
        
        for e in range(self.epochs):
            loss = 0.
            for review in range(len(self.corpus)):
                context, label = get_context_word(self.vocab[train_s:train_e], self.rating[train_s:train_e], self.vocab_size,
                                                  self.window)
                W1, W2, loss = self.cbow(context, label, W1, W2, loss)
            loss_vs_epoch.append(loss)
        return W1, W2, loss_vs_epoch
    
    def predict(self, x, W1, W2):
        # returns a vector representation
        h = np.mean([np.matmul(W1.T, xx) for xx in x], axis=0)
        u = np.dot(W2.T, h)
        
        return softmax(u)
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_corpus(filename):
    # Takes a json file that is in a format of df[rating, review] and returns 
    # the reviews and ratings
    df_reviews = pd.read_json(filename)
    reviews = df_reviews['text'].tolist()
    context = df_reviews['stars'].tolist()
        
    return np.reshape(reviews, (len(reviews), 1)), context

def rate_star(stars):
    # Treats any review with 4 or 5 stars good and anything lower is bad.
    if stars > 3:
        return np.array([1, 0])
    else: # stars <= 3
        return np.array([0, 1])

def get_vocab(text_file):
    # Takes the all the reviews and returns:
    # word2id : the reviews but in its id representation
    # vocab_size : vocabulary size
    # tokenizer : the tokenizer that has the id:word dictionary 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_file)
    word2id = tokenizer.texts_to_sequences(text_file)
    
    vocab_size = len(tokenizer.word_index)
    return word2id, vocab_size, tokenizer

def get_context_word(corpus, rating, vocab_size, window_size):
    # Takes review that has it's words represented as id and returns one-hot-vector 
    # representation of each word.
    for i, words in enumerate(corpus):
        length = len(words)
        contexts = []
        
        contexts.append([words[i]-1 for i in range(0, length)])
        
        # Transform it into one-hot vectors representation
        x = np_utils.to_categorical(contexts, vocab_size)
        y = rate_star(rating[i])
            
        return x, y
    
def vectorize_words(words, vocab_size, tokenizer):    
    # Take any review (list of words) and transform to the one-hot vector representation
    # TODO: Remove any words that does not exist in the vocab.
    
    wordid = tokenizer.texts_to_sequences(words)
    x = np_utils.to_categorical(wordid, vocab_size)
    return x

def get_predictions(reviews_list, cbow_model, W1, W2):
    # Takes a list of reviews and returns a list of predictions (vectors).
    res = []
    for review in reviews_list:
        one_hot_review = vectorize_words(review, cbow_model.vocab_size, cbow_model.tokenizer)
        rating = cbow_model.predict(np.array(one_hot_review).reshape(-1, cbow_model.vocab_size),
                                    W1,
                                    W2)
        res.append(rating.tolist())
    return res
 
            
def save_weights(weight, output):
    df_W = pd.DataFrame(weight)
    df_W.to_csv(output)
    
def csv_weights(filename):
    df_W = pd.read_csv(filename)
    return df_W.to_numpy()

    
    
