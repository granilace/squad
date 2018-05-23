from constants import *

from keras.layers import Bidirectional, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, merge, Lambda
from keras import backend as K

def Document_RNN(data):
    RNN_context = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, INPUT_CONTEXT_LEN) ) (data)
    RNN_context = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, HIDDEN_SIZE) ) (RNN_context)
    RNN_context = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, HIDDEN_SIZE) ) (RNN_context)
    return RNN_context

def Question_RNN(data):
    RNN_questions = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, INPUT_QUESTION_LEN)) (data)
    RNN_questions = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, HIDDEN_SIZE)) (RNN_questions)
    RNN_questions = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(None, HIDDEN_SIZE)) (RNN_questions)
    return RNN_questions

def Linear_Attention(data, mask):
    attention = Dense(1)(data)
    attention = tf.pad
    attention = Flatten()(attention)
    # TODO: apply masking
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*HIDDEN_SIZE)(attention)
    attention = Permute([2, 1])(attention)
    
    question_with_attention = merge([data, attention], mode='mul')
    question_with_attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2*HIDDEN_SIZE,))(question_with_attention)
    return question_with_attention

def Bilinear_Attention(context, question, mask_context):
    p_dot_W = Dense(2 * HIDDEN_SIZE) (context)
    # TODO: apply masking
    output = merge([p_dot_W, question], mode='dot', dot_axes=-1)
    return output

def Seq_Attention(context, question, mask_question):

    # Compute scores
    scores = Lambda(lambda x: K.batch_dot(x, question, axes=[2, 2])) (context)

    # TODO: apply masking
    # Mask padding
    #y_mask = y_mask.unsqueeze(1).expand(scores.size())
    #scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
    alpha_flat = Activation('softmax') (scores)
    
    matched_seq = Lambda(lambda x: K.batch_dot(question, x, axes=[1, 1])) (question)
    return matched_seq

##########
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint




import sys

########################################
## set directories and parameters
########################################



from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim