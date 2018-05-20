from constants import *

from keras.layers import Bidirectional, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, merge, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def Document_RNN(data):
    RNN_context = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(MAX_TEXT_LEN, INPUT_CONTEXT_LEN) ) (data)
    RNN_context = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(MAX_TEXT_LEN, HIDDEN_SIZE) ) (BatchNormalization() (RNN_context))
    return RNN_context

def Question_RNN(data):
    RNN_questions = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(MAX_QUESTION_LEN, INPUT_QUESTION_LEN)) (data)
    RNN_questions = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=RNN_DROPOUT_RATE),
                        input_shape=(MAX_QUESTION_LEN, HIDDEN_SIZE)) (BatchNormalization() (RNN_questions))
    return RNN_questions

def Linear_Attention(data, mask):
    attention = Dense(1)(data)
    attention = Lambda(lambda x: x)(attention)
    attention = Reshape((MAX_QUESTION_LEN,)) (attention)
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