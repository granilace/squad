from constants import *
from layers import *
from preprocessing import *
from losses import *
from quality_measuring import *

from keras.layers import Input, Embedding, Dropout, Concatenate, Activation
from keras import Model

import logging
logging.basicConfig(filename='train.log',level=logging.DEBUG)

import time

def attention_model_3():
    # inputs
    question_indices = Input(shape=(MAX_QUESTION_LEN, ))
    question_mask = Input(shape=(MAX_QUESTION_LEN, ))
    
    context_indices = Input(shape=(MAX_TEXT_LEN, ))
    context_features = Input(shape=(MAX_TEXT_LEN, NUM_FEATURES))
    context_pos_tags = Input(shape=(MAX_TEXT_LEN, ))
    context_entity_tags = Input(shape=(MAX_TEXT_LEN, ))
    context_mask = Input(shape=(MAX_TEXT_LEN, ))
    # onehots
    embedding_map, pos_onehot_embeddings, entity_onehot_embeddings = load_embeddings(EMB_PATH, POS_PATH, ENT_PATH)
    onehot_pos_layer = Embedding(POS_SIZE,
                                 POS_SIZE,
                                 weights=[pos_onehot_embeddings], 
                                 trainable=False)
    onehot_entity_layer = Embedding(ENTITY_SIZE, 
                                    ENTITY_SIZE,
                                    weights=[entity_onehot_embeddings], 
                                    trainable=False)
    context_pos_tags_oh = onehot_pos_layer(context_pos_tags)
    context_entity_tags_oh = onehot_entity_layer(context_entity_tags)
    # embedding layer  
    embedding_layer = Embedding(WORDS_IN_VOCABULARY, 
                                EMBEDDING_SIZE,
                                weights=[embedding_map], 
                                trainable=False)
    embedded_question = embedding_layer(question_indices)
    embedded_context = embedding_layer(context_indices)
    # dropout after embeddings
    embedded_question = Dropout(EMBEDDING_DROPOUT_RATE) (embedded_question)
    embedded_context = Dropout(EMBEDDING_DROPOUT_RATE) (embedded_context)
    # allignet question embedding
    #quemb_match = Seq_Attention(embedded_context, embedded_question, question_mask)
    # concating additional all contexts features
    '''context_data = Concatenate() ([embedded_context, 
                                   context_features, 
                                   quemb_match, 
                                   context_pos_tags_oh,
                                   context_entity_tags_oh])
    '''
    context_data = Concatenate() ([embedded_context, 
                                   context_features, 
                                   context_pos_tags_oh,
                                   context_entity_tags_oh])
    
    question_data = embedded_question
    # RNN for context
    context_hiddens = Document_RNN(context_data)
    # RNN for question
    question_hiddens = Question_RNN(question_data)
    # Attention for question
    question_with_attention = Linear_Attention(question_hiddens, question_mask)
    # Calculating output probas for start and end
    output_starts = Bilinear_Attention(context_hiddens,
                                       question_with_attention,
                                       context_mask)
    output_starts = Activation('softmax') (output_starts)
    
    output_ends = Bilinear_Attention(context_hiddens,
                                     question_with_attention,
                                     context_mask)
    output_ends = Activation('softmax') (output_ends)
    #
    model = Model(inputs=[context_indices, 
                          context_features, 
                          context_pos_tags, 
                          context_entity_tags, 
                          context_mask, 
                          question_indices, 
                          question_mask], 
                  outputs=[output_starts, 
                           output_ends])
    model.compile(loss=sum_of_losses,
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


class Attention_Model:
    def __init__(self, model_path=None):
        if model_path:
            self.model = keras.models.load_model(model_path)
            logging.info('---Model loaded---')
        else:
            self.model = attention_model_3()
            logging.info('---Model created---')

    def train(self, data=None, meta=None, n_epochs=40, from_scratch=False):
        model = self.model
        logging.info('Training started')
        if data:
            train_data = data['train']
            dev_data = data['dev']
        else:
            train_data = None
            dev_data = None
        train_contexts = get_contexts(train_data) # [word_indices, context_features, pos_tags, entity_tags, mask]
        train_questions = get_questions(train_data) # [word_indices, mask]
        train_answers_bin_list = get_bin_answers_train(train_data) # [starts, ends]
        self.train_answers_pairs = get_pairs_answers_train(train_data) # [[start_1, end_1]]
        logging.info('Train data loaded')
        #
        dev_contexts = get_contexts(dev_data, is_train=False) # [word_indices, context_features, pos_tags, entity_tags, mask]
        dev_questions = get_questions(dev_data, is_train=False) # [word_indices, mask]
        self.dev_answers_pairs = get_pairs_answers_dev(dev_data) # [[start_1, end_1], [start_2, end_2], ...]'''
        logging.info('Dev data loaded')
        #
        self.train_data = train_contexts + train_questions
        self.dev_data = dev_contexts + dev_questions
        #
        train_valid_x = get_valid_data(self.train_data)
        train_valid_y = self.train_answers_pairs[:VALID_SAMPLES]
        
        dev_valid_x = get_valid_data(self.dev_data)
        dev_valid_y = self.dev_answers_pairs[:VALID_SAMPLES]
        for i in range(n_epochs):
            logging.info(time.ctime() + '| epoch #', i + 1)
            model.fit(self.train_data, train_answers_bin_list, batch_size=BATCH_SIZE, epochs=1, verbose=0)
            logging.info(time.ctime() + '| epoch finished')
            logging.info('Train F1:' + str(measure_model_quality(model, 
                                                            train_valid_x, 
                                                            train_valid_y)))
            logging.info('Dev F1:' + str(measure_model_quality(model,
                                                            dev_valid_x,
                                                            dev_valid_y)))
            model_name = str(time.time())
            model.save(model_name)
            logging.info('Model saved with name:' + model_name)
        logging.info('Training finished')
