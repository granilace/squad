from constants import *
import layers_masking
import layers
from preprocessing import *
from losses import *
from quality_measuring import *

from keras.layers import Input, Embedding, Dropout, Concatenate, Activation, Masking, Dense
from keras import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization

import logging
logging.basicConfig(filename='train.log',level=logging.DEBUG)

import time
import keras
###

def attention_model_3_masking():
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
    #embedded_question = Masking() (embedded_question)
    #embedded_context = Masking() (embedded_context)
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
    context_hiddens = layers_masking.Document_RNN(BatchNormalization() (context_data))
    # RNN for question
    question_hiddens = layers_masking.Question_RNN(BatchNormalization() (question_data))
    # Attention for question
    question_with_attention = layers_masking.Linear_Attention(question_hiddens, question_mask)
    # Calculating output probas for start and end
    output_starts = layers_masking.Bilinear_Attention(context_hiddens,
                                       question_with_attention,
                                       context_mask)
    output_starts = Activation('softmax') (output_starts)
    
    output_ends = layers_masking.Bilinear_Attention(context_hiddens,
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])
    return model
##### Model without support of masking #####
def attention_model_3():
    # inputs
    question_indices = Input(shape=(MAX_QUESTION_LEN, ))
    
    context_indices = Input(shape=(MAX_TEXT_LEN, ))
    context_features = Input(shape=(MAX_TEXT_LEN, NUM_FEATURES))
    context_pos_tags = Input(shape=(MAX_TEXT_LEN, ))
    context_entity_tags = Input(shape=(MAX_TEXT_LEN, ))
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
    #embedded_question = Masking() (embedded_question)
    #embedded_context = Masking() (embedded_context)
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
    # RNN for question
    question_hiddens = layers.Question_RNN(BatchNormalization() (question_data))
    # Attention for question
    question_with_attention = layers.Linear_Attention(question_hiddens)
    # Make stretched version of question with attention(for initial state in  Document RNN)
    question_with_attention_stretched = Dense(HIDDEN_SIZE) (question_with_attention)
    # RNN for context
    print(question_with_attention_stretched.shape)
    context_hiddens = layers.Document_RNN(BatchNormalization() (context_data), question_with_attention_stretched)
    # Calculating output probas for start and end
    output_starts = layers.Bilinear_Attention(context_hiddens,
                                              question_with_attention)
    output_starts = Activation('softmax') (output_starts)
    
    output_ends = layers.Bilinear_Attention(context_hiddens,
                                            question_with_attention)
    output_ends = Activation('softmax') (output_ends)
    #
    model = Model(inputs=[context_indices, 
                          context_features, 
                          context_pos_tags, 
                          context_entity_tags, 
                          question_indices], 
                  outputs=[output_starts, 
                           output_ends])
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])
    return model
######

class AttentionModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = keras.models.load_model(model_path)
            logging.info('---Model loaded---')
        else:
            self.model = attention_model_3()
            logging.info('---Model created---')
        self.data_loaded = False
        
    def load_data(self, data=None, meta=None):
        model = self.model
        logging.info('Data loading started')
        if data:
            train_data = data['train']
            dev_data = data['dev']
        else:
            train_data = None
            dev_data = None
        train_contexts = get_contexts(train_data)   # [word_indices, context_features, pos_tags, entity_tags, mask]
        train_contexts = train_contexts[:4]         # we don't need masks
        train_questions = get_questions(train_data) # [word_indices, mask]
        train_questions = train_questions[:1]       # we don't need masks
        self.train_answers_bin_list = get_bin_answers_train(train_data) # [starts, ends]
        self.train_answers_pairs = get_pairs_answers_train(train_data)  # [[start_1, end_1]]
        logging.info('Train data loaded')
        #
        dev_contexts = get_contexts(dev_data, is_train=False)    # [word_indices, context_features, pos_tags, entity_tags, mask]
        dev_contexts = dev_contexts[:4]                          # we don't need masks
        dev_questions = get_questions(dev_data, is_train=False)  # [word_indices, mask]
        dev_questions = dev_questions[:1]                        # we don't need masks
        self.dev_answers_pairs = get_pairs_answers_dev(dev_data) # [[start_1, end_1], [start_2, end_2], ...]'''
        logging.info('Dev data loaded')
        #
        self.train_data = train_contexts + train_questions
        self.dev_data = dev_contexts + dev_questions
        #
        self.train_valid_x = get_valid_data(self.train_data)
        self.train_valid_y = self.train_answers_pairs[:VALID_SAMPLES]
        
        self.dev_valid_x = get_valid_data(self.dev_data)
        self.dev_valid_y = self.dev_answers_pairs[:VALID_SAMPLES]
        self.data_loaded = True

    def train(self, data=None, meta=None, n_epochs=40, from_scratch=False):
        model = self.model
        if not self.data_loaded:
            self.load_data()
        logging.info('Training started')
        for i in range(n_epochs):
            logging.info(time.ctime() + ' | epoch #' + str(i + 1))
            model.fit(self.train_data, self.train_answers_bin_list, batch_size=BATCH_SIZE, epochs=1, verbose=1)
            logging.info(time.ctime() + ' | epoch finished')
            self.quality()
            model_name = MODELS_PATH + time.asctime()
            self.save(model_name)
        logging.info('Training finished')
        
    def quality(self):
        model = self.model
        if not self.data_loaded:
            self.load_data()
        logging.info('Train F1: ' + str(measure_model_quality(model, 
                                                              self.train_valid_x, 
                                                              self.train_valid_y)))
        logging.info('Dev F1: ' + str(measure_model_quality(model,
                                                            self.dev_valid_x,
                                                            self.dev_valid_y)))
    def save(self, name):
        self.model.save(name)
        logging.info('Model saved with name:' + name)
