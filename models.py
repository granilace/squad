from constants import *

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
    onehot_pos_layer = Embedding(POS_SIZE,
                                 POS_SIZE,
                                 weights=[pos_onehot_embeddings], 
                                 trainable=False)
    onehot_entity_layer = Embedding(ENTITY_SIZE, 
                                    ENTITY_SIZE,
                                    weights=[entity_onehot_embeddings], 
                                    trainable=False)
    context_pos_tags = onehot_pos_layer(context_pos_tags)
    context_entity_tags = onehot_entity_layer(context_entity_tags)
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
                                   context_pos_tags,
                                   context_entity_tags])
    '''
    context_data = Concatenate() ([embedded_context, 
                                   context_features, 
                                   context_pos_tags,
                                   context_entity_tags])
    
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
        else:
            self.model = attention_model_3()

    def train(self, data, meta, n_epochs=40):
        model = self.model
        log.info('Train started')
        train_data = data['train']
        dev_data = data['dev']

        train_contexts = get_contexts(train_data) # [word_indices, context_features, pos_tags, entity_tags, mask]
        train_questions = get_questions(train_data) # [word_indices, mask]
        train_answers_bin_list = get_bin_answers_train(train_data) # [starts, ends]
        train_answers_pairs = get_pairs_answers_train(train_data) # [[start_1, end_1]]
        log.info('Train data loaded')
        #
        dev_contexts = get_contexts(dev_data, is_train=False) # [word_indices, context_features, pos_tags, entity_tags, mask]
        dev_questions = get_questions(dev_data, is_train=False) # [word_indices, mask]
        dev_answers_pairs = get_pairs_answers_dev(dev_data, is_train=False) # [[start_1, end_1], [start_2, end_2], ...]
        log.info('Dev data loaded')
        #
        train_data = train_contexts + train_questions
        for i in range(n_epochs):
            log.info(get_curr_time(), '| epoch #', i + 1)
            model.fit(train_data, train_answers_bin_list, batch_size=BATCH_SIZE, epochs=1, verbose=0)
            log.info(get_curr_time(), '| epoch finished')
            log.info('Train F1:', measure_model_quality(model, 
                                                        train_data[:VALID_SAMPLES], 
                                                        train_answer_pairs[:VALID_SAMPLES]))
            log.info('Dev F1:',   measure_model_quality(model,
                                                        dev_data[:VALID_SAMPLES],
                                                        dev_answer_pairs[:VALID_SAMPLES]))
            model_name = str(time.time())
            model.save(model_name)
            log.info('Model saved with name:', model_name)
        log.info('Train finished')
