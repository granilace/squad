from constants import *

import pickle
import numpy as np

import logging

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
#########
def padd(inp_list, padding_data, target_len):
    for i in range(target_len - len(inp_list)):
        inp_list.append(padding_data)
        
def get_context_masks(data):
    context_masks = list()
    for i in range(len(data)):
        mask = [1] * len(data[i][1]) + [0] * (MAX_TEXT_LEN - len(data[i][1]))
        context_masks.append(mask)
    return np.array(context_masks)

def get_context_ids(data):
    context_ids_list = list()
    for i, context in enumerate(data):
        if DO_PADDING:
            padd(context[1], 0, MAX_TEXT_LEN)
        context_ids_list.append(context[1])
    return np.array(context_ids_list)

def get_context_features(data):
    context_features = list()
    for i, context in enumerate(data):
        for feature_list in context[2]:
            feature_list[0] = float(int(feature_list[0]))
            feature_list[1] = float(int(feature_list[1])) # [False, True, False, 0.5] -> [0.0, 1.0, 0.0, 0.5]
            feature_list[2] = float(int(feature_list[2]))
        if DO_PADDING:
            padd(data[i][2], [0.0, 0.0, 0.0, 0.0], MAX_TEXT_LEN) # Maybe try another empty vector
        context_features.append(data[i][2])
    return np.array(context_features)

def get_context_pos_tags(data):
    context_pos_tags = list()
    for i, context in enumerate(data):
        if DO_PADDING:
            padd(context[3], POS_SIZE - 1, MAX_TEXT_LEN)
        context_pos_tags.append(context[3])
    return np.array(context_pos_tags)

def get_context_ent_tags(data):
    context_ent_tags = list()
    for i, context in enumerate(data):
        if DO_PADDING:
            padd(context[4], ENTITY_SIZE - 1, MAX_TEXT_LEN)
        context_ent_tags.append(context[4])
    return np.array(context_ent_tags)

def get_question_masks(data):
    question_masks = list()
    for i in range(len(data)):
        mask = [1] * len(data[i][5]) + [0] * (MAX_QUESTION_LEN - len(data[i][5]))
        question_masks.append(mask)
    return np.array(question_masks)

def get_question_ids(data):
    question_ids_list = list()
    for i, context in enumerate(data):
        if DO_PADDING:
            padd(context[5], 0, MAX_QUESTION_LEN)
        question_ids_list.append(context[5])
    return np.array(question_ids_list)
#####
def get_contexts(data=None, is_train=True):
    if is_train:
        logging.info('Getting contexts for train')
    else:
        logging.info('Getting contexts for dev')
    #
    if TRAIN_CONTEXT_PATH and is_train:
        return load_pickle(TRAIN_CONTEXT_PATH)
    if DEV_CONTEXT_PATH and not is_train:
        return load_pickle(DEV_CONTEXT_PATH)
    assert data is not None

    context_masks = get_context_masks(data)
    logging.info('Context masks loaded')

    context_ids_list = get_context_ids(data)
    logging.info('Context ids loaded')

    context_features = get_context_features(data)
    logging.info('Context features loaded')

    context_pos_tags = get_context_pos_tags(data)
    logging.info('Context pos tags loaded')

    context_ent_tags = get_context_ent_tags(data)
    logging.info('Context entity tags loaded')
    return [context_ids_list, context_features, context_pos_tags, context_ent_tags, context_masks]

def get_questions(data=None, is_train=True):
    if is_train:
        logging.info('Getting questions for train')
    else:
        logging.info('Getting questions for dev')
    if TRAIN_QUESTION_PATH and is_train:
        return load_pickle(TRAIN_QUESTION_PATH)
    if DEV_QUESTION_PATH and not is_train:
        return load_pickle(DEV_QUESTION_PATH)
    assert data is not None

    question_masks = get_question_masks(data)
    logging.info('Context masks loaded')

    question_ids_list = get_question_ids(data)
    logging.info('Context ids loaded')
    
    return [question_ids_list, question_masks]

def get_bin_answers_train(data=None):
    logging.info('Getting binary answers for train')
    if TRAIN_ANSWER_BIN_PATH:
        return load_pickle(TRAIN_ANSWER_BIN_PATH)
    assert data is not None
    
    bin_answers_start = list()
    bin_answers_end = list()
    for i, context in enumerate(data):
        answ_start = data[i][8]
        answ_end = data[i][9]
        if DO_PADDING:
            bin_list_start = [0] * MAX_TEXT_LEN
            bin_list_end = [0] * MAX_TEXT_LEN
        else:
            bin_list_start = [0] * len(data[i][1])
            bin_list_end = [0] * len(data[i][1])
        bin_list_start[answ_start] = 1
        bin_answers_start.append(bin_list_start)
        #
        bin_list_end[answ_end] = 1
        bin_answers_end.append(bin_list_end)
    return [np.array(bin_answers_start), np.array(bin_answers_end)]

def get_pairs_answers_train(data=None):
    logging.info('Getting answer pairs for train')
    if TRAIN_ANSWER_PAIRS_PATH:
        return load_pickle(TRAIN_ANSWER_PAIRS_PATH)
    assert data is not None
   
    answ_list = list()
    for i, context in enumerate(data):
        answ_start = data[i][8]
        answ_end = data[i][9]
        answ_list.append([[answ_start, answ_end]])
    return answ_list

def get_pairs_answers_dev(data=None):
    logging.info('Getting answer pairs for dev')
    if DEV_ANSWER_PAIRS_PATH:
        return load_pickle(DEV_ANSWER_PAIRS_PATH)
    assert data is not None
    all_pairs = list()
    for elem in data:
        text_answ_list = elem[8]
        answ_pairs = list()
        for text_answ in text_answ_list:
            start_pos_in_text = elem[6].find(text_answ)
            end_pos_in_text = start_pos_in_text + len(text_answ)
            while start_pos_in_text != -1:
                answ_pair = [-1, -1]
                for index, pair in enumerate(elem[7]):
                    if pair[0] == start_pos_in_text:
                        answ_pair[0] = index
                    if pair[1] == end_pos_in_text:
                        answ_pair[1] = index
                if answ_pair[0] != -1 and answ_pair[1] != -1:
                    answ_pairs.append(answ_pair)
                    break
                else:
                    start_pos_in_text = elem[6].find(text_answ, start_pos_in_text + 1)
                    end_pos_in_text = start_pos_in_text + len(text_answ)
                    print(start_pos_in_text)
        all_pairs.append(answ_pairs)
    return all_pairs

def load_embeddings(path1, path2, path3):
    return load_pickle(path1), load_pickle(path2), load_pickle(path3)

def get_valid_data(data):
    valid_data = []
    for j in range(len(data)):
        valid_data.append([])
    for i in range(VALID_SAMPLES):
        for j in range(len(data)):
            valid_data[j].append(data[j][i])
    for j in range(len(data)):
        valid_data[j] = np.array(valid_data[j])
    return valid_data
