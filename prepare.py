import numpy as np
import sys
sys.path.insert(0, 'scripts')

from constants import *
from preprocessing import *

def get_raw_data():
    wget('link -O ' + RAW_DATA_PATH)
    with open(RAW_DATA_PATH, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    return data

def get_raw_meta():
    wget('link -O ' + RAW_META_PATH)
    with open(RAW_META_PATH, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    return meta

def load_last_model():
    wget('link -O ' + MODELS_PATH + LAST_MODEL_NAME)

def build_dev_answ_pairs(data):
    pairs = get_pairs_answers_dev(data['dev'])
    save_pickle(pairs, DEV_ANSWER_PAIRS_PATH)
    free_mem(pairs)
    
def build_dev_contexts(data):
    contexts = get_contexts(data['dev'], is_train=False)
    save_pickle(contexts, DEV_CONTEXT_PATH)
    free_mem(contexts)

def build_dev_questions(data):
    questions = get_questions(data['dev'], is_train=False)
    save_pickle(questions, DEV_QUESTION_PATH)
    free_mem(questions)

def build_embeddings(meta):
    ent_emb = np.zeros((ENTITY_SIZE, ENTITY_SIZE))
    for i in range(ENTITY_SIZE - 1):
        ent_emb[i][i] = 1
    save_pickle(ent_emb, ENT_PATH)
    free_mem(ent_emb)
    #
    pos_emb = np.zeros((POS_SIZE, POS_SIZE))
    for i in range(POS_SIZE - 1):
        pos_emb[i][i] = 1
    save_pickle(pos_emb, POS_PATH)
    free_mem(pos_emb)
    #
    embeddings = meta['embedding']
    save_pickle(embeddings, EMB_PATH)
    free_mem(embeddings)
    
def build_train_answ_bin(data):
    answs = get_bin_answers_train(data['train'])
    save_pickle(answs, TRAIN_ANSWER_BIN_PATH)
    free_mem(answs)

def build_train_answ_pairs(data):
    answs = get_pairs_answers_train(data['train'])
    save_pickle(answs, TRAIN_ANSWER_PAIRS_PATH)
    free_mem(answs)

def build_train_contexts(data):
    contexts = get_contexts(data['dev'], is_train=True)
    save_pickle(contexts, TRAIN_CONTEXT_PATH)
    free_mem(contexts)
    
def build_train_questions(data):
    questions = get_questions(data['dev'], is_train=True)
    save_pickle(questions, TRAIN_QUESTION_PATH)
    free_mem(questions)
    

def main():
    download_all = os[1]
    if dowload_all:
        data = get_raw_data()
        build_dev_answ_pairs(data)
        build_dev_contexts(data)
        build_dev_questions(data)
        build_train_answ_bin(data)
        build_train_answ_pairs(data)
        build_train_contexts(data)
        build_train_questions(data)
    meta = get_raw_meta()
    build_embeddings(meta)
    load_last_model()
    
if __name__ == '__main__':
    main()
