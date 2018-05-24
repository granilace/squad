import sys
sys.path.insert(0, 'scripts')

import time
import msgpack
import spacy
import re
import unicodedata
import collections
import numpy as np

from constants import *
from preprocessing import *
from quality_measuring import *
import models


nlp = spacy.load('en', parser=False)

def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def annotate(row, wv_cased):
    global nlp
    id_, context, question = row[:3]
    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span) + row[3:]

def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]

def generate_batch(data):
    padd(data[1], 0, MAX_TEXT_LEN)
    context_ids = data[1]
    for i in range(len(data[2])):
        data[2][i] = list(map(float, data[2][i]))
    for i in range(MAX_TEXT_LEN - len(data[2])):
        data[2].append([0.,0.,0.,0.])
    context_features = data[2]
    padd(data[3], POS_SIZE - 1, MAX_TEXT_LEN)
    tag_ids = data[3]
    ent_ids = padd(data[4], ENTITY_SIZE - 1, MAX_TEXT_LEN)
    ent_ids = data[4]
    question_ids = padd(data[5], 0, MAX_QUESTION_LEN)
    question_ids = data[5]
    return [np.array(context_ids, ndmin=2), 
            np.array(context_features, ndmin=3), 
            np.array(tag_ids, ndmin=2), 
            np.array(ent_ids, ndmin=2), 
            np.array(question_ids, ndmin=2)]

def main():
    path = input('Enter file path of your model:')

    with open(RAW_META_PATH, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    embedding = load_pickle(EMB_PATH)

    model = models.AttentionModel(path)

    w2id = {w: i for i, w in enumerate(meta['vocab'])}
    tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
    ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}

    while True:
        id_ = 0
        try:
            while True:
                context = input('Enter context: ')
                if context.strip():
                    break
            while True:
                question = input('Enter question: ')
                if question.strip():
                    break
        except EOFError:
            break
        id_ += 1
        annotated = annotate(('interact-{}'.format(id_), context, question), meta['wv_cased'])
        model_in_raw = to_id(annotated, w2id, tag2id, ent2id)
        model_in = generate_batch(model_in_raw)
        start_probas, end_probas = model.model.predict(model_in)
        answ_pair = get_preds2(start_probas, end_probas, MAX_ANSW_LEN)
        print('Answer:', end=' ')
        for i in range(answ_pair[0][0], answ_pair[0][1] + 1):
            print(model_in_raw[6][model_in_raw[7][i][0]:model_in_raw[7][i][1]+1], end='')
        print('\n\n\n-------------|||-------------\n\n\n')
    
    
if __name__ == '__main__':
    main()
