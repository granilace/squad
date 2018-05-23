from constants import *

import numpy as np

def get_preds(start_probas, end_probas, max_len):
    predicted_answers = list()
    for i in range(len(end_probas)):
        best_end_pos = np.argmax(end_probas[i])
        best_start_pos = best_end_pos
        for start_pos in range(best_end_pos, max(-1, best_end_pos - max_len), -1):
            if start_probas[i][start_pos] > start_probas[i][best_start_pos]:
                best_start_pos = start_pos
        predicted_answers.append([best_start_pos, best_end_pos])
    return predicted_answers
    
def TP(pred_pair, true_pair):
    true_positive = 0
    for i in range(pred_pair[0], pred_pair[1] + 1):
        if i >= true_pair[0] and i <= true_pair[1]:
            true_positive += 1
    return true_positive
    
def get_preds2(start_probas, end_probas, max_len):
    predicted_answers = list()
    for i in range(len(start_probas)):
        best_pair = [0, 0]
        best_prob = 0.0
        for start in range(len(start_probas[i])):
            for end in range(start, min(start + max_len, len(start_probas[i]))):
                if start_probas[i][start] * end_probas[i][end] > best_prob:
                    best_prob = start_probas[i][start] * end_probas[i][end]
                    best_pair = [start, end]
        predicted_answers.append(best_pair)
    return predicted_answers
                             
    
def precision(pred_pair, true_pair):
    return TP(pred_pair, true_pair) / (pred_pair[1] - pred_pair[0] + 1)

def recall(pred_pair, true_pair):
    return TP(pred_pair, true_pair) / (true_pair[1] - true_pair[0] + 1)

def F1_score(pred_pairs, true_pairs):
    F1_sum = 0.0
    for i in range(len(pred_pairs)):
        best_F1 = 0.0
        for true_answ in true_pairs[i]:
            p = precision(pred_pairs[i], true_answ)
            r = recall(pred_pairs[i], true_answ)
            if p != 0 or r != 0:
                best_F1 = max(2 * p * r / (p + r), best_F1)
        F1_sum += best_F1
    return F1_sum / len(pred_pairs)

def measure_model_quality(model, data, true_pairs):
    start_probas, end_probas = model.predict(data, batch_size=100)
    pred_pairs = get_preds2(start_probas, end_probas, MAX_ANSW_LEN)
    return F1_score(pred_pairs, true_pairs)