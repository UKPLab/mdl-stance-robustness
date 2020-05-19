# Copyright (c) Microsoft. All rights reserved.
# Modified Copyright by Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

def compute_acc(predicts, labels):
    return accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return f1_score(labels, predicts)

def compute_f1_clw(predicts, labels):
    return f1_score(labels, predicts, average=None).tolist()

def compute_precision_clw(predicts, labels):
    return precision_score(labels, predicts, average=None).tolist()

def compute_recall_clw(predicts, labels):
    return recall_score(labels, predicts, average=None).tolist()

def compute_f1_macro(predicts, labels):
    return f1_score(labels, predicts, average="macro")

def compute_f1_micro(predicts, labels):
    return f1_score(labels, predicts, average="micro")

def compute_precision_macro(predicts, labels):
    return precision_score(labels, predicts, average="macro")

def compute_recall_macro(predicts, labels):
    return recall_score(labels, predicts, average="macro")

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_fnc1(predicts, labels):
    from .label_map import GLOBAL_MAP
    # implementation modified from https://github.com/FakeNewsChallenge/fnc-1-baseline/blob/master/utils/score.py
    RELATED = ['agree', 'disagree', 'discuss']
    label_dict = GLOBAL_MAP['fnc1'] # same order than ARC

    def score_submission(gold_labels, test_labels):
        score = 0.0

        for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
            g_stance, t_stance = g, t
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25

        return score

    def report_score(actual, predicted):
        score = score_submission(actual, predicted)
        best_score = score_submission(actual, actual)

        return score * 100 / best_score

    return report_score([label_dict.ind2tok[e] for e in labels],[label_dict.ind2tok[e] for e in predicts])