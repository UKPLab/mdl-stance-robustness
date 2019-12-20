# Copyright (c) Microsoft. All rights reserved.

from .vocab import Vocabulary
from .metrics import *

# snopes
SnopesLabelMapper = Vocabulary(True)
SnopesLabelMapper.add('refute')
SnopesLabelMapper.add('agree')

# PERSPECTRUM
PERSPECTRUMLabelMapper = Vocabulary(True)
PERSPECTRUMLabelMapper.add('UNDERMINE')
PERSPECTRUMLabelMapper.add('SUPPORT')

# argmin
ArgMinLabelMapper = Vocabulary(True)
ArgMinLabelMapper.add('Argument_against')
ArgMinLabelMapper.add('Argument_for')

# IBM Claim Stance (ibmcs)
IBMCSLabelMapper = Vocabulary(True)
IBMCSLabelMapper.add('CON')
IBMCSLabelMapper.add('PRO')

# Internet Argument Corpus v1.1 (iac1)
IAC1LabelMapper = Vocabulary(True)
IAC1LabelMapper.add('anti')
IAC1LabelMapper.add('pro')
IAC1LabelMapper.add('other')

# fnc-1
FNC1LabelMapper = Vocabulary(True)
FNC1LabelMapper.add('unrelated')
FNC1LabelMapper.add('discuss')
FNC1LabelMapper.add('agree')
FNC1LabelMapper.add('disagree')

# arc
ARCLabelMapper = Vocabulary(True)
ARCLabelMapper.add('unrelated')
ARCLabelMapper.add('discuss')
ARCLabelMapper.add('agree')
ARCLabelMapper.add('disagree')

# semeval2016 task 6
SemEval2016T6LabelMapper = Vocabulary(True)
SemEval2016T6LabelMapper.add('AGAINST')
SemEval2016T6LabelMapper.add('FAVOR')
SemEval2016T6LabelMapper.add('NONE')

# Stance classification dataset
SCDLabelMapper = Vocabulary(True)
SCDLabelMapper.add('against')
SCDLabelMapper.add('for')

# semeval2019 task 7
SemEval2019T7LabelMapper = Vocabulary(True)
SemEval2019T7LabelMapper.add('support')
SemEval2019T7LabelMapper.add('deny')
SemEval2019T7LabelMapper.add('query')
SemEval2019T7LabelMapper.add('comment')

GLOBAL_MAP = {
 'snopes': SnopesLabelMapper,
 'perspectrum': PERSPECTRUMLabelMapper,
 'argmin': ArgMinLabelMapper,
 'semeval2019t7': SemEval2019T7LabelMapper,
 'semeval2016t6': SemEval2016T6LabelMapper,
 'fnc1': FNC1LabelMapper,
 'iac1': IAC1LabelMapper,
 'arc': ARCLabelMapper,
 'scd': SCDLabelMapper,
 'ibmcs': IBMCSLabelMapper,
}

# number of class
DATA_META = {
 'snopes': 2,
 'argmin': 2,
 'ibmcs': 2,
 'fnc1': 4,
 'arc': 4,
 'iac1': 3,
 'perspectrum': 2,
 'semeval2016t6': 3,
 'semeval2019t7': 4,
 'scd': 2
}

DATA_TYPE = {
 'snopes': 0,
 'argmin': 0,
 'fnc1': 0,
 'iac1': 0,
 'arc': 0,
 'ibmcs': 0,
 'perspectrum': 0,
 'semeval2016t6': 0,
 'semeval2019t7': 1,
 'scd': 1,
}

DATA_SWAP = {
 'snopes': 0,
 'argmin': 0,
 'semeval2016t6': 0,
 'semeval2019t7': 0,
 'scd': 0,
 'fnc1': 0,
 'arc': 0,
 'iac1': 0,
 'ibmcs': 0,
 'perspectrum': 0,
}

# classification/regression
TASK_TYPE = {
 'snopes': 0,
 'argmin': 0,
 'semeval2016t6': 0,
 'semeval2019t7': 0,
 'fnc1': 0,
 'arc': 0,
 'iac1': 0,
 'ibmcs': 0,
 'perspectrum': 0,
 'scd': 0
}

METRIC_META = {
 'snopes': [0, 5, 6, 7, 8, 10, 11, 12],
 'fnc1': [0, 5, 6, 7, 8, 9, 10, 11, 12],
 'arc': [0, 5, 6, 7, 8, 9, 10, 11, 12],
 'iac1': [0, 5, 6, 7, 8, 10, 11, 12],
 'ibmcs': [0, 5, 6, 7, 8, 10, 11, 12],
 'perspectrum': [0, 5, 6, 7, 8, 10, 11, 12],
 'argmin': [0, 5, 6, 7, 8, 10, 11, 12],
 'scd': [0, 5, 6, 7, 8, 10, 11, 12],
 'semeval2016t6': [0, 5, 6, 7, 8, 10, 11, 12],
 'semeval2019t7': [0, 5, 6, 7, 8, 10, 11, 12]
}

METRIC_NAME = {
 0: 'ACC',
 1: 'F1',
 2: 'MCC',
 3: 'Pearson',
 4: 'Spearman',
 5: 'F1_macro',
 6: 'Precision_macro',
 7: 'Recall_macro',
 8: 'F1_micro',
 9: 'FNC-1',
 10: 'Recall_clw',
 11: 'Precision_clw',
 12: 'F1_clw',
}

METRIC_FUNC = {
 0: compute_acc,
 1: compute_f1,
 2: compute_mcc,
 3: compute_pearson,
 4: compute_spearman,
 5: compute_f1_macro,
 6: compute_precision_macro,
 7: compute_recall_macro,
 8: compute_f1_micro,
 9: compute_fnc1,
 10: compute_recall_clw,
 11: compute_precision_clw,
 12: compute_f1_clw,
}

SAN_META = {
    'snopes': 1,
    'argmin': 1,
    'semeval2016t6': 1,
    'semeval2019t7': 0,
    'scd': 0,
    'perspectrum': 1,
    'ibmcs': 1,
    'fnc1': 1,
    'arc': 1,
    'iac1': 1
}

def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v