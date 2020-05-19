# Copyright (c) Microsoft. All rights reserved.
# Modified Copyright by Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
import argparse
from data_utils.log_wrapper import create_logger
from data_utils.glue_utils import *
import random

LOW_RESOURCE_DATA_RATES = [0.1, 0.3, 0.7]

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_512.log')

def get_random_indices(dump_path, data, rate):
    rdm_indices = []
    if rate < 1.0:
        with open("data_splits/low_resource_splits/"+dump_path.rsplit("/", 1)[1].split("_")[0]+"_train.csv", "r") as split_file_in:
            rdm_indices = [int(idx) for idx in split_file_in.readline().split(",")[:int(len(data)*rate)]]
    return rdm_indices

def build_data(data, dump_path, rate=1.0, MAX_SEQ_LEN=100, is_train=True, tolower=True):
    """Build data of sentence pair tasks
    """

    if os.path.isfile(dump_path):
        print(dump_path + " already existis. Skipping!")
        return

    rdm_indices = get_random_indices(dump_path, data, rate)

    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in tqdm(enumerate(data)):

            if rate < 1.0 and idx not in rdm_indices:
                continue

            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis = bert_tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']

            if "spelling" in dump_path or "negation" in dump_path: # have already been truncated in preprocessing step
                truncate_len = 512-3
            else:
                truncate_len = MAX_SEQ_LEN - 3

            truncate_seq_pair(premise, hypothesis, truncate_len)
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'hypothesis': sample['hypothesis'], 'premise': sample['premise']}
            writer.write('{}\n'.format(json.dumps(features)))

def build_data_single(data, dump_path, rate=1.0, MAX_SEQ_LEN=100):
    """Build data of single sentence tasks
    """

    if os.path.isfile(dump_path):
        print(dump_path + " already existis. Skipping!")
        return

    rdm_indices = get_random_indices(dump_path, data, rate)

    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in tqdm(enumerate(data)):

            if rate < 1.0 and idx not in rdm_indices:
                continue

            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            label = sample['label']
            if "spelling" in dump_path or "negation" in dump_path: # have already been truncated in preprocessing step
                premise = premise[:512 - 3]
            elif len(premise) >  MAX_SEQ_LEN - 3:
                premise = premise[:MAX_SEQ_LEN - 3]
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(premise) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'premise': sample['premise']}
            writer.write('{}\n'.format(json.dumps(features)))

def print_status(train_data, dev_data, test_data, test_negation, test_spelling, test_paraphrase, dataset):
    logger.info('Loaded {0} {1} train samples'.format(len(train_data), dataset))
    logger.info('Loaded {0} {1} dev samples'.format(len(dev_data), dataset))
    logger.info('Loaded {0} {1} test samples'.format(len(test_data), dataset))
    if test_negation != None:
        logger.info('Loaded {0} {1} test negation samples'.format(len(test_negation), dataset))
    if test_spelling != None:
        logger.info('Loaded {0} {1} test spelling samples'.format(len(test_spelling), dataset))
    if test_paraphrase != None:
        logger.info('Loaded {0} {1} test paraphras samples'.format(len(test_paraphrase), dataset))
    logger.info('\n')

def build_handler(train_data, dev_data, test_data, test_negation, test_spelling, test_paraphrase,
                  dataset, mt_dnn_root, single=False, MAX_SEQ_LEN=100):
    build_fn = build_data_single if single == True else build_data

    logger.info('Start converting and storing train/dev/test files for the model')
    train_fout = os.path.join(mt_dnn_root, '{0}_train.json'.format(dataset))
    dev_fout = os.path.join(mt_dnn_root, '{0}_dev.json'.format(dataset))
    test_fout = os.path.join(mt_dnn_root, '{0}_test.json'.format(dataset))
    test_negation_fout = os.path.join(mt_dnn_root, '{0}_test_negation.json'.format(dataset))
    test_spelling_fout = os.path.join(mt_dnn_root, '{0}_test_spelling.json'.format(dataset))
    test_paraphrase_fout = os.path.join(mt_dnn_root, '{0}_test_paraphrase.json'.format(dataset))

    for rate in LOW_RESOURCE_DATA_RATES:
        build_fn(train_data, train_fout.split(".json")[0]+str(int(rate*100))+"p.json", rate, MAX_SEQ_LEN=MAX_SEQ_LEN)

    build_fn(train_data, train_fout)
    build_fn(dev_data, dev_fout)
    build_fn(test_data, test_fout)
    if test_negation != None:
        build_fn(test_negation, test_negation_fout)
    if test_spelling != None:
        build_fn(test_spelling, test_spelling_fout)
    if test_paraphrase != None:
        build_fn(test_paraphrase, test_paraphrase_fout)
    logger.info('done with {0}'.format(dataset))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing datasets.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--generate_adversarial', type=int, default=0, help="0 if no adversarial samples should be created, otherwise 1")
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--opennmt_gpu', type=int, default=-1, help="-1 for CPU, else GPU index.")
    parser.add_argument('--opennmt_batch_size', type=int, default=30)
    args = parser.parse_args()
    return args

def main(args):
    root = args.root_dir
    seed = args.seed
    CREATE_ADVERSARIAL = True if args.generate_adversarial == 1 else False
    MAX_SEQ_LEN = args.max_seq_len
    OPENNMT_GPU = args.opennmt_gpu  # Machine translation on GPU
    OPENNMT_BATCH_SIZE = args.opennmt_batch_size  # Machine translation model batch size

    assert os.path.exists(root)
    random.seed(seed)
    np.random.seed(seed)

    mt_dnn_root = os.path.join(root, 'mt_dnn')
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    ######################################
    # Stance Detection Tasks
    ######################################
    snopes_path = os.path.join(root, 'Snopes/snopes_corpus_2.csv')

    argmin_path = os.path.join(root, 'ArgMin/')

    fnc1_path = os.path.join(root, 'FNC-1/')

    arc_path = os.path.join(root, 'ARC/')

    ibm_claim_stance_path = os.path.join(root, 'IBM_CLAIM_STANCE/claim_stance_dataset_v1.csv')

    perspectrum_path = os.path.join(root, 'PERSPECTRUM/')

    semeval2016t6_path = os.path.join(root, 'SemEval2016Task6/')

    semeval2019t7_path = os.path.join(root, 'SemEval2019Task7/')

    scd_path = os.path.join(root, 'SCD/')

    iac1_path = os.path.join(root, 'IAC/')

    ######################################
    # Loading / Building DATA
    ######################################

    scd_train_data, scd_dev_data, scd_test_data, scd_test_negation, scd_test_spelling, scd_test_paraphrase = \
        load_scd(scd_path, GLOBAL_MAP['scd'], create_adversarial=CREATE_ADVERSARIAL, dataset='scd', MAX_SEQ_LEN=MAX_SEQ_LEN,
                 OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(scd_train_data, scd_dev_data, scd_test_data, scd_test_negation, scd_test_spelling,
                 scd_test_paraphrase, 'scd')
    build_handler(scd_train_data, scd_dev_data, scd_test_data, scd_test_negation,
                  scd_test_spelling, scd_test_paraphrase, 'scd', mt_dnn_root, single=True, MAX_SEQ_LEN=MAX_SEQ_LEN)

    semeval2016t6_train_data, semeval2016t6_dev_data, semeval2016t6_test_data, semeval2016t6_test_negation, \
        semeval2016t6_test_spelling, semeval2016t6_test_paraphrase\
        = load_semeval2016t6(semeval2016t6_path, GLOBAL_MAP['semeval2016t6'], create_adversarial=CREATE_ADVERSARIAL,
                             dataset='semeval2016t6', MAX_SEQ_LEN=MAX_SEQ_LEN,
                             OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(semeval2016t6_train_data, semeval2016t6_dev_data, semeval2016t6_test_data, semeval2016t6_test_negation,
        semeval2016t6_test_spelling, semeval2016t6_test_paraphrase, 'semeval2016t6')
    build_handler(semeval2016t6_train_data, semeval2016t6_dev_data, semeval2016t6_test_data, semeval2016t6_test_negation,
        semeval2016t6_test_spelling, semeval2016t6_test_paraphrase, 'semeval2016t6', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    ibmcs_train_data, ibmcs_dev_data, ibmcs_test_data, ibmcs_test_negation, ibmcs_test_spelling,\
        ibmcs_test_paraphrase = load_ibmcs(ibm_claim_stance_path, GLOBAL_MAP['ibmcs'], create_adversarial=CREATE_ADVERSARIAL,
                                           dataset='ibmcs', MAX_SEQ_LEN=MAX_SEQ_LEN,
                                           OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(ibmcs_train_data, ibmcs_dev_data, ibmcs_test_data, ibmcs_test_negation,
                 ibmcs_test_spelling, ibmcs_test_paraphrase, 'ibmcs')
    build_handler(ibmcs_train_data, ibmcs_dev_data, ibmcs_test_data, ibmcs_test_negation, ibmcs_test_spelling,
        ibmcs_test_paraphrase, 'ibmcs', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    iac1_train_data, iac1_dev_data, iac1_test_data, iac1_test_negation, iac1_test_spelling, iac1_test_paraphrase = \
        load_iac1(iac1_path, GLOBAL_MAP['iac1'], create_adversarial=CREATE_ADVERSARIAL, dataset='iac1', MAX_SEQ_LEN=MAX_SEQ_LEN,
                 OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(iac1_train_data, iac1_dev_data, iac1_test_data, iac1_test_negation, iac1_test_spelling,
                 iac1_test_paraphrase, 'iac1')
    build_handler(iac1_train_data, iac1_dev_data, iac1_test_data, iac1_test_negation, iac1_test_spelling,
                  iac1_test_paraphrase, 'iac1', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    snopes_train_data, snopes_dev_data, snopes_test_data, snopes_test_negation,\
        snopes_test_spelling, snopes_test_paraphrase = load_snopes(snopes_path, GLOBAL_MAP['snopes'],
                                                                  create_adversarial=CREATE_ADVERSARIAL,
                                                                  dataset='snopes', MAX_SEQ_LEN=MAX_SEQ_LEN,
                                                                  OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(snopes_train_data, snopes_dev_data, snopes_test_data, snopes_test_negation,
                 snopes_test_spelling, snopes_test_paraphrase, 'snopes')
    build_handler(snopes_train_data, snopes_dev_data, snopes_test_data, snopes_test_negation,
        snopes_test_spelling, snopes_test_paraphrase, 'snopes', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    arc_train_data, arc_dev_data, arc_test_data, arc_test_negation, arc_test_spelling, arc_test_paraphrase = \
        load_fnc1_arc(arc_path, GLOBAL_MAP['arc'], create_adversarial=CREATE_ADVERSARIAL,
                      dataset='arc', MAX_SEQ_LEN=MAX_SEQ_LEN,
                      OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(arc_train_data, arc_dev_data, arc_test_data, arc_test_negation, arc_test_spelling,
                 arc_test_paraphrase, 'arc')
    build_handler(arc_train_data, arc_dev_data, arc_test_data, arc_test_negation, arc_test_spelling,
                  arc_test_paraphrase, 'arc', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    semeval2019t7_train_data, semeval2019t7_dev_data, semeval2019t7_test_data, semeval2019t7_test_negation, \
        semeval2019t7_test_spelling, semeval2019t7_test_paraphrase\
        = load_semeval2019t7(semeval2019t7_path, GLOBAL_MAP['semeval2019t7'], create_adversarial=CREATE_ADVERSARIAL,
                             dataset='semeval2019t7', MAX_SEQ_LEN=MAX_SEQ_LEN,
                             OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(semeval2019t7_train_data, semeval2019t7_dev_data, semeval2019t7_test_data, semeval2019t7_test_negation,
        semeval2019t7_test_spelling, semeval2019t7_test_paraphrase, 'semeval2019t7')
    build_handler(semeval2019t7_train_data, semeval2019t7_dev_data, semeval2019t7_test_data, semeval2019t7_test_negation,
        semeval2019t7_test_spelling, semeval2019t7_test_paraphrase, 'semeval2019t7', mt_dnn_root, single=True, MAX_SEQ_LEN=MAX_SEQ_LEN)

    perspectrum_train_data, perspectrum_dev_data, perspectrum_test_data, perspectrum_test_negation, \
        perspectrum_test_spelling, perspectrum_test_paraphrase = load_perspectrum(perspectrum_path, GLOBAL_MAP['perspectrum'],
                                                                                 create_adversarial=CREATE_ADVERSARIAL,
                                                                                 dataset='perspectrum', MAX_SEQ_LEN=MAX_SEQ_LEN,
                                                                                 OPENNMT_GPU=OPENNMT_GPU,
                                                                                 OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(perspectrum_train_data, perspectrum_dev_data, perspectrum_test_data, perspectrum_test_negation,
        perspectrum_test_spelling, perspectrum_test_paraphrase, 'perspectrum')
    build_handler(perspectrum_train_data, perspectrum_dev_data, perspectrum_test_data, perspectrum_test_negation,
        perspectrum_test_spelling, perspectrum_test_paraphrase, 'perspectrum', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    argmin_train_data, argmin_dev_data, argmin_test_data, argmin_test_negation, \
        argmin_test_spelling, argmin_test_paraphrase = load_argmin(argmin_path, GLOBAL_MAP['argmin'],
                                                                  create_adversarial=CREATE_ADVERSARIAL,
                                                                  dataset='argmin', MAX_SEQ_LEN=MAX_SEQ_LEN,
                                                                  OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(argmin_train_data, argmin_dev_data, argmin_test_data, argmin_test_negation,
        argmin_test_spelling, argmin_test_paraphrase, 'argmin')
    build_handler(argmin_train_data, argmin_dev_data, argmin_test_data, argmin_test_negation,
        argmin_test_spelling, argmin_test_paraphrase, 'argmin', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

    fnc1_train_data, fnc1_dev_data, fnc1_test_data, fnc1_test_negation, fnc1_test_spelling,\
        fnc1_test_paraphrase = load_fnc1_arc(fnc1_path, GLOBAL_MAP['fnc1'], create_adversarial=CREATE_ADVERSARIAL,
                                             dataset='fnc1', MAX_SEQ_LEN=MAX_SEQ_LEN,
                                             OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)
    print_status(fnc1_train_data, fnc1_dev_data, fnc1_test_data, fnc1_test_negation,
                 fnc1_test_spelling, fnc1_test_paraphrase, 'fnc1')
    build_handler(fnc1_train_data, fnc1_dev_data, fnc1_test_data, fnc1_test_negation, fnc1_test_spelling,
        fnc1_test_paraphrase, 'fnc1', mt_dnn_root, MAX_SEQ_LEN=MAX_SEQ_LEN)

if __name__ == '__main__':
    args = parse_args()
    main(args)
