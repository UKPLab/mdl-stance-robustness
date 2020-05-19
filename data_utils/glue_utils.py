# Copyright (c) Microsoft. All rights reserved.
# Modified Copyright by Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
import json
import numpy as np
import random
import csv
import ast

import shutil
import zipfile
from tqdm import tqdm
import pandas as pd

from pytorch_pretrained_bert.tokenization import BertTokenizer

from .label_map import METRIC_FUNC, METRIC_META, METRIC_NAME, GLOBAL_MAP
import regex as re
from time import time
import os
from collections import defaultdict
import nltk
from collections import Counter
from .metrics import compute_f1_macro, compute_acc, compute_fnc1, compute_f1_micro, compute_f1_clw

import errno

import sentencepiece as spm
from data_utils import translate

### OpenNMT settings
VERBOSE = False # extra info for machine translation

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_sent_original_seq_len(hypo_orig, prem_orig, MAX_SEQ_LEN):
    # check how many wordpieces would be taken if no adversarial changes were applied
    hypo_temp = bert_tokenizer.tokenize(hypo_orig)
    prem_temp = bert_tokenizer.tokenize(prem_orig)
    truncate_seq_pair(prem_temp, hypo_temp, MAX_SEQ_LEN-3)
    return len(prem_temp), len(hypo_temp)

def create_sample_list_pair(X, y):
    sample_list = []
    for i, (prem, hypo) in enumerate(X):
        sample = {'uid': i, 'premise': prem, 'hypothesis': hypo, 'label': y[i]}
        sample_list.append(sample)
    return sample_list

def create_sample_list_single(X, y):
    sample_list = []
    for i, prem in enumerate(X):
        sample = {'uid': i, 'premise': prem, 'label': y[i]}
        sample_list.append(sample)
    return sample_list

def sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                    create_adversarial=False, pair_classification=True, dataset=None, MAX_SEQ_LEN=100,
                    OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):
    """
    Handles the single/pair sampling of train/dev/test and all adversarial sets and returns the results
    """
    def exists(dataset, adversarial_test):
        if os.path.isfile("data/mt_dnn/{0}_test_{1}.json".format(dataset, adversarial_test)):
            print("data/mt_dnn/{0}_test_{1}.json".format(dataset, adversarial_test) + " already existis. Skipping!")
            return True
        else:
            return False


    sampling_fct = create_sample_list_single if pair_classification == False else create_sample_list_pair

    samples_train, samples_dev, samples_test = sampling_fct(X_train, y_train), \
                                               sampling_fct(X_dev, y_dev), \
                                               sampling_fct(X_test, y_test)

    samples_test_negation, samples_test_spelling, sampels_test_paraphrase = None, None, None
    if create_adversarial == True:
        if not exists(dataset, "negation"):
            samples_test_negation = create_adversarial_negation(sampling_fct(X_test, y_test), MAX_SEQ_LEN)
        if not exists(dataset, "spelling"):
            samples_test_spelling = create_adversarial_spelling(sampling_fct(X_test, y_test), MAX_SEQ_LEN)
        if not exists(dataset, "paraphrase"):
            sampels_test_paraphrase = create_adversarial_paraphrase(sampling_fct(X_test, y_test), dataset=dataset,
                                                                    OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

        print("Finish preprocessing file: " + file + " in " + str(time() - t_total) + "s\n")
        return samples_train, samples_dev, samples_test, samples_test_negation, samples_test_spelling, \
               sampels_test_paraphrase

    print("Finish preprocessing file: " + file + " in " + str(time() - t_total) + "s\n")
    return samples_train, samples_dev, samples_test, None, None, None

def create_adversarial_spelling(sample_list, MAX_SEQ_LEN=100):
    """
    Swaps letters in words longer than three tokens. Adapted from:
    Reference:
        Naik, Aakanksha, Abhilasha Ravichander, Norman Sadeh, Carolyn Rose, and Graham Neubig.
        "Stress Test Evaluation for Natural Language Inference." In Proceedings of the 27th International
        Conference on Computational Linguistics, pp. 2340-2353. 2018.
    """
    def perturb_word_swap(word):
        char_ind = int(np.random.uniform(1, len(word) - 2))
        new_word = list(word)
        first_char = new_word[char_ind]
        new_word[char_ind] = new_word[char_ind + 1]
        new_word[char_ind + 1] = first_char
        new_word = "".join(new_word)
        return new_word

    def perturb_word_kb(word):
        keyboard_char_dict = {"a": ['s'], "b": ['v', 'n'], "c": ['x', 'v'], "d": ['s', 'f'], "e": ['r', 'w'],
                              "f": ['g', 'd'], "g": ['f', 'h'], "h": ['g', 'j'], "i": ['u', 'o'], "j": ['h', 'k'],
                              "k": ['j', 'l'], "l": ['k'], "m": ['n'], "n": ['m', 'b'], "o": ['i', 'p'], "p": ['o'],
                              "q": ['w'], "r": ['t', 'e'], "s": ['d', 'a'], "t": ['r', 'y'],
                              "u": ['y', 'i'], "v": ['c', 'b'], "w": ['e', 'q'], "x": ['z', 'c'], "y": ['t', 'u'],
                              "z": ['x']}

        new_word = list(word)
        acceptable_subs = []
        for ind, each_char in enumerate(new_word):
            if each_char.lower() in keyboard_char_dict.keys():
                acceptable_subs.append(ind)

        if len(acceptable_subs) == 0:
            return word

        char_ind = random.choice(acceptable_subs)

        first_char = new_word[char_ind]

        new_word[char_ind] = random.choice(keyboard_char_dict[first_char.lower()])
        final_new_word = "".join(new_word)
        return final_new_word

    def swap(sent):
        tokens = sent.split(" ")
        token_used = -1
        # we cut the sent later at a len of 100, so we don't want to have the adversarial changes after the cut
        rand_indices = random.sample(range(15 if len(tokens) > 15 else len(tokens)), (15 if len(tokens) > 15 else len(tokens)))
        for rand_index in rand_indices:
            if len(tokens[rand_index]) > 3:
                tokens[rand_index] = perturb_word_kb(tokens[rand_index])
                token_used = rand_index
                break

        for rand_index in rand_indices:
            if len(tokens[rand_index]) > 3 and rand_index != token_used:
                tokens[rand_index] = perturb_word_swap(tokens[rand_index])
                break

        return " ".join(tokens)

    def undo_wp(sent_wp):
        sent_redo = ""
        for index, t in enumerate(sent_wp):
            if t.startswith("##"):
                sent_redo += t[2:]
            elif index == 0:
                sent_redo += t
            else:
                sent_redo += " " + t
        return sent_redo

    def add_lost_info(sent_orig, sent_swap, orig_wp_len):

        sent_orig_len = len(bert_tokenizer.tokenize(sent_orig))
        sent_wp = bert_tokenizer.tokenize(sent_swap)

        additional_len = len(sent_wp) - sent_orig_len

        sent_wp = sent_wp[:orig_wp_len+additional_len]
        sent_wp = undo_wp(sent_wp)
        return sent_wp

    print("Swap letters from test set sentences.")
    for sample in tqdm(sample_list):
        if "hypothesis" in sample_list[0].keys():
            # check how many wordpieces would be taken if no adversarial changes were applied
            prem_orig_wp_len, hypo_orig_wp_len = get_sent_original_seq_len(sample['hypothesis'], sample['premise'], MAX_SEQ_LEN)

            # get swapped sentence and the indices of the swapped tokens
            temp_swap = swap(sample['hypothesis'])

            # add additional info to sentence that would be cut off due to lengthening the sent with spelling errors
            sample['hypothesis'] = add_lost_info(sample['hypothesis'], temp_swap, hypo_orig_wp_len)

            # same for premise
            temp_swap = swap(sample['premise'])
            sample['premise'] = add_lost_info(sample['premise'], temp_swap, prem_orig_wp_len)
        else:
            temp_swap = swap(sample['premise'])
            sample['premise'] = add_lost_info(sample['premise'], temp_swap,
                                              min(len(bert_tokenizer.tokenize(sample['premise'])), MAX_SEQ_LEN-3))
    return sample_list

def create_adversarial_paraphrase(sample_list, dataset=None, MAX_SEQ_LEN=100, OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):
    """
    Translate sample sentences into german and back to english to generate paraphrases. Two OpenNMT models
    were especially trained for this purpose.
    """
    def create_df(sample_list, num_samples=50):
        random_ids = random.sample(range(len(sample_list)), num_samples)
        if "hypothesis" in sample_list[0].keys():
            return pd.DataFrame(columns=['hypo_orig', 'hypo_trans', 'prem_orig', 'prem_trans', 'stance'], index=random_ids)
        else:
            return pd.DataFrame(columns=['hypo_orig', 'hypo_trans', 'stance'], index=random_ids)

    def safe_rm_file(file):
        try:
            os.remove(file)
        except OSError:
            pass

    def translate_from_to(src, tgt, backtranslated=False, verbose=False):
        """
        Translates all sentencepiece preprocessed samples from src to tgt language.
        Backtranslated parameter only influences the file name of the result files.
        """
        start = time()

        bt_placeholder = "" if backtranslated == False else ".backtranslated"
        verb_placeholder = "" if verbose == False else "-verbose "

        if "hypothesis" in sample_list[0].keys():
            print("Translating hypotheses to {0}".format(tgt))
            parser = translate._get_parser()
            opt = parser.parse_args(
                args="-model translation_models/{0}-{1}/model/opennmt_{2}-{3}.final.pt -replace_unk {4} -gpu {5} -batch_size {6} "
                         .format(src, tgt, src, tgt, verb_placeholder, OPENNMT_GPU, OPENNMT_BATCH_SIZE)
                     +"-src translation_models/hypotheses-{0}.txt -output translation_models/hypotheses-{1}{2}.txt"
                         .format(src, tgt, bt_placeholder))
            translate.main(opt)

        print("Translating premises to {0}".format(tgt))
        parser = translate._get_parser()
        opt = parser.parse_args(
            args="-model translation_models/{0}-{1}/model/opennmt_{2}-{3}.final.pt -gpu 1 -replace_unk {4} -gpu {5} -batch_size {6} "
                     .format(src, tgt, src, tgt, verb_placeholder, OPENNMT_GPU, OPENNMT_BATCH_SIZE)
                     +"-src translation_models/premises-{0}.txt -output translation_models/premises-{1}{2}.txt"
                     .format(src, tgt, bt_placeholder))
        translate.main(opt)

        print("Took {0}sec to translate data from {1} to {2}.".format("{0:.2f}".format(round(time()-start, 2)), src, tgt))

    def save_and_encode_sents(file, MAX_SEQ_LEN):
        """
        Save hypotheses and premises to text files and encode with sentencepiece.
        As the MT model has trouble with long sentences, we split the sentences and store a mapping to revert them back after translation
        """

        # maps original index of sentence to new, temporary, indices of split sentence
        map = defaultdict(list)
        empty_lines = []

        with open(file, "w") as out_f:
            s = spm.SentencePieceProcessor()
            s.Load('translation_models/en-de/model/sentencepiece-en.model')

            real_count = 0
            for i, sample in tqdm(enumerate(sample_list)):
                count_tokens = 0
                sample = sample_list[i]
                sents = nltk.sent_tokenize(sample[("hypothesis" if "hypotheses" in file else "premise")])

                if len(sents) == 0: #e.g. semeval2019t7 actually has a few empty samples that have to be treated accordingly
                    empty_lines.append(real_count)
                    map[i].append(real_count)
                    out_f.write("\n")
                    real_count += 1

                for j, sent in enumerate(sents):
                    if count_tokens >= MAX_SEQ_LEN:
                        # We can safely stop at MAX_SEQ_LEN tokens input
                        break

                    count_tokens += len(sent.split(" "))
                    map[i].append(real_count)
                    out_f.write(" ".join([t for t in s.EncodeAsPieces(sent)]) +
                                ("" if (i == len(sample_list) - 1 and j == len(sents) - 1) else "\n"))
                    real_count += 1

        return map, empty_lines

    def decode_and_load_sents(map, empty_lines, file, original_src, sample_list, sample_list_key, random_picks, dataset):
        """
        Sentencepiece decodes samples back to original, loads them, and replaces them with current sample list
        """
        with open(file, "r") as in_p_f:
            s = spm.SentencePieceProcessor()
            s.Load('translation_models/en-de/model/sentencepiece-{0}.model'.format(original_src))

            reverted_map = {value:key for key, values in map.items() for value in values}
            new_samples_dict = defaultdict(list)

            for i, sample in tqdm(enumerate(in_p_f.readlines())):
                if i in empty_lines:
                    new_samples_dict[reverted_map[i]].append("") # these lines were empty before
                else:
                    new_samples_dict[reverted_map[i]].append(sample.rstrip())

            assert len(new_samples_dict.items()) == len(sample_list), "Uneven sizes of old and new samples when " \
                                                                              "re-assembling split sentences of dataset."

            # fill sample_list with translated sentences, also add random samples to dataframe for analysis
            for id, sents in sorted(new_samples_dict.items()):
                # decode sentence
                temp = " ".join(sents).split(" ")
                temp = s.DecodePieces(temp)

                # add random picks of sentences for analysis
                if id in random_picks.index.tolist() and sample_list_key == "premise":
                    random_picks.at[id, "hypo_orig"] = sample_list[id]["premise"]
                    random_picks.at[id, "hypo_trans"] = temp
                    random_picks.at[id, "stance"] = GLOBAL_MAP[dataset][sample_list[id]["label"]]
                elif id in random_picks.index.tolist() and sample_list_key == "hypothesis": # hypo is always second, so the dataframe is filled
                    random_picks.at[id, "prem_orig"] = sample_list[id]["hypothesis"]
                    random_picks.at[id, "prem_trans"] = temp

                # add backtranslated sentence to original sample list
                sample_list[id][sample_list_key] = temp

        return random_picks

    print("Create paraphrase adversarial samples (this can take a long time. For monitoring of the translation process, "
          "set VERBOSE=True in glue_utils.py).")

    # sentencepiece encode sentences and store. If samples have more than one sentence, they are split and treated separately,
    # as the model has problems with long sentences (and takes much longer to translate them)
    if "hypothesis" in sample_list[0].keys():
        hypo_map, hypo_empty_lines = save_and_encode_sents("translation_models/hypotheses-en.txt", MAX_SEQ_LEN)
    prem_map, prem_empty_lines = save_and_encode_sents("translation_models/premises-en.txt", MAX_SEQ_LEN)

    # translate to german
    translate_from_to("en", "de", backtranslated=False, verbose=VERBOSE)

    # translate back to english
    translate_from_to("de", "en", backtranslated=True, verbose=VERBOSE)

    # we will pick random samples and save the corresponding original and translated data for later analysis
    random_picks = create_df(sample_list, num_samples=50)

    # decode sentencepiece sentences, load, and re-assemble to old structure
    random_picks = decode_and_load_sents(prem_map, prem_empty_lines, "translation_models/premises-en.backtranslated.txt",
                          "en", sample_list, "premise", random_picks, dataset)
    if "hypothesis" in sample_list[0].keys():
        random_picks = decode_and_load_sents(hypo_map, hypo_empty_lines, "translation_models/hypotheses-en.backtranslated.txt",
                              "en", sample_list, "hypothesis", random_picks, dataset)

    # save random picks of translated samples for analysis
    make_dir("analysis")
    random_picks.sort_index(inplace=True)
    random_picks.to_csv(r'analysis/{0}.csv'.format(dataset), header=True)

    # remove all files all files
    safe_rm_file("translation_models/premises-en.backtranslated.txt")
    safe_rm_file("translation_models/premises-en.txt")
    safe_rm_file("translation_models/premises-de.txt")
    safe_rm_file("translation_models/hypotheses-en.backtranslated.txt")
    safe_rm_file("translation_models/hypotheses-en.txt")
    safe_rm_file("translation_models/hypotheses-de.txt")

    return sample_list

def create_adversarial_negation(sample_list, MAX_SEQ_LEN):
    """
    Add tautology "and false is not true" at the beginning of the hypothesis or premise
    Reference:
        Naik, Aakanksha, Abhilasha Ravichander, Norman Sadeh, Carolyn Rose, and Graham Neubig.
        "Stress Test Evaluation for Natural Language Inference." In Proceedings of the 27th
        International Conference on Computational Linguistics, pp. 2340-2353. 2018.
    """

    def cut_at_max_seq_len(sent, orig_wp_len):
        # prevents new information to follow into the sequence through removing stopword
        def undo_wp(sent_wp):
            sent_redo = ""
            for index, t in enumerate(sent_wp):
                if t.startswith("##"):
                    sent_redo += t[2:]
                elif index == 0:
                    sent_redo += t
                else:
                    sent_redo += " " + t
            return sent_redo

        sent_wp = bert_tokenizer.tokenize(sent)
        sent_wp = sent_wp[:orig_wp_len]
        sent_wp = undo_wp(sent_wp)
        return sent_wp

    print("Add negation word to test set sentences.")
    if "hypothesis" in sample_list[0].keys():
        for sample in tqdm(sample_list):
            prem_orig_wp_len, hypo_orig_wp_len = get_sent_original_seq_len(sample['hypothesis'], sample['premise'], MAX_SEQ_LEN)
            sample['premise'] = cut_at_max_seq_len(sample['premise'], prem_orig_wp_len)
            sample['hypothesis'] = cut_at_max_seq_len(sample['hypothesis'], hypo_orig_wp_len)
            sample['hypothesis'] = "false is not true and " + sample['hypothesis']
    else:
        for sample in tqdm(sample_list):
            sample['premise'] = cut_at_max_seq_len(sample['premise'], MAX_SEQ_LEN-3)
            sample['premise'] = "false is not true and " + sample['premise']

    return sample_list

def load_argmin(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):
    t_total = time()

    # load all into a topic-separated dict with train, dev, test lists
    gold_files = os.listdir(file)
    X_train, X_dev, X_test = [], [], []
    y_train, y_dev, y_test = [], [], []
    for data_file in gold_files:
        if data_file.endswith(".tsv"):
            topic = data_file.replace(".tsv", "")
            with open(file + data_file, 'r') as f_in:
                reader = csv.reader(f_in, delimiter="\t", quoting=3)
                next(reader, None)
                for row in reader:
                    if label_dict[row[5]] != None:
                        if topic == "death_penalty":
                            X_dev.append((row[0], row[4]))
                            y_dev.append(label_dict[row[5]])
                        elif topic == "school_uniforms" or topic == "gun_control":
                            X_test.append((row[0], row[4]))
                            y_test.append(label_dict[row[5]])
                        else:
                            X_train.append((row[0], row[4]))
                            y_train.append(label_dict[row[5]])

    print("\n===================== Start preprocessing file: "+ file + " =====================")

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_iac1(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    # load all into a topic-separated dict with train, dev, test lists
    def read_and_tokenize_csv(file, label_dict, is_test=False):

        def get_label(id):
            return {
                0: "pro",
                1: "anti",
                2: "other"
            }[id]

        # get sample rows with discussion ids and labels
        sample_dict = defaultdict(list)
        with open(file+"author_stance.csv", "r") as in_f:
            reader = csv.reader(in_f)
            next(reader)
            labels = [0, 0, 0]  # pro con other
            topics_count = []
            for row in reader:
                label_id = np.argmax([row[3], row[4], row[5]])
                labels[label_id] += 1

                topics_count.append(row[0])

                sample_dict[row[1]].append({
                    "author": row[2],
                    "topic": row[0],
                    "label": label_dict[get_label(label_id)]
                })

            print(labels)

        # get dicussion_id -> author_sentences dict
        discussion_dict = {}
        for d_id in sample_dict.keys():
            with open(file+"discussions/"+str(d_id)+".json", "r") as in_f:
                discussion_data = json.load(in_f)
                user_posts = defaultdict(list)
                for post in discussion_data[0]:
                    user_posts[post[2]].append(post[3])
                discussion_dict[d_id] = user_posts

        # create X,y dample lists
        X = []
        y = []
        text_lens = []
        for d_id in sample_dict:
            for author_data in sample_dict[d_id]:
                text = " ".join([s for s in discussion_dict[d_id][author_data["author"]]])
                text_lens.append(len(text.split()))
                X.append((author_data["topic"], text))
                y.append(author_data["label"])

        return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    X, y = read_and_tokenize_csv(file, label_dict, is_test=False)

    # split without topic leakage
    topic_set_dict = {
        "evolution": "train",
        "death penalty": "train",
        "gay marriage": "train",
        "climate change": "dev",
        "gun control": "train",
        "healthcare": "train",
        "abortion": "train",
        "existence of god": "test",
        "communism vs capitalism": "dev",
        "marijuana legalization": "test"
    }

    X_train, X_dev, X_test = [], [], []
    y_train, y_dev, y_test = [], [], []
    for i in range(len(X)):
        if topic_set_dict[X[i][0]] == "train":
            X_train.append(X[i])
            y_train.append(y[i])
        elif topic_set_dict[X[i][0]] == "dev":
            X_dev.append(X[i])
            y_dev.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_fnc1_arc(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def create_data(split_path, data_sents, bodies_lookup, label_dict):
        X = []
        y = []
        with open(split_path, "r") as in_dev_split:
            ids = ast.literal_eval(in_dev_split.readline())
            for id in ids:
                X.append((data_sents[id][0], bodies_lookup[int(data_sents[id][1])]))
                y.append(label_dict[data_sents[id][2]])
        return X, y

    def read_csv_from_split(file, label_dict):
        # train and dev set are not fixed, hence, read from split_file

        if "FNC-1" in file:
            sents_path = file+"train_stances.csv"
            bodies_path = file+"train_bodies.csv"
            dev_split_path = "data_splits/fnc1_dev.csv"
            train_split_path = "data_splits/fnc1_train.csv"
        else: # if ARC corpus is preprocessed
            sents_path = file+"arc_stances_train.csv"
            bodies_path = file + "arc_bodies.csv"
            dev_split_path = "data_splits/arc_dev.csv"
            train_split_path = "data_splits/arc_train.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}
            data_sents = list(data_sents)

        X_train, y_train = create_data(train_split_path, data_sents, bodies_lookup, label_dict)
        X_dev, y_dev = create_data(dev_split_path, data_sents, bodies_lookup, label_dict)

        return X_train, y_train, X_dev, y_dev

    def read_csv(file, label_dict):
        # test set is fixed, hence, read from file

        X = []
        y = []

        if "FNC-1" in file:
            sents_path = file+"competition_test_stances.csv"
            bodies_path = file+"competition_test_bodies.csv"
        else: # if ARC corpus is preprocessed
            sents_path = file+"arc_stances_test.csv"
            bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}

            # generate train instances
            for row in data_sents:
                if row[2] not in ["unrelated", "discuss", "disagree", "agree"]:
                    continue

                X.append((row[0], bodies_lookup[int(row[1])]))
                y.append(label_dict[row[2]])

            return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read and tokenize data
    X_train, y_train, X_dev, y_dev = read_csv_from_split(file, label_dict)
    X_test, y_test = read_csv(file, label_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict, fnc_arc=True)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_perspectrum(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def read_and_tokenize_csv(file, label_dict, is_test=False):

        with open(file+"dataset_split_v1.0.json", "r") as split_in, \
                open(file+"perspectrum_with_answers_v1.0.json","r") as claims_in, \
                open(file+"perspective_pool_v1.0.json", "r") as perspectives_in:

            # load files
            data_split = json.load(split_in)
            claims = json.load(claims_in)
            perspectives = json.load(perspectives_in)

            # lookup for perspective ids
            perspectives_dict = {}
            for p in perspectives:
                perspectives_dict[p['pId']] = p['text']

            # init
            X_train, X_dev, X_test = [], [], []
            y_train, y_dev, y_test = [], [], []

            count_sup_cluster = 0
            count_und_cluster = 0

            # fill train/dev/test
            for claim in claims:
                cId = str(claim['cId'])
                for p_cluster in claim['perspectives']:
                    cluster_label = label_dict[p_cluster['stance_label_3']]
                    for pid in p_cluster['pids']:
                        if data_split[cId] == 'train':
                            X_train.append((claim['text'], perspectives_dict[pid]))
                            y_train.append(cluster_label)
                        elif data_split[cId] == 'dev':
                            X_dev.append((claim['text'], perspectives_dict[pid]))
                            y_dev.append(cluster_label)
                        elif data_split[cId] == 'test':
                            X_test.append((claim['text'], perspectives_dict[pid]))
                            y_test.append(cluster_label)
                        else:
                            print("Incorrect set type: "+data_split[claim['cId']])
                    if cluster_label == 1:
                        count_sup_cluster += 1
                    if cluster_label == 0:
                        count_und_cluster += 1



        print('\n# of "support" perspectives: ' + str(count_sup_cluster))
        print('# of "opposing" perspectives: ' + str(count_und_cluster))
        print('Note: Counts cluster-wise, not perspective-wise!\n')

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read and tokenize data
    X_train, y_train, X_dev, y_dev, X_test, y_test = read_and_tokenize_csv(file, label_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_ibmcs(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):
    def read(file, split_file, label_dict):
        X, y = [], []

        with open(file, "r") as in_f, open(split_file, "r") as split_in:
            data = csv.reader(in_f, delimiter = ',', quotechar = '"')
            data = list(data)

            for split_id in split_in.readlines():
                id = int(split_id.rstrip())
                X.append((data[id][2], data[id][7]))
                y.append(label_dict[data[id][6]])

        return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read and tokenize data
    X_train, y_train = read(file, "data_splits/ibmcs_train.csv", label_dict)
    X_dev, y_dev = read(file, "data_splits/ibmcs_dev.csv", label_dict)
    X_test, y_test = read(file, "data_splits/ibmcs_test.csv", label_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_semeval2016t6(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def read(train_file, trial_file, test_file, split_file, label_dict):
        X, y = [], []

        with open(file+train_file, "r", encoding="ISO-8859-1") as in_train_f,\
        open(file+trial_file, "r", encoding="ISO-8859-1") as in_trial_f,\
        open(file+test_file, "r", encoding="ISO-8859-1") as in_test_f,\
        open(split_file, "r") as in_split_file:

            data_train = list(csv.reader(in_train_f, delimiter='\t', quotechar = '"', ))
            data_trial = list(csv.reader(in_trial_f, delimiter='\t', quotechar = '"', ))
            data_test = list(csv.reader(in_test_f, delimiter='\t', quotechar = '"', ))

            for row in in_split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)

                if data_file == "train":
                    X.append((data_train[line_number][1], data_train[line_number][2]))
                    y.append(label_dict[data_train[line_number][3]])
                elif data_file == "trial":
                    X.append((data_trial[line_number][1], data_trial[line_number][2]))
                    y.append(label_dict[data_trial[line_number][3]])
                else:
                    X.append((data_test[line_number][1], data_test[line_number][2]))
                    y.append(label_dict[data_test[line_number][3]])

        return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read and tokenize data
    X_train, y_train = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "data_splits/semeval2016t6_train.csv", label_dict)
    X_dev, y_dev = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "data_splits/semeval2016t6_dev.csv", label_dict)
    X_test, y_test = read("trainingdata-all-annotations.txt",
                            "trialdata-all-annotations.txt",
                            "testdata-taskA-all-annotations.txt",
                            "data_splits/semeval2016t6_test.csv", label_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict, semeval2016t6=True)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_semeval2019t7(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def parse_tweets(folder_path):
        # create a dict with key = reply_tweet_id and values = source_tweet_id, source_tweet_text, reply_tweet_text
        tweet_dict = {}
        with zipfile.ZipFile(file+folder_path, 'r') as z:
            for filename in z.namelist():
                if not filename.lower().endswith(".json") or filename.rsplit("/", 1)[1] in ['raw.json', 'structure.json', 'dev-key.json', 'train-key.json']:
                    continue
                with z.open(filename) as f:
                    data = f.read()
                    d = json.loads(data.decode("ISO-8859-1"))

                    if "data" in d.keys(): #reddit
                        if "body" in d['data'].keys(): # reply
                            tweet_dict[d['data']['id']] = d['data']['body']
                        elif "children" in d['data'].keys() and isinstance(d['data']['children'][0], dict):
                            tweet_dict[d['data']['children'][0]['data']['id']] = d['data']['children'][0]['data']['title']
                        else: # source
                            try:
                                tweet_dict[d['data']['children'][0]] = ""
                            except Exception as e:
                                print(e)

                    if "text" in d.keys(): # twitter
                        tweet_dict[str(d['id'])] = d['text']

        return tweet_dict

    def read_and_tokenize_json(file_name, label_dict, tweet_dict):
        X, y = [], []

        with open(file+file_name, "r") as in_f:
            split_dict = json.load(in_f)['subtaskaenglish']

            for tweet_id, label in split_dict.items():
                try:
                    X.append(tweet_dict[tweet_id])
                except:
                    continue
                y.append(label_dict[label])

        return X, y

    def read_and_tokenize_zip(folder_path, set_file, label_dict, tweet_dict):
        X, y = [], []

        with zipfile.ZipFile(file + folder_path, 'r') as z:
            with z.open("rumoureval-2019-training-data/"+set_file) as in_f:
                split_dict = json.load(in_f)['subtaskaenglish']

                for tweet_id, label in split_dict.items():
                    try:
                        X.append(tweet_dict[tweet_id])
                    except:
                        continue
                    y.append(label_dict[label])

            return X, y

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read train/dev data
    tweet_dict = parse_tweets("rumoureval-2019-training-data.zip")
    X_train, y_train = read_and_tokenize_zip("rumoureval-2019-training-data.zip", "train-key.json", label_dict, tweet_dict)
    X_dev, y_dev = read_and_tokenize_zip("rumoureval-2019-training-data.zip", "dev-key.json", label_dict, tweet_dict)

    # read test data
    tweet_dict = parse_tweets("rumoureval-2019-test-data.zip")
    X_test, y_test = read_and_tokenize_json("final-eval-key.json", label_dict, tweet_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=False, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_scd(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def read_and_tokenize_zip(file_name):
        X_train, X_dev, X_test = [], [], []
        y_train, y_dev, y_test = [], [], []
        split_dict = {
            "abortion": "train",
            "gayRights": "train",
            "marijuana": "dev",
            "obama": "test"
        }
        label_converter = {"Stance=+1": "for", "Stance=-1": "against"}
        with zipfile.ZipFile(file+file_name, 'r') as z:
            z.extractall(file)

            for topic in split_dict.keys():
                topic_files = os.listdir(file+file_name[:-4]+"/"+topic+"/")

                for topic_file in topic_files:
                    if topic_file.endswith(".meta"):
                        continue

                    with open(file+file_name[:-4]+"/"+topic+"/"+topic_file, "r", encoding="ISO-8859-1") as data_f,\
                        open(file+file_name[:-4]+"/"+topic+"/"+topic_file[:-5]+".meta", "r") as meta_f:
                        text = data_f.readlines()
                        if topic_file == "G35.data" and topic == "gayRights":
                            text_label = "for"  # missing label: supports its parent, which is G34 and has stance +1
                        else:
                            text_label = label_converter[meta_f.readlines()[2].rstrip("\n")]

                    # the dataset contains duplicates, sometimes with contradicting labels
                    # -> we take only the first appearance of a sentence
                    if text[0] in X_train or text[0] in X_dev or text[0] in X_test:
                        continue

                    if split_dict[topic] == "train":
                        X_train.extend(text)
                        y_train.append(label_dict[text_label])
                    elif split_dict[topic] == "dev":
                        X_dev.extend(text)
                        y_dev.append(label_dict[text_label])
                    else:
                        X_test.extend(text)
                        y_test.append(label_dict[text_label])
        try:
            shutil.rmtree(file+file_name[:-4], ignore_errors=True)
        except Exception as e:
            print("Cannot delete folder at " + file+file_name[:-4])
        return X_train, X_dev, X_test, y_train, y_dev, y_test


    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read train/dev data
    X_train, X_dev, X_test, y_train, y_dev, y_test = read_and_tokenize_zip("stance.zip")

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=False, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def load_snopes(file, label_dict, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def read(file, label_dict):
        data_dict = {}
        count_sup = 0
        count_ref = 0
        with open(file, "r") as in_f, open("data_splits/snopes_train.csv", "r") as train_split_out, \
                open("data_splits/snopes_dev.csv", "r") as dev_split_out, open("data_splits/snopes_test.csv", "r") as test_split_out:
            data = csv.reader(in_f, delimiter = ',', quotechar = '"')
            next(data)

            for row in data:
                if row[11] == "nostance" or "no_evidence" in row[12]:
                    continue

                evid_ids = row[12].split("_")[1:]
                fges = re.findall("[0-9]*_{(.+?)}", row[9])
                claim = row[6].lower()

                if len(evid_ids) == 0:
                    continue

                assert len(re.findall("[0-9]*_{(.+?)}", row[9])) == len(re.findall("[0-9]*_{.+?}", row[9])), "Error in regex"

                for evid_id in evid_ids:
                    split_fge = fges[int(evid_id)].lower()


                    if row[0] not in data_dict.keys():
                        data_dict[row[0]] = {
                            "claim": claim,
                            "label": label_dict[row[11]],
                            "evidence": {evid_id: split_fge}
                        }
                    else:
                        data_dict[row[0]]["evidence"][evid_id] = split_fge


                    if row[11] == "agree":
                        count_sup += 1
                    else:
                        count_ref += 1

            X_train, X_dev, X_test = [], [], []
            y_train, y_dev, y_test = [], [], []

            for line in train_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_train.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_train.append(data_dict[claim_id]["label"])

            for line in dev_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_dev.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_dev.append(data_dict[claim_id]["label"])

            for line in test_split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X_test.append((data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]))
                y_test.append(data_dict[claim_id]["label"])

        return X_train, y_train, X_dev, y_dev, X_test, y_test


    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read and tokenize data
    X_train, y_train, X_dev, y_dev, X_test, y_test = read(file, label_dict)

    print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict)

    return sample_handling(X_train, y_train, X_dev, y_dev, X_test, y_test, file, t_total,
                           create_adversarial=create_adversarial, pair_classification=True, dataset=dataset,
                           MAX_SEQ_LEN=MAX_SEQ_LEN, OPENNMT_GPU=OPENNMT_GPU, OPENNMT_BATCH_SIZE=OPENNMT_BATCH_SIZE)

def print_stats(X_train, X_dev, X_test, y_train, y_dev, y_test, label_dict, semeval2016t6=False, fnc_arc=False):
    def count_classes(y):
        """
        Returns a dictionary with class balances for each class (absolute and relative)
        :param y: gold labels as array of int
        """
        count_dict = Counter([label_dict[l] for l in y])
        return {l:str(c)+" ({0:.2f}".format((c/len(y))*100)+"%)" for l, c  in count_dict.items()}

    def get_random_baseline_predictions(y):
        """
        Returns array of random integers of len(y) between 0 and max(y)
        :param y:
        """
        return np.random.randint(0, max(y)+1, len(y))

    def get_majority_baseline_predictions(y):
        """
        Returns array of len(y) with integer value of majority class
        :param y: gold labels as array of int
        """
        class_counter = Counter(y)
        return [max(class_counter, key=class_counter.get)]*len(y)

    # unique claims per set
    if len(X_train[0]) == 2: # check if dataset has a topic
        print("# unique topics in train: " + str(len(set([t for t, x in X_train]))))
        print("# unique topics in dev: " + str(len(set([t for t, x in X_dev]))))
        print("# unique topics in test: " + str(len(set([t for t, x in X_test]))))

    # class balance per set
    print("\nClass balance for train: " + str(count_classes(y_train)))
    print("Class balance for dev: " + str(count_classes(y_dev)))
    print("Class balance for test: " + str(count_classes(y_test)))
    print("Class balance for all data: " + str(count_classes(y_train+y_dev+y_test)))

    # calc rmd baseline on test
    print("\nRandom Baseline F1 macro on test: " + str(compute_f1_macro(get_random_baseline_predictions(y_test), y_test)))
    print("Random Baseline F1 micro on test: " + str(compute_f1_micro(get_random_baseline_predictions(y_test), y_test)))
    print("Random Baseline ACC on test: " + str(compute_acc(get_random_baseline_predictions(y_test), y_test)))
    if fnc_arc == True:
        print("Random Baseline FNC1 on test: " + str(compute_fnc1(get_random_baseline_predictions(y_test), y_test)))
    if semeval2016t6 == True:
        f1_clw = compute_f1_clw(get_random_baseline_predictions(y_test), y_test)
        print("Random Baseline F1 macro w\o None class: " + str((f1_clw[label_dict["AGAINST"]]+f1_clw[label_dict["FAVOR"]])/2.0))

    # calc majority baseline on test
    print("Majority Baseline F1 macro on test: " + str(compute_f1_macro(get_majority_baseline_predictions(y_test), y_test)))
    print("Majority Baseline F1 micro on test: " + str(compute_f1_micro(get_majority_baseline_predictions(y_test), y_test)))
    print("Majority Baseline ACC on test: " + str(compute_acc(get_majority_baseline_predictions(y_test), y_test)))
    if fnc_arc == True:
        print("Majority Baseline FNC1 on test: " + str(compute_fnc1(get_majority_baseline_predictions(y_test), y_test)))
    if semeval2016t6 == True:
        f1_clw = compute_f1_clw(get_majority_baseline_predictions(y_test), y_test)
        print("Majority Baseline F1 macro w\o None class: " + str((f1_clw[label_dict["AGAINST"]]+f1_clw[label_dict["FAVOR"]])/2.0))

    # train/dev/test size
    print("\nTrain set size: " + str(len(X_train)))
    print("Dev set size: " + str(len(X_dev)))
    print("Test set size: " + str(len(X_test)))

    # avg topic/hypothesis length
    if len(X_train[0]) == 2: # check if dataset has a topic
        split_topics, split_hypos = zip(*[(len(t.split()), len(h.split())) for t, h in X_train+X_dev+X_test])
        print("\nAvg. topic length: "+str(int(np.average(split_topics))))
        print("Avg. #tokens in hypothesis: "+str(int(np.average(split_hypos))))
        print("Max #tokens in topic: "+str(int(np.max(split_topics))))
        print("Max #tokens in hypothesis: "+str(int(np.max(split_hypos))))
    else:
        split_hypos = [len(h.split()) for h in X_train + X_dev + X_test]
        print("\nAvg. hypothesis length: " + str(int(np.average(split_hypos))))
        print("Max hypothesis length: " + str(int(np.max(split_hypos))))

    print()

def submit(path, data, label_dict=None):
    header = 'index\tprediction\tgold\tpremise\thypothesis'
    with open(path ,'w') as writer:
        predictions, uids, premises, hypotheses, golds = data['predictions'], data['uids'], data['premises'], data['hypotheses'], data['golds']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx], golds[idx], premises[idx], hypotheses[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred, gold, prem, hypo in paired:
            if label_dict is None:
                writer.write('{}\t{}\t{}\t{}\t{}\n'.format(uid, pred, gold, prem, hypo))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\t{}\t{}\t{}\n'.format(uid, label_dict[pred], label_dict[gold], prem, hypo))

def eval_model(model, data, dataset, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    premises = []
    hypotheses = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
        premises.extend(batch_meta['premises'])
        hypotheses.extend(batch_meta['hypotheses'])
    mmeta = METRIC_META[dataset]
    if with_label:
        for mm in mmeta:
            metric_name = METRIC_NAME[mm]
            metric_func = METRIC_FUNC[mm]

            if mm < 3 or mm in [5, 6, 7, 8, 9]:
                metric = metric_func(predictions, golds)
                metrics[metric_name] = metric
            elif mm in [10, 11, 12]:
                metric = metric_func(predictions, golds)
                for i, res in enumerate(metric):
                    metrics[METRIC_NAME[mm] + "_" + GLOBAL_MAP[dataset].ind2tok[i]] = res
            else:
                metric = metric_func(scores, golds)
                metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids, premises, hypotheses
