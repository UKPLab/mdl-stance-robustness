# Copyright by Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt
import json
import numpy as np

"""
Aggregates the test or dev results by averaging them over all seeds and creating a new file
"""

def add_seed_avg_key(dictionary, set_key="test"):

    # initilize dict with all-key
    temp_seed_dict = dictionary[list(dictionary.keys())[0]]
    init_keys = temp_seed_dict[set_key]['metrics'].keys()
    avg_dict = {"all_seed_avg": {set_key: {}}}
    for k in init_keys: # add all metrics
        avg_dict["all_seed_avg"][set_key][k] = []


    # get the scores for the topics and add them to a list in the all-dict
    for seed, values in dictionary.items():
        if seed == 'config':
            continue

        for k in init_keys: # averaged macro scores over all seeds
            avg_dict["all_seed_avg"][set_key][k].append(values[set_key]['metrics'][k])

    # calculate the stdev, add them to the dict, and the replace the list of values with the final averaged result for the metric
    for k in list(avg_dict["all_seed_avg"][set_key].keys()):
        if k == "topics":
            continue

        avg_dict["all_seed_avg"][set_key][k+"_stdev"] = np.std(avg_dict["all_seed_avg"][set_key][k])
        avg_dict["all_seed_avg"][set_key][k] = np.average(avg_dict["all_seed_avg"][set_key][k])

    # update result dict with averages over all seeds
    dictionary.update(avg_dict)

    return dictionary

if __name__ == '__main__':
    """
    Creates a new file for each result file, which has an additional key "all_seed_avg", which holds the averaged results
    over all seeds. Files are stored in the original folder.
    """
    # variables to set
    path = "../results/" # path to results folder
    folder = "EVAL_ONLY_mt-dnn-arc_ST_ep5_bert_model_large_answer_opt1_trainratio10_maxlen100_trainratio10"
    sets = "negation,spelling,paraphrase,test" #negation,spelling,paraphrase,test
    eval_datasets = "arc,argmin,fnc1,ibmcs,iac1,perspectrum,semeval2016t6,semeval2019t7,scd,snopes" # dataset for which results are to be aggregated over all seeds

    for eval_dataset in eval_datasets.split(","):
        for set in sets.split(","):
            # set paths
            result_file = "results_" + eval_dataset + ".json"
            eval_file_name = "eval_results_" + eval_dataset + "_" + set + ".json"

            with open(path+folder+"/"+result_file, "r") as f_in:
                results = json.load(f_in)

            results = add_seed_avg_key(results, set_key=set)

            with open(path+folder+"/"+eval_file_name, "w") as f_out:
                json.dump(results, f_out, indent=4, sort_keys=True)