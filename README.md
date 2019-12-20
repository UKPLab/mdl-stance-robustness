# Analyzing Multi-Dataset Learning and its Robustness for Stance Detection
##### Introduction
This repository modifies and adapts the [official mt-dnn repository](https://github.com/namisan/mt-dnn) to multi-dataset stance detection and robustness experiments. Ten stance detection datasets of different domains are trained via single-dataset and multi-dataset learning. Three adversarial attacks to probe and compare the robustness of both settings have been added. The framework can easily be adapted to include more datasets and adversarial attack sets.

##### Fine-tuned models
The BERT and MT-DNN weights, fine-tuned on all ten stance detection datasets, can be downloaded [here](https://public.ukp.informatik.tu-darmstadt.de/bes/fine_tuned_models.zip). The models can be placed in folder _mt_dnn_models_ and used by modifying the run scripts (see below).


Contact person: [Benjamin Schiller](mailto:schiller@ukp.informatik.tu-darmstadt.de)

https://www.ukp.tu-darmstadt.de/


### Install requirements

The repository requires Python 3.6+

##### 1. (Optional) Set up OpenNMT-py to create data for the paraphrasing attack 

        git clone https://github.com/OpenNMT/OpenNMT-py.git
        cd OpenNMT-py/
        python3 setup.py install

##### 2. Install python requirements
    
        pip install -r requirements.txt
       

### Preprocessing
 
##### 1. Download all data sets
   
        bash download.sh
        
Note: three of the datasets have to be manually retrieved due to license reasons. Please follow the instructions at the bottom of the download.sh file
 
##### 2. Start preprocessing of all datasets

        python prepro.py
        
Note: see _parse_args()_ in prepro.py for additional parameters, e.g.:
- for additional low resource experiment ratios, add/remove ratios to _LOW_RESOURCE_DATA_RATES_ variable
- to create the data for adversarial test sets, add --generate_adversarial 1 --opennmt_gpu 0
    - please note that the paraphrasing can take several hours to complete for all datasets
    - use a GPU to fasten up translation (export CUDA_VISIBLE_DEVICES=0)
         
### Run experiments and evaluate
All scripts can be found in the _scripts_ folder.

##### 1. Start training 

        # Train all stance detection datasets in a multi dataset fashion 
        bash run_mt_dnn_all_stance_MTL_seed_loop.sh (see bash file for parameters)
        E.g.: bash run_mt_dnn_all_stance_MTL_seed_loop.sh 0 mt_dnn_large 100
        
        # Train single datasets
        bash run_mt_dnn_ST_seed_loop.sh (see bash file for parameters)
        E.g.: bash run_mt_dnn_ST_seed_loop.sh 1 bert_model_base ibmcs 30
    
##### 2. Run adversarial attack sets on previously trained model
    
        # For multi dataset model 
        bash evaluate_mt_dnn_all_stance_MTL_seed_loop.sh (see bash file for parameters)
        
        # For single dataset model
        bash evaluate_mt_dnn_ST_seed_loop.sh (see bash file for parameters)
        E.g.: bash evaluate_mt_dnn_ST_seed_loop.sh 1 bert_model_base ibmcs 2019-12-19T2001 30

Note: 
- adapt the configuration in this file to your needs, e.g. set _stress_tests_ variable if adversarial attack sets have been created.
- the timestamp can be found in the _checkpoint_ folder for the specific model (folder name)


##### 3. Evaluate results
        
The predictions for each seed can be found in the _checkpoints_ folder. The scores for each seed can be found in the _results_ folder. To average the results over all seeds, use the following script:
        
        python eval_results/eval_results.py
        
Note: Adapt the parameters in the script to your needs. A new file for the test and each adversarial attack set will be created in the specified model folder in _results_. Each file has an additional key _all_seed_avg_ with the averaged results.
        

For more infos of the underlying Framework itself, please refer to the [official mt-dnn repository](https://github.com/namisan/mt-dnn).

### Add you own datasets
For all steps in the following, please use the other dataset entries for guidance.

1. Add your dataset file to the _data_ folder
2. Add an entry for your dataset in the _prepro.py_ 
3. Add your dataset configuration in _data_utils/label_map.py_ into the dictionaties, as well as in _train.py_ and _predict.py_ to _dump_result_files()_.
4. Add a dataset reader in _data_utils/glue_utils.py_ 
5. Add your dataset key (e.g. like "snopes") to _scripts/run_mt_dnn_all_stance_MTL_seed_loop.sh_ and execute the script

### Add you own adversarial attack
For all steps in the following, please use the other adversarial attacks for guidance.

1. Add your function in _data_utils/glue_utils.py_ (e.g. like _create_adversarial_negation()_)
2. Add the function call to _sample_handling()_ in _data_utils/glue_utils.py_
3. Add your additional adversarial attack as an additional returned parameter for the datasets in _prepro.py_
4. Pass the adversarial attack data into _build_handler()_ in _prepro.py_ and add another entry for your attack in this function

Note: If the attack modifies the length of the original sentences, please consider this for the cutoff that takes place in functions _build_data()_ and _build_data_single()_ in _prepro.py_ in order to avoid information loss.
