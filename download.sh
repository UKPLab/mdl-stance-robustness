#!/usr/bin/env bash
##############################################################
# This script is used to download resources for the stance detection experiments
##############################################################

DATA_DIR=$(pwd)/data
echo "Create a folder $DATA_DIR"
mkdir ${DATA_DIR}

BERT_DIR=$(pwd)/mt_dnn_models
echo "Create a folder BERT_DIR"
mkdir ${BERT_DIR}

############ DOWNLOAD MODELS ############

## DOWNLOAD BERT
cd ${BERT_DIR}
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O "uncased_bert_base.zip"
unzip uncased_bert_base.zip
mv uncased_L-12_H-768_A-12/vocab.txt "${BERT_DIR}/"
rm *.zip
rm -rf uncased_L-12_H-768_A-12

## Download bert models
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_base_v2.pt -O "${BERT_DIR}/bert_model_base.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_large_v2.pt -O "${BERT_DIR}/bert_model_large.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large.pt"


## Download translation models
wget https://public.ukp.informatik.tu-darmstadt.de/bes/translation_models.zip
unzip translation_models.zip
rm -rf translation_models.zip

## Download fine-tuned StD bert models
#wget https://public.ukp.informatik.tu-darmstadt.de/bes/fine_tuned_models.zip
#unzip fine_tuned_models.zip -d $BERT_DIR
#rm -rf fine_tuned_models.zip

############ DOWNLOAD DATA ############

# Download ARC corpus
mkdir $DATA_DIR/ARC
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC
rm -rf coling2018_fake-news-challenge

## Download FNC-1
mkdir $DATA_DIR/FNC-1
git clone https://github.com/FakeNewsChallenge/fnc-1.git
mv fnc-1/README.md $DATA_DIR/FNC-1
mv fnc-1/train_bodies.csv $DATA_DIR/FNC-1
mv fnc-1/train_stances.csv $DATA_DIR/FNC-1
mv fnc-1/competition_test_bodies.csv $DATA_DIR/FNC-1
mv fnc-1/competition_test_stances.csv $DATA_DIR/FNC-1
rm -rf fnc-1

## Download IAC
mkdir $DATA_DIR/IAC
wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip
unzip iac_v1.1.zip -d $DATA_DIR/IAC
mv $DATA_DIR/IAC/iac_v1.1/data/fourforums/discussions $DATA_DIR/IAC
mv $DATA_DIR/IAC/iac_v1.1/data/fourforums/annotations/author_stance.csv $DATA_DIR/IAC
rm -rf $DATA_DIR/IAC/iac_v1.1
rm -rf iac_v1.1.zip

## Download PERSPECTRUM
mkdir $DATA_DIR/PERSPECTRUM
git clone https://github.com/CogComp/perspectrum.git
mv perspectrum/data/dataset/* $DATA_DIR/PERSPECTRUM
rm -rf perspectrum

## Download SCD
mkdir $DATA_DIR/SCD
wget http://www.hlt.utdallas.edu/~saidul/stance/stance.zip
mv stance.zip $DATA_DIR/SCD

## Download SemEval2016Task6
mkdir $DATA_DIR/SemEval2016Task6
wget http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip
unzip -p stance-data-all-annotations.zip data-all-annotations/trialdata-all-annotations.txt >$DATA_DIR/SemEval2016task6/trialdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/trainingdata-all-annotations.txt >$DATA_DIR/SemEval2016task6/trainingdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/testdata-taskA-all-annotations.txt >$DATA_DIR/SemEval2016task6/testdata-taskA-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/readme.txt >$DATA_DIR/SemEval2016task6/readme.txt
rm -rf stance-data-all-annotations.zip

## Download SemEval2019Task7
mkdir $DATA_DIR/SemEval2019Task7
wget https://ndownloader.figshare.com/files/16188500
mv 16188500 SemEval2019Task7.tar.bz2
tar -xvf SemEval2019Task7.tar.bz2 -C $DATA_DIR/SemEval2019Task7
mv $DATA_DIR/SemEval2019Task7/rumoureval2019/* $DATA_DIR/SemEval2019Task7/
rm -rf $DATA_DIR/SemEval2019Task7/rumoureval2019
rm -rf SemEval2019Task7.tar.bz2

echo "############################################################"
echo "############# PLEASE FOLLOW INSTRUCTIONS BELOW #############"
echo "############################################################"
echo ""

## Download IBM claim stance
mkdir $DATA_DIR/IBM_CLAIM_STANCE
echo "IBM claim stance dataset: Please go to https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml,
download "IBM Debater - Claim Stance Dataset" dataset, and put claim_stance_dataset_v1.csv into folder ${DATA_DIR}/IBM_CLAIM_STANCE/"
echo ""

## Download UKP Sentential Argument Mining and Snopes Corpus
mkdir $DATA_DIR/ArgMin
mkdir $DATA_DIR/Snopes
echo "UKP Sentential Argument Mining and Snopes corpus: Please send an e-Mail to schiller@ukp.informatik.tu-darmstadt.de
and ask for the UKP Sentential Argument Mining Corpus and Snopes corpus. Copy all 8 topic files of the ArgMin corpus
into ${DATA_DIR}/ArgMin/. For Snopes, copy the file snopes_corpus_2.csv into ${DATA_DIR}/Snopes/."