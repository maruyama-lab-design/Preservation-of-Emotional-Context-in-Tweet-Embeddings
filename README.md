# Preparation
Download wrime-ver1.tsv from https://github.com/ids-cv/wrime, and save it into the same directory of our source codes. 

## Make clusters of tweets based on eight emotional intensities.
intensity_kmeans.ipynb

## Make clusters of tweets using word2vec embedding vectors.
word2vec_kmeans.ipynb

## Make clusters of tweets using pre-trained BERT embedding vectors.
pretrained_bert_kmeans.ipynb

## Make clusters of tweets using fine-tuned BERT embedding vectors.
fine_tune_Japanese_bert.ipynb
fine_tuned_bert_kmeans.ipynb

## Note
lib.py is a module called by the above scripts. 