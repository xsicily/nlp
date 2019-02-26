#------------Wikipedia based similarity----------------#
### fuctions of getting vector and sentence similarity refers to https://github.com/neomoha/python-lsi-similarity/blob/master/model.py

""" Compute the similarty between defined two sentences using Wikipedia dump files """

import gensim
from gensim.corpora import MmCorpus
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models import LsiModel
from math import *
import numpy as np
from nltk.corpus import stopwords

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

STOPWORDS=set(stopwords.words('english'))

def tokenize(text):
    return [token for token in gensim.utils.simple_preprocess(text) if token not in STOPWORDS]

id2word = gensim.corpora.Dictionary.load_from_text ('./wikiresults/results_wordids.txt.bz2')
#mm = gensim.corpora.MmCorpus('./results/results_tfidf.mm')-22.4GB
#print(mm)

# train LSI model
# model_lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)
# model_lsi.save('./wikiresult/model_lsi.model')

# load LSI model

model_lsi = LsiModel.load('./wikiresults/lsi.lsi_model')

# doc to bag of words vector
def get_vector(sentence):
    """ compute lsivectors using LSI model """
    vec_bow = id2word.doc2bow(tokenize(sentence)) #or item.lower().split()
    return vec_bow

# cosine similarity
def cosine(v1,v2):
	dot_product = np.dot(v1, v2)
	norm_v1 = np.linalg.norm(v1)
	norm_v2 = np.linalg.norm(v2)
	return dot_product / (norm_v1 * norm_v2)

# get lsi vectors to compute sentence similarity
def get_sentence_sim(S1,S2):
    vec_bow1=get_vector(S1)
    vec_bow2=get_vector(S2)
    vec_lsi1 = [val for idx,val in model_lsi[vec_bow1]]
    vec_lsi2 = [val for idx,val in model_lsi[vec_bow2]]
    return cosine(vec_lsi1,vec_lsi2)


if __name__ == "__main__":
    # test sentence similarity
    S1="I have a hammer."
    S2="Take some apples."
    wiki_sim=get_sentence_sim(S1,S2)
    print(wiki_sim)
    