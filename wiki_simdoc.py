#------------Wikipedia based similarity----------------#

# Wikipedia based similarity
""" Compute the similarty between defined two sentences using Wikipedia dump files """
import gensim
#from gensim.corpora import WikiCorpus, MmCorpus
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models import LsiModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text ('./wikiresults/results_wordids.txt.bz2')
#mm = gensim.corpora.MmCorpus('./results/results_tfidf.mm')-22.4GB
#print(mm)

# load LSI model

model_lsi = LsiModel.load('./wikiresults/lsi.lsi_model')

# load index of wikicorpus documents

index = similarities.MatrixSimilarity.load('./wikiresults/lsi_index.index')

# Wiki_similarity
def get_similardocuments(sentence):
    vec_bow = id2word.doc2bow(gensim.utils.simple_preprocess(sentence)) #or item.lower().split()
    vec_lsi = model_lsi[vec_bow]
    sims = index[vec_lsi] 
    return sims # (output: Document No, cosine similarity)


def Wiki_similarity(s1,s2):
    """ retrieve all the similar documents from Wikipedia dump files for the input sentences """
    # only search the similarity value >0
    s_1 = list([index for index in get_similardocuments(s1) if index > 0])
    #print(s_1)
    s_2 = list([index for index in get_similardocuments(s2) if index > 0])
    #print(s_2)

# calculate the similary based document numbers
    DS1 = len(s_1)/(len(s_1)+len(s_2))
    DS2 = len(s_2)/(len(s_1)+len(s_2))
    return DS1, DS2

# test
sentence1 = 'I have a hammer.'
sentence2= 'Take some apples.'


Simresult = Wiki_similarity(sentence1,sentence2) ############### 
print('wikiSim=',Simresult) # 


sentence=['I like that bachelor.','I like that unmarried man.'] 
sentence=['I have a pen.','Where do you live?']   
sentence=['John is very nice.','Is John very nice.']
sentence=['Canis familiaris are animals.','Dogs are common pets.']
sentence=['It is a dog.','It is a log.']
sentence=['It is a dog.','It is a pig.']
sentence=['It is a dog.','That must be your dog.']
sentence=['I have a hammer.','Take some nails.']
sentence=['I have a hammer.','Take some apples.']