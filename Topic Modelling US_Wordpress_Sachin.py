In [1]:
from google.colab import drive
drive.mount('/content/drive')
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly

Enter your authorization code:
··········
Mounted at /content/drive
In [2]:
import pandas as pd
import numpy as np

# Import the Data Frame
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# nltk.download('all')

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set( stopwords.words('english'))
from spacy.lang.en import English

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import spacy
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
In [0]:
# Function to remove stopwords and replacing model names with brand names
# This function will be applied to all comments in our data frame

def filterReview(review):
    nlp = English()
    tokens = word_tokenize(review)
    word_str = ""

    # Removing numbers
    for w in tokens:
        try:
            float(w)
        
        # If not a number, we will consider the words
        except Exception as e:
            if w.lower() not in stop_words: 
                word_str = word_str + " " + w.lower()
        continue
    
    # Lemmatization
    lemmatized = []
    doc = nlp(' '.join(word_str.split()))
    for word in doc:
        lemmatized.append(word.lemma_)
    
    return lemmatized

# Backup fundtion to clean the data 
def removePunctuations(review):
    nlp = English()
    tokens = word_tokenize(review)
    stop_words_punct = set(stopwords.words('english') + list(punctuation))
    no_stopwords = [w.lower() for w in tokens if w.lower() not in stop_words_punct]
    word_str = ""
    for w in no_stopwords:
        # Removing numbers
        try:
            float(w)
        except Exception as e:
            word_str = word_str + " " + w
        continue
    
    # Lemmatization
    lemmatized = []
    doc = nlp(' '.join(word_str.split()))
    for word in doc:
        lemmatized.append(word.lemma_)
    
    return lemmatized
In [0]:
df_wpUS = pd.read_excel('/content/drive/My Drive/Text Project/Wordpress/usWP_compile_V1.xlsx')
df_wpUS.columns = ['Title','Comments']

# Combining title and comments
df_wpUS['Comments'] = df_wpUS['Title'] + df_wpUS['Comments']
In [5]:
df_wpUS.head()
Out[5]:
Title	Comments
0	A brief history of how the American healthcare...	A brief history of how the American healthcare...
1	The Disabled States of America: regional dispa...	The Disabled States of America: regional dispa...
2	Why American Presidents (and Some Oscar Winner...	Why American Presidents (and Some Oscar Winner...
3	NaN	NaN
4	The Eleven Most Implanted Medical Devices In A...	The Eleven Most Implanted Medical Devices In A...
In [6]:
df_wpUS = df_wpUS.dropna(axis=0, subset=['Comments']).reset_index(drop=True)
df_wpUS.head()
Out[6]:
Title	Comments
0	A brief history of how the American healthcare...	A brief history of how the American healthcare...
1	The Disabled States of America: regional dispa...	The Disabled States of America: regional dispa...
2	Why American Presidents (and Some Oscar Winner...	Why American Presidents (and Some Oscar Winner...
3	The Eleven Most Implanted Medical Devices In A...	The Eleven Most Implanted Medical Devices In A...
4	In Their Words – Stories of Chinese Immigrants...	In Their Words – Stories of Chinese Immigrants...
In [0]:
# Filtering comments and converting to a list of words
df_wpUS['Filtered comments'] = df_wpUS['Comments'].map(removePunctuations)
In [8]:
df_wpUS.head()
Out[8]:
Title	Comments	Filtered comments
0	A brief history of how the American healthcare...	A brief history of how the American healthcare...	[brief, history, american, healthcare, system,...
1	The Disabled States of America: regional dispa...	The Disabled States of America: regional dispa...	[disable, state, america, regional, disparity,...
2	Why American Presidents (and Some Oscar Winner...	Why American Presidents (and Some Oscar Winner...	[american, president, oscar, winner, live, lon...
3	The Eleven Most Implanted Medical Devices In A...	The Eleven Most Implanted Medical Devices In A...	[eleven, implant, medical, device, america5, m...
4	In Their Words – Stories of Chinese Immigrants...	In Their Words – Stories of Chinese Immigrants...	[word, –, story, chinese, immigrant, americaim...
In [9]:
# Word Count
from collections import Counter
s = df_wpUS['Filtered comments']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
common_words[:20]
Out[9]:
[('-', 418),
 ('’', 354),
 ('health', 321),
 ('healthcare', 296),
 ('care', 233),
 ('“', 162),
 ('”', 160),
 ('.', 156),
 ('medical', 125),
 ('/', 122),
 ('system', 114),
 ('people', 113),
 ('year', 109),
 ('–', 105),
 ('would', 100),
 ('cost', 98),
 ('make', 94),
 ('one', 90),
 ('insurance', 86),
 ('patient', 84)]
In [23]:
import os
os.getcwd()
Out[23]:
'/content'
In [0]:
additional_stop = ['-', '’', '“', '”', '.', '/', '–',  '—', '`', '…', "''", '‘', 'e.g.','https', 'http']
def remove_Stop1(word_list):
    return [word for word in word_list if word not in additional_stop]
In [0]:
df_wpUS['Filtered comments V1'] = df_wpUS['Filtered comments'].map(remove_Stop1)
In [26]:
df_wpUS.head()
Out[26]:
Title	Comments	Filtered comments	Filtered comments V1
0	A brief history of how the American healthcare...	A brief history of how the American healthcare...	[brief, history, american, healthcare, system,...	[brief, history, american, healthcare, system,...
1	The Disabled States of America: regional dispa...	The Disabled States of America: regional dispa...	[disable, state, america, regional, disparity,...	[disable, state, america, regional, disparity,...
2	Why American Presidents (and Some Oscar Winner...	Why American Presidents (and Some Oscar Winner...	[american, president, oscar, winner, live, lon...	[american, president, oscar, winner, live, lon...
3	The Eleven Most Implanted Medical Devices In A...	The Eleven Most Implanted Medical Devices In A...	[eleven, implant, medical, device, america5, m...	[eleven, implant, medical, device, america5, m...
4	In Their Words – Stories of Chinese Immigrants...	In Their Words – Stories of Chinese Immigrants...	[word, –, story, chinese, immigrant, americaim...	[word, story, chinese, immigrant, americaimmig...
In [27]:
# Word Count
s = df_wpUS['Filtered comments V1']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
common_words[:20]
Out[27]:
[('health', 321),
 ('healthcare', 296),
 ('care', 233),
 ('medical', 125),
 ('system', 114),
 ('people', 113),
 ('year', 109),
 ('would', 100),
 ('cost', 98),
 ('make', 94),
 ('one', 90),
 ('insurance', 86),
 ('patient', 84),
 ('u.s', 81),
 ('service', 79),
 ('also', 78),
 ('time', 77),
 ('american', 76),
 ('drug', 76),
 ('state', 75)]
In [0]:
comm_words_wpUS = pd.DataFrame(common_words)
comm_words_wpUS.to_csv('/content/drive/My Drive/Text Project/comm_words_wpUS.csv')
Topic Modelling
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
In [0]:
dictionary_US = gensim.corpora.Dictionary(df_wpUS['Filtered comments V1'])
dictionary_US.filter_extremes(no_below=1, no_above=0.5)
bow_corpus = [dictionary_US.doc2bow(words) for words in df_wpUS['Filtered comments V1']]
In [31]:
# 5 topics
lda_model_US = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary_US, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[31]:
[(0,
  '0.007*"drug" + 0.005*"social" + 0.005*"death" + 0.005*"factor" + 0.004*"u.s" + 0.004*"president" + 0.004*"use" + 0.004*"prescription" + 0.004*"insurance" + 0.004*"program"'),
 (1,
  '0.010*"waste" + 0.009*"medication" + 0.006*"bill" + 0.005*"fraud" + 0.005*"cancer" + 0.005*"spend" + 0.005*"billion" + 0.005*"price" + 0.005*"cost" + 0.005*"program"'),
 (2,
  '0.010*"infologix" + 0.007*"solution" + 0.006*"pioneer" + 0.006*"freedom" + 0.005*"technology" + 0.005*"business" + 0.004*"include" + 0.004*"private" + 0.004*"hle" + 0.004*"hospital"'),
 (3,
  '0.006*"insurance" + 0.005*"pay" + 0.005*"cost" + 0.005*"medicine" + 0.005*"get" + 0.005*"medieval" + 0.004*"physician" + 0.004*"doctor" + 0.004*"increase" + 0.004*"price"'),
 (4,
  '0.009*"u.s" + 0.008*"cost" + 0.006*"high" + 0.006*"spend" + 0.005*"pay" + 0.005*"per" + 0.005*"country" + 0.004*"compare" + 0.004*"doctor" + 0.004*"procedure"')]
In [32]:
# 3 topics
lda_model_US = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary_US, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[32]:
[(0,
  '0.007*"u.s" + 0.005*"cost" + 0.005*"drug" + 0.005*"insurance" + 0.005*"high" + 0.004*"disease" + 0.004*"death" + 0.004*"program" + 0.004*"social" + 0.004*"factor"'),
 (1,
  '0.006*"bill" + 0.005*"pay" + 0.005*"insurance" + 0.005*"plan" + 0.005*"get" + 0.004*"clinton" + 0.004*"cost" + 0.004*"waste" + 0.003*"can" + 0.003*"medicine"'),
 (2,
  '0.008*"infologix" + 0.005*"solution" + 0.005*"pioneer" + 0.004*"freedom" + 0.004*"technology" + 0.004*"procedure" + 0.004*"include" + 0.004*"business" + 0.003*"hospital" + 0.003*"private"')]
In [33]:
# 2 topics
lda_model_US = gensim.models.LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary_US, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[33]:
[(0,
  '0.006*"u.s" + 0.005*"drug" + 0.005*"cost" + 0.005*"high" + 0.004*"disease" + 0.004*"death" + 0.003*"program" + 0.003*"insurance" + 0.003*"country" + 0.003*"spend"'),
 (1,
  '0.005*"insurance" + 0.004*"hospital" + 0.004*"cost" + 0.004*"infologix" + 0.004*"pay" + 0.003*"use" + 0.003*"include" + 0.003*"solution" + 0.003*"technology" + 0.003*"medicine"')]
In [34]:
# 4 topics with paramater tuning
lda_model_US = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary_US, passes=25, workers=2, chunksize=64, random_state=1000)
lda_model_US.print_topics()
Out[34]:
[(0,
  '0.006*"drug" + 0.004*"social" + 0.004*"u.s" + 0.004*"death" + 0.004*"program" + 0.004*"factor" + 0.004*"study" + 0.003*"president" + 0.003*"community" + 0.003*"use"'),
 (1,
  '0.006*"film" + 0.006*"insurance" + 0.005*"pay" + 0.004*"cost" + 0.004*"photo" + 0.004*"bill" + 0.004*"live" + 0.004*"pexels.com" + 0.003*"quality" + 0.003*"medium"'),
 (2,
  '0.007*"infologix" + 0.005*"solution" + 0.005*"technology" + 0.005*"pioneer" + 0.005*"medication" + 0.004*"waste" + 0.004*"billion" + 0.004*"freedom" + 0.004*"cost" + 0.004*"company"'),
 (3,
  '0.007*"cost" + 0.007*"u.s" + 0.007*"pay" + 0.006*"high" + 0.006*"insurance" + 0.005*"spend" + 0.005*"doctor" + 0.004*"hospital" + 0.004*"physician" + 0.004*"medicine"')]
In [35]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary_US, passes=25, workers=2, chunksize=64, random_state=1000)
lda_model_US.print_topics()
Out[35]:
[(0,
  '0.007*"u.s" + 0.005*"cost" + 0.005*"drug" + 0.005*"insurance" + 0.005*"high" + 0.004*"disease" + 0.004*"program" + 0.004*"death" + 0.004*"social" + 0.004*"factor"'),
 (1,
  '0.006*"bill" + 0.005*"pay" + 0.005*"insurance" + 0.005*"plan" + 0.004*"get" + 0.004*"clinton" + 0.004*"waste" + 0.004*"cost" + 0.004*"can" + 0.003*"medicine"'),
 (2,
  '0.007*"infologix" + 0.005*"solution" + 0.005*"pioneer" + 0.004*"freedom" + 0.004*"technology" + 0.004*"procedure" + 0.004*"include" + 0.004*"business" + 0.003*"cost" + 0.003*"hospital"')]
In [0]:
df_wpUS.to_csv('/content/drive/My Drive/Text Project/Wordpress/Filtered_US_wordpress.csv')
In [0]:
from nltk.tokenize import sent_tokenize
sentences = []
for s in df_wpUS['Comments']:
  sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list
In [38]:
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
--2019-12-01 06:02:59--  http://nlp.stanford.edu/data/glove.6B.zip
Resolving nlp.stanford.edu (nlp.stanford.edu)... 171.64.67.140
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:80... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://nlp.stanford.edu/data/glove.6B.zip [following]
--2019-12-01 06:03:00--  https://nlp.stanford.edu/data/glove.6B.zip
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip [following]
--2019-12-01 06:03:00--  http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
Resolving downloads.cs.stanford.edu (downloads.cs.stanford.edu)... 171.64.64.22
Connecting to downloads.cs.stanford.edu (downloads.cs.stanford.edu)|171.64.64.22|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 862182613 (822M) [application/zip]
Saving to: ‘glove.6B.zip’

glove.6B.zip        100%[===================>] 822.24M  2.08MB/s    in 6m 29s  

2019-12-01 06:09:30 (2.11 MB/s) - ‘glove.6B.zip’ saved [862182613/862182613]

Archive:  glove.6B.zip
  inflating: glove.6B.50d.txt        
  inflating: glove.6B.100d.txt       
  inflating: glove.6B.200d.txt       
  inflating: glove.6B.300d.txt       
In [0]:
# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
In [0]:
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]
In [0]:
# function to remove stopwords
def remove_stopwords_sentence(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
In [0]:
# remove stopwords from the sentences
clean_sentences = [remove_stopwords_sentence(r.split()) for r in clean_sentences]
In [0]:
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
In [0]:
# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
In [0]:
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
In [0]:
import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
In [0]:
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
In [48]:
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
No less than 6 different statements were made by Mr Obama forcefully claiming in 2009 while addressing the American Medical Association:
“If you like your doctor, you will be able to keep your doctor, period, If you like your health care plan, you’ll be able to keep your health care plan, period.
At long last, after decades of false starts, we must make this our most urgent priority, giving every American health security — health care that can never be taken away, health care that is always there.
“I have always found a bit cruel the much-mouthed suggestion that patients should have ‘more skin in the game’ and ‘shop around for cost-effective health care’ in the health care market,” said Uwe E. Reinhardt, a health policy expert and professor at Princeton University, “when patients have so little information easily available on prices and quality to those things.”
President Obama’s Affordable Care Act, the health care overhaul law passed in 2010, tries to make some improvements (though the Supreme Court is expected to rule whether all or some of the law is constitutional this month).
While a European national health system seemed politically out of the question, pilot models in which employees of a Dallas school system were part of a collective payment system seemed to offer the best model to emulate more broadly in order to bring up hospital service demand as well as mitigate adverse selection of the sickest patients opting into insurance plans.
Another recent paper is a major new report from the National Bureau of Economic Research (NBER), which shows that when the poor have medical insurance, they not only find regular doctors and see them more frequently for preventive care, but also end up feeling healthier, less depressed and are better able to maintain financial stability.
Why getting sick in America is a really bad ideaMay_30_Health_Care_Rally_NP (641) (Photo credit: seiuhealthcare775nw)
One element of living in the United States sickens me to my core — the persistent inequality of access to affordable quality health care, something citizens of virtually every other developed nation take for granted.
As the heated debate over health care continues, there has been plenty of talk about how the U.S. system stacks up to that of other countries, and how much American doctors earn compared with M.D.s in other parts of the world.
Expanding this program to include everyone regardless of their age would be necessary; signing up for Medicare at the same time one applies for a Social Security number (typically at a very young age) seems simple enough.
Expanding this program to include everyone regardless of their age would be necessary; signing up for Medicare at the same time one applies for a Social Security number (typically at a very young age) seems simple enough.
)How did we get to the point where the discussion on health care has come down to how much less coverage can be provided and how much tax relief can be granted to already rich people?
