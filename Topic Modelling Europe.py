
Europe Reddit
In [1]:
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
In [0]:
import pandas as pd
import numpy as np
In [0]:
df_redEU = pd.read_csv('/content/drive/My Drive/Text Project/reddit/Reddit_europe.csv', header=None)
df_redEU.columns = ['Comments']
In [4]:
df_redEU.head()
Out[4]:
Comments
0	This is one of those things that's more promin...
1	Dude. Of course Massachusetts is in the US. I’...
2	Dismissing facts as "Libertarian nonsense" doe...
3	Canada just passed a law where you can get fin...
4	- Massive efforts to fight against sexism, emp...
In [5]:
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
import numpy as np
import pandas as pd
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
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
df_redEU['Filter Comments NO punct'] = df_redEU['Comments'].map(removePunctuations)
In [8]:
df_redEU.head()
Out[8]:
Comments	Filter Comments NO punct
0	This is one of those things that's more promin...	[one, thing, 's, prominent, us, europe, 's, va...
1	Dude. Of course Massachusetts is in the US. I’...	[dude, course, massachusetts, us, ’, make, poi...
2	Dismissing facts as "Libertarian nonsense" doe...	[dismiss, fact, `, `, libertarian, nonsense, '...
3	Canada just passed a law where you can get fin...	[canada, pass, law, get, fine, misgendering, s...
4	- Massive efforts to fight against sexism, emp...	[massive, effort, fight, sexism, empowerement,...
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
dictionary_EU = gensim.corpora.Dictionary(df_redEU['Filter Comments NO punct'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_redEU['Filter Comments NO punct']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=6, id2word=dictionary_EU, passes=2, workers=2, chunksize=100, random_state=1000)
In [10]:
import pprint
pprint.pprint(...)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_EU.print_topics())
Ellipsis
[   (   0,
        '0.020*"`" + 0.020*"circumcision" + 0.019*"medical" + '
        '0.014*"association" + 0.014*"\'\'" + 0.010*"child" + 0.010*"eu" + '
        '0.010*"-" + 0.009*"procedure" + 0.007*"netherlands"'),
    (   1,
        '0.016*"pay" + 0.012*"*" + 0.010*"cost" + 0.009*"get" + 0.009*"system" '
        '+ 0.008*"insurance" + 0.008*"would" + 0.007*"us" + 0.007*"health" + '
        '0.006*"tax"'),
    (   2,
        '0.091*"’" + 0.016*"“" + 0.016*"”" + 0.010*"trump" + 0.008*"-" + '
        '0.006*"war" + 0.004*"new" + 0.004*"vote" + 0.004*"party" + '
        '0.004*"gt"'),
    (   3,
        '0.020*"not" + 0.016*"\'s" + 0.014*"`" + 0.014*"people" + '
        '0.012*"country" + 0.010*"us" + 0.010*"like" + 0.009*"be" + 0.009*"gt" '
        '+ 0.009*"would"'),
    (   4,
        '0.181*"-" + 0.110*"/" + 0.033*"*" + 0.033*"https" + 0.014*"gt" + '
        '0.009*"amp" + 0.007*"http" + 0.006*"`" + 0.005*"wiki" + 0.005*"the"'),
    (   5,
        '0.073*"-" + 0.029*"trump" + 0.017*"military" + 0.013*"/" + '
        '0.010*"veteran" + 0.010*"fid" + 0.007*"not" + 0.007*"troop" + '
        '0.007*"\'s" + 0.006*"say"')]
In [0]:
new_stopword_list = ['eu', 'us', 'work', 'people', 'would', 'wiki', 'the', 'new', 'war', 'black', 'group' , '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument', 'amitheasshole', 'https', 'http']
def additionalStop(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list]
In [0]:
df_redEU['filter V1'] = df_redEU['Filter Comments NO punct'].map(additionalStop)
In [0]:
dictionary_EU = gensim.corpora.Dictionary(df_redEU['filter V1'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_redEU['filter V1']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=10, workers=2, chunksize=100, random_state=1000)
In [14]:
lda_model_EU.print_topics()
Out[14]:
[(0,
  '0.059*"trump" + 0.019*"military" + 0.010*"veteran" + 0.008*"news" + 0.008*"border" + 0.007*"troop" + 0.006*"july" + 0.006*"obama" + 0.006*"family" + 0.005*"sign"'),
 (1,
  '0.019*"country" + 0.015*"like" + 0.010*"think" + 0.009*"good" + 0.009*"well" + 0.008*"live" + 0.008*"thing" + 0.008*"even" + 0.008*"make" + 0.007*"want"'),
 (2,
  '0.011*"right" + 0.008*"leave" + 0.008*"state" + 0.007*"socialist" + 0.007*"socialism" + 0.007*"policy" + 0.007*"party" + 0.006*"government" + 0.006*"capitalism" + 0.006*"social"'),
 (3,
  '0.016*"cost" + 0.014*"system" + 0.013*"insurance" + 0.012*"health" + 0.010*"private" + 0.009*"care" + 0.008*"company" + 0.008*"price" + 0.008*"government" + 0.008*"doctor"'),
 (4,
  '0.028*"circumcision" + 0.027*"medical" + 0.020*"association" + 0.015*"child" + 0.013*"procedure" + 0.010*"netherlands" + 0.009*"male" + 0.008*"surgeon" + 0.008*"society" + 0.008*"benefit"')]
In [0]:
common_stops = ['july', 'sign', 'like', 'think', 'good', 'well', 'live', 'thing', 'even', 'make', 'want', 'male']
new_stopword_list_v1 = ['//www.reddit.com', 'july', 'sign', 'like', 'think', 'good', 'well', 'live', 'thing', 'even', 'make', 'want', 'male']

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list_v1]

df_redEU['filter V2'] = df_redEU['filter V1'].map(additionalStop_V1)
In [0]:
dictionary_EU = gensim.corpora.Dictionary(df_redEU['filter V2'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_redEU['filter V2']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_EU.print_topics()
Out[0]:
[(0,
  '0.051*"trump" + 0.023*"military" + 0.013*"veteran" + 0.009*"troop" + 0.007*"news" + 0.007*"border" + 0.006*"family" + 0.006*"raise" + 0.006*"fund" + 0.006*"order"'),
 (1,
  '0.018*"country" + 0.010*"system" + 0.008*"much" + 0.007*"free" + 0.007*"also" + 0.007*"cost" + 0.007*"high" + 0.007*"care" + 0.007*"little" + 0.006*"health"'),
 (2,
  '0.017*"right" + 0.010*"leave" + 0.007*"party" + 0.005*"wing" + 0.005*"white" + 0.004*"vote" + 0.004*"trump" + 0.004*"world" + 0.004*"country" + 0.004*"policy"'),
 (3,
  '0.032*"circumcision" + 0.030*"medical" + 0.023*"association" + 0.015*"procedure" + 0.014*"child" + 0.011*"netherlands" + 0.009*"surgeon" + 0.008*"society" + 0.008*"benefit" + 0.008*"norwegian"'),
 (4,
  '0.011*"government" + 0.007*"country" + 0.007*"social" + 0.007*"state" + 0.007*"socialist" + 0.006*"socialism" + 0.006*"economy" + 0.006*"market" + 0.006*"system" + 0.005*"capitalism"')]
In [0]:
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
sentences = []
for s in df_redEU['Comments']:
  sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
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
In [0]:
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
Europe Wordpress
In [0]:
df_wpEU = pd.read_excel('/content/drive/My Drive/Text Project/Wordpress/WP EU compiled.xlsx',header=None)
df_wpEU.columns = ['Comments']
In [0]:
df_wpEU = pd.DataFrame(df_wpEU['Comments'])
In [19]:
df_wpEU.head()
Out[19]:
Comments
0	The importance of innovative technologies in t...
1	The future of healthcare in Europe\t\t\t\t\t\t...
2	In the midst of financial crisis: an invitatio...
3	FUTURE OF HEALTHCARE &VALUE-BASED RATIONING\t\...
4	4th edition of the European Patient Group Dire...
In [0]:
df_wpEU['Filtered Comments wo punct'] = df_wpEU['Comments'].map(removePunctuations)
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
dictionary_EU = gensim.corpora.Dictionary(df_wpEU['Filtered Comments wo punct'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_wpEU['Filtered Comments wo punct']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=6, id2word=dictionary_EU, passes=2, workers=2, chunksize=100, random_state=1000)
In [22]:
import pprint
pprint.pprint(...)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_EU.print_topics())
Ellipsis
[   (   0,
        '0.017*"care" + 0.012*"palliative" + 0.008*"patient" + 0.006*"new" + '
        '0.005*"service" + 0.004*"`" + 0.004*"." + 0.004*"life" + '
        '0.004*"share" + 0.004*"open"'),
    (   1,
        '0.018*"quot" + 0.017*"\'\'" + 0.008*"woman" + 0.008*"/" + '
        '0.007*"care" + 0.006*"…" + 0.006*"nurse" + 0.005*"patient" + '
        '0.005*"state" + 0.005*"medical"'),
    (   2,
        '0.011*"hospital" + 0.008*"care" + 0.008*"medical" + 0.007*"patient" + '
        '0.006*"conference" + 0.005*"sustainable" + 0.004*"”" + 0.004*"“" + '
        '0.004*"work" + 0.004*"new"'),
    (   3,
        '0.010*"quot" + 0.008*"\'\'" + 0.007*"." + 0.007*"care" + '
        '0.006*"system" + 0.006*"/" + 0.006*"medicine" + 0.006*"spend" + '
        '0.006*"u.s" + 0.005*"datum"'),
    (   4,
        '0.104*"quot" + 0.076*"\'\'" + 0.043*"/" + 0.028*"datum" + 0.012*"wp" '
        '+ 0.012*"https" + 0.010*"randomcriticalanalysis.com" + '
        '0.009*"uploads/2016/10" + 0.009*"content" + 0.009*"ssl=1"'),
    (   5,
        '0.009*"research" + 0.006*"patient" + 0.006*"country" + '
        '0.006*"conference" + 0.005*"university" + 0.005*"policy" + '
        '0.005*"project" + 0.004*"clinical" + 0.004*"service" + 0.004*"site"')]
In [0]:
new_stopword_list = ['eu', 'us', 'work', 'people', 'would', 'wiki', 'the', 'new', 'war', 'black', 'group' , '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument', 'amitheasshole', 'https', 'http']
def additionalStop(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list]
df_wpEU['filter V1'] = df_wpEU['Filtered Comments wo punct'].map(additionalStop)
In [0]:
dictionary_EU = gensim.corpora.Dictionary(df_wpEU['filter V1'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_wpEU['filter V1']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=10, workers=2, chunksize=100, random_state=1000)
In [25]:
lda_model_EU.print_topics()
Out[25]:
[(0,
  '0.010*"service" + 0.009*"system" + 0.008*"country" + 0.006*"research" + 0.005*"project" + 0.005*"hospital" + 0.005*"high" + 0.005*"spend" + 0.005*"care" + 0.005*"cost"'),
 (1,
  '0.011*"patient" + 0.010*"medical" + 0.008*"care" + 0.005*"time" + 0.005*"court" + 0.004*"also" + 0.004*"research" + 0.004*"site" + 0.004*"study" + 0.004*"mental"'),
 (2,
  '0.027*"care" + 0.019*"palliative" + 0.014*"hospital" + 0.010*"university" + 0.009*"conference" + 0.005*"example" + 0.005*"patient" + 0.005*"beauty" + 0.005*"life" + 0.005*"education"'),
 (3,
  '0.012*"medicine" + 0.008*"woman" + 0.008*"research" + 0.008*"patient" + 0.006*"policy" + 0.006*"conference" + 0.006*"include" + 0.006*"disease" + 0.006*"medical" + 0.006*"nurse"'),
 (4,
  '0.195*"quot" + 0.051*"datum" + 0.018*"randomcriticalanalysis.com" + 0.018*"content" + 0.018*"uploads/2016/10" + 0.018*"ssl=1" + 0.014*"file=" + 0.013*"image" + 0.009*"orig" + 0.008*"life"')]
In [0]:
common_stops = ['also', 'datum', 'randomcriticalanalysis.com', 'quot', 'uploads/2016/10', 'ssl=1', 'file=', 'image', 'orig', 'life', 'want', 'male', 'eapc', 'akos', 'large', '//i0.wp.com', '//i2.wp.com', '//i1.wp.com', '//randomcriticalanalysis.com/2016/11/06']
new_stopword_list_v1 = ['//www.reddit.com', 'also', 'datum', 'randomcriticalanalysis.com', 'quot', 'uploads/2016/10', 'ssl=1', 'file=', 'image', 'orig', 'life', 'want', 'male', 'eapc', 'akos', 'large', '//i0.wp.com', '//i2.wp.com', '//i1.wp.com', '//randomcriticalanalysis.com/2016/11/06']

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list_v1]

df_wpEU['filter V2'] = df_wpEU['filter V1'].map(additionalStop_V1)
In [27]:
dictionary_EU = gensim.corpora.Dictionary(df_wpEU['filter V2'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_wpEU['filter V2']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_EU.print_topics()
Out[27]:
[(0,
  '0.009*"sustainable" + 0.009*"hospital" + 0.006*"conference" + 0.006*"spend" + 0.006*"service" + 0.006*"high" + 0.006*"site" + 0.005*"system" + 0.005*"cost" + 0.005*"solution"'),
 (1,
  '0.011*"woman" + 0.010*"university" + 0.007*"nurse" + 0.007*"digital" + 0.005*"conference" + 0.005*"innovation" + 0.005*"project" + 0.005*"female" + 0.005*"research" + 0.004*"chapter"'),
 (2,
  '0.008*"research" + 0.006*"service" + 0.005*"medicine" + 0.005*"court" + 0.005*"policy" + 0.005*"project" + 0.005*"state" + 0.005*"workshop" + 0.004*"conference" + 0.004*"university"'),
 (3,
  '0.015*"hospital" + 0.008*"system" + 0.006*"beauty" + 0.006*"mental" + 0.006*"example" + 0.005*"report" + 0.004*"public" + 0.004*"illness" + 0.004*"floor" + 0.004*"cost"'),
 (4,
  '0.015*"palliative" + 0.015*"content" + 0.006*"medicine" + 0.006*"expectancy" + 0.005*"expectation" + 0.005*"open" + 0.005*"mostly" + 0.005*"share" + 0.004*"naive" + 0.004*"develope"')]
In [29]:
df_wpEU[:1]
Out[29]:
Comments	Filtered Comments wo punct	filter V1	filter V2
0	The importance of innovative technologies in t...	[importance, innovative, technology, healthcar...	[importance, innovative, technology, healthcar...	[importance, innovative, technology, healthcar...
In [0]:
# Word Count
from collections import Counter
s = df_wpEU['filter V2']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
common_words[:20]
comm_words_wpUS = pd.DataFrame(common_words)
comm_words_wpUS.to_csv('/content/drive/My Drive/Text Project/comm_words_wpEU.csv')
In [0]:
from nltk.tokenize import sent_tokenize
sentences = []
for s in df_wpEU['Comments']:
  sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-9-b1a80a76572c> in <module>()
      1 from nltk.tokenize import sent_tokenize
      2 sentences = []
----> 3 for s in df_wpEU['Comments']:
      4   sentences.append(sent_tokenize(s))
      5 sentences = [y for x in sentences for y in x] # flatten list

NameError: name 'df_wpEU' is not defined
In [0]:
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
--2019-12-01 04:38:25--  http://nlp.stanford.edu/data/glove.6B.zip
Resolving nlp.stanford.edu (nlp.stanford.edu)... 171.64.67.140
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:80... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://nlp.stanford.edu/data/glove.6B.zip [following]
--2019-12-01 04:38:25--  https://nlp.stanford.edu/data/glove.6B.zip
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip [following]
--2019-12-01 04:38:26--  http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
Resolving downloads.cs.stanford.edu (downloads.cs.stanford.edu)... 171.64.64.22
Connecting to downloads.cs.stanford.edu (downloads.cs.stanford.edu)|171.64.64.22|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 862182613 (822M) [application/zip]
Saving to: ‘glove.6B.zip’

glove.6B.zip        100%[===================>] 822.24M  2.22MB/s    in 6m 27s  

2019-12-01 04:44:53 (2.13 MB/s) - ‘glove.6B.zip’ saved [862182613/862182613]

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
In [0]:
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
We hope the project will make a significant contribution to the history of health care provision by opening up local records in East-Central Europe for researchers and by placing hospital development in these countries in a Europe wide context.
Conference Dates: 13-14 September 2013 This conference seeks to bring together scholars working on topics related…02/08/2013In "history"The Value of Time and the  Temporality of Value in Socialities of WasteDrawing from long-term ethnographic research on a 25-year-old medical aid program linking the U.S. and Madagascar, I use this brief essay to trace how Malagasy and American participants engender different orientations to time through their work with discards, as they transform both discards’ value and the social relations surrounding them.09/21/2015In "Economics/Economies"Conference on waste and pollution in healthcareHealthcare Without Harm is an NGO dedicated to promoting environmental justice ?inside?
Today, the best tool we have for attempting any large scale changes in the way we view and talk about mental health is education.
I am certain that this latest edition of the Directory will be used by many actors in health and will provide useful guidance and open access to patient groups in the European Union and beyond.Share this:MoreEmailPrint                                ShareLike this:Like Loading...	RelatedLaunch of myhealthapps.netWith 1 commentWhite paper: Health apps: where do they make sense?
We should be cautious that even within Europe, systems like Germany that have some degree of private insurance are quite different from systems like the U.K. where the National Health Service was once quite universally dominant and is now under flux with calls for reform (despite evidence that it works much better than the post-reform version would) from the new conservative government.
Over 30000 registered doctors in the UK gained their primary medical qualification in another European Economic Area state and over 150000 EU citizens work in the UK’s health and social care sphere – with free movement playing a crucial role in both professional development and in meeting varying health and social care, including medical, workforce requirements across Europe.
During the next network event, a study visit to Centralsjukhuset Kristianstad on September 27th, we will be able to look at some of the hospital’s energy efficient solutions, gain an unique insight to investments in sustainable healthcare, meet key persons in the process as well as have a closer look on real-life sustainability projects in a hospital environment.
The aim of this event was to show how innovation influences the medical and healthcare sector and how, ultimately, the improvements obtained have positive repercussions on patients’ lives.Over breakfast, Ms Denjoy began her presentation by giving a picture of the role COCIR plays and by underlining the main sectors of its influence.
In order to do that, we will be bringing on stage those who are shaking things:

Empowered physicians and nurses who are using digital tools in their daily routine
Investors who have been investing in digital health projects this past year
Successful entrepreneurs whose solution is improving efficiency, reducing costs and proved to be a positive return on investment.
The discussion among participants showed that this vision is already reality for many and that the right tools and strategies to manage a digital work environment will become even more relevant in the future.
Europe Reuters
In [0]:
df_reEU = pd.read_csv('/content/drive/My Drive/Text Project/Reuters/Reuters_Europe.csv', header=None)
df_reEU.columns = ['Topic', 'Comments']
df_reEU = pd.DataFrame(df_reEU['Comments'])
In [0]:
df_reEU.head()
Out[0]:
Comments
0	Field1
1	LONDON/MADRID/PARIS (Reuters) - Some Britons l...
2	DHAKA (Thomson Reuters Foundation) - Living on...
3	BUDAPEST (Reuters) - Csilla Balla became anxio...
4	LONDON (Reuters) - Europe’s listed companies a...
In [0]:
df_reEU['Filtered Comments wo punct'] = df_reEU['Comments'].map(removePunctuations)
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
dictionary_EU = gensim.corpora.Dictionary(df_reEU['Filtered Comments wo punct'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_reEU['Filtered Comments wo punct']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=6, id2word=dictionary_EU, passes=2, workers=2, chunksize=100, random_state=1000)
In [0]:
import pprint
pprint.pprint(...)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_EU.print_topics())
Ellipsis
[   (   0,
        '0.004*"health" + 0.003*"private" + 0.002*"state" + 0.002*"insurance" '
        '+ 0.002*"eu" + 0.002*"hospital" + 0.002*"year" + 0.002*"medical" + '
        '0.002*"care" + 0.002*"service"'),
    (   1,
        '0.025*"private" + 0.018*"health" + 0.009*"eastern" + 0.008*"year" + '
        '0.008*"poland" + 0.007*"hospital" + 0.007*"service" + 0.007*"hungary" '
        '+ 0.007*"budapest" + 0.007*"romania"'),
    (   2,
        '0.010*"brexit" + 0.010*"insurance" + 0.009*"health" + 0.008*"live" + '
        '0.008*"deal" + 0.007*"british" + 0.006*"eu" + 0.006*"spain" + '
        '0.006*"britain" + 0.006*"britons"'),
    (   3,
        '0.016*"company" + 0.009*"revenue" + 0.008*"u.s" + 0.008*"trade" + '
        '0.008*"state" + 0.007*"sector" + 0.007*"unite" + 0.006*"may" + '
        '0.006*"index" + 0.005*"gain"'),
    (   4,
        '0.013*"hospital" + 0.012*"char" + 0.011*"ship" + 0.009*"change" + '
        '0.009*"climate" + 0.008*"bangladesh" + 0.007*"people" + 0.006*"live" '
        '+ 0.006*"two" + 0.006*"island"'),
    (   5,
        '0.004*"health" + 0.003*"revenue" + 0.003*"company" + 0.003*"private" '
        '+ 0.003*"people" + 0.003*"year" + 0.003*"medical" + 0.003*"hospital" '
        '+ 0.002*"sector" + 0.002*"u.s"')]
In [0]:
new_stopword_list = ['eu', 'us', 'work', 'people', 'would', 'wiki', 'the', 'new', 'war', 'black', 'group' , '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument', 'amitheasshole', 'https', 'http']
def additionalStop(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list]
df_reEU['filter V1'] = df_reEU['Filtered Comments wo punct'].map(additionalStop)
In [0]:
dictionary_EU = gensim.corpora.Dictionary(df_reEU['filter V1'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_reEU['filter V1']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=10, workers=2, chunksize=100, random_state=1000)
In [0]:
lda_model_EU.print_topics()
Out[0]:
[(0,
  '0.016*"hospital" + 0.016*"char" + 0.015*"ship" + 0.012*"change" + 0.012*"climate" + 0.010*"bangladesh" + 0.007*"live" + 0.007*"island" + 0.007*"khan" + 0.006*"surgery"'),
 (1,
  '0.013*"company" + 0.011*"revenue" + 0.008*"index" + 0.007*"stock" + 0.007*"trade" + 0.007*"sector" + 0.007*"state" + 0.007*"unite" + 0.006*"brexit" + 0.006*"economy"'),
 (2,
  '0.019*"private" + 0.018*"health" + 0.009*"year" + 0.008*"insurance" + 0.006*"medical" + 0.006*"live" + 0.006*"however" + 0.006*"service" + 0.006*"eastern" + 0.006*"company"'),
 (3,
  '0.001*"hospital" + 0.001*"char" + 0.001*"private" + 0.001*"ship" + 0.001*"medical" + 0.001*"health" + 0.001*"company" + 0.001*"climate" + 0.001*"doctor" + 0.001*"change"'),
 (4,
  '0.001*"health" + 0.001*"private" + 0.001*"insurance" + 0.001*"medical" + 0.001*"year" + 0.001*"live" + 0.001*"state" + 0.001*"britons" + 0.001*"service" + 0.001*"brexit"')]
In [0]:
common_stops = ['also', 'datum', 'randomcriticalanalysis.com', 'quot', 'uploads/2016/10', 'ssl=1', 'file=', 'image', 'orig', 'life', 'want', 'male', 'eapc', 'akos', 'large', '//i0.wp.com', '//i2.wp.com', '//i1.wp.com', '//randomcriticalanalysis.com/2016/11/06', 'char', 'year', 'khan', 'ship', 'live', 'field1', 'trade', 'know']
new_stopword_list_v1 = ['//www.reddit.com', 'also', 'datum', 'randomcriticalanalysis.com', 'quot', 'uploads/2016/10', 'ssl=1', 'file=', 'image', 'orig', 'life', 'want', 'male', 'eapc', 'akos', 'large', '//i0.wp.com', '//i2.wp.com', '//i1.wp.com', '//randomcriticalanalysis.com/2016/11/06', 'char', 'year', 'khan', 'ship', 'live', 'field1', 'trade', 'know']

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list_v1]

df_reEU['filter V2'] = df_reEU['filter V1'].map(additionalStop_V1)
In [0]:
dictionary_EU = gensim.corpora.Dictionary(df_reEU['filter V2'])
dictionary_EU.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_EU.doc2bow(words) for words in df_reEU['filter V2']]
lda_model_EU = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_EU, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_EU.print_topics()
Out[0]:
[(0,
  '0.012*"gain" + 0.012*"company" + 0.008*"sector" + 0.008*"index" + 0.008*"week" + 0.008*"talk" + 0.008*"stock" + 0.008*"profit" + 0.005*"stoxx" + 0.005*"pound"'),
 (1,
  '0.013*"company" + 0.012*"insurance" + 0.010*"health" + 0.007*"unite" + 0.007*"brexit" + 0.007*"spain" + 0.007*"global" + 0.007*"britons" + 0.006*"britain" + 0.006*"british"'),
 (2,
  '0.001*"health" + 0.001*"private" + 0.001*"insurance" + 0.001*"brexit" + 0.001*"service" + 0.001*"britain" + 0.001*"doctor" + 0.001*"cost" + 0.001*"however" + 0.001*"hospital"'),
 (3,
  '0.024*"private" + 0.017*"health" + 0.009*"eastern" + 0.008*"poland" + 0.007*"service" + 0.007*"romania" + 0.007*"budapest" + 0.007*"hungary" + 0.007*"clinic" + 0.007*"hospital"'),
 (4,
  '0.018*"hospital" + 0.013*"change" + 0.013*"climate" + 0.011*"bangladesh" + 0.008*"island" + 0.007*"system" + 0.007*"doctor" + 0.007*"government" + 0.007*"organization" + 0.007*"surgery"')]
In [0]:
from nltk.tokenize import sent_tokenize
sentences = []
for s in df_reEU['Comments']:
  sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list
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
In [0]:
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
The British government last month said 180,000 people already living in the EU who have their healthcare funded by the UK, including pensioners and students, would have their costs covered in the case of a no-deal Brexit.
UNDERFUNDED Poland’s health ministry said it was working to improve access to medical services and cut waiting times, adding that spending on state healthcare was expected to rise by nearly 9%  this year.
Allianz Care (ALVG.DE), which offers an international private medical insurance policy, said enquiries from British citizens, often living in or with properties in France or Spain, had risen by 20% so far this year from the same 2018 period.
The change is being driven by low public health spending as a share of the economy - which has often led to staff shortages and longer waiting times for tests and surgery - coupled with rising wages, which is making private care a viable alternative.
Britain and Spain have a further arrangement giving people living in each country continued access to local healthcare until at least the end of 2020.
Many Eastern Europeans, whose net wages pale in comparison to Western Europeans’, even after rapid rises in recent years, have responded by shelling out their own money so they can cut waiting times for procedures or screening services.
However the EU Commission says the high private health spending in Eastern European countries is leading to inequality in access to medical services.
Their biggest concern is over Britain leaving without a deal, which could affect some of the healthcare arrangements enjoyed by Britons living in Europe, many of them pensioners, as well as by more than three million EU citizens in Britain.
Vienna Insurance Group’s Hungarian division, Union, said private health cover was becoming an essential benefit for many employers to offer.
Like Buda Health, all five major healthcare providers interviewed by Reuters said they were expanding their operations to keep up with demand.
