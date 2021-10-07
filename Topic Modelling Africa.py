In [1]:
from google.colab import drive
drive.mount('/content/drive')
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly

Enter your authorization code:
··········
Mounted at /content/drive
In [0]:
import pandas as pd
import numpy as np
In [0]:
df_red_africa = pd.read_csv('/content/drive/My Drive/Text/Text Project/reddit/Reddit_africa.csv', header=None)
df_red_africa.columns = ['Comments']
In [5]:
df_red_africa.head()
Out[5]:
Comments
0	Those dying and going into bankruptcy under th...
1	That's what makes me worried. Ive lived all ar...
2	Haha you mean the states that we all prop up a...
3	Wow USA so rich, but still cant afford univers...
4	Trinkets and material goods are of no comfort ...
In [6]:
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
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
In [0]:
df_red_africa['Filter Comments NO punct'] = df_red_africa['Comments'].map(removePunctuations)
# df_redUS['Filtered Comments'] = df_redUS['Comments'].map(filterReview)
In [0]:
df_red_africa.to_csv("/content/drive/My Drive/Text/Text Project/reddit/Reddit_africa_c1.csv")
Topic Modelling
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
In [0]:
dictionary_africa = gensim.corpora.Dictionary(df_red_africa['Filter Comments NO punct'])
dictionary_africa.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_africa.doc2bow(words) for words in df_red_africa['Filter Comments NO punct']]
lda_model_africa = gensim.models.LdaMulticore(bow_corpus_before, num_topics=6, id2word=dictionary_africa, passes=2, workers=2, chunksize=100, random_state=1000)
In [12]:
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_africa.print_topics())
[   (   0,
        '0.041*"*" + 0.019*"circumcision" + 0.017*"medical" + 0.016*"`" + '
        '0.014*"-" + 0.014*"\'\'" + 0.013*"katie" + 0.012*"/" + '
        '0.010*"association" + 0.010*".."'),
    (   1,
        '0.216*"-" + 0.139*"/" + 0.030*"http" + 0.027*"https" + 0.010*"=" + '
        '0.010*"*" + 0.008*"gt" + 0.006*"the" + 0.006*"amp" + 0.005*"news"'),
    (   2,
        '0.013*"-" + 0.009*"state" + 0.009*"trump" + 0.008*"war" + '
        '0.008*"clinton" + 0.007*"hillary" + 0.007*"’" + 0.006*"military" + '
        '0.006*"president" + 0.006*"vote"'),
    (   3,
        '0.010*"\'s" + 0.010*"not" + 0.009*"people" + 0.008*"would" + '
        '0.008*"get" + 0.007*"country" + 0.007*"pay" + 0.006*"money" + '
        '0.006*"go" + 0.006*"work"'),
    (   4,
        '0.019*"not" + 0.018*"`" + 0.017*"\'s" + 0.015*"people" + 0.012*"gt" + '
        '0.009*"\'\'" + 0.009*"be" + 0.009*"country" + 0.008*"would" + '
        '0.008*"like"'),
    (   5,
        '0.077*"*" + 0.042*"gt" + 0.040*".." + 0.014*"`" + 0.014*"-" + '
        '0.010*"/" + 0.008*"\'\'" + 0.008*"_" + 0.007*"walk" + 0.006*"woman"')]
Additional Stop Words removal
In [0]:
new_stopword_list = ['gt', '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument', 'amitheasshole', 'https', 'http']
def additionalStop(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list]
In [0]:
df_red_africa['filter V1'] = df_red_africa['Filter Comments NO punct'].map(additionalStop)
In [0]:
df_red_africa.to_csv("/content/drive/My Drive/Text/Text Project/reddit/Reddit_africa_c1.csv")
In [0]:
dictionary_africa = gensim.corpora.Dictionary(df_red_africa['filter V1'])
dictionary_africa.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_africa.doc2bow(words) for words in df_red_africa['filter V1']]
In [0]:
lda_model_africa2 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_africa, passes=10, workers=2, chunksize=100, random_state=1000)
In [17]:
lda_model_africa2 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=3, id2word=dictionary_africa, passes=10, workers=2, chunksize=100, random_state=1000)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_africa2.print_topics())
[   (   0,
        '0.020*"medical" + 0.017*"circumcision" + 0.012*"child" + '
        '0.011*"woman" + 0.009*"association" + 0.009*"health" + 0.008*"doctor" '
        '+ 0.007*"procedure" + 0.006*"male" + 0.006*"disease"'),
    (   1,
        '0.016*"people" + 0.010*"country" + 0.010*"would" + 0.008*"like" + '
        '0.006*"good" + 0.006*"make" + 0.006*"world" + 0.006*"think" + '
        '0.005*"even" + 0.005*"work"'),
    (   2,
        '0.007*"clinton" + 0.006*"trump" + 0.005*"hillary" + 0.005*"state" + '
        '0.005*"obama" + 0.005*"president" + 0.004*"health" + 0.004*"libya" + '
        '0.004*"year" + 0.004*"million"')]
In [18]:
lda_model_africa3 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_africa, passes=10, workers=2, chunksize=100, random_state=1000)
lda_model_africa3.print_topics()
Out[18]:
[(0,
  '0.024*"circumcision" + 0.020*"medical" + 0.013*"association" + 0.012*"child" + 0.010*"male" + 0.010*"procedure" + 0.008*"sexual" + 0.007*"woman" + 0.007*"abortion" + 0.007*"evidence"'),
 (1,
  '0.018*"people" + 0.011*"would" + 0.010*"country" + 0.009*"like" + 0.007*"make" + 0.007*"good" + 0.007*"think" + 0.007*"world" + 0.006*"even" + 0.006*"work"'),
 (2,
  '0.012*"trump" + 0.009*"state" + 0.009*"clinton" + 0.009*"president" + 0.007*"hillary" + 0.007*"vote" + 0.007*"obama" + 0.007*"israel" + 0.007*"support" + 0.006*"policy"'),
 (3,
  '0.014*"china" + 0.009*"million" + 0.008*"capitalism" + 0.008*"world" + 0.008*"cuba" + 0.008*"india" + 0.007*"capitalist" + 0.007*"year" + 0.007*"nation" + 0.006*"country"'),
 (4,
  '0.014*"health" + 0.006*"walk" + 0.006*"medical" + 0.006*"hospital" + 0.005*"care" + 0.005*"disease" + 0.005*"doctor" + 0.005*"edge" + 0.005*"mcgarry" + 0.005*"katie"')]
In [19]:
lda_model_africa4 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_africa, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_africa4.print_topics()
Out[19]:
[(0,
  '0.028*"circumcision" + 0.022*"medical" + 0.015*"association" + 0.012*"child" + 0.012*"procedure" + 0.011*"male" + 0.008*"evidence" + 0.007*"netherlands" + 0.007*"college" + 0.007*"surgeon"'),
 (1,
  '0.018*"people" + 0.012*"would" + 0.010*"country" + 0.010*"like" + 0.007*"good" + 0.007*"make" + 0.007*"think" + 0.006*"world" + 0.006*"even" + 0.006*"thing"'),
 (2,
  '0.015*"trump" + 0.011*"clinton" + 0.009*"state" + 0.009*"vote" + 0.009*"obama" + 0.009*"hillary" + 0.009*"president" + 0.007*"support" + 0.006*"policy" + 0.005*"unite"'),
 (3,
  '0.012*"china" + 0.010*"country" + 0.008*"capitalism" + 0.008*"world" + 0.008*"state" + 0.007*"nation" + 0.007*"million" + 0.007*"cuba" + 0.006*"capitalist" + 0.006*"year"'),
 (4,
  '0.014*"health" + 0.006*"medical" + 0.006*"disease" + 0.005*"hospital" + 0.005*"woman" + 0.005*"care" + 0.005*"doctor" + 0.005*"walk" + 0.004*"year" + 0.004*"rate"')]
In [0]:
# df_redUS.to_csv('RedditUSA.csv')
Preprocessing to remove the most common trivial words
In [0]:
from collections import Counter
s = df_red_africa['filter V1']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
In [0]:
common_stops = ['would', 'like','make','think','even','also','right','thing','year','well','much','take','know','time','many','still','come','look','really','have','fuck','will']
In [0]:
new_stopword_list_v1 = ['//www.reddit.com', 'people', 'would', 'like', 'make', 'work', 'good', 'think', 'want', 'right', 'well', 'thing', 'year', 'take', 'also', 'know', 'time', 'come', 'every', 'life', 'look', 'have', 'will', 'tell', 'believe', 'talk', 'seem', 'since', 'show', 'else'] 

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in common_stops]

df_red_africa['filter V2'] = df_red_africa['filter V1'].map(additionalStop_V1)
In [0]:
df_red_africa.to_csv("/content/drive/My Drive/Text/Text Project/reddit/Reddit_africa_c1.csv")
In [25]:
df_red_africa.head()
Out[25]:
Comments	Filter Comments NO punct	filter V1	filter V2
0	Those dying and going into bankruptcy under th...	[dye, go, bankruptcy, aca, healthcare, pharma,...	[bankruptcy, healthcare, pharma, ceos, profit,...	[bankruptcy, healthcare, pharma, ceos, profit,...
1	That's what makes me worried. Ive lived all ar...	['s, make, worry, -PRON-, have, live, around, ...	[make, worry, -PRON-, have, live, around, worl...	[worry, -PRON-, live, around, world, nigeria, ...
2	Haha you mean the states that we all prop up a...	[haha, mean, state, prop, would, fail, like, g...	[haha, mean, state, prop, would, fail, like, g...	[haha, mean, state, prop, fail, greece, federa...
3	Wow USA so rich, but still cant afford univers...	[wow, usa, rich, still, can, not, afford, univ...	[rich, still, afford, universal, healthcare, p...	[rich, still, afford, universal, healthcare, p...
4	Trinkets and material goods are of no comfort ...	[trinket, material, good, comfort, deathbed, y...	[trinket, material, good, comfort, deathbed, w...	[trinket, material, comfort, deathbed, riot, n...
In [0]:
dictionary_africa = gensim.corpora.Dictionary(df_red_africa['filter V2'])
dictionary_africa.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_africa.doc2bow(words) for words in df_red_africa['filter V2']]
In [28]:
lda_model_africa5 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_africa, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_africa5.print_topics()
Out[28]:
[(0,
  '0.013*"white" + 0.010*"black" + 0.007*"woman" + 0.007*"immigrant" + 0.006*"live" + 0.005*"culture" + 0.005*"even" + 0.005*"south" + 0.005*"edge" + 0.004*"many"'),
 (1,
  '0.014*"world" + 0.009*"china" + 0.009*"nation" + 0.008*"population" + 0.007*"state" + 0.007*"europe" + 0.006*"million" + 0.006*"rate" + 0.005*"high" + 0.005*"india"'),
 (2,
  '0.024*"medical" + 0.023*"circumcision" + 0.014*"child" + 0.013*"association" + 0.010*"procedure" + 0.009*"male" + 0.007*"sexual" + 0.007*"evidence" + 0.007*"health" + 0.007*"doctor"'),
 (3,
  '0.007*"government" + 0.007*"money" + 0.007*"need" + 0.006*"even" + 0.006*"system" + 0.006*"world" + 0.006*"much" + 0.005*"live" + 0.005*"little" + 0.005*"give"'),
 (4,
  '0.010*"clinton" + 0.008*"katie" + 0.008*"mcgarry" + 0.008*"hillary" + 0.007*"trump" + 0.007*"health" + 0.006*"obama" + 0.006*"walk" + 0.006*"president" + 0.006*"book"')]
Africa Word Press Data
In [30]:
df_wp_africa = pd.read_excel('/content/drive/My Drive/Text/Text Project/Wordpress/africa compiled.xlsx')
df_wp_africa.columns = ['Comments']
Out[30]:
0
In [31]:
df_wp_africa.head()
Out[31]:
Comments
0	Have health inequalities worsened in South Afr...
1	Tackling the looming epidemic of non-infectiou...
2	Foreign Aid, Gay Rights, and Public Health inÂ...
3	Is there a future for the healthcare system in...
4	"By Mark Williams, Club President | Strand Rot...
In [0]:
df_wp_africa['Filtered Comments'] = df_wp_africa['Comments'].map(removePunctuations)
In [33]:
df_wp_africa.head()
Out[33]:
Comments	Filtered Comments
0	Have health inequalities worsened in South Afr...	[health, inequality, worsen, south, africa, si...
1	Tackling the looming epidemic of non-infectiou...	[tackle, loom, epidemic, non, -, infectious, d...
2	Foreign Aid, Gay Rights, and Public Health inÂ...	[foreign, aid, gay, right, public, health, inâ...
3	Is there a future for the healthcare system in...	[future, healthcare, system, inâ, africa, acco...
4	"By Mark Williams, Club President | Strand Rot...	[`, `, mark, williams, club, president, strand...
In [0]:
from collections import Counter
s = df_wp_africa['filter V2']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
In [0]:
df = pd.DataFrame(common_words)
df.to_csv('/content/drive/My Drive/Text/Text Project/Wordpress/africa_wp_wordcount.csv')
In [0]:
new_stopword_list_v1 = ['â', '™', 'would', '-', 's', 'â€', '“', 'itâ€', 'go', 't', 'also', 'year', 're', 'take', 'would', 'thing', 'be', 'day', '¦', 'come', '/', 'many', 'give', 'tell', 'know', 'can', 'iâ€', 'thereâ€', 'thatâ€', '\ufeff1','well'] 

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list_v1]

df_wp_africa['filter V2'] = df_wp_africa['Filtered Comments'].map(additionalStop_V1)
In [0]:
df_wp_africa.to_csv("/content/drive/My Drive/Text/Text Project/Wordpress/wp_africa_c1.csv")
In [0]:
dictionary_wp_africa = gensim.corpora.Dictionary(df_wp_africa['filter V2'])
dictionary_wp_africa.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_wp_africa.doc2bow(words) for words in df_wp_africa['filter V2']]
In [43]:
lda_model_wpafrica5 = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_wp_africa, passes=50, workers=2, chunksize=100, random_state=100)
lda_model_wpafrica5.print_topics()
Out[43]:
[(0,
  '0.014*"global" + 0.012*"right" + 0.011*"fund" + 0.011*"human" + 0.007*"project" + 0.007*"program" + 0.007*"grant" + 0.007*"organization" + 0.006*"research" + 0.006*"support"'),
 (1,
  '0.009*"south" + 0.007*"country" + 0.007*"disease" + 0.005*"family" + 0.005*"work" + 0.005*"care" + 0.004*"rotary" + 0.004*"service" + 0.004*"like" + 0.004*"medical"'),
 (2,
  '0.016*"care" + 0.014*"migrant" + 0.010*"south" + 0.008*"palliative" + 0.007*"country" + 0.005*"hospital" + 0.005*"service" + 0.005*"cancer" + 0.005*"african" + 0.004*"access"'),
 (3,
  '0.016*"mental" + 0.015*"think" + 0.011*"medical" + 0.010*"humanity" + 0.008*"base" + 0.007*"care" + 0.006*"work" + 0.006*"science" + 0.006*"international" + 0.006*"social"'),
 (4,
  '0.012*"research" + 0.007*"fund" + 0.006*"grant" + 0.005*"work" + 0.005*"award" + 0.005*"tiba" + 0.005*"challenge" + 0.004*"mental" + 0.004*"global" + 0.004*"country"')]
Summarization
In [0]:
from nltk.tokenize import sent_tokenize
sentences = []
for s in df_wp_africa['Comments']:
  sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list
In [46]:
sentences[:5]
Out[46]:
['Have health inequalities worsened in South Africa sinceÂ\xa0apartheid?',
 '"Weâ€™ve all heard about the infamous apartheid-era health system in South Africa.',
 'As a middle-income country, richer than many in sub-Saharan Africa, the Republic of South Africa provided world-class care for White elites, including the worldâ€™s first heart transplant.',
 'But the majority of people were denied appropriate access to health care.',
 'Spatial segregation between populations was a prominent method to sustain inequality during apartheid, with racially-biased policies leading to the creation of â€˜â€˜Black homelandsâ€™â€™ that detached the poorest areas from regions with better health care infrastructure.Whatâ€™s happened since apartheid ended?']
In [47]:
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
--2019-11-30 23:13:01--  http://nlp.stanford.edu/data/glove.6B.zip
Resolving nlp.stanford.edu (nlp.stanford.edu)... 171.64.67.140
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:80... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://nlp.stanford.edu/data/glove.6B.zip [following]
--2019-11-30 23:13:01--  https://nlp.stanford.edu/data/glove.6B.zip
Connecting to nlp.stanford.edu (nlp.stanford.edu)|171.64.67.140|:443... connected.
HTTP request sent, awaiting response... 301 Moved Permanently
Location: http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip [following]
--2019-11-30 23:13:01--  http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
Resolving downloads.cs.stanford.edu (downloads.cs.stanford.edu)... 171.64.64.22
Connecting to downloads.cs.stanford.edu (downloads.cs.stanford.edu)|171.64.64.22|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 862182613 (822M) [application/zip]
Saving to: ‘glove.6B.zip’

glove.6B.zip        100%[===================>] 822.24M  2.01MB/s    in 6m 28s  

2019-11-30 23:19:29 (2.12 MB/s) - ‘glove.6B.zip’ saved [862182613/862182613]

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
In [57]:
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
I donâ€™t know whether itâ€™s more of an umbrella concept â€“ just if I look at the journal, thereâ€™s so many different [topics], from poetry to ethics, to finance, to how to train medical students, to the arts, so it seems to hold a lot of things together â€“ but it does seem maybe interdisciplinary and multidisciplinary.From my very limited understanding I think one of the goals would be to help medical students and medical practitioners act in more â€“ in less mechanical ways, more humane ways; so maybe ultimately the goal is to produce medical practitioners that are moreâ€¦ that listen better, that have more of an understanding of the whole person.VHÂ Â Â Â Â Â  Thinking about it in those broadish terms, do you think that there is any particular value for this kind of work relating to South Africa, or to southern Africa more generally?AdSÂ Â Â Â  You know, yesterday there was an interview on CNN with somebody whoâ€™d worked with ebola; I was listening more with a research methodology hat on but maybe it applies here as well: you can have all the medical knowledge in the world, but if a community doesnâ€™t trust you, the information youâ€™re giving them about how to manage the body of your deceased loved one â€“ theyâ€™re not going to listen to you.
He sayâ€™s â€œAlways aim at getting the results wherever u find yourselfâ€.SHORT PROFILE:Demilade is a man of many parts â€“ trained petroleum engineer, health systems consultant by day, writer by night, project manager by practice, and drummer by weekend.Heâ€™s worked as the associate editor of over 5 magazines and online platforms reaching audience in over 40 countries.In his current role, he led a project that provided HIV testing for over 420, 000 people and put over 2,000 people living with HIV on life saving anti-retroviral therapy across FCT, Benue and Nasarawa state.
It has held six successful elections since the 1992 end of military rule.Its economy has also been strong in recent years, with 14% growth in 2011 and some eight percent in 2012.But Gates and others point out that one of the real secrets of its success has been the way its health service is organised, with efficient community health centres that conduct outreach to ensure as many babies are vaccinated as possible.Subscribe to the Africa Research Bulletin today
As a participant you will have the opportunity to learn how to use the limited resources at your disposal to deliver health care at all times.Â One of the questions asked most by participants who want to take part in our programs is of how a typical day with Elective Africa healthcare internships is like.
In the case of South Africa, it is clear that the ideal health professional â€˜fit for purposeâ€™, and ready to deliver health services as proposed by Health Plan 2030, needs to use their hearts and their heads to bridge the different chasms between theory and practice, between disease and the patients experience of illness and lastly between the body as object that needs fixing and the person as subject of our care and intervention focus.
So I think that bridges that gap.VHÂ Â Â Â Â Â Â Â Â Â Â  Thereâ€™s a long history â€“ particularly in this country, of the use of theatre in that way, in medical education; but I suppose the issue is that itâ€™s not mainstreamed; it comes down to the initiative of a particular clinic, doctor â€“EPÂ Â Â Â Â Â Â Â Â Â Â Â  Itâ€™s targeted niches, or targeted schools, because not all schools have the privilege of having people coming to perform.
Over the next several days, thousands of Rotarians and partners rolled up their sleeves to work together, resultingÂ in the provision of diseaseÂ prevention and treatment services to over 66,500 folks in need.TheÂ power of these partnerships, which includedÂ the South Africa Department of Health taking full responsibility for follow up on referrals, is what makes RFHDs such an excellent model for high impact, measurable and sustainable interventions.As Rotary continues to explore models for projects that provide large, measurable impacts and are sustainable, we have in our â€˜front yardâ€™ a living, evolving masterpiece.
Itâ€™s very difficult to change their beliefs because this is what they have believed since they were born,â€ he says.The government has enforced security in affected areas, but many feel that it should have taken a stauncher response, reassuring those who felt threatened by the wave of violence.â€œAs a country I think weâ€™re very ashamed,â€ says Dr Nyaka, adding that the Malawi Society of Medical Doctors â€œwill take this as a challenge to communicate to our people and to reassure them that they are not blood suckers.â€Malawi will from November 1st host a three-day high-level meeting on promoting policy coherence on health technology innovation and access for the African Regional Intellectual Property Office (ARIPO), said Malawi News Agency.The meeting brings together a range of leaders, policymakers and institutions including representatives from Ministries of Trade, Health and Justice, civil society, international experts and academics.Minister of Health Atupele Muluzi has said Malawi and many of the 18 ARIPO member states have made great strides in improving public health and by consequence, human development outcomes in recent years.He said that despite significant progress, the burden of infectious diseases, particularly HIV, malaria and Tuberculosis pose a threat to public health.Minister of Justice and Constitutional Affairs, Samuel Tembenu said the meeting comes at a time when countries around the world are pursuing various means to ensure availability and access to medicines for their citizens.The meeting aims at providing a forum for ARIPO countries to exchange views and to share experiences on best practices that promote availability and access to affordable health coverage, Tembenu added.The high-level meeting has been jointly organised by the Malawi government and the United Nations Development Programme (UNDP).Find out more in the Africa Research Bulletin:Malawi â€“ Refugee Clashes DeploredPolitical, Social and Cultural series
More than 90% of Rwandans are now covered under the community-based health insurance scheme known as Mutuelle de SantÃ©, reports The East African.It becomes one of the few developing countries that have successfully achieved universal healthcare.Following problems in recent years with low real uptake numbers and poor service delivery, the government transferred management of the scheme to the Rwanda Social Security Board (RSSB).According to the Ministry of Health, management of the scheme has improved since it was moved to RSSB in 2015, but some citizens say the quality of services is still wanting.â€œIt is not easy to secure a transfer from a district hospital if you need to go to a referral hospital,â€ says Sarafina Mukasarasi, 40, a card holder, though she says services at the district hospitals have improved, despite the long queues.Mukasarasi however says that thanks to the insurance scheme, for which she pays about $3.36 (Rwf3,000) a year based on her household income, she does not pay a single penny when she falls ill or gives birth.The scheme has been touted as one of the most successful on the continent and is credited for the countryâ€™s lower maternal and infant mortality rates of 77% and 70%, respectively, since 2000.In November, experts had called for proactive measures to achieve universal healthcare coverage in order to reverse the trend whereby millions of people are pushed into extreme poverty by unaffordable healthcare.Some of the main challenges are non-communicable diseases (NCDs) such as cancer, diabetes and kidney disease, which are costly to treat, according to a report by The New Times.Â Maternal and child health are the main areas to be covered in Kenyaâ€™s new pilot programme.
Or was that during the [HIV/AIDS] crisis?DALÂ Â Â Â Â Â Â Â  Yeah, but also more generally the greater emphasis on primary healthcare â€¦ in informal settlements around Khayelitsha in Cape Town â€“ the outreach dimensions, the social health dimensions, and even practical things like going into ECD [Early Childhood Development] facilities for preventive medicine screening, inoculation, all those kinds of things â€“ a lot of thatâ€™s fallen apart since the clinic structure has improved; which means that the networks, social relations, are also less; and itâ€™s a deeply moralistic space.
In [62]:
!pip install vaderSentiment
Collecting vaderSentiment
  Downloading https://files.pythonhosted.org/packages/86/9e/c53e1fc61aac5ee490a6ac5e21b1ac04e55a7c2aba647bb8411c9aadf24e/vaderSentiment-3.2.1-py2.py3-none-any.whl (125kB)
     |████████████████████████████████| 133kB 3.4MB/s 
Installing collected packages: vaderSentiment
Successfully installed vaderSentiment-3.2.1
In [0]:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def get_sentiment(review, **kwargs):
 sentiment_score = analyser.polarity_scores(review)
 positive_meter = round((sentiment_score['pos'] * 100), 2)
 negative_meter = round((sentiment_score['neg'] * 100), 2)
 return positive_meter if kwargs['k'] == 'positive' else negative_meter
In [0]:
df_red_africa['positive'] = df_red_africa['Comments'].apply(get_sentiment, k='positive')
df_red_africa['negative'] = df_red_africa['Comments'].apply(get_sentiment, k='negative')
In [0]:
df_wp_africa['positive'] = df_wp_africa['Comments'].apply(get_sentiment, k='positive')
df_wp_africa['negative'] = df_wp_africa['Comments'].apply(get_sentiment, k='negative')
In [67]:
df_red_africa['positive'].mean()
Out[67]:
10.962250000000006
In [68]:
df_red_africa['negative'].mean()
Out[68]:
9.856080000000043
Reuters
In [0]:
df_ru_africa = pd.read_csv('/content/drive/My Drive/Text/Text Project/Reuters/Reuters_Africa.csv')
In [61]:

Out[61]:
Title	Field1
0	UPDATE 1-South Africa's Life Healthcare posts ...	HealthcareMay 30, 2019 / 2:03 AM / 6 months a...
1	South Africa puts initial universal healthcare...	Health NewsAugust 8, 2019 / 8:44 AM / 4 month...
2	South Africa's Life Healthcare plans Poland ex...	HealthcareNovember 20, 2019 / 11:53 PM / 9 da...
3	South Africa's Netcare expects FY HEPS to more...	HealthcareNovember 14, 2019 / 9:35 AM / 16 da...
4	Japan to drive Africa investment with enhanced...	Business NewsAugust 28, 2019 / 4:59 AM / 3 mo...
5	South Africa's Life Healthcare posts marginal ...	HealthcareMay 30, 2019 / 12:58 AM / 6 months ...
6	S Africa puts initial universal healthcare cos...	FinancialsAugust 8, 2019 / 8:34 AM / 4 months...
7	South Africa's Life Healthcare plans Poland ex...	HealthcareNovember 20, 2019 / 11:53 PM / 9 da...
8	TPG healthcare fund CEO to leave -memo	Funds NewsMay 14, 2019 / 12:39 PM / 7 months ...
9	Japan to drive Africa investment with enhanced...	Business NewsAugust 28, 2019 / 4:59 AM / 3 mo...
10	South Africa to roll out sweeping health refor...	World NewsAugust 23, 2019 / 4:26 AM / 3 month...
11	UPDATE 1-South Africa's Netcare posts higher H...	HealthcareNovember 18, 2019 / 1:01 AM / 12 da...
12	South Africa's Life Healthcare FY earnings up ...	HealthcareNovember 22, 2018 / 11:43 PM / a ye...
13	UPDATE 1-South Africa's slow growth hurts Impe...	Cyclical Consumer GoodsFebruary 28, 2019 / 8:...
14	UPDATE 1-South Africa's Life Healthcare FY ear...	HealthcareNovember 23, 2018 / 12:20 AM / a ye...
15	UPDATE 1-South Africa's Long4Life enters healt...	FinancialsOctober 24, 2018 / 10:17 AM / a yea...
16	South Africa's Life Healthcare H1 profit more ...	HealthcareJune 1, 2018 / 12:29 AM / a year ag...
17	Abraaj sets up $200 mln North Africa healthcar...	FinancialsMarch 2, 2015 / 9:40 AM / 5 years a...
18	Abraaj scouts for Africa healthcare deals afte...	FinancialsJanuary 30, 2013 / 7:20 AM / 7 year...
19	UPDATE 1-South Africa's Life Healthcare posts ...	HealthcareMay 30, 2019 / 2:03 AM / 6 months a...
20	South Africa puts initial universal healthcare...	Health NewsAugust 8, 2019 / 8:44 AM / 4 month...
21	TPG healthcare fund CEO to leave -memo	Funds NewsMay 14, 2019 / 12:39 PM / 7 months ...
22	UPDATE 1-South Africa's slow growth hurts Impe...	Cyclical Consumer GoodsFebruary 28, 2019 / 2:...
23	Abraaj scouts for Africa healthcare deals afte...	FinancialsJanuary 30, 2013 / 7:20 AM / 7 year...
