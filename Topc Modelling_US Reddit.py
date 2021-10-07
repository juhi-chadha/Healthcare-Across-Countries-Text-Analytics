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
df_redUS = pd.read_csv('/content/drive/My Drive/Text Project/reddit/Reddit_us.csv', header=None)
In [13]:
df_redUS.head()
Out[13]:
Comments
0	I had a kidney transplant last year after 12 y...
1	Wow, I envy you your being in Canada, healthca...
2	That's...not my argument at all.\n\nFurther, t...
3	I'd like to point out that this comment doesn'...
4	Holy Christ the US healthcare system is brutal...
In [14]:
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
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
In [0]:
df_redUS['Filter Comments NO punct'] = df_redUS['Comments'].map(removePunctuations)
# df_redUS['Filtered Comments'] = df_redUS['Comments'].map(filterReview)
In [72]:
df_redUS.head()
Out[72]:
Comments	Filtered Comments	Filter Comments NO punct
0	I had a kidney transplant last year after 12 y...	[kidney, transplant, last, year, year, dialysi...	[kidney, transplant, last, year, year, dialysi...
1	Wow, I envy you your being in Canada, healthca...	[wow, ,, envy, canada, ,, healthcare, ., ’, gr...	[wow, envy, canada, healthcare, ’, great, eith...
2	That's...not my argument at all.\n\nFurther, t...	['s, ..., argument, ., ,, 's, evidence, single...	['s, ..., argument, 's, evidence, single, paye...
3	I'd like to point out that this comment doesn'...	[would, like, point, comment, not, defend, us,...	[would, like, point, comment, not, defend, us,...
4	Holy Christ the US healthcare system is brutal...	[holy, christ, us, healthcare, system, brutal,...	[holy, christ, us, healthcare, system, brutal,...
Topic Modelling
In [0]:
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
In [0]:
dictionary_US = gensim.corpora.Dictionary(df_redUS['Filtered Comments'])
dictionary_US.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_US.doc2bow(words) for words in df_redUS['Filtered Comments']]
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=6, id2word=dictionary_US, passes=2, workers=2, chunksize=100, random_state=1000)
In [65]:
# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(lda_model_US.print_topics())
[   (   0,
        '0.097*"*" + 0.060*"/" + 0.022*":" + 0.021*"[" + 0.021*"]" + '
        '0.020*"subreddit" + 0.017*"suicide" + 0.016*"(" + 0.016*")" + '
        '0.015*"post"'),
    (   1,
        '0.166*"-" + 0.062*"/" + 0.036*":" + 0.033*")" + 0.032*"(" + '
        '0.023*"https" + 0.016*"[" + 0.016*"]" + 0.013*"*" + 0.007*"trump"'),
    (   2,
        '0.020*"not" + 0.018*"’" + 0.017*"people" + 0.014*"\'s" + 0.013*"?" + '
        '0.011*"get" + 0.010*"would" + 0.009*"like" + 0.008*"be" + '
        '0.008*"work"'),
    (   3,
        '0.029*"`" + 0.022*"*" + 0.017*"\'\'" + 0.015*"\'s" + 0.011*"not" + '
        '0.009*")" + 0.008*"(" + 0.007*"\'" + 0.007*"right" + 0.006*"be"'),
    (   4,
        '0.047*";" + 0.041*"&" + 0.034*"gt" + 0.010*"country" + 0.008*"-" + '
        '0.007*"%" + 0.007*")" + 0.007*"state" + 0.007*"?" + 0.006*"("'),
    (   5,
        '0.019*"pay" + 0.017*"cost" + 0.017*"$" + 0.017*"insurance" + '
        '0.014*")" + 0.013*"(" + 0.010*"health" + 0.009*"hospital" + '
        '0.008*"would" + 0.008*"-"')]
In [0]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=10, workers=2, chunksize=100, random_state=1000)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_US.print_topics())
In [67]:

[   (   0,
        '0.057*"/" + 0.052*"*" + 0.023*"subreddit" + 0.020*"[" + 0.019*"]" + '
        '0.019*":" + 0.019*"suicide" + 0.017*"message" + 0.017*"post" + '
        '0.016*"mention"'),
    (   1,
        '0.141*"-" + 0.060*"/" + 0.057*"*" + 0.036*")" + 0.035*":" + 0.035*"(" '
        '+ 0.020*"https" + 0.017*"[" + 0.017*"]" + 0.007*"health"'),
    (   2,
        '0.017*"not" + 0.016*"’" + 0.013*"people" + 0.013*"\'s" + 0.012*"get" '
        '+ 0.010*"would" + 0.010*"pay" + 0.009*"system" + 0.008*"?" + '
        '0.008*"go"'),
    (   3,
        '0.023*"`" + 0.015*"\'s" + 0.014*"not" + 0.013*"\'\'" + 0.010*"*" + '
        '0.010*"?" + 0.009*"people" + 0.009*"right" + 0.009*"-" + 0.009*";"'),
    (   4,
        '0.039*";" + 0.034*"&" + 0.028*"gt" + 0.012*"country" + 0.010*"%" + '
        '0.009*"spend" + 0.009*"tax" + 0.009*"?" + 0.007*"$" + 0.007*"*"')]
Topic Modelling after removing punctuations
In [0]:
dictionary_US = gensim.corpora.Dictionary(df_redUS['Filter Comments NO punct'])
dictionary_US.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_US.doc2bow(words) for words in df_redUS['Filter Comments NO punct']]
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=10, workers=2, chunksize=100, random_state=1000)
In [74]:
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_US.print_topics())
[   (   0,
        '0.013*"country" + 0.011*"gt" + 0.009*"world" + 0.007*"`" + '
        '0.007*"capitalism" + 0.006*"government" + 0.005*"market" + '
        '0.005*"socialism" + 0.005*"society" + 0.005*"economic"'),
    (   1,
        '0.151*"-" + 0.067*"/" + 0.064*"*" + 0.023*"https" + 0.008*"trump" + '
        '0.006*"health" + 0.005*"gt" + 0.005*"`" + 0.005*"amp" + 0.004*"•"'),
    (   2,
        '0.066*"/" + 0.046*"*" + 0.028*"subreddit" + 0.024*"suicide" + '
        '0.021*"message" + 0.021*"post" + 0.019*"mention" + '
        '0.019*"amitheasshole" + 0.015*"someone" + 0.015*"issue"'),
    (   3,
        '0.025*"’" + 0.013*"pay" + 0.011*"get" + 0.010*"people" + '
        '0.010*"system" + 0.010*"would" + 0.009*"cost" + 0.009*"not" + '
        '0.009*"insurance" + 0.007*"go"'),
    (   4,
        '0.024*"not" + 0.021*"`" + 0.021*"\'s" + 0.013*"people" + 0.011*"be" + '
        '0.011*"\'\'" + 0.009*"would" + 0.009*"\'" + 0.008*"like" + '
        '0.008*"gt"')]
In [75]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=30, workers=2, chunksize=100, random_state=1000)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_US.print_topics())
[   (   0,
        '0.010*"gt" + 0.008*"country" + 0.007*"`" + 0.007*"capitalism" + '
        '0.007*"right" + 0.006*"world" + 0.006*"-" + 0.006*"society" + '
        '0.005*"“" + 0.005*"”"'),
    (   1,
        '0.153*"-" + 0.068*"/" + 0.067*"*" + 0.023*"https" + 0.008*"trump" + '
        '0.006*"health" + 0.006*"`" + 0.005*"gt" + 0.005*"amp" + 0.004*"•"'),
    (   2,
        '0.066*"/" + 0.047*"*" + 0.027*"subreddit" + 0.023*"suicide" + '
        '0.021*"message" + 0.021*"post" + 0.019*"mention" + '
        '0.019*"amitheasshole" + 0.014*"someone" + 0.014*"issue"'),
    (   3,
        '0.025*"’" + 0.013*"pay" + 0.010*"people" + 0.010*"get" + '
        '0.010*"system" + 0.009*"cost" + 0.009*"would" + 0.009*"insurance" + '
        '0.008*"not" + 0.007*"go"'),
    (   4,
        '0.025*"not" + 0.022*"\'s" + 0.021*"`" + 0.014*"people" + 0.012*"be" + '
        '0.011*"\'\'" + 0.010*"would" + 0.009*"\'" + 0.009*"like" + '
        '0.009*"get"')]
Additional Stop Words removal
In [0]:
new_stopword_list = ['gt', '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument', 'amitheasshole', 'https', 'http']
def additionalStop(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list]
In [0]:
df_redUS['filter V1'] = df_redUS['Filter Comments NO punct'].map(additionalStop)
In [0]:
dictionary_US = gensim.corpora.Dictionary(df_redUS['filter V1'])
dictionary_US.filter_extremes(no_below=1, no_above=0.7) #, keep_n=100000)
bow_corpus_before = [dictionary_US.doc2bow(words) for words in df_redUS['filter V1']]
In [105]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=10, workers=2, chunksize=100, random_state=1000)
[   (   0,
        '0.015*"people" + 0.009*"like" + 0.009*"work" + 0.008*"would" + '
        '0.008*"make" + 0.008*"know" + 0.007*"time" + 0.007*"need" + '
        '0.007*"good" + 0.007*"think"'),
    (   1,
        '0.031*"suicide" + 0.029*"message" + 0.029*"post" + 0.026*"mention" + '
        '0.026*"amitheasshole" + 0.020*"someone" + 0.020*"issue" + '
        '0.018*"suicidal" + 0.015*"https" + 0.015*"//www.reddit.com"'),
    (   2,
        '0.039*"https" + 0.013*"health" + 0.008*"trump" + 0.008*"medical" + '
        '0.007*"veteran" + 0.007*"http" + 0.007*"military" + 0.006*"care" + '
        '0.006*"news" + 0.006*"report"'),
    (   3,
        '0.015*"people" + 0.013*"system" + 0.013*"would" + 0.012*"country" + '
        '0.010*"cost" + 0.010*"government" + 0.010*"insurance" + 0.008*"money" '
        '+ 0.008*"make" + 0.007*"like"'),
    (   4,
        '0.011*"right" + 0.009*"people" + 0.008*"trump" + 0.007*"vote" + '
        '0.007*"like" + 0.007*"would" + 0.007*"want" + 0.007*"think" + '
        '0.005*"leave" + 0.005*"bernie"')]
In [111]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=3, id2word=dictionary_US, passes=10, workers=2, chunksize=100, random_state=1000)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda_model_US.print_topics())
[   (   0,
        '0.016*"people" + 0.009*"like" + 0.008*"would" + 0.008*"right" + '
        '0.007*"think" + 0.007*"make" + 0.007*"country" + 0.006*"work" + '
        '0.006*"thing" + 0.006*"want"'),
    (   1,
        '0.028*"suicide" + 0.026*"post" + 0.025*"message" + 0.025*"mention" + '
        '0.019*"someone" + 0.018*"issue" + 0.015*"suicidal" + 0.013*"care" + '
        '0.013*"story" + 0.013*"//www.reddit.com"'),
    (   2,
        '0.011*"would" + 0.010*"system" + 0.010*"cost" + 0.009*"insurance" + '
        '0.009*"people" + 0.008*"health" + 0.008*"care" + 0.007*"make" + '
        '0.006*"year" + 0.006*"country"')]
In [113]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=10, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[113]:
[(0,
  '0.017*"people" + 0.014*"would" + 0.012*"system" + 0.012*"country" + 0.010*"government" + 0.009*"like" + 0.009*"cost" + 0.008*"make" + 0.008*"insurance" + 0.008*"work"'),
 (1,
  '0.032*"suicide" + 0.030*"message" + 0.030*"post" + 0.027*"mention" + 0.021*"issue" + 0.021*"someone" + 0.018*"suicidal" + 0.016*"//www.reddit.com" + 0.015*"story" + 0.014*"emergency"'),
 (2,
  '0.010*"vote" + 0.009*"right" + 0.009*"trump" + 0.009*"bernie" + 0.006*"policy" + 0.006*"like" + 0.006*"party" + 0.006*"support" + 0.006*"warren" + 0.006*"people"'),
 (3,
  '0.012*"people" + 0.008*"like" + 0.007*"make" + 0.007*"know" + 0.007*"would" + 0.007*"work" + 0.006*"right" + 0.006*"want" + 0.006*"think" + 0.006*"good"'),
 (4,
  '0.019*"health" + 0.012*"medical" + 0.010*"care" + 0.008*"patient" + 0.008*"hospital" + 0.006*"datum" + 0.006*"rate" + 0.006*"study" + 0.005*"doctor" + 0.005*"high"')]
In [114]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[114]:
[(0,
  '0.019*"people" + 0.015*"would" + 0.012*"country" + 0.011*"system" + 0.010*"like" + 0.010*"government" + 0.009*"make" + 0.008*"work" + 0.007*"good" + 0.007*"well"'),
 (1,
  '0.032*"suicide" + 0.030*"message" + 0.030*"post" + 0.028*"mention" + 0.021*"someone" + 0.021*"issue" + 0.018*"suicidal" + 0.015*"//www.reddit.com" + 0.015*"story" + 0.014*"emergency"'),
 (2,
  '0.012*"trump" + 0.012*"vote" + 0.011*"bernie" + 0.008*"right" + 0.008*"party" + 0.008*"policy" + 0.008*"warren" + 0.007*"democrat" + 0.007*"support" + 0.007*"sander"'),
 (3,
  '0.009*"people" + 0.008*"like" + 0.007*"make" + 0.007*"know" + 0.007*"work" + 0.006*"time" + 0.006*"would" + 0.006*"good" + 0.005*"want" + 0.005*"think"'),
 (4,
  '0.021*"health" + 0.017*"insurance" + 0.014*"care" + 0.013*"medical" + 0.013*"hospital" + 0.012*"cost" + 0.012*"doctor" + 0.008*"patient" + 0.006*"treatment" + 0.006*"plan"')]
In [0]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, chunksize = 64, eval_every = 5, per_word_topics = 3, minimum_probability = 0.1, passes=25, random_state=23)
In [117]:
lda_model_US.print_topics()
Out[117]:
[(0,
  '0.023*"trump" + 0.013*"military" + 0.010*"veteran" + 0.007*"news" + 0.006*"report" + 0.006*"border" + 0.005*"datum" + 0.005*"troop" + 0.005*"fund" + 0.004*"caput"'),
 (1,
  '0.018*"people" + 0.011*"like" + 0.010*"work" + 0.010*"would" + 0.009*"make" + 0.008*"know" + 0.008*"good" + 0.008*"live" + 0.007*"time" + 0.007*"think"'),
 (2,
  '0.027*"insurance" + 0.025*"health" + 0.024*"cost" + 0.019*"care" + 0.018*"system" + 0.016*"medical" + 0.016*"doctor" + 0.016*"hospital" + 0.010*"high" + 0.009*"company"'),
 (3,
  '0.033*"suicide" + 0.030*"post" + 0.030*"message" + 0.028*"mention" + 0.021*"someone" + 0.021*"issue" + 0.018*"suicidal" + 0.015*"story" + 0.014*"//www.reddit.com" + 0.014*"question"'),
 (4,
  '0.014*"people" + 0.013*"would" + 0.013*"government" + 0.012*"country" + 0.009*"like" + 0.008*"system" + 0.007*"money" + 0.007*"make" + 0.007*"think" + 0.006*"right"')]
In [0]:
# df_redUS.to_csv('RedditUSA.csv')
Preprocessing to remove the most common trivial words
In [0]:
from collections import Counter
s = df_redUS['filter V1']
text = s.apply(pd.Series).stack().reset_index(drop=True)
word_counts = Counter(text)
common_words = word_counts.most_common()
In [0]:
common_stops = ['people', 'would', 'like', 'make', 'work', 'good', 'think', 'want', 'right', 'well', 'thing', 'year', 'take', 'also', 'know', 'time', 'come', 'every', 'life', 'look', 'have', 'will', 'tell', 'believe', 'talk', 'seem', 'since', 'show', 'else']
In [0]:
new_stopword_list_v1 = ['//www.reddit.com', 'people', 'would', 'like', 'make', 'work', 'good', 'think', 'want', 'right', 'well', 'thing', 'year', 'take', 'also', 'know', 'time', 'come', 'every', 'life', 'look', 'have', 'will', 'tell', 'believe', 'talk', 'seem', 'since', 'show', 'else'] 

def additionalStop_V1(comment):
    return [w for w in comment if len(w)>3 and w not in new_stopword_list_v1]

df_redUS['filter V2'] = df_redUS['filter V1'].map(additionalStop_V1)
In [140]:
df_redUS.head()
Out[140]:
Comments	Filtered Comments	Filter Comments NO punct	filter V1	filter V2
0	I had a kidney transplant last year after 12 y...	[kidney, transplant, last, year, year, dialysi...	[kidney, transplant, last, year, year, dialysi...	[kidney, transplant, last, year, year, dialysi...	[kidney, transplant, last, dialysis, medicare,...
1	Wow, I envy you your being in Canada, healthca...	[wow, ,, envy, canada, ,, healthcare, ., ’, gr...	[wow, envy, canada, healthcare, ’, great, eith...	[envy, canada, healthcare, great, either, know...	[envy, canada, healthcare, great, either, luck...
2	That's...not my argument at all.\n\nFurther, t...	['s, ..., argument, ., ,, 's, evidence, single...	['s, ..., argument, 's, evidence, single, paye...	[evidence, single, payer, healthcare, reduce, ...	[evidence, single, payer, healthcare, reduce, ...
3	I'd like to point out that this comment doesn'...	[would, like, point, comment, not, defend, us,...	[would, like, point, comment, not, defend, us,...	[would, like, point, comment, defend, private,...	[point, comment, defend, private, healthcare, ...
4	Holy Christ the US healthcare system is brutal...	[holy, christ, us, healthcare, system, brutal,...	[holy, christ, us, healthcare, system, brutal,...	[holy, christ, healthcare, system, brutal, wro...	[holy, christ, healthcare, system, brutal, wro...
In [0]:
dictionary_US = gensim.corpora.Dictionary(df_redUS['filter V2'])
dictionary_US.filter_extremes(no_below=1, no_above=0.5) #, keep_n=100000)
bow_corpus_before = [dictionary_US.doc2bow(words) for words in df_redUS['filter V2']]
In [138]:
lda_model_US = gensim.models.LdaMulticore(bow_corpus_before, num_topics=5, id2word=dictionary_US, passes=50, workers=2, chunksize=100, random_state=1000)
lda_model_US.print_topics()
Out[138]:
[(0,
  '0.009*"country" + 0.008*"even" + 0.007*"need" + 0.007*"government" + 0.006*"live" + 0.006*"system" + 0.006*"little" + 0.005*"mean" + 0.005*"much" + 0.005*"free"'),
 (1,
  '0.035*"suicide" + 0.033*"message" + 0.032*"post" + 0.030*"mention" + 0.023*"someone" + 0.023*"issue" + 0.020*"suicidal" + 0.016*"story" + 0.016*"question" + 0.015*"emergency"'),
 (2,
  '0.016*"trump" + 0.009*"vote" + 0.008*"bernie" + 0.007*"state" + 0.007*"policy" + 0.007*"military" + 0.007*"warren" + 0.006*"party" + 0.006*"democrat" + 0.005*"support"'),
 (3,
  '0.022*"cost" + 0.021*"insurance" + 0.017*"system" + 0.017*"health" + 0.014*"care" + 0.010*"country" + 0.010*"high" + 0.009*"hospital" + 0.009*"spend" + 0.009*"company"'),
 (4,
  '0.013*"medical" + 0.012*"woman" + 0.009*"patient" + 0.008*"health" + 0.007*"doctor" + 0.007*"trans" + 0.006*"datum" + 0.006*"treatment" + 0.005*"gender" + 0.005*"mental"')]
In [0]:
