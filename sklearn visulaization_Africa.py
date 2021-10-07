In [2]:
!pip install pyLDAvis

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline
Collecting pyLDAvis
  Downloading https://files.pythonhosted.org/packages/a5/3a/af82e070a8a96e13217c8f362f9a73e82d61ac8fff3a2561946a97f96266/pyLDAvis-2.1.2.tar.gz (1.6MB)
     |████████████████████████████████| 1.6MB 3.5MB/s 
Requirement already satisfied: wheel>=0.23.0 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (0.33.6)
Requirement already satisfied: numpy>=1.9.2 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (1.17.4)
Requirement already satisfied: scipy>=0.18.0 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (1.3.2)
Requirement already satisfied: pandas>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (0.25.3)
Requirement already satisfied: joblib>=0.8.4 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (0.14.0)
Requirement already satisfied: jinja2>=2.7.2 in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (2.10.3)
Requirement already satisfied: numexpr in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (2.7.0)
Requirement already satisfied: pytest in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (3.6.4)
Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyLDAvis) (0.16.0)
Collecting funcy
  Downloading https://files.pythonhosted.org/packages/ce/4b/6ffa76544e46614123de31574ad95758c421aae391a1764921b8a81e1eae/funcy-1.14.tar.gz (548kB)
     |████████████████████████████████| 552kB 37.1MB/s 
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.17.0->pyLDAvis) (2018.9)
Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.17.0->pyLDAvis) (2.6.1)
Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2>=2.7.2->pyLDAvis) (1.1.1)
Requirement already satisfied: pluggy<0.8,>=0.5 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (0.7.1)
Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (19.3.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (41.6.0)
Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (1.12.0)
Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (1.8.0)
Requirement already satisfied: more-itertools>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (7.2.0)
Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.6/dist-packages (from pytest->pyLDAvis) (1.3.0)
Building wheels for collected packages: pyLDAvis, funcy
  Building wheel for pyLDAvis (setup.py) ... done
  Created wheel for pyLDAvis: filename=pyLDAvis-2.1.2-py2.py3-none-any.whl size=97711 sha256=0fd31e02dc1c7c8bedb7d30e76248ce16ac965a7bcae53f98c68d01d472db49f
  Stored in directory: /root/.cache/pip/wheels/98/71/24/513a99e58bb6b8465bae4d2d5e9dba8f0bef8179e3051ac414
  Building wheel for funcy (setup.py) ... done
  Created wheel for funcy: filename=funcy-1.14-py2.py3-none-any.whl size=32040 sha256=81b686ebd50b96641876a8507ff5b4632292caa55b93dd5071fd13d8e0817b2c
  Stored in directory: /root/.cache/pip/wheels/20/5a/d8/1d875df03deae6f178dfdf70238cca33f948ef8a6f5209f2eb
Successfully built pyLDAvis funcy
Installing collected packages: funcy, pyLDAvis
Successfully installed funcy-1.14 pyLDAvis-2.1.2
In [1]:
from google.colab import drive
drive.mount('/content/drive')
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly

Enter your authorization code:
··········
Mounted at /content/drive
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
In [0]:
# Converting the data to list
data = df_red_africa.Comments.values.tolist()
In [7]:
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', word) for word in data]
<input>:1: DeprecationWarning: invalid escape sequence \S
<input>:1: DeprecationWarning: invalid escape sequence \S
<input>:1: DeprecationWarning: invalid escape sequence \S
<ipython-input-7-13916e437db0>:1: DeprecationWarning: invalid escape sequence \S
  data = [re.sub('\S*@\S*\s?', '', word) for word in data]
In [8]:
# Remove new line characters
data = [re.sub('\s+', ' ', word) for word in data]
<input>:1: DeprecationWarning: invalid escape sequence \s
<input>:1: DeprecationWarning: invalid escape sequence \s
<input>:1: DeprecationWarning: invalid escape sequence \s
<ipython-input-8-6a1dd0234014>:1: DeprecationWarning: invalid escape sequence \s
  data = [re.sub('\s+', ' ', word) for word in data]
In [0]:
# Remove distracting single quotes
data = [re.sub("\'", "", word) for word in data]
In [10]:
data[16]
Out[10]:
'I am a corporate auditor for a large multinational pharmaceutical company. I travel internationally for two week periods in business class and five star hotels. I also bring tremendous value to the company by performing financial, FCPA, advisory, IT, etc. audits and provide recommendations to upper management. My 2.5 week honeymoon will be entirely paid for via points accumulated, in addition to other awesome trips I’ve taken. I’ve been to Morocco, Thailand, Panama, France, Belgium, Japan, China, South Korea, South Africa, Turkey, Poland, and Dubai in the two years I’ve worked here. I am relatively young at 26, but credentialed and know my shit. I love my job, because I can also take vacation to extend my trip at a location and fly my fiancé out. We also get comp days each time we travel to make up for time lost while traveling. I also have learned a ton about pharmaceuticals and how there are systematic issues in the way healthcare operates, and am glad to say I work at an ethical and moral company despite its size. My goal is to work in upper management to figure out how we can increase access to medicines and cures to more people who need them while continuing to operate as a business in this extremely regulated and nuanced industry. Immunotherapy and inoculation are truly innovative life changing fields. I am only in my mid twenties.'
In [0]:
#text = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in data]
In [0]:
# text[16]
In [11]:
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[0])
['those', 'dying', 'and', 'going', 'into', 'bankruptcy', 'under', 'the', 'aca', 'while', 'healthcare', 'and', 'pharma', 'ceos', 'profit', 'would', 'like', 'word', 'too', 'but', 'obamas', 'abandonment', 'of', 'the', 'public', 'option', 'after', 'campaigning', 'on', 'it', 'is', 'just', 'one', 'issue', 'he', 'advanced', 'and', 'expanded', 'bushs', 'most', 'malignant', 'economic', 'war', 'and', 'police', 'state', 'policies', 'across', 'the', 'board', 'see', 'the', 'list', 'below', 'corporate', 'money', 'flows', 'into', 'both', 'parties', 'now', 'and', 'that', 'is', 'why', 'the', 'corporate', 'agenda', 'keeps', 'advancing', 'no', 'matter', 'which', 'party', 'is', 'in', 'power', 'we', 'are', 'propagandized', 'to', 'fight', 'red', 'team', 'versus', 'blue', 'team', 'so', 'that', 'we', 'will', 'never', 'unite', 'against', 'the', 'policies', 'that', 'hurt', 'us', 'all', 'as', 'long', 'as', 'we', 'close', 'our', 'eyes', 'to', 'what', 'the', 'guy', 'on', 'our', 'side', 'does', 'the', 'corporate', 'elite', 'can', 'always', 'be', 'sure', 'of', 'having', 'about', 'half', 'the', 'electorate', 'circling', 'the', 'wagons', 'around', 'their', 'malignant', 'policies', 'no', 'matter', 'who', 'is', 'in', 'power', 'thats', 'why', 'its', 'so', 'important', 'now', 'to', 'be', 'honest', 'about', 'whats', 'happening', 'and', 'unite', 'against', 'and', 'for', 'policy', 'instead', 'of', 'team', 'obama', 'biden', 'legacy', 'filled', 'his', 'cabinet', 'with', 'wall', 'street', 'and', 'monsanto', 'reps', 'according', 'to', 'citibank', 'recommendations', 'bailouts', 'and', 'settlements', 'for', 'corrupt', 'banks', 'with', 'personal', 'pressure', 'from', 'obama', 'to', 'attorneys', 'general', 'to', 'approve', 'them', 'delaying', 'prosecutions', 'until', 'statutes', 'of', 'limitations', 'were', 'expiring', 'failing', 'to', 'prosecute', 'mortgage', 'abusers', 'passionate', 'speeches', 'and', 'press', 'conferences', 'promoting', 'austerity', 'for', 'americans', 'eat', 'your', 'peas', 'extended', 'the', 'bush', 'tax', 'cuts', 'for', 'billionaires', 'then', 'made', 'much', 'of', 'it', 'permanent', 'support', 'for', 'the', 'payroll', 'tax', 'holiday', 'tying', 'ss', 'to', 'the', 'general', 'fund', 'support', 'for', 'the', 'vicious', 'chained', 'cpi', 'cuts', 'in', 'social', 'security', 'and', 'benefits', 'for', 'the', 'disabled', 'failed', 'to', 'veto', 'bipartisan', 'vote', 'in', 'congress', 'to', 'gut', 'more', 'financial', 'regulations', 'social', 'security', 'medicare', 'and', 'medicaid', 'offered', 'up', 'immediately', 'as', 'bargaining', 'chips', 'in', 'budget', 'negotiations', 'with', 'no', 'mention', 'of', 'cutting', 'corporate', 'welfare', 'or', 'the', 'military', 'budget', 'his', 'administration', 'oversaw', 'huge', 'increases', 'in', 'wealth', 'inequality', 'wealth', 'of', 'minorities', 'in', 'particular', 'collapsed', 'and', 'the', 'wealth', 'gap', 'between', 'whites', 'and', 'non', 'whites', 'skyrocketed', 'advocacy', 'of', 'multiple', 'new', 'free', 'trade', 'agreements', 'including', 'the', 'trans', 'pacific', 'otherwise', 'known', 'as', 'nafta', 'on', 'steroids', 'support', 'of', 'drilling', 'pipelines', 'and', 'selling', 'off', 'portions', 'of', 'the', 'gulf', 'of', 'mexico', 'corporate', 'education', 'policy', 'including', 'high', 'stakes', 'corporate', 'testing', 'and', 'closures', 'of', 'public', 'schools', 'entrenchment', 'of', 'exorbitant', 'for', 'profit', 'health', 'insurance', 'companies', 'into', 'healthcare', 'through', 'mandate', 'legal', 'assault', 'on', 'union', 'rights', 'of', 'hundreds', 'of', 'thousands', 'of', 'federal', 'workers', 'signed', 'cuts', 'in', 'food', 'stamps', 'for', 'the', 'most', 'desperate', 'americans', 'marijuana', 'users', 'and', 'medical', 'marijuana', 'clinics', 'under', 'assault', 'drone', 'campaigns', 'in', 'multiple', 'countries', 'with', 'whom', 'we', 'are', 'not', 'at', 'war', 'expanded', 'bombing', 'to', 'countries', 'and', 'militarized', 'across', 'africa', 'carried', 'out', 'multiple', 'disastrous', 'illegal', 'regime', 'changes', 'resulting', 'in', 'massive', 'slaughter', 'dispossession', 'refugee', 'crises', 'terrorism', 'and', 'open', 'slavery', 'renewed', 'public', 'advocacy', 'for', 'the', 'concept', 'of', 'preemptive', 'war', 'kill', 'lists', 'and', 'claiming', 'of', 'the', 'right', 'to', 'assassinate', 'even', 'american', 'citizens', 'without', 'trial', 'he', 'droned', 'an', 'american', 'teenager', 'his', 'administration', 'authorized', 'double', 'taps', 'war', 'crimes', 'in', 'which', 'drone', 'bombs', 'once', 'and', 'then', 'again', 'several', 'minutes', 'later', 'when', 'distraught', 'family', 'members', 'and', 'first', 'responders', 'have', 'arrived', 'on', 'the', 'scene', 'his', 'military', 'also', 'acknowledged', 'targeting', 'children', 'with', 'bombs', 'when', 'it', 'was', 'suspected', 'that', 'terrorists', 'were', 'using', 'children', 'as', 'shields', 'continued', 'bushs', 'signature', 'strikes', 'which', 'target', 'with', 'bombs', 'not', 'known', 'terrorists', 'but', 'general', 'areas', 'in', 'which', 'terrorist', 'activity', 'is', 'suspected', 'to', 'occur', 'and', 'which', 'conceal', 'civilian', 'death', 'counts', 'by', 'retroactively', 'declaring', 'every', 'murdered', 'man', 'within', 'certain', 'age', 'range', 'in', 'the', 'area', 'to', 'have', 'been', 'an', 'enemy', 'combatant', 'presided', 'over', 'more', 'drone', 'attacks', 'in', 'the', 'first', 'year', 'of', 'his', 'presidency', 'than', 'bush', 'did', 'during', 'his', 'entire', 'presidency', 'signing', 'ndaa', 'to', 'allow', 'indefinite', 'detention', 'new', 'massive', 'spy', 'center', 'in', 'utah', 'for', 'warrantless', 'access', 'to', 'americans', 'phone', 'calls', 'emails', 'and', 'internet', 'activity', 'use', 'of', 'surveillance', 'to', 'target', 'political', 'dissenters', 'coordination', 'between', 'wall', 'street', 'and', 'homeland', 'security', 'surveillance', 'to', 'monitor', 'and', 'brutally', 'suppress', 'the', 'occupy', 'movement', 'militarized', 'response', 'to', 'peaceful', 'protesters', 'support', 'of', 'legal', 'immunity', 'for', 'telecoms', 'warrantless', 'wiretapping', 'fighting', 'all', 'the', 'way', 'to', 'the', 'supreme', 'court', 'for', 'warrantless', 'surveillance', 'fighting', 'all', 'the', 'way', 'to', 'the', 'supreme', 'court', 'for', 'strip', 'searches', 'for', 'any', 'arrestee', 'dea', 'use', 'of', 'nsa', 'spy', 'data', 'to', 'arrest', 'americans', 'training', 'of', 'police', 'to', 'create', 'false', 'evidence', 'trails', 'to', 'hide', 'use', 'of', 'nsa', 'data', 'continued', 'bushs', 'secret', 'laws', 'and', 'courts', 'use', 'of', 'terrorism', 'laws', 'against', 'political', 'protesters', 'fbi', 'labeling', 'peace', 'activists', 'as', 'potential', 'terrorists', 'support', 'of', 'legislation', 'to', 'specifically', 'legalize', 'mass', 'surveillance', 'of', 'americans', 'supporting', 'and', 'signing', 'internet', 'censoring', 'and', 'privacy', 'violating', 'measures', 'like', 'acta', 'support', 'for', 'tsa', 'groping', 'and', 'naked', 'scanning', 'of', 'americans', 'seeking', 'to', 'travel', 'proliferation', 'of', 'military', 'drones', 'in', 'us', 'skies', 'aggressively', 'militarized', 'police', 'across', 'the', 'nation', 'with', 'federal', 'grants', 'and', 'incentives', 'huge', 'budget', 'increases', 'for', 'private', 'prisons', 'massive', 'expansion', 'of', 'private', 'prisons', 'even', 'hiring', 'private', 'prison', 'executive', 'as', 'head', 'of', 'the', 'us', 'marshalls', 'service', 'corporate', 'media', 'widely', 'reported', 'his', 'plan', 'to', 'phase', 'out', 'private', 'prisons', 'near', 'the', 'end', 'of', 'his', 'presidency', 'but', 'most', 'people', 'are', 'unaware', 'of', 'his', 'role', 'in', 'massively', 'growing', 'the', 'system', 'in', 'the', 'first', 'place', 'contracts', 'ensured', 'the', 'filling', 'of', 'prison', 'beds', 'to', 'maintain', 'profits', 'and', 'he', 'filled', 'those', 'beds', 'with', 'immigrants', 'aggressive', 'persecution', 'of', 'whistleblowers', 'and', 'journalists', 'abuse', 'of', 'the', 'espionage', 'act', 'to', 'charge', 'more', 'people', 'than', 'in', 'its', 'entire', 'history', 'his', 'administration', 'changed', 'the', 'conditions', 'for', 'using', 'the', 'law', 'in', 'order', 'to', 'make', 'charges', 'easier', 'our', 'national', 'ranking', 'on', 'press', 'freedom', 'plummeted', 'during', 'his', 'administration', 'whitewashed', 'torture', 'we', 'tortured', 'some', 'folks', 'and', 'even', 'when', 'public', 'pressure', 'forced', 'ban', 'on', 'torture', 'in', 'our', 'prisons', 'he', 'continued', 'to', 'allow', 'it', 'at', 'black', 'sites', 'added', 'an', 'amendment', 'to', 'the', 'ndaa', 'to', 'end', 'ban', 'on', 'targeting', 'propaganda', 'at', 'americans', 'sought', 'legal', 'authority', 'to', 'lie', 'in', 'response', 'to', 'freedom', 'of', 'information', 'requests', 'snowdens', 'leaks', 'show', 'massive', 'campaign', 'within', 'his', 'government', 'to', 'manipulate', 'americans', 'control', 'discourse', 'and', 'smear', 'dissenters', 'through', 'relentless', 'online', 'astroturfing', 'and', 'propaganda', 'assaults']
In [12]:
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])
['die go bankruptcy aca healthcare pharma ceo profit would like word too obamas abandonment public option campaign be just issue advance expand bush most malignant economic war police state policy board see list corporate money flow party now be why corporate agenda keep advance no matter party be power be propagandize fight red team blue team will never unite policy hurt as long close eye guy side do corporate elite can always be sure have electorate circle wagon malignant policy no matter be power s why so important now be honest s happen unite policy instead team obama biden legacy fill cabinet wall street monsanto rep accord citibank recommendation bailout settlement corrupt bank personal pressure obama attorney general approve delay prosecution statute limitation be expire fail prosecute mortgage abuser passionate speech press conference promote austerity american eat pea extend bush tax cut billionaire then make much permanent support payroll tax holiday tie ss general fund support vicious chain cpi cut social security benefit disabled fail veto bipartisan vote congress gut more financial regulation social security medicare medicaid offer immediately bargaining chip budget negotiation mention cut corporate welfare military budget administration oversee huge increase wealth inequality wealth minority particular collapse wealth gap white non white skyrocketed advocacy multiple new free trade agreement include tran pacific otherwise know nafta steroid support drill pipeline sell portion gulf mexico corporate education policy include high stake corporate testing closure public school entrenchment exorbitant profit health insurance company healthcare mandate legal assault union right hundred thousand federal worker sign cut food stamp most desperate american marijuana user medical marijuana clinic assault drone campaign multiple country be not war expand bombing country militarize africa carry multiple disastrous illegal regime change result massive slaughter dispossession refugee crise terrorism open slavery renew public advocacy concept preemptive war kill list claiming right assassinate even american citizen trial drone american teenager administration authorize double tap war crime drone bomb once then again several minute later when distraught family member first responder have arrive scene military also acknowledge target child bomb when be suspect terrorist be use child shield continue bushs signature strike target bomb not know terrorist general area terrorist activity be suspect occur conceal civilian death count retroactively declare murder man certain age range area have be enemy combatant preside more drone attack first year presidency bush do entire presidency signing ndaa allow indefinite detention new massive spy center utah warrantless access american phone call email internet activity use surveillance target political dissenter coordination wall street homeland security surveillance monitor brutally suppress occupy movement militarize response peaceful protester support legal immunity telecom warrantless wiretapping fight way supreme court warrantless surveillance fight way supreme court strip search arrestee use nsa spy datum arrest american training police create false evidence trail hide use nsa datum continue bush secret law court use terrorism law political protester fbi labeling peace activist potential terrorist support legislation specifically legalize mass surveillance american support sign internet censoring privacy violate measure acta support tsa grope naked scanning american seek travel proliferation military drone sky aggressively militarize police nation federal grant incentive huge budget increase private prison massive expansion private prison even hire private prison executive head marshall service corporate medium widely report plan phase private prison end presidency most people be unaware role massively grow system first place contract ensure filling prison bed maintain profit fill bed immigrant aggressive persecution whistleblower journalist abuse espionage act charge more people entire history administration change condition use law order make charge easy national ranking press freedom plummet administration whitewash torture torture folk even when public pressure force ban torture prison continue allow black site add amendment ndaa end ban target propaganda american seek legal authority lie response freedom information request snowden leak show massive campaign government manipulate american control discourse smear dissenter relentless online astroturfing propaganda assault', 's make worried have live all world nigeria botswana kenya south africa spain uk spend significant amount time birmingham manchester plymouth bristol suggest visit time can experience good healthcare system world can think case be nhs fail partner friend be relatively small issue have see sicko say france sweden could understand have not live be either sound good try experience healthcare uk least read brit actually think take so long see anyone time seem just want get have more people see']
In [0]:
def removeStops(comment):
    word_list = []
    for word in comment.split():
        if word not in common_stops:
            word_list.append(word)
    
    return ' '.join(word_list)

common_stops = ['people', 'would', 'like', 'make', 'work', 'good', 'think', 'want', 'right', 'well', 'thing', 'year', 'take', 'also', 
                'know', 'time', 'come', 'every', 'life', 'look', 'have', 'will', 'tell', 'believe', 'talk', 'seem', 'since', 'show', 
                'else', '//www.reddit.com', 'gt', '`', '-', '“', '-', '/', 'amp', '•', 'subreddit', '’', '\'s',"\'\'", "\'", '”', 'argument',
                'amitheasshole', 'https', 'http', 'just', 'say','www', 'use', 'com', 'would', 'like','make','think','even','also','right','thing','year','well','much','take','know','time','many','still','come','look','really','have','fuck','will']

df = pd.DataFrame(data_lemmatized, columns=['Comments'])
df['Filter comments'] = df['Comments'].map(removeStops)
data_lemmatized = list(df['Filter comments'])
In [0]:
# Converting the data to vectorized form

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)
In [15]:
# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
Sparsicity:  1.166374282433984 %
LDA model with sklearn
In [16]:
# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=5,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1)               # Use all available CPUs

lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)  # Model attributes
LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.7,
                          learning_method='online', learning_offset=10.0,
                          max_doc_update_iter=100, max_iter=10,
                          mean_change_tol=0.001, n_components=5, n_jobs=-1,
                          perp_tol=0.1, random_state=100, topic_word_prior=None,
                          total_samples=1000000.0, verbose=0)
In [17]:
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))
Log Likelihood:  -11030605.620649353
Perplexity:  2076.8030762676676
In [18]:
# Use Grid Search to find the bets parms

# Define Search Param
search_params = {'n_components': [3, 4, 5, 7, 10, 15], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)
/usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
Out[18]:
GridSearchCV(cv='warn', error_score='raise-deprecating',
             estimator=LatentDirichletAllocation(batch_size=128,
                                                 doc_topic_prior=None,
                                                 evaluate_every=-1,
                                                 learning_decay=0.7,
                                                 learning_method='batch',
                                                 learning_offset=10.0,
                                                 max_doc_update_iter=100,
                                                 max_iter=10,
                                                 mean_change_tol=0.001,
                                                 n_components=10, n_jobs=None,
                                                 perp_tol=0.1,
                                                 random_state=None,
                                                 topic_word_prior=None,
                                                 total_samples=1000000.0,
                                                 verbose=0),
             iid='warn', n_jobs=None,
             param_grid={'learning_decay': [0.5, 0.7, 0.9],
                         'n_components': [3, 4, 5, 7, 10, 15]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
In [19]:
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
Best Model's Params:  {'learning_decay': 0.5, 'n_components': 3}
Best Log Likelihood Score:  -3763610.198215896
Model Perplexity:  2172.8076671559843
In [20]:
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics
Out[20]:
Topic0	Topic1	Topic2	dominant_topic
Doc0	1	0	0	0
Doc1	0.01	0.98	0.01	1
Doc2	0.07	0.92	0.01	1
Doc3	0.28	0.71	0.01	1
Doc4	0.02	0.96	0.02	1
Doc5	0.32	0.67	0.02	1
Doc6	0.13	0.87	0	1
Doc7	0.28	0.14	0.58	2
Doc8	0.99	0.01	0	0
Doc9	0.05	0.94	0.01	1
Doc10	0.02	0.98	0	1
Doc11	0.39	0.34	0.27	0
Doc12	0	0.43	0.56	2
Doc13	0.88	0.12	0	0
Doc14	0.33	0.66	0.01	1
In [21]:
# Topic Distribution
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution
Out[21]:
Topic Num	Num Documents
0	1	7876
1	0	1336
2	2	788
In [22]:
# Visualize LDA model
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
panel
/usr/local/lib/python3.6/dist-packages/pyLDAvis/_prepare.py:257: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.

  return pd.concat([default_term_info] + list(topic_dfs))
Out[22]:
In [0]:

Second Model
In [0]:
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))
Log Likelihood:  -6258203.3490843475
Perplexity:  1481.0707321179195
In [0]:
# Use Grid Search to find the bets parms

# Define Search Param
search_params = {'n_components': [3,4, 5], 'learning_decay': [.5, .7]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)
/usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
  warnings.warn(CV_WARNING, FutureWarning)
Out[0]:
GridSearchCV(cv='warn', error_score='raise-deprecating',
             estimator=LatentDirichletAllocation(batch_size=128,
                                                 doc_topic_prior=None,
                                                 evaluate_every=-1,
                                                 learning_decay=0.7,
                                                 learning_method='batch',
                                                 learning_offset=10.0,
                                                 max_doc_update_iter=100,
                                                 max_iter=10,
                                                 mean_change_tol=0.001,
                                                 n_components=10, n_jobs=None,
                                                 perp_tol=0.1,
                                                 random_state=None,
                                                 topic_word_prior=None,
                                                 total_samples=1000000.0,
                                                 verbose=0),
             iid='warn', n_jobs=None,
             param_grid={'learning_decay': [0.5, 0.7],
                         'n_components': [3, 4, 5]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
In [0]:
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
Best Model's Params:  {'learning_decay': 0.5, 'n_components': 4}
Best Log Likelihood Score:  -2137524.1324504977
Model Perplexity:  1506.3585041334263
In [0]:
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics
Out[0]:
Topic0	Topic1	Topic2	Topic3	dominant_topic
Doc0	0	0.75	0.06	0.19	1
Doc1	0.07	0.46	0.15	0.32	1
Doc2	0.01	0.97	0.01	0.01	1
Doc3	0.01	0.62	0.01	0.36	1
Doc4	0.01	0.48	0.01	0.51	3
Doc5	0.01	0.67	0.01	0.31	1
Doc6	0.19	0.26	0.01	0.54	3
Doc7	0.01	0.77	0.03	0.19	1
Doc8	0.01	0.6	0.01	0.38	1
Doc9	0	0	0.2	0.8	3
Doc10	0.01	0.97	0.01	0.01	1
Doc11	0.04	0.5	0.41	0.04	1
Doc12	0.01	0.01	0.15	0.83	3
Doc13	0.01	0.68	0.01	0.31	1
Doc14	0.01	0.98	0.01	0.01	1
In [0]:
# Topic Distribution
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution
Out[0]:
Topic Num	Num Documents
0	1	4830
1	3	4291
2	2	481
3	0	398
In [0]:
# Topics keywords

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()
Out[0]:
aafp	aap	abandon	abc	abhorrent	abide	ability	abject	able	abolish	abolition	abortion	abroad	absence	absent	absolute	absolutely	absorb	abstract	absurd	absurdly	abundance	abundant	abuse	abused	abuser	abusive	abysmal	aca	academic	academy	accelerate	accent	accept	acceptable	acceptance	access	accessibility	accessible	accident	...	worship	worst	worth	worthless	worthwhile	worthy	wouldn	wound	wrap	wreck	wrestle	write	writer	writing	wrong	wsj	wtf	wwi	wwii	xenophobia	xenophobic	yacht	yahoo	yang	yank	yard	yea	yearly	yell	yeman	yesterday	yield	york	young	youth	youtu	youtube	zealand	zone	zoning
Topic0	0.251012	0.250664	0.250449	0.252307	0.251073	0.251197	0.251341	0.250947	0.251856	0.250907	0.250139	0.252773	0.250821	0.252467	0.251184	0.251277	0.251817	0.251516	0.252287	0.250809	0.250408	0.250431	0.250426	0.251472	0.251100	0.250906	0.251039	0.251594	0.250837	0.251906	0.250716	0.250891	0.250525	0.251357	0.252149	0.251778	0.252248	0.250075	0.254078	0.251338	...	0.250801	0.250440	0.251434	0.251138	0.250678	0.250916	0.250901	0.252126	0.252590	0.252318	0.254685	0.253801	0.251441	0.251710	0.251658	0.254693	0.250995	0.251108	0.250447	0.250421	0.252280	0.250152	0.250003	0.250927	0.250985	0.251069	0.250715	0.250595	0.250347	0.250447	0.252755	0.251195	0.250616	0.251562	0.251085	0.251899	0.251497	0.251153	0.250861	0.252568
Topic1	0.250103	0.250583	2.031890	8.681321	1.922375	1.132516	128.935175	0.258243	574.113933	25.517355	0.255938	0.262649	32.472665	7.739252	6.784007	52.598386	204.617284	20.076349	0.788930	42.681474	22.246774	2.432858	0.256812	33.349526	0.300719	0.252929	10.422213	10.858250	175.832076	20.826454	0.252113	9.513642	0.251591	132.202466	17.338141	5.207180	593.734765	16.131933	41.291471	99.849869	...	0.253621	0.260454	264.172108	8.022397	3.183288	9.189679	87.550791	8.592876	7.744673	14.938549	4.819104	110.864253	1.239885	0.653779	239.290389	0.435360	10.745281	0.252229	0.264624	0.262175	0.251401	10.710126	0.251358	322.208136	0.901813	2.306114	3.436398	63.103177	0.448079	0.253313	7.742976	16.719629	28.544337	64.152852	1.343450	13.669212	6.083934	26.974009	12.855264	7.717437
Topic2	28.248705	41.248529	22.582684	13.798607	0.251857	0.252512	30.593864	0.256192	49.004171	6.658046	3.321678	12.079461	42.243483	1.859913	1.651238	21.935478	14.803740	0.257798	35.000115	0.575070	0.252322	0.251027	0.250623	82.164846	0.251324	5.892200	3.221894	0.406194	130.544015	35.000900	61.244014	3.468856	0.251577	70.353412	20.720136	18.868624	83.953244	2.360995	0.293645	0.269311	...	0.261621	34.228791	35.572029	8.251139	0.258128	0.259009	2.129872	0.262334	0.255148	0.262216	5.335167	70.939294	1.938878	0.444800	11.093027	16.057783	2.699281	23.235123	19.377749	0.258855	0.589253	0.267347	14.248349	111.271850	0.252174	0.295055	0.252559	0.257700	0.274122	30.897563	0.265623	0.267344	101.154035	88.070842	82.515951	36.930642	249.802202	0.255342	12.941366	0.251308
Topic3	0.250180	0.250223	34.134977	0.267765	10.574696	30.363774	202.219620	14.234618	509.630041	63.573691	9.172245	302.405117	47.033032	13.148368	3.313572	122.214858	366.327159	16.414337	14.958668	38.492648	1.250496	20.065684	10.242139	170.234157	11.196857	11.603965	21.104854	18.483962	4.373073	9.920740	0.253157	16.766611	12.246307	238.192765	50.689574	18.672418	487.059742	0.256997	46.160806	29.629483	...	33.233957	0.260315	183.004429	16.475326	10.307906	18.300396	92.068437	28.892663	12.747589	17.546917	0.591044	244.942652	19.569796	22.649710	592.364926	0.252164	24.304442	0.261540	18.107180	19.228550	11.907067	7.772375	0.250290	22.269087	10.595028	15.147762	15.060329	5.388527	27.027452	5.598677	27.738646	5.761832	17.051012	305.524744	22.889515	22.148247	17.862367	14.519495	40.952510	4.778687
4 rows × 6143 columns

In [0]:
# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
Out[0]:
Word 0	Word 1	Word 2	Word 3	Word 4	Word 5	Word 6	Word 7	Word 8	Word 9	Word 10	Word 11	Word 12	Word 13	Word 14	Word 15	Word 16	Word 17	Word 18	Word 19
Topic 0	post	message	suicide	mention	question	issue	suicidal	reddit	story	healthcare	care	emergency	compose	danger	belong	hotline	suicidewatch	comment	link	info
Topic 1	healthcare	pay	cost	insurance	health	care	country	government	money	company	need	high	tax	doctor	private	taxis	hospital	medical	price	free
Topic 2	trump	bernie	org	news	policy	state	republican	vote	military	obama	health	american	sander	candidate	support	president	democrat	medical	warren	article
Topic 3	healthcare	country	need	government	way	world	mean	really	live	free	point	bad	try	state	issue	problem	lot	help	change	american
In [0]:
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
panel
/usr/local/lib/python3.6/dist-packages/pyLDAvis/_prepare.py:257: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.

  return pd.concat([default_term_info] + list(topic_dfs))
