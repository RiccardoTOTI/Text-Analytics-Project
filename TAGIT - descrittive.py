# =============================================================================
# Carico i file
# =============================================================================
train_file = r'C:\Users\giann\Desktop\Laurea_Magistrale\Text_Analytitcs\Progetto\Dati\training.txt'
train_url=r'C:\Users\giann\Desktop\Laurea_Magistrale\Text_Analytitcs\Progetto\Dati\training1.csv'
test_file = r'C:\Users\giann\Desktop\Laurea_Magistrale\Text_Analytitcs\Progetto\Dati\test.txt'
test_url = r'C:\Users\giann\Desktop\Laurea_Magistrale\Text_Analytitcs\Progetto\Dati\test1.csv'
delimiter = ','

import pandas as pd
df = pd.read_csv(r'C:\Users\giann\Desktop\Laurea_Magistrale\Text_Analytitcs\Progetto\Dati\training1.csv')
df

import csv
x_train = list()
y_train = list()
with open(train_url, encoding='utf-8', newline='') as infile:
    reader = csv.reader(infile, delimiter=delimiter)
    for row in reader:
        x_train.append(row[5])
        y_train.append(row[4])

x_test = list()
y_test = list()
with open(test_url, encoding='utf-8', newline='') as infile:
    reader = csv.reader(infile, delimiter=delimiter)
    for row in reader:
        x_test.append(row[5])
        y_test.append(row[4])
        
len(x_train),len(y_train),len(x_test),len(y_test)
y_train.pop(0) # elimino il primo elemento che è la parola "gender"
y_test.pop(0) # elimino il primo elemento che è la parola "gender"
set(y_train)
sample_idx = 10

x_train.pop(0) # elimino il primo elemento che è la parola "post"
x_test.pop(0) # elimino il primo elemento che è la parola "post"
x_train[sample_idx]
y_train[sample_idx]

print("Percentuale di maschi nel training set:", round(y_train.count("M")/len(y_train),3))
print("Percentuale di femmine nel training set:", round(y_train.count("F")/len(y_train),3))
print("Percentuale di maschi nel test set:", round(y_test.count("M")/len(y_test),3))
print("Percentuale di femmine nel test set:", round(y_test.count("F")/len(y_test),3))

import string
import re

# Elimino la parola "post"
regex = re.compile("\\bpost\\b")
for i in range(0, len(x_train)):
    x_train[i] = regex.sub('', x_train[i])
for i in range(0, len(x_test)):
    x_test[i] = regex.sub('', x_test[i])
# Elimino la punteggiatura
regex = re.compile('[%s]' % re.escape(string.punctuation))
for i in range(0, len(x_train)):
    x_train[i] = regex.sub('', x_train[i])
for i in range(0, len(x_test)):
    x_test[i] = regex.sub('', x_test[i])
# Elimino i numeri
regex = re.compile("[0-9]+")
for i in range(0, len(x_train)):
    x_train[i] = regex.sub('', x_train[i])
for i in range(0, len(x_test)):
    x_test[i] = regex.sub('', x_test[i])
# Rendo minuscola la prima parola dopo il punto
# for i in range(0, len(x_train)):
    # x_train[i] = re.sub('(?<=\.\s)(\w+)', lambda m: m.group().lower(), x_train[i])
# for i in range(0, len(x_test)):
    # x_test[i] = re.sub('(?<=\.\s)(\w+)', lambda m: m.group().lower(), x_test[i])
# Rendo minuscola la prima parola
# for i in range(0, len(x_train)):
    # x_train[i] = x_train[i][1].lower() + x_train[i][2:] # il primo carattere di ogni post è \n, quindi parto da 1, non da 0
# for i in range(0, len(x_test)):
    # x_test[i] = x_test[i][1].lower() + x_test[i][2:] # il primo carattere di ogni post è \n, quindi parto da 1, non da 0

# =============================================================================
# Funzioni
# =============================================================================
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

stopword_list = stopwords.words('italian')

from collections import defaultdict
tag_map = defaultdict(lambda : wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV
lemmatizer = WordNetLemmatizer()

doc_counter = 0
def reset_counter():
    global doc_counter
    doc_counter = 0

def increase_counter():
    global doc_counter
    doc_counter += 1
    if doc_counter % 100 == 0:
        print(doc_counter)

def nltk_ngram_tokenizer(text):
    increase_counter()

    # tokens, skipping stopwords
    tokens = [token for token in word_tokenize(text) if token not in stopword_list]

    # we use a simple nltk function to create ngrams
    bigrams = ['BI_'+w1+'_'+w2 for w1,w2 in nltk.ngrams(tokens,2)]
    trigrams = ['TRI_'+p1+'_'+p2+'_'+p3 for p1,p2,p3 in nltk.ngrams(tokens,3)]

    all_tokens = list()
    all_tokens.extend(tokens)
    all_tokens.extend(bigrams)
    all_tokens.extend(trigrams)
    return all_tokens

def nltk_nlp_tokenizer(text):
    increase_counter()

    # tokens, skipping stopwords
    tokens = [token for token in word_tokenize(text) if token not in stopword_list]

    # lemmatized tokens
    lemmas = list()
    for token, tag in pos_tag(tokens):
  	    lemmas.append('LEMMA_'+lemmatizer.lemmatize(token, tag_map[tag[0]]))

    # we use a simple nltk function to create ngrams
    lemma_bigrams = ['BI_'+p1+'_'+p2 for p1,p2 in nltk.ngrams(lemmas,2)]
    lemma_trigrams = ['TRI_'+p1+'_'+p2+'_'+p3 for p1,p2,p3 in nltk.ngrams(lemmas,3)]

    all_tokens = list()
    all_tokens.extend(lemmas)
    all_tokens.extend(lemma_bigrams)
    all_tokens.extend(lemma_trigrams)
    return all_tokens

import spacy
import re
nlp = spacy.load('it_core_news_sm')

def spacy_nlp_tokenizer(text):
    increase_counter()

    # substituting all space characters with a single space
    text = re.sub('\s+', ' ', text)

    # we use spacy for main nlp tasks
    doc = nlp(text)
    # lemmatized tokens, skipping stopwords
    lemmas = ['LEMMA_'+token.lemma_ for token in doc if not token.is_stop]
    # entity_types
    entity_types = ['NER_'+token.ent_type_ for token in doc if token.ent_type_]

    # in case an entity linker is available, we can use it do put actual entities as
    # features, e.g. Queen Elizabeth, Elizabeth II, Her Majesty -> KB2912
    # see https://spacy.io/usage/training#entity-linker
    # entities = ['ENT_'+token.ent_kb_id_ for token in doc if token.ent_kb_id_]

    # we use a simple nltk function to create ngrams
    lemma_bigrams = ['BI_'+p1+'_'+p2 for p1,p2 in nltk.ngrams(lemmas,2)]
    lemma_trigrams = ['TRI_'+p1+'_'+p2+'_'+p3 for p1,p2,p3 in nltk.ngrams(lemmas,3)]

    all_tokens = list()
    all_tokens.extend(lemmas)
    all_tokens.extend(lemma_bigrams)
    all_tokens.extend(lemma_trigrams)
    all_tokens.extend(entity_types)
    return all_tokens

#### ===============================================================================
#### ===============================================================================

#### FREQUENZA PAROLE IN ALTRO MODO
#nltk.download('punkt')
from collections import defaultdict
frequenze_tot = {}
frequenze_tot = defaultdict(int)

for i in range(0, len(x_train)):
    tokenized_text = nltk.word_tokenize(x_train[i].lower(), language = "italian")
    tokenized_sentences = list()
    for sent in nltk.sent_tokenize(x_train[i]):
        tokenized_sentences.append(nltk.word_tokenize(sent, language = "italian"))
    freq_dist = nltk.FreqDist(tokenized_text)
    for key in freq_dist:
        frequenze_tot[key] += freq_dist[key]

# Elimino le stopwords
frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if key not in stopword_list}
import collections
import matplotlib.pyplot as plt
f, ax = plt.subplots()
frequenze_tot = collections.Counter(frequenze_tot)

toplot = zip(*frequenze_tot.most_common(35))
toplot = list(toplot)
toplot[0] = toplot[0][::-1]
toplot[1] = toplot[1][::-1]
plt.figure(figsize = (9,7))
plt.barh(y = toplot[0], width = toplot[1], height=.5, color='b')
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 
plt.title("Parole più frequenti per tutto il training set")

## Grafico parole più frequenti per sesso
dati = pd.DataFrame({"post": x_train, "gender": y_train})
datiF = dati.loc[df['gender'] == "F"]
datiM = dati.loc[df['gender'] == "M"]

## FEMMINE
frequenze_tot = {}
frequenze_tot = defaultdict(int)
for i in datiF["post"]:
    tokenized_text = nltk.word_tokenize(i.lower(), language = "italian")
    tokenized_sentences = list()
    for sent in nltk.sent_tokenize(i):
        tokenized_sentences.append(nltk.word_tokenize(sent, language = "italian"))
    freq_dist = nltk.FreqDist(tokenized_text)
    for key in freq_dist:
        frequenze_tot[key] += freq_dist[key]

# Elimino le stopwords
frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if key not in stopword_list}
frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if len(key) != 1}
frequenze_tot = collections.Counter(frequenze_tot)

toplot = zip(*frequenze_tot.most_common(35))
toplot = list(toplot)
toplot[0] = toplot[0][::-1]
toplot[1] = toplot[1][::-1]
plt.figure(figsize = (9,7))
plt.barh(y = toplot[0], width = toplot[1], height=.5, color='teal')
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 
plt.title("Parole più frequenti per le femmine")

## MASCHI
frequenze_tot = {}
frequenze_tot = defaultdict(int)
for i in datiM["post"]:
    tokenized_text = nltk.word_tokenize(i.lower(), language = "italian")
    tokenized_sentences = list()
    for sent in nltk.sent_tokenize(i):
        tokenized_sentences.append(nltk.word_tokenize(sent, language = "italian"))
    freq_dist = nltk.FreqDist(tokenized_text)
    for key in freq_dist:
        frequenze_tot[key] += freq_dist[key]

# Elimino le stopwords
frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if key not in stopword_list}
frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if len(key) != 1}
frequenze_tot = collections.Counter(frequenze_tot)
toplot = zip(*frequenze_tot.most_common(35))
toplot = list(toplot)
toplot[0] = toplot[0][::-1]
toplot[1] = toplot[1][::-1]
plt.figure(figsize = (9,7))
plt.barh(y = toplot[0], width = toplot[1], height=.5, color='sandybrown')
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.title("Parole più frequenti per i maschi")

# Funzione per calcolare le parole più frequenti in un subset dei nostri dati
def freqtot(dati):
    frequenze_tot = {}
    frequenze_tot = defaultdict(int)
    for i in dati["post"]:
        tokenized_text = nltk.word_tokenize(i.lower(), language = "italian")
        tokenized_sentences = list()
        for sent in nltk.sent_tokenize(i):
            tokenized_sentences.append(nltk.word_tokenize(sent, language = "italian"))
        freq_dist = nltk.FreqDist(tokenized_text)
        for key in freq_dist:
            frequenze_tot[key] += freq_dist[key]
    return frequenze_tot

## Grafico parole più frequenti per age
dati = pd.DataFrame({"post": x_train, "age": df["age"]})
print(dati["age"].unique())
# dati_sub = dati.loc[dati['age'] == "0-19"]

eta = dati["age"].unique()
fig = plt.figure(figsize=(8, 8)) # per il plot
columns = 2
rows = 3
colore = ["firebrick", "darkmagenta", "lightsalmon", "springgreen", "slateblue"]
for i in range(0, len(eta)):
# Elimino le stopwords
    frequenze_tot = freqtot(dati.loc[dati["age"] == eta[i]])
    frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if key not in stopword_list}
    frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if len(key) != 1}
    frequenze_tot = collections.Counter(frequenze_tot)
# for i in range(1, columns*rows +1):
    toplot = zip(*frequenze_tot.most_common(35))
    toplot = list(toplot)
    toplot[0] = toplot[0][::-1]
    toplot[1] = toplot[1][::-1]
    plt.figure(figsize = (9,7))
    img = plt.barh(y = toplot[0], width = toplot[1], height=.5, color=colore[i])
    if eta[i] != "50-100":
        plt.title("Parole più frequenti per il gruppo di età " + eta[i])
        plt.rc('xtick', labelsize=13)
        plt.rc('ytick', labelsize=13)
        plt.show()
    else:
        plt.title("Parole più frequenti per il gruppo di età 50+")
        plt.rc('xtick', labelsize=13)
        plt.rc('ytick', labelsize=13)
        plt.show()
    # fig.add_subplot(rows, columns, i)
    # plt.imshow(img)
# plt.show()

## Grafico parole più frequenti per topic
dati = pd.DataFrame({"post": x_train, "topic": df["topic"]})
print(dati["topic"].unique())

topic = dati["topic"].unique()
fig = plt.figure(figsize=(8, 8)) # per il plot
columns = 2
rows = 3
colore = ["firebrick", "darkmagenta", "greenyellow", "springgreen", "slateblue",
          "salmon", "goldenrod", "lightslategrey", "indigo", "aqua", "hotpink"]
for i in range(0, len(topic)):
# Elimino le stopwords
    frequenze_tot = freqtot(dati.loc[dati["topic"] == topic[i]])
    frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if key not in stopword_list}
    frequenze_tot = {key: frequenze_tot[key] for key in frequenze_tot.keys() if len(key) != 1}
    frequenze_tot = collections.Counter(frequenze_tot)
# for i in range(1, columns*rows +1):
    toplot = zip(*frequenze_tot.most_common(35))
    toplot = list(toplot)
    toplot[0] = toplot[0][::-1]
    toplot[1] = toplot[1][::-1]
    plt.figure(figsize = (9,7))
    img = plt.barh(y = toplot[0], width = toplot[1], height=.5, color=colore[i])
    plt.title("Parole più frequenti per il topic " + topic[i])
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.show()
    # fig.add_subplot(rows, columns, i)
    # plt.imshow(img)
# plt.show()


