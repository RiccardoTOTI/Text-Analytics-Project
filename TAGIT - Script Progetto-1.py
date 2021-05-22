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

# =============================================================================
# Campiono i dati per rimediare allo sbilanciamento di questi
# =============================================================================
df_train = pd.DataFrame(data = {"x": x_train, "y": y_train})
df_test = pd.DataFrame(data = {"x": x_test, "y": y_test})

# Training Set
df_train = df_train.groupby('y', group_keys=False).apply(lambda x: x.sample(min(len(x), 125)))
df_train["y"].tolist().count("M")

x_train = df_train["x"].tolist()
y_train = df_train["y"].tolist()

# =============================================================================
# Operazioni sul testo
# =============================================================================
# Rendo minuscola la prima parola dopo il punto
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

### EXTRA: NON IMPORTANTE
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
###

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

# Rendo la risposta binaria
import numpy as np
# y_train[sample_idx] è una delle due classi, nel nostro caso M. Quando vado a fare y_train == y_train[sample_idx], in pratica
# metto TRUE per i maschi (M) e FALSE per le femmine (F).
y_train_bin = np.asarray(y_train)==y_train[sample_idx]
y_test_bin = np.asarray(y_test)==y_train[sample_idx]
y_train_bin,y_test_bin

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
# vect = CountVectorizer(analyzer=nltk_ngram_tokenizer, min_df=5)  # tokenization and frequency count
# vect = CountVectorizer(analyzer=nltk_n_gram_tokenizer)  
# vect = CountVectorizer(analyzer=nltk_nlp_tokenizer, min_df=5)  
vect = CountVectorizer(analyzer = nltk_nlp_tokenizer, min_df=5) # Passiamo la funzione spacy_nlp_tokenizer per ottenere sia parole singole che n-grams
reset_counter()
X_train_tok = vect.fit_transform(x_train)
reset_counter()
X_test_tok = vect.transform(x_test)

len(vect.vocabulary_)
X_train_tok[:5]
print(X_train_tok[:5])
vect.inverse_transform(X_train_tok[:5])
for feat,freq in zip(vect.inverse_transform(X_train_tok[:5])[1],X_train_tok[:5].data):
  print(feat,freq)

# FEATURE SELECTION
bin_sel = SelectKBest(chi2, k=5000)
bin_sel.fit(X_train_tok,y_train_bin)
X_train_sel_bin = bin_sel.transform(X_train_tok)
X_test_sel_bin = bin_sel.transform(X_test_tok)

bin_sel.get_support()
X_train_sel_bin
print(X_train_sel_bin[:5])
print(vect.inverse_transform(bin_sel.inverse_transform(X_train_sel_bin[:5])))

# PESI CON TF-IDF
tfidf = TfidfTransformer()
tfidf.fit(X_train_sel_bin)
X_train_vec_bin = tfidf.transform(X_train_sel_bin)
X_test_vec_bin =tfidf.transform(X_test_sel_bin)

print(X_train_vec_bin[:5])
for feat,weight,freq in zip(vect.inverse_transform(bin_sel.inverse_transform(X_train_vec_bin[:5]))[1],X_train_vec_bin[:5].data,X_train_sel_bin[:5].data):
  print(feat,weight,freq)

svm_bin = LinearSVC()  # linear svm with default parameters
svm_bin_clf = svm_bin.fit(X_train_vec_bin,y_train_bin)
bin_predictions = svm_bin_clf.predict(X_test_vec_bin)
len(bin_predictions)
bin_predictions
# ACCURACY
correct = 0
for prediction,true_label in zip(bin_predictions, y_test_bin):
    if prediction==true_label:
        correct += 1
print(correct/len(bin_predictions))

# =============================================================================
# Support Vector Machine ma con la Pipeline
# =============================================================================
bin_pipeline = Pipeline([
    ('sel', SelectKBest(chi2, k=850)),  # feature selection
    ('tfidf', TfidfTransformer()),  # weighting
    ('learner', LinearSVC())  # learning algorithm
])

reset_counter()
bin_pipeline.fit(X_train_tok,y_train_bin)

# ACCURACY
reset_counter()
bin_predictions = bin_pipeline.predict(X_test_tok)
correct = 0
for prediction,true_label in zip(bin_predictions, y_test_bin):
    if prediction==true_label:
        correct += 1
print(correct/len(bin_predictions))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
print('Classification report:')
print(classification_report(y_test_bin, bin_predictions))
print('Confusion matrix:')
cm = confusion_matrix(y_test_bin, bin_predictions)
print(cm)

tokenizer = vect
selector = bin_pipeline.named_steps['sel']
classifier = bin_pipeline.named_steps['learner']

feature_names = tokenizer.get_feature_names()
feats_w_score = list()
for index,(selected,score) in enumerate(zip(selector.get_support(),selector.scores_)):
    feats_w_score.append((score,selected,feature_names[index]))
feats_w_score = sorted(feats_w_score)
len(feats_w_score)

feats_w_score[:100],feats_w_score[-100:]
feats_w_classifier_weight = list()
for index,weight in enumerate(selector.inverse_transform(classifier.coef_)[0]):
    if weight!=0:
        feats_w_classifier_weight.append((weight,feature_names[index]))
feats_w_classifier_weight = sorted(feats_w_classifier_weight)
len(feats_w_classifier_weight)

feats_w_classifier_weight[-100:]
feats_w_classifier_weight[:100]

# =============================================================================
# Decision Tree
# =============================================================================
dt_bin_pipeline = Pipeline([
    ('sel', SelectKBest(chi2, k=850)),  # feature selection
    ('tfidf', TfidfTransformer()),  # weighting
    ('learner', DecisionTreeClassifier())  # learning algorithm
])

dt_bin_pipeline.fit(X_train_tok,y_train_bin)
bin_predictions = dt_bin_pipeline.predict(X_test_tok)

print('Classification report:')
print(classification_report(y_test_bin, bin_predictions))
print('Confusion matrix:')
cm = confusion_matrix(y_test_bin, bin_predictions)
print(cm)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(24, 24))
plot_tree(dt_bin_pipeline.named_steps['learner'])
plt.show()

# =============================================================================
# Multinomial NB
# =============================================================================
nb_bin_pipeline = Pipeline([
    ('sel', SelectKBest(chi2, k=850)),  # feature selection
    ('learner', MultinomialNB())  # learning algorithm
])

nb_bin_pipeline.fit(X_train_tok,y_train_bin)
bin_predictions = nb_bin_pipeline.predict(X_test_tok)

print('Classification report:')
print(classification_report(y_test_bin, bin_predictions))
print('Confusion matrix:')
cm = confusion_matrix(y_test_bin, bin_predictions)
print(cm)


tokenizer = vect
selector = nb_bin_pipeline.named_steps['sel']
classifier = nb_bin_pipeline.named_steps['learner']
classifier.class_log_prior_,classifier.feature_log_prob_, len(classifier.feature_log_prob_[0])
ratio = classifier.feature_log_prob_[0]/classifier.feature_log_prob_[1]

feats_w_classifier_weight = list()
feature_names = tokenizer.get_feature_names()
for index,weight in enumerate(selector.inverse_transform([ratio])[0]):
    if weight!=0:
        feats_w_classifier_weight.append((weight,feature_names[index]))
feats_w_classifier_weight = sorted(feats_w_classifier_weight)
len(feats_w_classifier_weight)

feats_w_classifier_weight[-100::-1]
feats_w_classifier_weight[:100]

# =============================================================================
# Linear SVC
# =============================================================================
pipeline = Pipeline([
    ('sel', SelectKBest(chi2, k=5000)),  # feature selection
    ('tfidf', TfidfTransformer()),  # weighting
    ('learner', LinearSVC())  # learning algorithm
])

classifier = pipeline.fit(X_train_tok,y_train)
predictions = classifier.predict(X_test_tok)
correct = 0
for prediction,true_label in zip(predictions, y_test):
    if prediction==true_label:
        correct += 1
print(correct/len(predictions))

from sklearn.metrics import confusion_matrix, classification_report
print('Classification report:')
print(classification_report(y_test, predictions))
print('Confusion matrix:')
cm = confusion_matrix(y_test, predictions)
print(cm)

# =============================================================================
# One Vs One
# =============================================================================
from sklearn.multiclass import OneVsOneClassifier

pipeline = Pipeline([
    ('sel', SelectKBest(chi2, k=850)),  # feature selection
    ('tfidf', TfidfTransformer()),  # weighting
    ('learner', OneVsOneClassifier(LinearSVC()))  # learning algorithm
])

classifier = pipeline.fit(X_train_tok,y_train)
predictions = classifier.predict(X_test_tok)

from sklearn.metrics import confusion_matrix, classification_report
print('Classification report:')
print(classification_report(y_test, predictions))
print('Confusion matrix:')
cm = confusion_matrix(y_test, predictions)
print(cm)

