"""count vectorizer"""

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(min_df=5,max_df=.99, ngram_range=(1, 2)) #remove rare and common words with df parameter
#include single and 2 word pairs
X_train_vec = bow.fit_transform(X_train[‘text’])
X_test_vec = bow.transform(X_test[‘text’])
cols = bow.get_feature_names() #if you need feature names
model = RandomForestClassifier(n_estimators=500, n_jobs=8)
model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)


""" TF_IDF- Term Frequency — Inverse Document Frequency"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(min_df=5,max_df=.99, ngram_range=(1, 2)) #remove rare and common words with df parameter
#include single and 2 word pairs
X_train_vec = tfidf.fit_transform(X_train[‘text’])
X_test_vec = tfidf.transform(X_test[‘text’])
cols = tfidf.get_feature_names() #if you need feature names
model = RandomForestClassifier(n_estimators=500, n_jobs=8)
model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)


"""word2vec"""

import gensim
import gensim.models as g
import gensim.downloader
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def vectorize_sentence(sentence,model):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    a = []
    for i in tokenizer(sentence):
        try:
            a.append(model.get_vector(str(i)))
        except:
            pass
        
    a=np.array(a).mean(axis=0)
    a = np.zeros(300) if np.all(a!=a) else a
    return a
word2vec = gensim.downloader.load('word2vec-google-news-300') #1.66 gb
# vectorize the data
X_train_vec = pd.DataFrame(np.vstack(X_train['text'].apply(vectorize_sentence, model=word2vec)))
X_test_vec = pd.DataFrame(np.vstack(X_test['text'].apply(vectorize_sentence, model=word2vec)))
# Word2Vec doesn't have feature names
model = RandomForestClassifier(n_estimators=500, n_jobs=8)
model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)


"""universal sentence encoder"""

import tensorflow_hub as hub
def embed_document(data):
    model = hub.load("../USE/")
    embeddings = np.array([np.array(model([i])) for i in data])
    return pd.DataFrame(np.vstack(embeddings))
# vectorize the data
X_train_vec = embed_document(X_train['text'])
X_test_vec = embed_document(X_test['text'])
# USE doesn't have feature names
model = RandomForestClassifier(n_estimators=500, n_jobs=8)
model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)


"""BERT"""

from sentence_transformers import SentenceTransformer
bert = SentenceTransformer('stsb-roberta-large') #1.3 gb
# vectorize the data
X_train_vec = pd.DataFrame(np.vstack(X_train['text'].apply(bert.encode)))
X_test_vec = pd.DataFrame(np.vstack(X_test['text'].apply(bert.encode)))
# BERT doesn't have feature names
model = RandomForestClassifier(n_estimators=500, n_jobs=8)
model.fit(X_train_vec, y_train)
model.score(X_test_vec, y_test)