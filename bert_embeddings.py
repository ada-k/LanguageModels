from flair.data import Sentence
from flair.embeddings import BertEmbeddings

s = Sentence('the dummy sentence to be embedded')

bert_embedding = BertEmbeddings()
bert_embedding.embed(s)

for token in s:
   print(token.embedding)