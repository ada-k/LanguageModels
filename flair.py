import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings

# using forward flair embeddingembedding
forward_flair_embedding= FlairEmbeddings('news-forward-fast')

# input the sentence
s = Sentence('Geeks for Geeks helps me study.')

# embed words in the input sentence
forward_flair_embedding.embed(s)

# print the embedded tokens
for token in s:
	print(token)
	print(token.embedding)


"""stacked embeddings"""
import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, WordEmbeddings
from flair.embeddings import StackedEmbeddings
# flair embeddings
forward_flair_embedding= FlairEmbeddings('news-forward-fast')
backward_flair_embedding= FlairEmbeddings('news-backward-fast')

# glove embedding
GloVe_embedding = WordEmbeddings('glove')

# create a object which combines the two embeddings
stacked_embeddings = StackedEmbeddings([forward_flair_embedding,
										backward_flair_embedding,
										GloVe_embedding,])

# input the sentence
s = Sentence('Geeks for Geeks helps me study.')
										
# embed the input sentence with the stacked embedding
stacked_embeddings.embed(s)

# print the embedded tokens
for token in s:
	print(token)
	print(token.embedding)
