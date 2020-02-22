from model import WordEmbedding
from config import config

model = WordEmbedding(config.embedding)

print("Model loaded!")
while True:
	text = input("Input: ")
	text = text.lower()
	if text in ('quit', 'q', 'exit', 'e'):
		break
	text = text.split(' ')
	if len(text) == 2:
		word1, word2 = text
		sim = model.model.wv.similarity(word1, word2)
		print('Similarity between {} and {} is {}'.format(word1, word2, sim))