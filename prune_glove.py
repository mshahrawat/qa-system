import pickle
import numpy as np

def glove_generator(glove_path):
	with open(glove_path) as f:
		for line in f:
			line = line.split()
			yield line

android_path = './data/part2/corpus.txt'
ubuntu_path = './data/part1/text_tokenized.txt'
glove_path = './data/part2/glove.840B.300d.txt'
vocab = set()

with open(android_path) as f:
	for line in f:
		line = line.split()
		vocab.update(line[1:])

with open(ubuntu_path) as f:
	for line in f:
		line = line.split()
		vocab.update(line[1:])


glove_gen = glove_generator(glove_path)
vocab_fv = {}

for line in glove_gen:
	if line[0] in vocab:
		vocab_fv[line[0]] = np.asarray(line[1:], dtype='float32')

with open("data/part2/glove_dict", "wb") as f:
    pickle.dump(vocab_fv, f)