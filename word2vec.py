import gensim

# class TrainingData(object):
#     def __init__(self, corpus):
#         self.corpus = corpus
 
#     def __iter__(self):
#     	for line in open(self.corpus):
#     		yield line.split()

# train_data = TrainingData("texts_raw_fixed.txt")
# word2vec = gensim.models.Word2Vec(iter=1, size=200)
# word2vec.build_vocab(train_data)
# word2vec.train(train_data, epochs=30,total_examples=26440254)
# word2vec.save('train_word2vec')

model = gensim.models.Word2Vec.load('./word2vec/train_word2vec')
print type(model['i'])