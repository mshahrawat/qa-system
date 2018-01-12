import pandas as pd
import numpy as np
import pickle
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from meter import *
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

source_text = './data/part1/raw/text_tokenized.txt'
android_csv = './data/part2/corpus.txt'
android_dev_pos = './data/part2/dev.pos.txt'
android_dev_neg = './data/part2/dev.neg.txt'
android_test_pos = './data/part2/test.pos.txt'
android_test_neg = './data/part2/test.neg.txt'

source_text_df = pd.read_csv(source_text, sep="\t", header=None)
android_df = pd.read_csv(android_csv, sep="\t", header=None)
android_dev_pos_df = pd.read_csv(android_dev_pos, sep=" ", header=None)
android_dev_neg_df = pd.read_csv(android_dev_neg, sep=" ", header=None)
android_test_pos_df = pd.read_csv(android_test_pos, sep=" ", header=None)
android_test_neg_df = pd.read_csv(android_test_neg, sep=" ", header=None)

source_text_df.columns = ['id', 'title', 'body']
source_text_df = source_text_df.dropna()
android_df.columns = ["id", "title", "body"]

android_dev_pos_df.columns = ["qid", "q_pos"]
android_dev_neg_df.columns = ["qid", "q_neg"]
android_test_pos_df.columns = ["qid", "q_pos"]
android_test_neg_df.columns = ["qid", "q_neg"]

# fit on ubuntu source data
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,1), max_df=0.8, min_df=2, stop_words='english')
vectorizer.fit(source_text_df['title'].tolist() + source_text_df['body'].tolist())
vocab = vectorizer.vocabulary_
print "source vocab size: ", len(vocab)

# create tfidf dictionary
tfidf_dict = {}
for row in tqdm(range(android_df.shape[0])):
    qid = android_df.loc[row, 'id']
    title_body = android_df.loc[row, 'title'] + ' ' + android_df.loc[row, 'body']
    tfidf_vector = vectorizer.transform([title_body])
    tfidf_dict[qid] = tfidf_vector

start_time = time.time()

# start going through pos and neg samples
y_actual, y_predicted = [], []

meter = AUCMeter()

for row in tqdm(range(android_test_pos_df.shape[0])):
    y_actual.append(1)
    q1_idx = android_test_pos_df.loc[row, 'qid']
    q2_idx = android_test_pos_df.loc[row, 'q_pos']
    score = cosine_similarity(tfidf_dict[q1_idx], tfidf_dict[q2_idx])
    y_predicted.append(score[0][0])
    meter.add(np.array([score[0][0]]), np.array([1]))

for row in tqdm(range(android_test_neg_df.shape[0])):
    y_actual.append(0)
    q1_idx = android_test_neg_df.loc[row, 'qid']
    q2_idx = android_test_neg_df.loc[row, 'q_neg']
    score = cosine_similarity(tfidf_dict[q1_idx], tfidf_dict[q2_idx])
    y_predicted.append(score[0][0])
    meter.add(np.array([score[0][0]]), np.array([0]))

print "auc value", meter.value(max_fpr=0.05)

end_time = time.time()
print 'Finished running in', (end_time - start_time) / 60
