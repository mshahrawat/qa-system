import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import pickle
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from meter import *
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(q, others):
    cos_sims = F.cosine_similarity(q.unsqueeze(0).expand_as(others), others).numpy()
    sorted_indices = cos_sims.argsort()[::-1]
    return cos_sims, sorted_indices

android_csv = './data/part2/corpus.txt'
android_dev_pos = './data/part2/dev.pos.txt'
android_dev_neg = './data/part2/dev.neg.txt'
android_test_pos = './data/part2/test.pos.txt'
android_test_neg = './data/part2/test.neg.txt'

def load_corpus():
    corpus = []
    with open(android_csv) as file:
        lines = file.readlines()
        for line in lines:
            line = line.split('\t')
            l = ' '.join([line[1], line[2]])
            corpus.append(l)
    return corpus

class AndroidQuestionsDataset(Dataset):
    def __init__(self, transform=None):
        self.android_df = pd.read_csv(android_csv, sep="\t", header=None)
        self.android_dev_pos_df = pd.read_csv(android_dev_pos, sep=" ", header=None)
        self.android_dev_neg_df = pd.read_csv(android_dev_neg, sep=" ", header=None)
        self.android_test_pos_df = pd.read_csv(android_test_pos, sep=" ", header=None)
        self.android_test_neg_df = pd.read_csv(android_test_neg, sep=" ", header=None)

        self.android_df.columns = ["qid", "title", "body"]
        self.android_dev_pos_df.columns = ["qid", "q_pos"]
        self.android_dev_neg_df.columns = ["qid", "q_neg"]
        self.android_test_pos_df.columns = ["qid", "q_pos"]
        self.android_test_neg_df.columns = ["qid", "q_neg"]

        self.android_dev_neg_df = self.android_dev_neg_df.groupby('qid')['q_neg'].apply(list).reset_index()
        self.android_test_neg_df = self.android_test_neg_df.groupby('qid')['q_neg'].apply(list).reset_index()

        self.vectorizer, self.vocab, self.bow = self.make_vectorizer()

        self.transform = transform

    def make_vectorizer(self):
        android_corpus = load_corpus()
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), max_features=1000)
        tfidf_vectors = tfidf_vectorizer.fit_transform(android_corpus)
        return tfidf_vectorizer, tfidf_vectorizer.vocabulary_, tfidf_vectors

    def __len__(self):
        return len(self.android_test_pos_df)

    def __getitem__(self, idx):
        pos_q_df = self.android_test_pos_df.iloc[[idx]]
        pos_q_id = int(pos_q_df['qid'])
        pos_p_id = list(pos_q_df['q_pos'])[0]

        neg_q_df = self.android_test_neg_df.loc[self.android_test_neg_df['qid'] == pos_q_id]
        neg_q_id = int(neg_q_df['qid'])
        neg_q_ids = list(neg_q_df['q_neg'])[0]

        title_sample = self.get_sample(pos_q_id, pos_p_id, neg_q_ids, is_title=True)
        body_sample = self.get_sample(pos_q_id, pos_p_id, neg_q_ids, is_title=False)

        labels = np.zeros((title_sample.shape[0],))
        labels[:2] = 1
        labels[2:] = 0

        return title_sample, body_sample, labels

    def get_sample(self, q_id, p_id, qneg_ids, is_title):
        q = self.id_to_text(q_id, is_title)
        q = self.string_to_tokens(q)
        q = self.tokens_to_vector(q)

        p = self.id_to_text(p_id, is_title)
        p = self.string_to_tokens(p)
        p = self.tokens_to_vector(p)

        qneg = map(lambda x: self.id_to_text(x, is_title), qneg_ids)
        qneg = map(self.string_to_tokens, qneg)
        qneg = map(self.tokens_to_vector, qneg)

        s = np.expand_dims([q, p] + qneg, axis=0)
        sample = np.concatenate(s, axis=1)
        return sample

    def get_question_title(self, question_id):
        title = list(self.android_df.loc[self.android_df['qid'] == question_id]['title'])[0]
        return title

    def get_question_body(self, question_id):
        body = list(self.android_df.loc[self.android_df['qid'] == question_id]['body'])[0]
        return body

    def id_to_text(self, id, is_title):
        # return (self.get_question_title(id) + ' ' + self.get_question_body(id))
        if is_title:
            return self.get_question_title(id)
        return self.get_question_body(id)

    # def vector_from_vocab(self, token):
    #     if token in self.vocab:
    #         return self.bow[self.vocab[token]].toarray()
    #     return np.asarray([None])

    def string_to_tokens(self, s):
        if type(s) != str:
            s = str(s)
        tokens = re.findall(r"\w+|[^\w\s]", s, re.UNICODE)[:100]
        return tokens

    def tokens_to_vector(self, tokens):
        feature_vector = self.vectorizer.transform(tokens).toarray()
        if feature_vector.shape[0] < 100:
            padding = [np.zeros((1000))] * (100-feature_vector.shape[0])
            feature_vector = np.concatenate((feature_vector, padding), axis = 0)
        return feature_vector

questions_dataset = AndroidQuestionsDataset()
dataloader = DataLoader(questions_dataset, batch_size=1, shuffle=True, num_workers=1)

start_time = time.time()

meter = AUCMeter()
num_samples = 0
for title_batch, body_batch, labels in tqdm(dataloader):
# for title_batch, labels in tqdm(dataloader):
    title_batch = Variable(torch.cat(title_batch, 0))
    body_batch = Variable(torch.cat(body_batch, 0))
    labels = torch.cat(labels, 0)

    title_mask = (title_batch != 0)
    body_mask = (body_batch != 0)
    title_mask = title_mask.type(torch.DoubleTensor)
    body_mask = body_mask.type(torch.DoubleTensor)
    # print title_mask.size()
    title_masked = torch.mul(title_mask, title_batch)
    body_masked = torch.mul(body_mask, body_batch)
    # print title_masked.size()
    titles_encoded = torch.sum(title_masked, 1)
    bodies_encoded = torch.sum(body_masked, 1)

    # print titles_encoded.size()

    # titles_encoded = torch.mean(title_batch, dim=1)
    # bodies_encoded = torch.mean(body_batch, dim=1)

    # qs_encoded = titles_encoded
    qs_encoded = (titles_encoded + bodies_encoded) / Variable(torch.DoubleTensor([2]))

    q = qs_encoded[0].data
    others = qs_encoded[1:].data
    labels = labels[1:].numpy()

    cos_sims, sorted_indices = cos_sim(q, others)

    meter.add(cos_sims, labels)

    num_samples += 1
    # if num_samples == 500:
    #     break

print meter.value(max_fpr=0.05)

end_time = time.time()
print 'Finished running in', (end_time - start_time) / 60

# load corpus file
# scan file and get vocabulary using CountVectorizer
# for all main questions in android test data set:
#   - change all questions (main, pos, neg) to BoW representation
#   - get cosine similarity between vector rep of query vs pos/neg examples
#   - add these similarities to a list
# make a target list with labels corresponding to the entries in the sim list
# feed both the similarities list and target list to meter.py
