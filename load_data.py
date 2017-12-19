import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import gensim
import pickle
from random import shuffle
import pdb

model = gensim.models.Word2Vec.load('./word2vec/train_word2vec')

def string_ids_to_list(ids):
    ids = ids.split()
    ids = [int(x) for x in ids]
    return ids

def one_pos_sample_transform(df):
    lens = [len(item) for item in df['p']]
    return pd.DataFrame( {"q" : np.repeat(df['q'].values,lens),
                          "p" : np.concatenate(df['p'].values),
                          "qi" : np.repeat(df['qi'].values,lens)})

def vector_from_model(token, glove=None):
    if glove:
        if token in glove:
            return glove[token]
        return np.zeros(300)

    if token in model:
        return model[token]
    return np.zeros(200)

def make_glove_dict(glove_path):
    with open(glove_path, "rb") as f:
        glove_dict = pickle.load(f)
    return glove_dict

class TrainQuestionsDataset(Dataset):
    def __init__(self, train_csv, questions_csv, evaluate=False, transform=None, glove=None):
        self.evaluate = evaluate
        self.train_df = pd.read_csv(train_csv, sep="\t", header=None, keep_default_na=False)
        if self.evaluate:
            self.train_df.columns = ["q", "p", "qi", "bm25"]
        else:
            self.train_df.columns = ["q", "p", "qi"]
        self.train_df['p'] = self.train_df['p'].apply(string_ids_to_list)
        if not evaluate:
            self.train_df = one_pos_sample_transform(self.train_df)
        self.train_df['qi'] = self.train_df['qi'].apply(string_ids_to_list)
        self.questions_df = pd.read_csv(questions_csv, sep="\t", header=None)
        self.questions_df.columns = ["qid", "title", "body"]
        self.transform = transform
        if glove:
            self.glove = make_glove_dict(glove)
        else:
            self.glove = None

    def __len__(self):
        return len(self.train_df)

    def get_sample(self, q_id, qneg_ids, p_id, is_title):
        q = self.id_to_text(q_id, is_title)
        qneg = map(lambda x: self.id_to_text(x, is_title), qneg_ids)
        # convert strings to list of tokens
        q = self.string_to_tokens(q, is_title)
        qneg = map(lambda x: self.string_to_tokens(x, is_title), qneg)
        # convert tokens to word2vec feature vector
        q, q_mask = self.tokens_to_vector(q, is_title)
        qneg = map(lambda x: self.tokens_to_vector(x, is_title), qneg)
        qneg_mask = [x[1] for x in qneg]
        qneg = [x[0] for x in qneg]

        if self.evaluate:
            p = map(lambda x: self.id_to_text(x, is_title), p_id)
            p = map(lambda x: self.string_to_tokens(x, is_title), p)
            p = map(lambda x: self.tokens_to_vector(x, is_title), p)
            p_mask = [x[1] for x in p]
            p = [x[0] for x in p]

            s = np.expand_dims([q] + p + qneg, axis=0)
            s_mask = np.expand_dims([q_mask] + p_mask + qneg_mask, axis=0)
        else:
            p = self.id_to_text(p_id, is_title)
            p = self.string_to_tokens(p, is_title)
            p, p_mask = self.tokens_to_vector(p, is_title)

            s = np.expand_dims([q, p] + qneg, axis=0)
            s_mask = np.expand_dims([q_mask, p_mask] + qneg_mask, axis=0)
        
        sample = np.concatenate(s, axis=1)
        sample_mask = np.concatenate(s_mask, axis=1)

        return sample, sample_mask

    def __getitem__(self, idx):
        q_df = self.train_df.iloc[[idx]]
        q_id = int(q_df['q'])
        p_id = list(q_df['p'])[0]
        qneg_ids = list(q_df['qi'])[0]
        if not self.evaluate:
            qneg_ids = np.random.choice(qneg_ids, 20)
        # take ids found and map to title + body of question word2vec feature vectors
        sample_titles, title_mask = self.get_sample(q_id, qneg_ids, p_id, True)
        sample_bodies, bodies_mask = self.get_sample(q_id, qneg_ids, p_id, False)

        sample_titles = np.transpose(sample_titles, (0, 2, 1))
        sample_bodies = np.transpose(sample_bodies, (0, 2, 1))
       
        if self.transform:
            sample = self.transform(sample)
        return sample_titles, sample_bodies, title_mask, bodies_mask

    def get_question_title(self, question_id):
        title = list(self.questions_df.loc[self.questions_df['qid'] == question_id]['title'])[0]
        return title

    def get_question_body(self, question_id):
        body = list(self.questions_df.loc[self.questions_df['qid'] == question_id]['body'])[0]
        return body

    def id_to_text(self, id, is_title):
        if is_title:
            return self.get_question_title(id)
        return self.get_question_body(id)

    def string_to_tokens(self, s, is_title):
        if type(s) != str:
            s = str(s)
        if is_title:
            trunc_len = 25
        else:
            trunc_len = 100
        tokens = re.findall(r"\w+|[^\w\s]", s, re.UNICODE)[:trunc_len]
        return tokens

    def tokens_to_vector(self, tokens, is_title):
        token_vectors = map(lambda x: vector_from_model(x, self.glove), tokens)
        token_vectors = map(lambda v: np.expand_dims(v, axis=0), token_vectors)
        if self.glove:
            embed_dim = 300
        else:
            embed_dim = 200
        if is_title:
            # length = 25
            length = 100
        else:
            length = 100
        token_vectors.extend([np.zeros((1,embed_dim))] * (length-len(token_vectors)))
        feature_vector = np.concatenate(token_vectors, axis = 0)
        mask = np.all(feature_vector, axis=1).astype(int)
        return feature_vector, mask

if __name__ == "__main__":
    train_file = './data/part1/raw/train_random.txt'
    questions_file = './data/part1/raw/text_tokenized.txt'
    dataloader_name = "data/part1/train_glove_dataloader"
    # dataloader_name = "data/part1/train_dataloader"
    batch_size = 64
    is_eval = False
    if is_eval:
        batch_size = 1
    glove_path = './data/part2/glove_dict'
    # glove_path = None

    questions_dataset = TrainQuestionsDataset(train_csv=train_file, 
        questions_csv=questions_file, evaluate=is_eval, glove=glove_path)
    # questions_dataset = TrainQuestionsDataset(train_csv=train_file, 
    #     questions_csv=questions_file, evaluate=is_eval)

    dataloader = DataLoader(questions_dataset, batch_size=batch_size, shuffle=True)

    with open(dataloader_name, "wb") as f:
        pickle.dump(dataloader, f)

    # with open("data/dataloader", "rb") as f:
    #     dataloader = pickle.load(f)




