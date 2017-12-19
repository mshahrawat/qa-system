import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import pickle
import pdb

def string_ids_to_list(ids):
    ids = ids.split()
    ids = [int(x) for x in ids]
    return ids

class AndroidEvalDataset(Dataset):
    def __init__(self, negatives_csv, positives_csv, question_csv, glove_path):
        self.glove = self.make_glove_dict(glove_path)

        negative_df = pd.read_csv(negatives_csv, sep=" ", header=None)
        negative_df.columns = ["q", "qi"]
        negative_df = negative_df.groupby(['q'])['qi'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

        positive_df = pd.read_csv(positives_csv, sep=" ", header=None)
        positive_df.columns = ["q", "p"]

        self.questions_df = pd.read_csv(question_csv, sep="\t", header=None)
        self.questions_df.columns = ["qid", "title", "body"]

        self.df = pd.merge(positive_df, negative_df, on="q", how = "inner")
        self.df['qi'] = self.df['qi'].apply(string_ids_to_list)

    def make_glove_dict(self, glove_path):
        with open(glove_path, "rb") as f:
            glove_dict = pickle.load(f)
        return glove_dict

    def __len__(self):
        return len(self.df)

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

        p = self.id_to_text(p_id, is_title)
        p = self.string_to_tokens(p, is_title)
        p, p_mask = self.tokens_to_vector(p, is_title)

        s = np.expand_dims([q, p] + qneg, axis=0)
        s_mask = np.expand_dims([q_mask, p_mask] + qneg_mask, axis=0)
        
        sample = np.concatenate(s, axis=1)
        sample_mask = np.concatenate(s_mask, axis=1)

        return sample, sample_mask

    def __getitem__(self, idx):
        q_df = self.df.iloc[[idx]]
        q_id = int(q_df['q'])
        p_id = list(q_df['p'])[0]
        qneg_ids = list(q_df['qi'])[0]
        # take ids found and map to title + body of question word2vec feature vectors
        sample_titles, title_mask = self.get_sample(q_id, qneg_ids, p_id, True)
        sample_bodies, bodies_mask = self.get_sample(q_id, qneg_ids, p_id, False)

        sample_titles = np.transpose(sample_titles, (0, 2, 1))
        sample_bodies = np.transpose(sample_bodies, (0, 2, 1))
       
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

    def vector_from_model(self, token):
        if token in self.glove:
            return self.glove[token]
        return np.zeros(300)

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
        token_vectors = map(lambda x: self.vector_from_model(x), tokens)
        token_vectors = map(lambda v: np.expand_dims(v, axis=0), token_vectors)
        embed_dim = 300
        if is_title:
            length = 25
        else:
            length = 100
        token_vectors.extend([np.zeros((1,embed_dim))] * (length-len(token_vectors)))
        feature_vector = np.concatenate(token_vectors, axis = 0)
        mask = np.all(feature_vector, axis=1).astype(int)
        return feature_vector, mask

if __name__ == "__main__":
    pos_file = './data/part2/dev.pos.txt'
    neg_file = './data/part2/dev.neg.txt'
    questions_file = './data/part2/corpus.txt'
    glove_model_path = './data/part2/glove_dict'
    dataloader_path = './data/part2/dev_dataloader'

    questions_dataset = AndroidEvalDataset(positives_csv=pos_file, negatives_csv=neg_file,
        question_csv=questions_file, glove_path=glove_model_path)

    dataloader = DataLoader(questions_dataset, batch_size=1, shuffle=True, num_workers=1)

    with open(dataloader_path, "wb") as f:
        pickle.dump(dataloader, f)
