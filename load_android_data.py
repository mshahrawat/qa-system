import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import pickle
import pdb

class AndroidQuestionsDataset(Dataset):
    def __init__(self, ubuntu_csv, android_csv, glove_path):
        self.glove = self.make_glove_dict(glove_path)

        self.ubuntu_df = pd.read_csv(ubuntu_csv, sep="\t", header=None)
        self.ubuntu_df.columns = ["qid", "title", "body"]

        self.android_df = pd.read_csv(android_csv, sep="\t", header=None)
        self.android_df.columns = ["qid", "title", "body"]

    def make_glove_dict(self, glove_path):
        with open(glove_path, "rb") as f:
            glove_dict = pickle.load(f)
        return glove_dict

    def __len__(self):
        return len(self.ubuntu_df)

    def transform_samples(self, samples, is_title):
        samples = samples.apply(lambda x: self.string_to_tokens(x, is_title))
        samples = samples.apply(lambda x: self.tokens_to_vector(x, is_title)).tolist() 
        samples_mask = [x[1] for x in samples]
        samples = [x[0] for x in samples]
        samples = np.concatenate(np.expand_dims(samples, axis=0))
        samples_mask = np.concatenate(np.expand_dims(samples_mask, axis=0))
        return samples, samples_mask

    def __getitem__(self, idx):
        ubuntu_title_samples = self.ubuntu_df['title'].sample(n=20)
        android_title_samples = self.android_df['title'].sample(n=20)

        ubuntu_body_samples = self.ubuntu_df['body'].sample(n=20)
        android_body_samples = self.android_df['body'].sample(n=20)

        ubuntu_title_samples, ubuntu_title_mask = self.transform_samples(ubuntu_title_samples, True)
        android_title_samples, android_title_mask = self.transform_samples(android_title_samples, True)

        ubuntu_body_samples, ubuntu_body_mask = self.transform_samples(ubuntu_body_samples, False)
        android_body_samples, android_body_mask = self.transform_samples(android_body_samples, False)

        title_samples = np.concatenate([ubuntu_title_samples, android_title_samples])
        body_samples = np.concatenate([ubuntu_body_samples, android_body_samples])
        title_mask = np.concatenate([ubuntu_title_mask, android_title_mask])
        body_mask = np.concatenate([ubuntu_body_mask, android_body_mask])
        labels = np.zeros((40, 2))
        labels[:20,0] = 1
        labels[20:,1] = 1

        p = np.random.permutation(labels.shape[0])

        title_samples = np.transpose(title_samples, (0, 2, 1))
        body_samples = np.transpose(body_samples, (0, 2, 1))

        return title_samples[p], body_samples[p], title_mask[p], body_mask[p], labels[p]

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
    ubuntu_file = './data/part1/raw/text_tokenized.txt'
    android_file = './data/part2/corpus.txt'
    model_path = './data/part2/glove_dict'
    batch_size = 64

    questions_dataset = AndroidQuestionsDataset(ubuntu_csv=ubuntu_file, 
        android_csv=android_file, glove_path=model_path)

    dataloader = DataLoader(questions_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    with open("data/part2/dataloader", "wb") as f:
        pickle.dump(dataloader, f)
