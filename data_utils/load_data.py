from torch.utils.data import Dataset, DataLoader
from data_utils.vocab import Vocab

import pandas as pd
from torch.utils.data import random_split

class MyDataset(Dataset):
    def __init__(self, dataset_path: str, vocab: Vocab):
        super(MyDataset, self).__init__()

        self.vocab = vocab

        data = pd.read_csv(dataset_path, encoding= 'latin1')
        data = data.fillna(method= 'ffill')

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]

        grouped = data.groupby("Sentence #").apply(agg_func)
        obj = [s for s in grouped]

        self.sentences = [[w[0] for w in s] for s in obj]
        self.tags = [[w[2] for w in s] for s in obj]

    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        tag = self.tags[index]

        # tokenize
        sentence_padding = self.vocab.convert_tokens_to_ids(sentence)
        tag_padding = self.vocab.convert_tags_to_ids(tag)

        return {
            "sentence": sentence_padding,
            "tag": tag_padding
        }


class Load_Data:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.vocab = Vocab(config)

        self.dataset_path = config["dataset_path"]
        self.dataset = MyDataset(self.dataset_path, self.vocab)

        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(self.dataset, [0.7, 0.1, 0.2])

    def load_train_dev(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size= self.train_batch,
                                      shuffle= True)
        
        dev_dataloader = DataLoader(self.dev_dataset,
                                    batch_size= self.dev_batch,
                                    shuffle= False)
        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataloader = DataLoader(self.test_dataset,
                                    batch_size= self.test_batch,
                                    shuffle= False)
        return test_dataloader