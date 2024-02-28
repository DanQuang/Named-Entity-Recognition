import pandas as pd
from typing import List
from data_utils.utils import padding

class Vocab:
    def __init__(self, config):

        self.word_to_idx = {}
        self.idx_to_word = {}

        self.tag_to_idx = {}
        self.idx_to_tag = {}

        self.dataset_path = config["dataset_path"]
        self.max_length = config["text_embedding"]["max_length"]

        self.build_vocab()

    def build_words_and_tags(self):
        dataset = pd.read_csv(self.dataset_path, encoding= 'latin1')
        dataset = dataset.fillna(method= 'ffill')
        
        list_words = list(set(dataset["Word"].values))

        # lis tags
        list_tags = list(set(dataset["Tag"].values))

        return list_words, list_tags
    
    def build_vocab(self):
        list_words, list_tags = self.build_words_and_tags()
        
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(list_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.tag_to_idx = {tag: idx + 1 for idx, tag in enumerate(list_tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

    def convert_tokens_to_ids(self, tokens: List):
        ids = [self.word_to_idx.get(token) for token in tokens]
        return padding(ids, self.max_length, self.pad_token_idx())
    
    def convert_ids_to_tokens(self, ids: List):
        return [self.idx_to_word[idx] for idx in ids]
    
    def convert_tags_to_ids(self, tags: List):
        ids = [self.tag_to_idx.get(tag) for tag in tags]
        return padding(ids, self.max_length, self.tag_to_idx['O'])
    
    def convert_ids_to_tags(self, ids: List):
        return [self.idx_to_tag[idx] for idx in ids]
    
    def vocab_size(self):
        return len(self.word_to_idx)
    
    def num_tags(self):
        return len(self.tag_to_idx)
    
    def pad_token_idx(self):
        return 0