import pandas as pd
import numpy as np

class Vocab:
    def __init__(self, config):

        self.word_to_idx = {}
        self.idx_to_word = {}

        self.tag_to_idx = {}
        self.idx_to_tag = {}

        self.dataset_path = config["dataset_path"]

        self.build_vocab()

    def build_words_and_tags(self):
        dataset = pd.read_csv(self.dataset_path)
        
        list_words = set()
        list_tags = set()

        for words in dataset["Word"]:
            for word in words:
                list_words.add(word)

        for tags in dataset["Tag"]:
            for tag in tags:
                list_tags.add(tag)

        list_words = list(list_words)
        list_tags = list(list_tags)

        return list_words, list_tags
    
    def build_vocab(self):
        list_words, list_tags = self.build_words_and_tags()
        
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(list_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.tag_to_idx = {tag: idx + 1 for idx, tag in enumerate(list_tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]
    
    def convert_tags_to_ids(self, tags):
        return [self.tag_to_idx.get(tag) for tag in tags]
    
    def convert_ids_to_tags(self, ids):
        return [self.idx_to_tag[idx] for idx in ids]
    
    def vocab_size(self):
        return len(self.word_to_idx) + 1
    
    def num_tags(self):
        return len(self.tag_to_idx)
    
    def pad_token_idx(self):
        return 0