import torch
from torch import nn
from data_utils.vocab import Vocab

class WordEmbedding(nn.Module):
    def __init__(self, config):
        super(WordEmbedding, self).__init__()
        self.vocab = Vocab(config)
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.max_length = config["text_embedding"]["max_length"]
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim, self.vocab.pad_token_idx())
        self.droput = nn.Dropout(config["text_embedding"]["dropout"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, input_texts):
        output = self.embedding(input_texts)
        output = self.droput(output)
        return output