import torch
from torch import nn
from data_utils.vocab import Vocab
from data_utils.utils import padding_sequence

class WordEmbedding(nn.Module):
    def __init__(self, config):
        self.vocab = Vocab(config)
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.max_length = config["text_embedding"]["max_length"]
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim, self.vocab.pad_token_idx())
        self.droput = nn.Dropout(config["text_embedding"]["dropout"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, input_texts):
        sequence_vectors = []
        for input_text in input_texts:
            padding_seq = padding_sequence(input_text, self.max_length, self.vocab.pad_token_idx())
            sequence_vectors.append(padding_seq)
        
        padding_sequences = torch.stack(sequence_vectors, dim= 0).to(self.device)
        output = self.embedding(padding_sequences)
        output = self.droput(output)
        return output