import torch
from torch import nn
import torch.nn.functional as F
from text_module.word_embedding import WordEmbedding
from data_utils.vocab import Vocab
from data_utils.utils import padding_tags

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()

        self.vocab = Vocab(config)
        self.num_tags = self.vocab.num_tags()
        self.text_embedding = WordEmbedding(config)
        self.hidden_units = config["model"]["hidden_units"]
        self.dropout = config["model"]["dropout"]
        self.num_layers = config["model"]["num_layers"]
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.max_length = config["text_embedding"]["max_length"]
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_units,
                          num_layers= self.num_layers,dropout=self.dropout)
        self.fc = nn.LazyLinear(self.num_tags)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, texts, tags):
        embbed = self.text_embedding(texts)
        rnn_output, _ = self.rnn(embbed)
        out = rnn_output[-1]
        logits = self.fc(out)
        tags_out = padding_tags(tags, self.max_length, self.vocab.tag_to_idx['O'])
        loss = self.criterion(logits, tags_out)
        return logits, loss