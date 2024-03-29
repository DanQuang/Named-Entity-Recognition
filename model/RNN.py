import torch
from torch import nn
import torch.nn.functional as F
from text_module.word_embedding import WordEmbedding
from data_utils.vocab import Vocab

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
                          num_layers= self.num_layers,dropout=self.dropout, batch_first= True)
        self.fc = nn.Linear(self.hidden_units, self.num_tags)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, texts, tags):
        embbed = self.text_embedding(texts)
        rnn_output, _ = self.rnn(embbed)
        logits = self.fc(rnn_output)
        loss = self.criterion(logits.permute(0, 2, 1), tags)
        return logits, loss