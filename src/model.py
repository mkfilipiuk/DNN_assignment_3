import torch
import torch.nn as nn
from .data_loading import DataLoader
import torch.optim as optim
import numpy as np

import torch.nn.functional as F

class WordEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.grus = nn.GRU(input_size=26,
                           hidden_size=40,
                           num_layers=2,
                           bidirectional=True)
    
    def forward(self, x):
        _, hidden_states = self.grus(x)
        hidden_states = hidden_states.view(2, 2, -1, 40)[-1]
        return torch.cat([hidden_states[0], hidden_states[1]], dim=-1)
    
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedder = WordEmbedder()
        self.grus = nn.GRU(input_size = 80,
                           hidden_size = 40,
                           num_layers=2,
                           bidirectional=True)
        
        self.fc0 = nn.Linear(160, 100)
        self.fc1 = nn.Linear(100, 2)

    
    def forward(self, sents, props):
        p = nn.utils.rnn.pad_sequence(props)
        p = self.word_embedder(p)
        
        x = [self.word_embedder(sent) for sent in sents]
        x = nn.utils.rnn.pad_sequence(x)
        _, hidden_states = self.grus(x)
        hidden_states = hidden_states.view(2, 2, -1, 40)[-1]
        
        x = torch.cat([hidden_states[0], hidden_states[1], p], dim=-1)
        x = self.fc0(x)
        x = F.relu(x)
        
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
    def train(self, data_loader, n_epoch = 10):
        print("Starting training...")
        optimizer = optim.Adam(self.parameters())
        for epoch in range(1, 1+n_epoch):
            losses = []
            for sents, props, targs in data_loader.iterate():
                optimizer.zero_grad()
                outputs = self(sents, props)
                loss = F.nll_loss(outputs, targs)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            lloss = round(np.log(np.mean(losses)), 5)
            print(f"epoch: {epoch}, lloss: {lloss}")
            self.validate(data_loader)
            
    def validate(self, data_loader):
        c, a = 0, 0
        with torch.no_grad():
            for sents, props, targs in data_loader.iterate_validation():
                outputs = self(sents, props)
                c += torch.sum(torch.argmax(outputs, 1) == targs)
                a += torch.argmax(outputs, 1).shape[0]
        print(float(c)/a)