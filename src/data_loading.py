import json
import os
import torch.nn as nn
import torch

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config
    
def read_sentences(config, n=int(5e4)):
    sentences = []
    with open(os.path.join(*[os.getcwd(), config["DATA_PATH"], config["INPUT_FILE_NAME"]]), encoding='utf-8') as f:
        for i in range(n):
            line = f.readline()
            if line:
                sentences.append(line.strip())
            else:
                print("The whole file was read. It was " + str(i) + " exactly")
                break
    return sentences

class DataLoader():
    def __init__(self, data, validation_size = 0.1):
        self.split = int(len(data) * (1-validation_size))
        self.data = data[:self.split]
        self.data_validation = data[self.split:]
        self.batch_size = 100
        self.prepare_train()
        self.prepare_validation()
    
    def prepare_train(self):
        self.sentences_train, self.proposals_train, self.targets_train = [], [], []
        for s in self.data:
            self.sentences_train.append(nn.utils.rnn.pad_sequence(s[0]).float().cuda())
            self.proposals_train.append(s[1].float().cuda())
            self.targets_train.append(s[2])
        self.targets_train = torch.Tensor(self.targets_train).long().cuda()     
    
    def prepare_validation(self):
        self.sentences_validation, self.proposals_validation, self.targets_validation = [], [], []
        for s in self.data:
            self.sentences_validation.append(nn.utils.rnn.pad_sequence(s[0]).float().cuda())
            self.proposals_validation.append(s[1].float().cuda())
            self.targets_validation.append(s[2])
        self.targets_validation = torch.Tensor(self.targets_validation).long().cuda() 
            
    def iterate(self):
        for st_idx in range(0, len(self.data), self.batch_size):
            sents = self.sentences_train[st_idx:st_idx + self.batch_size]
            props = self.proposals_train[st_idx:st_idx + self.batch_size]
            targs = self.targets_train[st_idx:st_idx + self.batch_size]
            yield sents, props, targs
    
    def iterate_validation(self):
        for st_idx in range(0, int(len(self.data_validation)), self.batch_size):
            sents = self.sentences_validation[st_idx:st_idx + self.batch_size]
            props = self.proposals_validation[st_idx:st_idx + self.batch_size]
            targs = self.targets_validation[st_idx:st_idx + self.batch_size]
            yield sents, props, targs