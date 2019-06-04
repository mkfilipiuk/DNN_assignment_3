import string
import unicodedata
import random
import torch.nn.functional as F
import numpy as np
import torch

class CorpusPreprocessor():
    def __init__(self):
        self.legal_characters = set(list(string.ascii_lowercase + ' '))
        self.mask = "xxxxxxxx"

    def _random_word(self):
        return self.vocabulary[np.random.randint(self.vocabulary_len)]

    def _unicode_to_ascii_lowercase(self, text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def transform_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = ''.join(['l' if c == 'ł' else c for c in sentence]) # 'ł' requires a special treatment
        sentence = unicodedata.normalize('NFKD', sentence).encode('ascii','ignore').decode('ascii')
        sentence = ''.join([c for c in sentence if c in self.legal_characters])
        return sentence
        
    def transform_text(self, text):
        return [self.transform_sentence(sentence) for sentence in text]

    def mask_text(self, text):
        self.to_sample = list(set([y for x in [s.split() for s in text] for y in x]))
        return [self.mask_sentence(sentence) for sentence in text]
    
    def get_random_word(self):
        return random.choice(self.to_sample)
    
    def mask_sentence(self, sentence):
        sentence = sentence.split()
        
        orig_word_index = np.random.randint(len(sentence))
        orig_word = sentence[orig_word_index]

        sentence[orig_word_index] = self.mask
        sentence = ' '.join(sentence)
        
        true_or_false = bool(random.getrandbits(1))
        
        return sentence, (orig_word if true_or_false else self.get_random_word()), true_or_false
                             
    def encode_text(self, text):
        return [self.encode_sentence(s) for s in text]
    
    def encode_sentence(self, sentence):
        return [self.encode_word(w) for w in sentence[0].split()], self.encode_word(sentence[1]), sentence[2]
                              
    def encode_word(self, text):
        indices = np.array([self.encode_char(c) for c in text])
        indices = torch.LongTensor(indices)
        return F.one_hot(indices, 26)
                             
    def encode_char(self, char):
        return string.ascii_lowercase.index(char)
