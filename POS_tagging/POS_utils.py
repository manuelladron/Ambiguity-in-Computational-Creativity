#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from spacy import displacy
import spacy
import json
from progress.bar import IncrementalBar


nlp = spacy.load('en_core_web_sm')


class POS_Utils(object):
    def __init__(self, file_path):
        self.json = self.load_json(file_path)
        self.num_samples = len(self.json)
        self.new_name = self.make_file_name(file_path)
        self.sample_data = []
        self.BOW_adj = []
        self.bar = self.track_progess(self.num_samples)

    def BOW_unique(self):
        return list(set(self.BOW_adj))

    def extract_adjectives(self, doc):
        adj = []
        for token in doc:
            pos = token.pos_
            if pos == 'ADJ':
                adj.append(token.text)
        return adj 

    def has_text(self, sample):
        if sample['text'] != []:
            return True
        else: return False

    def parse_sample(self, text):
        # receives a list with strings 
        adjectives = []
        for phrase in text:
            doc = nlp(phrase)
            list_of_adj = self.extract_adjectives(doc)
            adjectives.extend(list_of_adj)
        return adjectives

    def load_json(self, file_path):
        with open(file_path) as f:
            data = json.loads(json.load(f))
        f.close()
        return data
    
    def make_file_name(self, file_path):
        return '%s-adjectives' % file_path.split('.json')[0] + '.json'

    def save_json2(self, file_path, data):
        data_ = json.dumps(data, ensure_ascii=False)
        with open(file_path, 'w') as json_file:
            json.dump(data_, json_file)

    def save_json(self, file_path, data):
        with open(file_path, "w", encoding='utf-8') as write_file:
            json.dump(data, write_file, ensure_ascii=False)

    def track_progess(self, maxVal):
        return IncrementalBar('Processing', max=maxVal)