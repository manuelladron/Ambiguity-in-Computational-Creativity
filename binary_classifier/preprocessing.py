import torch
import numpy as np
import pdb
import string
import json
import spacy
import pandas as pd
import numpy as np
# from torch.utils.data import TensorDataset, random_split

# nlp = spacy.load('en_core_web_lg')
    
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NUMBERS = '0123456789'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

class PreprocessedData(object):
    def __init__(self, train_file_path, test_file_path, max_sentence_length, eqs):
        """
        train_file_path = list with files
        test_file_path = list with files
        """
        
        self.train_path = train_file_path
        self.test_path = test_file_path
        
        self.VOCAB = None
        self.VOCAB_SIZE = None
        self.stop_words = self.getStopWords('./data/stopword.list')
#         self.max_sentence_length = max_sentence_length
        self.eqs = eqs # whether or not +,- has same size
                
        # Automatically run it when making an instance
        self.RUN_for_dataset()

    ############ utils #########################
    
    def getStopWords(self, file_name):
        with open(file_name) as f:
            file = f.read().splitlines()
            stop_words = set(file)
        return stop_words
    
    def get_file(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.loads(json.load(f))
        return data
    
    def text_from_json(self, json_file):
        all_text = []
        for file in json_file:
            for sample in file:
                text_l = sample['text']
                for sentence in text_l:
                    sent = sentence.lower()
                    all_text.append(sent)
        return all_text
    

    ############## PROCESSING DATA ##############
    
    def getNegativeTags(self, alphabet, positive_tags, all_text):
        """
        Takes in an alphabet, a list of tags and a list of sentences. Returns an alphabet that correspond to the 
        negative samples by substracting the positive tags
        """
        # 1) Get alphabet cropped to the length of sentences
        idx_len = len(all_text)
        negative_tags = alphabet[:idx_len]
        
        # 2) Iterate over positive tags and remove them from cropped_alphabet. 
        for tag in positive_tags:
            negative_tags = negative_tags.replace(tag, "")

        return negative_tags
    
    def cleanText(self, cur_text):
        new_list = []
        for sentence in cur_text:

            # (1) Make everything lowercase
            new_s = sentence.lower()

            # (2) Remove any punctuation
            # https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace
            new_s = new_s.replace('\'', '') # remove apostrophe first
            punctuation_removed = new_s.translate(new_s.maketrans(string.punctuation, ' '*len(string.punctuation)))
            
            list_of_words = punctuation_removed.split(' ')
            
            # (4) Remove words: in stop list, that are now len=0, or contain numbers
            new_s = [word for word in list_of_words if word not in self.stop_words and len(word) > 0 and word.isalpha()]
            if len(new_s) == 0:
                new_list.append(None)
            else:
                new_list.append(" ".join(new_s))
    
        return new_list
    
    def getTextValidity(self, positive_tags, negative_tags, text_list):
        v = [text_list[ALPHABET.index(letter)].lower() for letter in positive_tags if text_list[ALPHABET.index(letter)] != None]
        n = [ text_list[ALPHABET.index(letter)].lower() for letter in negative_tags if text_list[ALPHABET.index(letter)] != None]
        return v, n
        
        
    def get_all_text(self, files):
        """
        Parse json file and outputs train_data (text) and numpy array labels for binary classification
        """
        self.neg_sentences = []
        self.neg_sentences_labels = []
        self.pos_sentences = []
        self.pos_sentences_labels = []
        for file in files:
            # iterate over the examples in file and grab positive and negative samples
            for i in range(len(file)):
                positive_tags = file[i]['text-tags']
                text_list = file[i]['text']
                negative_tags = self.getNegativeTags(ALPHABET, positive_tags, text_list)
                text_list = self.cleanText(text_list)
                valid_text, nonvalid_text = self.getTextValidity(positive_tags, negative_tags, text_list) 
                
                # append sentences and labels
                for nv_text in nonvalid_text:
                    neg_label = np.array([0])
                    self.neg_sentences.append(nv_text)
                    self.neg_sentences_labels.append(neg_label)

                for v_text in valid_text:
                    pos_label = np.array([1])
                    self.pos_sentences.append(v_text)
                    self.pos_sentences_labels.append(pos_label)
        
        if self.eqs:
            min_val = min(len(self.neg_sentences), len(self.pos_sentences))
            self.neg_sentences = self.neg_sentences[:min_val]
            self.pos_sentences = self.pos_sentences[:min_val]
            self.neg_sentences_labels = self.neg_sentences_labels[:min_val]
            self.pos_sentences_labels = self.pos_sentences_labels[:min_val]
        
        self.sentences = self.pos_sentences + self.neg_sentences
        self.sentences_labels = self.pos_sentences_labels + self.neg_sentences_labels

    def tokenize_sentences(self):
        word2index = dict()
        index2word = dict()
        cur_index = 1
        data = []
        for idx, sent in enumerate(self.sentences): 
            sent_list = sent.split(' ')
            tokenized_text = [0] * len(sent_list)
            for word_idx, word in enumerate(sent_list):
                if word not in word2index:
                    word2index[word] = cur_index
                    index2word[cur_index] = word
                    cur_index += 1
                tokenized_text[word_idx] = word2index[word]
            data.append(np.array(tokenized_text))
        self.data = np.array(data)
        self.vocab_size = cur_index - 1
        self.sentences_labels = np.array(self.sentences_labels, dtype='int64')
        
    def partition_data(self, train_percentage):
        assert len(self.data) == len(self.sentences_labels)
        
        train_size = int(train_percentage * len(self.data))
        dev_size = len(self.data) - train_size
        
        # Shuffle the data
        idx = np.random.permutation(len(self.data))
        self.data, self.sentences_labels = self.data[idx], self.sentences_labels[idx]
        
        self.train_data = self.data[0:train_size]
        self.train_labels = self.sentences_labels[0:train_size]
        self.dev_data = self.data[train_size:]
        self.dev_labels = self.sentences_labels[train_size:]
        
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(dev_size))
    
    def RUN_for_dataset(self):
        # 1) get jsons
        train_raw = []
        for i in range(len(self.train_path)): # list with all training data from different sections
            train_raw.append(self.get_file(self.train_path[i]))
        
        # 2) get text
        self.get_all_text(train_raw)
        print(len(self.sentences), len(self.sentences_labels))
        
        # 3) tokenize at word-level
        self.tokenize_sentences()
        
        # 4) partition data
        self.partition_data(.8)