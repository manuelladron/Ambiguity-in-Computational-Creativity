import torch
import numpy as np
import pdb
import string
import json
from torch.utils.data import TensorDataset, random_split

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NUMBERS = '0123456789'

class PreprocessedData_wordlevel(object):
    def __init__(self, train_file_path, test_file_path, tokenizer):
        """
        train_file_path = list with files
        test_file_path = list with files
        """
        
        self.train_path = train_file_path
        self.test_path = test_file_path
        self.tokenizer = tokenizer
        
        self.VOCAB = None
        self.VOCAB_SIZE = None
        self.stop_words = self.getStopWords('./data/stopword.list')
                
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
                new_list.append('')
            else:
                new_list.append(" ".join(new_s))
            
        return new_list
        
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

                valid_text = [ text_list[ALPHABET.index(letter)].lower() for letter in positive_tags ]
                nonvalid_text = [ text_list[ALPHABET.index(letter)].lower() for letter in negative_tags ] 
                # append sentences and labels
                for nv_text in nonvalid_text:
                    neg_label = np.array([0])
                    self.neg_sentences.append(nv_text)
                    self.neg_sentences_labels.append(neg_label)

                for v_text in valid_text:
                    pos_label = np.array([1])
                    self.pos_sentences.append(v_text)
                    self.pos_sentences_labels.append(pos_label)
                    
        min_val = min(len(self.neg_sentences), len(self.pos_sentences))
        self.neg_sentences = self.neg_sentences[:min_val]
        self.pos_sentences = self.pos_sentences[:min_val]
        self.neg_sentences_labels = self.neg_sentences_labels[:min_val]
        self.pos_sentences_labels = self.pos_sentences_labels[:min_val]
        
        self.sentences = self.pos_sentences + self.neg_sentences
        self.sentences_labels = self.pos_sentences_labels + self.neg_sentences_labels

        # from list to array
        self.sentences_labels = np.array(self.sentences_labels, dtype='int64')

    def tokenize_sentences(self):
        
        # Tokenize all sentences and map the tokens to their word ID 
        self.inputs_ids = []
        self.attention_masks = []
        
        for sent in self.sentences:                
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                add_special_tokens = True, # add [CLS] and [SEP]
                                                max_length = 64,
                                                pad_to_max_length = True,  # pad and truncate
                                                return_attention_mask = True,
                                                return_tensors = 'pt'
                                                )
            self.inputs_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
            
#             print('original: ', sent)
#             print('token IDs: ', encoded_dict['input_ids'])

        # Convert list to tensors
        self.inputs_ids = torch.cat(self.inputs_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.sentences_labels)
        print(self.labels.shape)
        print(self.labels.type())
        
    def partition_data(self, train_percentage):
        assert len(self.inputs_ids) == len(self.sentences_labels)
        dataset = TensorDataset(self.inputs_ids, self.attention_masks, self.labels)
        
        train_size = int(train_percentage * len(dataset))
        dev_size = len(dataset) - train_size
        
        self.train_dataset, self.dev_dataset = random_split(dataset, [train_size, dev_size])
        
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
        self.partition_data(.6)