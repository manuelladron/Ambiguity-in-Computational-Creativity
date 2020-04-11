import json
import numpy as np

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NUMBERS = '0123456789'

class PreprocessedData(object):
    def __init__(self, train_file_path, test_file_path):
        """
        train_file_path = list with files
        test_file_path = list with files
        """

        self.train_path = train_file_path
        self.test_path = test_file_path

        self.VOCAB = None
        self.VOCAB_SIZE = None

        # Dictionary to convert char to integers
        self.char2index = None

        # Dataset
        self.train_data = None
        self.dev_data = None
        self.train_labels = None
        self.dev_labels = None

        # Automatically run it when making an instance
        self.RUN_for_vocab()
        self.RUN_for_dataset()


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

    ################# VOCABULARY ##############

    def get_all_chars(self, text_list):
        all_chars = []
        for sentences in text_list:
            for char in sentences:
                all_chars.append(char)
        chars = sorted(list(set(all_chars)))
        return chars

    def get_vocabulary(self, json_files):
        """
        from test json file (includes elements in training), get all unique chars
        """
        text = self.text_from_json(json_files)
        chars = self.get_all_chars(text)
        return chars

    def word_2_index(self, VOCAB):
        char_to_int = dict((c,i) for i,c in enumerate(VOCAB))
        self.char2index = char_to_int

    def RUN_for_vocab(self):
        # 1) Get json for vocabulary
        train_and_test_samples = []
        for i in range(len(self.test_path)):
            sample = self.get_file(self.test_path[i]) # Again, testpath includes train samples
            train_and_test_samples.append(sample)

        # 2) Get vocabulary
        self.VOCAB = self.get_vocabulary(train_and_test_samples)
        self.VOCAB_SIZE = len(self.VOCAB)

        # 3) Get dictionary
        self.word_2_index(self.VOCAB)

    ############## PROCESSING DATA ##############

    def remove_all_letters_in_text_tags_from_alphabet(self, alphabet, positive_tags, all_text):
        """
        Takes in an alphabet, a list of tags and a list of sentences. Returns an alphabet that correspond to the
        negative samples by substracting the positive tags
        """
        # 1) Get alphabet cropped to the length of sentences
        idx_len = len(all_text)
        cropped_alphabet = alphabet[:idx_len]


        # 2) If not positive tags, return cropped_alpha
        if positive_tags == []:
            return cropped_alphabet

        # 3) Iterate over positive tags and remove them from cropped_alphabet.
        for tag in positive_tags:
            new_alphabet = cropped_alphabet.replace(tag, "")
            cropped_alphabet = new_alphabet

        # 4) The result is the negative tags! :)
        return new_alphabet


    def get_all_text(self, files):
        """
        Parse json file and outputs train_data (text) and numpy array labels for binary classification
        """
        train_data = []
        train_labels = []

        for file in files:
            # iterate over the examples in file and grab positive and negative samples
            for i in range(len(file)):

                # elements from dictionary
                positive_tags = file[i]['text-tags']
                text_list = file[i]['text']

                # valid text
                valid_text = [ text_list[ALPHABET.index(letter)].lower() for letter in positive_tags ]

                # nonvalid text
                negative_tags = self.remove_all_letters_in_text_tags_from_alphabet(ALPHABET, positive_tags, text_list)
                nonvalid_text = [ text_list[ALPHABET.index(letter)].lower() for letter in negative_tags ]

                # labels
                pos_label = np.array([0,1])
                neg_label = np.array([1,0])

    #             pos_label = np.array([1])
    #             neg_label = np.array([0])

                # store samples and labels that are not empty lists
                if len(nonvalid_text) != 0:
                    train_data.append(nonvalid_text)
                    train_labels.append(pos_label)

                if len(valid_text) != 0:
                    train_data.append(valid_text)
                    train_labels.append(neg_label)


        return train_data, np.array(train_labels)

    def convert_text_to_int_array(self, text, dic):
        """
        Convert text dataset to int array
        """
        all_ints = []
        for sample in text:
            for sentence in sample:
                sent_len = len(sentence)
                sent_array = np.zeros(sent_len, dtype = int)
                for i, char in enumerate(sentence):
                    val = dic[char]
                    sent_array[i] = val
            all_ints.append(sent_array)
        return np.array(all_ints)


    def partition_data(self, data_set, label_set, train_percentage):
        train_len = int(train_percentage*data_set.size)
        dev_len = data_set.size - train_len

        # train
        self.train_data = data_set[:train_len]
        self.train_labels = label_set[:train_len]

        # development
        self.dev_data = data_set[train_len:]
        self.dev_labels = label_set[train_len:]

        return train_set, dev_set, train_labels, dev_labels

    def RUN_for_dataset(self):
        train_raw = []
        for i in range(len(self.train_path)): # list with all training data from different sections
            train_raw.append(self.get_file(self.train_path[i]))

        raw_dataset, labels_dataset = self.get_all_text(train_raw)
        data_set = self.convert_text_to_int_array(raw_dataset, self.char2index)
        self.partition_data(data_set, labels_dataset, .8)
