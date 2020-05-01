import numpy as np
import utils

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NUMBERS = '0123456789'


class PreprocessedData(object):
    def __init__(self, files_paths, images_paths, new_folder, nlp, vocab):
        """
        train_file_path = list with files
        test_file_path = list with files
        """

        # Create pipe
        self.nlp = nlp

        # Inherit vocabulary
        self.VOCAB = vocab

        # Call to utils
        self.Utils = utils.Utils()

        # paths to jsons
        self.files_paths = files_paths

        # path to image folders
        self.images_paths = images_paths

        # path to new image_path
        self.new_image_path = new_folder

        # Dataset
        self.train_data = None
        self.dev_data = None
        self.train_labels = None
        self.dev_labels = None

        # Run
        print("\nRunner....")
        self.runner()

    ############## PROCESSING DATA ##############
    def get_adj_from_sentence(self, sentence):
        """
        Parses sentence and get a list of cleaned lemmatized adj longer than 1 character. No duplicates
        """
        doc = self.nlp(sentence)
        adj = []
        for token in doc:
            if token.pos_ == 'ADJ':
                adj_ = token.lemma_.lower()
                if len(adj_) > 1:
                    adj.append(adj_)
        return list(set(adj))

    def get_adj_from_all_sentences(self, sentences):
        """
        Calls the function get_adj_from_sentence a sentences number of times
        """
        adj = []
        for sentence in sentences:
            sent_adj = self.get_adj_from_sentence(sentence)
            adj.extend(sent_adj)
        return adj

    def is_valid(self, images, adj):
        """
        Avoids empty data samples
        """
        if adj == []: return False
        if images == []: return False
        return True

    def handle_N_labels(self, list_of_labels, N_labels):
        """
        Truncates or pads list of labels according to N_labels
        """
        pad_with = list_of_labels[0]
        if len(list_of_labels) == N_labels:
            return list_of_labels

        elif len(list_of_labels) < N_labels:
            diff = N_labels - len(list_of_labels)
            for i in range(diff):
                list_of_labels.append(pad_with)
        else:
            list_of_labels = list_of_labels[:N_labels]

        return list_of_labels

    def get_images_labels(self, files):
        """
        Parse json files and outputs train_data (image) + numpy array labels for multi-label classification
        """
        train_data = []
        labels = dict()

        for f, file in enumerate(files):  # 0_arch 1_des 2_tech
            im_per_section = []  # these ones have the same length
            for i in range(len(file)):  # 902 samples architecture, 675 samples design
                sample_dict = file[i]  # dictionary
                sample_text = sample_dict['text']  # list with sentences (strings)
                text_tags = sample_dict['text-tags']
                sample_images = sample_dict['images']
                image_tags = sample_dict['image-tags']

                # 2) Select valid text
                tagged_text = [sample_text[ALPHABET.index(letter)].lower() for letter in text_tags]

                # 3) Get adjectives from valid text
                adj = self.get_adj_from_all_sentences(tagged_text)

                # 4) Select valid images
                tagged_images = [sample_images[int(letter)] for letter in image_tags]

                if self.is_valid(tagged_images, adj):
                    # Add special token '<end>' as a label
                    #                     adj.append('<end>')
                    #                     adj = self.handle_N_labels(adj, 3)
                    for image in tagged_images:
                        labels[image[5:]] = adj
                    im_per_section.append(tagged_images)

            train_data.append(im_per_section)

        return train_data, labels

    def convert_labels_to_int(self, labels):
        """
        Convert labels to int array
        """
        self.labels_int = dict()
        for key, val in labels.items():
            label_array = np.zeros(len(val), dtype=int)
            for i, label in enumerate(val):
                # Try add label in vocabulary. If already exists, nothing happens, just get the idx
                self.VOCAB.add_word(label)
                idx = self.VOCAB.word2idx[label]
                label_array[i] = idx
            self.labels_int[key] = label_array

        return self.labels_int

    def copy_wrapper(self, list_dataset_per_section, curr_folders, dest_folder):
        self.all_failed_samples = []
        for i, dataset in enumerate(list_dataset_per_section):
            self.all_failed_samples.append(self.copy_dataset(dataset, curr_folders[i], dest_folder))

    def copy_dataset(self, image_dataset, curr_folder, dest_folder):
        fail_samples = self.Utils.copy_files(image_dataset, curr_folder, dest_folder)
        return fail_samples

    def flatten(self, S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return self.flatten(S[0]) + self.flatten(S[1:])
        return S[:1] + self.flatten(S[1:])

    def change_name(self, dataset):
        dataset = self.flatten(dataset)
        for i, image in enumerate(dataset):
            newname = image[5:]
            dataset[i] = newname
        return dataset

    def remove_dups(self):
        seen = []
        dups = []
        for sample in self.train_data:
            if sample not in seen:
                seen.append(sample)
            else:
                dups.append(sample)

        for dup in dups:
            if dup in self.train_data:
                self.train_data.remove(dup)

    def partition_data(self, data_set, label_set, train_percentage):
        train_len = int(train_percentage * data_set.size)
        dev_len = data_set.size - train_len

        # train
        train_set = data_set[:train_len]
        train_labels = label_set[:train_len]

        # development
        dev_set = data_set[train_len:]
        dev_labels = label_set[train_len:]

    def runner(self):
        files = self.Utils.jsons_to_list(self.files_paths)
        self.train_data, self.labels = self.get_images_labels(files)  # keep track of the length of these variables

        # This puts all the images in one folder.
        self.copy_wrapper(self.train_data, self.images_paths, self.new_image_path)

        # Change names of self.train
        self.train_data = self.change_name(self.train_data)

        # This needs to be done after flattening
        self.convert_labels_to_int(self.labels)

        # Removes potential duplicates
        self.remove_dups()
