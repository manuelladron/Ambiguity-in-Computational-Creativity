import pickle
import utils
import spacy

nlp = spacy.load('en_core_web_lg')

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, nlp):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.nlp = nlp

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

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
        #         print("adj: ", adj)
        return list(set(adj))

    def get_adj_from_all_sentences(self, sentences):
        """
        Calls the function get_adj_from_sentence a sentences number of times
        """
        adj = []
        for sentence in sentences:
            #             print("sentence: ", sentence)
            sent_adj = self.get_adj_from_sentence(sentence)
            adj.extend(sent_adj)
        return adj



def build_vocabulary(path, vocab_dict_path, nlp):
    """
    Builds vocabulary with the entirity of the datasets
    """
    # Create Utils instance
    U = utils.Utils()
    data_raw = U.jsons_to_list(path)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary(nlp)
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    all_adjs = []
    for f, file in enumerate(data_raw):  # 0_arch 1_des 2_...
        print("Starting building vocab corresponding to file: \n", f)
        for i in range(len(file)):  # 900 samples
            sample_dict = file[i]
            sample_text = sample_dict['text']  # this is a list with strings
            adj_list = vocab.get_adj_from_all_sentences(sample_text)
            all_adjs.append(adj_list)
        print("....Finishing vocab file: \n", f)

    for sublist in all_adjs:
        for adj in sublist:
            vocab.add_word(adj)

    # Save vocab in dict
    with open(vocab_dict_path, 'wb') as f:
        pickle.dump(vocab, f)
    f.close()
    return vocab


def load_vocab(file_path):
    # open file
    f = open(file_path, 'rb')

    # dump info to that file
    data = pickle.load(f)

    # close file
    f.close()

    # return vocab
    return data
