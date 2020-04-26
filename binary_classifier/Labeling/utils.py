import pdb, sys, os, io
from html.parser import HTMLParser
import json
import unicodedata
from progress.bar import Bar

# https://stackoverflow.com/questions/11061058/using-htmlparser-in-python-3-2
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

class Utils(object):
    def __init__(self, file_path=None):
        if file_path is None:
            pass
        else:
            self.stripper = MLStripper()
            self.json = self.load_json(file_path)
            self.num_examples = len(self.json)
            self.data = []
            self.new_name = self.make_file_name(file_path)
            self.bar = self.track_progess(self.num_examples)

    def run(self):
        raise NotImplementedError

    def load_output(self):
        return json.loads(self.json)

    def load_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data

    def save_json(self, file_path, data):
        data = json.dumps(data, ensure_ascii=False)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

    def clean_images(self, images):
        ims = []
        for i in range(len(images)):
            ims.append(images[i]['path'])
        return ims

    def get_quotes(self, text):
        data = self.clean_html(text)
        quotes = []
        has_quote = data.find('"')
        while has_quote != -1:
            data = data[has_quote+1:]
            next_quote = data.find('"')
            if next_quote != -1:
                quotes.append(data[:next_quote])
            else:
                break
            data = data[next_quote+1:]
            has_quote = data.find('"')
        return quotes

    def clean_html(self, text):
        html = ' '.join(text)
        self.stripper.feed(html)
        data = self.stripper.get_data()
        data = unicodedata.normalize("NFKD", data)
        self.stripper.fed = []
        return data

    def track_progess(self, maxVal):
        return Bar('Processing', max=maxVal)

    def make_file_name(self, file_path):
        return '%s-cleaned' % file_path.split('.json')[0] + '.json'
