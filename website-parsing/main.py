import pdb, sys, os
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
    def __init__(self, file_path):
        self.stripper = MLStripper()
        self.json = self.load_json(file_path)
        self.num_examples = len(self.json)
        self.data = []
        self.new_name = self.make_file_name(file_path)
        self.bar = self.track_progess(num_examples)

    def run(self):
        # Iterste over the examples, clean the text and image lists/dicts
        for i in range(self.num_examples):
            example = self.json[i]
            text = example['text']
            images = example['images']
            cleaned_text = self.clean_html(text)
            cleaned_images = self.clean_images(images)
            new_example = {'text': cleaned_text, 'images': cleaned_images, 'title': example['title']}
            self.data.append(new_example)
            self.bar.next()

        # Save new json & cleanup
        self.save_json(new_name, data)
        self.bar.finish()

    def load_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data

    def save_json(self, file_path, data):
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
            next_quote = data[has_quote+1:].find('"')
            if next_quote != -1:
                quotes.append(data[has_quote:next_quote])
            else:
                break
            has_quote = data[next_quote+1:].find('"')
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


def main(file_path):
    u = Utils(file_path)
    u.run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('*'*20, '\nIncorrect number of arguemenets...')
        print('Example: python3 main.py filePath.json\n'+'*'*20)
    elif not os.path.exists(sys.argv[-1]):
        print('File %s does not exist.' % sys.argv[-1])
    else:
        main(sys.argv[-1])

