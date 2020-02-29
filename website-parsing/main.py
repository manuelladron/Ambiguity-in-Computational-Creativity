import pdb, sys, os
from html.parser import HTMLParser
import json
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
    def __init__(self):
        self.stripper = MLStripper()

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

    def clean_html(self, text):
        html = ' '.join(text)
        self.stripper.feed(html)
        data = self.stripper.get_data()
        self.stripper.fed = []
        return data

    def track_progess(self, maxVal):
        return Bar('Processing', max=maxVal)


def main(file_path):

    # Initialize the json data and other variables
    u = Utils()
    json_art = u.load_json(file_path)
    num_examples = len(json_art)
    data = []
    new_name = '%s-cleaned' % file_path.split('.json')[0] + '.json'
    bar = u.track_progess(num_examples)

    # Iterste over the examples, clean the text and image lists/dicts
    for i in range(num_examples):
        example = json_art[i]
        text = example['text']
        images = example['images']
        cleaned_text = u.clean_html(text)
        cleaned_images = u.clean_images(images)
        new_example = {'text': cleaned_text, 'images': cleaned_images}
        data.append(new_example)
        bar.next()

    # Save new json & cleanup
    u.save_json(new_name, data)
    bar.finish()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('*'*20, '\nIncorrect number of arguemenets...')
        print('Example: python3 main.py filePath.json\n'+'*'*20)
    elif not os.path.exists(sys.argv[-1]):
        print('File %s does not exist.' % sys.argv[-1])
    else:
        main(sys.argv[-1])
