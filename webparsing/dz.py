from utils import Utils
import pdb

class dz(Utils):
    def __init__(self, file_path):
        Utils.__init__(self, file_path)

    def run(self):
        for i in range(self.num_examples):
            example = self.json[i]
            text = example['text']
            images = example['images']
            cleaned_text = self.get_quotes(text)
            cleaned_images = self.clean_images(images)
            new_example = {'text': cleaned_text, 'images': cleaned_images, 'title': example['title']}
            self.data.append(new_example)
            self.bar.next()

        # Save new json & cleanup
        self.save_json(self.new_name, self.data)
        self.bar.finish()
