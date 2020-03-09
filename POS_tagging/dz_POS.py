from POS_utils import POS_Utils
import pdb

import json
class DZ_POS(POS_Utils):
    def __init__(self, file_path):
        POS_Utils.__init__(self, file_path)

    def run(self):
        for i in range(self.num_samples):
            sample = self.json[i]
            if self.has_text(sample):
                text = sample['text']
                sample['adjectives'] = self.parse_sample(text)
                self.BOW_adj.extend(sample['adjectives'])

            self.sample_data.append(sample)
            self.bar.next()

        adj_dict = dict()
        adj_dict['all_adj'] = self.BOW_unique()
        self.sample_data.append(adj_dict)
        # Save new json & cleanup
        self.save_json(self.new_name, self.sample_data)
        self.bar.finish()

# fur = '/Users/manuelladron/phd_cd/DL_11785/project/data_collection/fashion_dz-cleaned-adjectives.json'
#
#
#
# def load_json(file_path):
#     with open(file_path) as f:
#         # data = json.loads(json.load(f))
#         data = json.load(f)
#     f.close()
#     return data
#
#
# d = load_json(fur)
# print((d[-1].values()))
# furniture = DZ_POS(fur)
# furniture.run()

