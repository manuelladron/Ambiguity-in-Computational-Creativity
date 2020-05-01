from shutil import copyfile
from sys import exit
import json

curr_folder_des = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/design/design/thumbs/big"
dest_folder_des = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/design/design/thumbs/tagged"
curr_folder_arch = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/architecture/thumbs/big"
dest_folder_ach = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/architecture/thumbs/tagged"

class Utils():

    def __init__(self):
        print("Utils object created...")

    def open_file(self, path):
        with open(path) as f:
            output = json.load(f)
        f.close()
        return output

    def load_json(self, file_path):
        with open(file_path) as f:
            data = json.loads(json.load(f))
        f.close()
        return data

    def jsons_to_list(self, paths):
        """
        Returns a list of dictionaries corresponding to json files
        """
        data = []
        for i in range(len(paths)):  # list with all training data from different sections
            json_file = self.load_json(paths[i])
            print("Json_file: {}, length: {}, type: {}".format(i, len(json_file), type(json_file)))
            data.append(json_file)
        return data


    def flatten(self, S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return self.flatten(S[0]) + self.flatten(S[1:])
        return S[:1] + self.flatten(S[1:])



    def copy_files(self, files, curr_folder, dest_folder):
        """
        Files is a 2d list with file names
        """
        failed_samples = []
        for i in range(len(files)):
            for j in range(len(files[i])):
                image = files[i][j]
                image = image[4:] #remove full --> so they are /0030227348273427.jpg

                old_address = curr_folder + image
                new_address = dest_folder + image
                try:
                    copyfile(old_address, new_address)
                except IOError as e:
                    print("unable to copy file. %s" % e)
                    #exit(1)
                    failed_samples.append(image)
                    continue
                except:
                    print("Unexpected error: ", sys.exc_info())
                    exit(1)

        print("Done!")
        return failed_samples

    def change_file_name(self, files, new_name):
        """
        Files is a 2d list with file names
        """
        for i in range(len(files)):
            for j in range(len(files[i])):
                image = files[i][j]
                oldname = image[:5]
                image = image[4:]
                if oldname == new_name: return
                image = new_name + image
                files[i][j] = image

        print("Done!")


    def partition_numbers(self, percentage, datalen):
        train_set_n = int(datalen * percentage)
        val_set_n = datalen - train_set_n
        return train_set_n, val_set_n
