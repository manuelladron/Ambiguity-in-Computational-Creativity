from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

IMAGES_PATH = "/Users/manuelladron/phd_cd/DL_11785/project/Github_Ambiguity-in-Computational-Creativity-master/multilabel_classifier/dataset/images"

class ImageTextDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.image_labels_string = dataset.labels  # dictionary
        self.image_labels_int = dataset.labels_int
        self.image_names = dataset.train_data  # list with image_names
        self.name2idx()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.idx2name[index]
        #         name = dataset_new_folder + "/" + self.image_labels[self.idx2name[index]]
        name = IMAGES_PATH + "/" + self.image_names[index]
        img = Image.open(name)

        img = transforms.Compose([
            transforms.Resize((64, 64)),
            #         transforms.RandomResizedCrop(100),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])(img)

        #         img = transforms.ToTensor()(img)
        label = torch.from_numpy(self.image_labels_int[image_name])
        return img, label

    def name2idx(self):
        self.name2idx = dict()
        self.idx2name = dict()
        for i, key in enumerate(self.image_labels_string.keys()):
            self.name2idx[key] = i
            self.idx2name[i] = key


def collate(sequence):
    """
    "the input of this function is the output of function __getitem__"
    "this gets BATCH_SIZE times GETITEM! "
    if batch_Size == 2 --> sequence is a list with length 2.
    Each list is a tuple (image, label) = ((3,64,64), label_length)
    """

    """
    print("\nCollate function....")
    print("Sequence: ")
    print(len(sequence))
    print("seq[0] = ", sequence[0][0].shape, sequence[0][1].shape)
    print("")
    print("Seq[1] = ", sequence[1][0].shape, sequence[1][1].shape)
    """

    # Concatenate all images in the batch
    inputs = torch.cat(([batch_[0].view(-1, 3, 64, 64) for batch_ in sequence]), dim=0)

    # Pad labels with max_sequence_label
    targets = pad_sequence([batch_[1] for batch_ in sequence], batch_first=True)
    targets_length = torch.LongTensor([len(batch_[1]) for batch_ in sequence])

    #     print("\nInputs: {}\nTargets: {}\nTargets length:{}\n".format(len(inputs), targets.shape, targets_length))
    return inputs, targets, targets_length

def partition_numbers(percentage, datalen):
    train_set_n = int(datalen*percentage)
    val_set_n = datalen - train_set_n
    return train_set_n, val_set_n


"""
def getDataLoader(batch_size, num_workers, dataset, cuda, isTrain):


    number_train_samples, number_val_samples = partition_numbers(.8, len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [number_train_samples, number_val_samples])

    if cuda:
        loader_args = dict(shuffle=isTrain, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate)
    else:
        loader_args =  dict(shuffle=isTrain, batch_size=batch_size, collate_fn=collate)
    if isTrain:
        cur_dataset = ImageTextDataset(train_set)
    else:
        cur_dataset = ImageTextDataset(val_set)
    loader = DataLoader(cur_dataset, **loader_args)
    return loader

"""