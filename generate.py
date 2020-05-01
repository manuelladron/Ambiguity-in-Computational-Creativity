import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

cuda = torch.cuda.is_available()
cuda

def load_image(image_path, xform=None):
    image = Image.open(image_path)
    image = image.resize([64, 64], Image.LANCZOS)

    if xform is not None:
        image = xform(image).unsqueeze(0)

    return image


def word_ids2words(list_of_labels_ids):
    # Convert word_ids to words
    sampled_caption = []
    for word_id in list_of_labels_ids:
        word = VOCAB.idx2word[word_id]
        print("word: ", word)
        sampled_caption.append(word)
    #         if word == '<end>':
    #             break
    print("sampled caption: ", sampled_caption)
    sentence = ' '.join(sampled_caption)
    return sentence


def generate_caption_from_image(encoder, decoder, image_tensor):
    # Set model to eval
    encoder = encoder.eval()

    # Encode image
    feature = encoder(image_tensor)

    # Get captions
    sampled_ids = decoder.sample_topk(feature)
    print("sampled_ids: ", sampled_ids)
    # Convert tensor to numpy in cpu
    sampled_ids = sampled_ids[0]  # (1, max_seq_length) -> (max_seq_length)
    print("sampled_ids after moving to cpu and numpy: ", sampled_ids)
    return sampled_ids


def generate_labels(images):
    labels = []
    XFORMS = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomResizedCrop(100),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    # Load models
    encoder = EncoderCNN(64)
    enc_load = './saved_models/encoder-19.pth'
    dec_load = './saved_models/decoder-19.pth'
    encoder.load_state_dict(torch.load(enc_load, map_location=DEVICE))
    decoder.load_state_dict(torch.load(dec_load, map_location=DEVICE))

    for image_ in images:
        # Attach path to image
        #         image_name = dataset_new_folder + "/" + dataset.train_data[1]
        print("\nimage: ", image_)
        image_name = dataset_new_folder + "/" + image_
        print("image_name: ", image_name)
        # Load image and apply transformations
        image = load_image(image_name, XFORMS)

        # Move to proper device
        image_tensor = image.to(device)

        # Generate an caption from the image
        sampled_ids = generate_caption_from_image(encoder, decoder, image_tensor)

        # Decode ids to words
        caption = word_ids2words(sampled_ids)

        # Print out the image and the generated caption
        print("\nGenerated labels: ", caption)
        image = Image.open(image_name)
        plt.imshow(np.asarray(image))
        plt.show(image)
        # Print ground truth labels
        print("\nGround truth labels: ", dataset.labels[image_])


