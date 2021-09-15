import os
import glob
import shutil

from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import scipy.io as scio
import pandas as pd

target_root = "dataset"
target_ann = os.path.join(target_root, "Annotations")
if not os.path.exists(target_ann):
    os.makedirs(target_ann)

target_image = os.path.join(target_root, "JPEGImages")
if not os.path.exists(target_image):
    os.makedirs(target_image)

data = pd.read_csv('/Users/alex/Downloads/steel/train.csv')
data['ClassId'] = data['ClassId'].astype(np.uint8)

squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# squash multiple rows per image into a list
squashed = (
    data[['ImageId', 'EncodedPixels', 'ClassId']]
        .groupby('ImageId', as_index=False)
        .agg(list)
)

# count the amount of class labels per image
squashed['DistinctDefectTypes'] = squashed['ClassId'].apply(lambda x: len(x))

def rle_to_mask(lre, shape=(1600, 256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return

    returns: numpy array with dimensions of shape parameter
    '''
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])

    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1

    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]

    # build the mask
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1

    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)

def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels,
        and turns them into a 3d numpy array of shape (256, 1600, 4)
    """

    # initialise an empty numpy array
    mask = np.zeros((256, 1600, 5), dtype=np.uint8)

    # building the masks
    for rle, label in zip(encodings, labels):
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label

        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for
        # numpy and openCV handling width and height in reverse order
        mask[:, :, index] = rle_to_mask(rle).T

    return mask

def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map

color_map = get_color_map_list(256)
from PIL import Image

for i in range(squashed.shape[0]):
    row = squashed.iloc[i].values
    classes = row[2]
    mask = build_mask(encodings=row[1], labels=row[2])
    mask = np.argmax(mask, axis=-1)
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    lbl_pil.putpalette(color_map)
    label_name = os.path.join(target_ann, row[0])
    parts = label_name.split('.')
    label_name = parts[0] + '.png'
    lbl_pil.save(label_name)
    shutil.copy(os.path.join('/Users/alex/Downloads/steel/train_images/', row[0]),
                os.path.join(target_image, row[0]))
