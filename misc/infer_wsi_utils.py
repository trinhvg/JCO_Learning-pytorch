import os
import shutil  # High-level file operations
from itertools import chain
from sklearn.metrics import f1_score
import random
import cv2
import numpy as np
import torch.utils.data as data
from torchvision import transforms


def color_mask(a, r, g, b):
    ch_r = a[..., 0] == r
    ch_g = a[..., 1] == g
    ch_b = a[..., 2] == b
    return ch_r & ch_g & ch_b


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    return x


# to make it easier for visualization
def randomize_label(label_map):
    label_list = np.unique(label_map)
    label_list = label_list[1:]  # exclude the background
    label_rand = list(label_list)  # dup frist cause shuffle is done in place
    random.shuffle(label_rand)
    new_map = np.zeros(label_map.shape, dtype=label_map.dtype)


"""Recursive directory creation function. Like mkdir(), 
but makes all intermediate-level directories needed to contain the leaf directory.
A leaf is a node on a tree with no child nodes."""


def rm_n_mkdir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


###
# test

# import cv2
# import matplotlib.pyplot as plt
#
# img = cv2.imread('/media/vtltrinh/Data1/COLON_MANUAL_PATCHES/v1/1010711/000_3.jpg')
# im = np.array(img)
# im_mask = color_mask(im, 1, 1, 1)
#
# bound = bounding_box(im)
# print(bound)

def findExtension(directory, extension='.txt'):
    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files


def generate_patch_list_(roi, patch_size, stride):
    min_height, min_width, max_height, max_width = roi
    min_height, min_width, max_height, max_width = min_height - stride, min_width - stride, max_height + stride, max_width + stride
    h_list = np.arange(min_height, max_height - patch_size, stride)
    w_list = np.arange(min_width, max_width - patch_size, stride)
    out = [[[h_list[h], w_list[w]] for w in range(len(w_list))] for h in range(len(h_list))]
    return list(chain(*out))


def generate_patch_list(ano, roi, patch_size, stride):
    min_height, min_width, max_height, max_width = roi
    min_height, min_width, max_height, max_width = min_height - stride, min_width - stride, max_height + stride, max_width + stride
    h_list = np.arange(min_height, max_height - patch_size, stride)
    w_list = np.arange(min_width, max_width - patch_size, stride)
    out = [[[h_list[h], w_list[w]] for w in range(len(w_list))] for h in range(len(h_list))]
    path_list = list(chain(*out))
    # print(len(path_list))
    infer_dataset = DatasetSelectPatch(ano, path_list, patch_size)
    path_loader = data.DataLoader(infer_dataset, num_workers=31, batch_size=1144, shuffle=False, drop_last=False)
    for keeps, loca in path_loader:
        keeps_ = keeps.to('cuda')
        keeps_ += 1
        for idx in range(len(keeps)):
            if keeps[idx] == 1:
                a = eval(loca[idx])
                path_list.remove(a)
    # print('hi', len(path_list))
    return path_list


def read_ano_text(text_path):
    list_labels = {
        "BG": 0,
        "BN": 1,
        "WD": 2,
        "MD": 3,
        "PD": 4,
        "Ad": 5,
    }
    text_file = open(text_path, "r")
    lines = text_file.readlines()
    lines = [line.replace('\n', '').replace('\t', '') for line in lines]
    anos_dict = {}
    count_ROIs = np.zeros(shape=5, dtype=int)
    for label in list_labels:
        anos_dict.__setitem__(label, {})

    for line in lines[1:-1]:
        if line[1:3] in list_labels:
            label_id = line[1:3]
            coordinates = []
            count_ROIs[list_labels[label_id] - 1] += 1
            ROIs_id = count_ROIs[list_labels[label_id] - 1]
        else:
            if 'X' in line:
                dims_val = eval(line.replace("},", "}"))
                coordinates.append([int(dims_val[dim]) for dim in dims_val.keys()])
            else:
                anos_dict[label_id].__setitem__(ROIs_id, coordinates)

    keys_to_remove = ["BG", "Ad"]
    for key in keys_to_remove:
        del anos_dict[key]
    return anos_dict


def find_roi(anos_dict):
    min_height = []
    min_width = []
    max_height = []
    max_width = []
    valid_ano = ['BN', 'WD', 'MD', 'PD']
    for label_key in anos_dict.keys():
        if label_key in valid_ano:
            for polygon_key in anos_dict[label_key]:
                region = anos_dict[label_key][polygon_key]
                min_height.append(np.int32([region])[0, :, 1].min())  # np(height, width) while openslide (with,height)
                min_width.append(np.int32([region])[0, :, 0].min())
                max_height.append(np.int32([region])[0, :, 1].max())  # np(height, width) while openslide (with,height)
                max_width.append(np.int32([region])[0, :, 0].max())
    min_height = min(min_height)
    min_width = min(min_width)
    max_height = max(max_height)
    max_width = max(max_width)
    return [min_height, min_width, max_height, max_width]


def compute_f1(pred, ano):
    pred, ano = pred.flatten(), ano.flatten()
    pred = pred[ano != 0]
    ano = ano[ano != 0]
    f1 = f1_score(ano, pred, average='macro', labels=np.unique(ano))
    return int(f1 * 10000)


class DatasetSelectPatch(data.Dataset):
    def __init__(self, ano, path_list, patch_size):
        self.ano = ano
        self.path_list = path_list
        self.patch_size = patch_size

    def __getitem__(self, idx):
        w = self.path_list[idx][0]//16
        h = self.path_list[idx][1]//16
        patch_size = self.patch_size//16
        input_img = self.ano[w: w + patch_size, h: h + patch_size]

        if input_img.size == 0:
            keep = np.array([0])
        elif input_img.mean() > 0:
            keep = np.array([1])
        else:
            keep = np.array([0])
        return keep, str(self.path_list[idx])

    def __len__(self):
        return len(self.path_list)
