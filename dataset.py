import os
import csv
import glob
import random
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from imgaug import augmenters as iaa

####


class DatasetSerial(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None, has_aux=False, test_aux=False):
        self.test_aux = test_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if not self.test_aux:

            # shape must be deterministic so it can be reused
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_img = shape_augs.augment_image(input_img)

            # additional augmenattion just for the input
            if self.input_augs is not None:
                input_img = self.input_augs.augment_image(input_img)

            input_img = np.array(input_img).copy()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
            ])

            out_img = np.array(transform(input_img)).transpose(1, 2, 0)
        else:
            out_img = []
            for idx in range(5):
                input_img_ = input_img.copy()
                if self.shape_augs is not None:
                    shape_augs = self.shape_augs.to_deterministic()
                    input_img_ = shape_augs.augment_image(input_img_)
                input_img_ = iaa.Sequential(self.input_augs[idx]).augment_image(input_img_)
                input_img_ = np.array(input_img_).copy()
                input_img_ = np.array(transform(input_img_)).transpose(1, 2, 0)
                out_img.append(input_img_)
        return np.array(out_img), img_label

    def __len__(self):
        return len(self.pair_list)


class DatasetSerialWSI(data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list

    def __getitem__(self, idx):
        input_img = cv2.imread(self.path_list[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.array(input_img).copy()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])
        input_img = np.array(transform(input_img)).transpose(1, 2, 0)
        location = self.path_list[idx].split('/')[-1].split('.')[0].split('_')
        return input_img, location

    def __len__(self):
        return len(self.path_list)

def prepare_colon_tma_data():
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = '/media/data1/member1/projects/workspace_data/COLON_MANUAL_512/COLON_MANUAL_512'

    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir, parse_label=True,
                               label_value=0)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017
    return train_set, valid_set, test_set


def prepare_colon_wsi_patch(data_visual=False):
    def load_data_info_from_list(data_dir, path_list):
        file_list = []
        for WSI_name in path_list:
            pathname = glob.glob(f'{data_dir}/{WSI_name}/*/*.png')
            file_list.extend(pathname)
            label_list = [int(file_path.split('_')[-1].split('.')[0]) - 1 for file_path in file_list]
        print(Counter(label_list))
        list_out = list(zip(file_list, label_list))
        return list_out

    data_root_dir = '/media/data1/trinh/data/workspace_data/colon_wsi/patches_colon_edit_MD/colon_45WSIs_1144_08_step05_05'
    data_visual = '/media/data1/trinh/data/workspace_data/colon_wsi/patches_colon_edit_MD/colon_45WSIs_1144_01_step05_visualize/patch_512/'

    df_test = [] #Note: Will be update later

    if data_visual:
        test_set = load_data_info_from_list(data_visual, df_test)
    else:
        test_set = load_data_info_from_list(data_root_dir, df_test)
    return test_set


def prepare_prostate_uhu_data():
    def load_data_info(pathname, parse_label=True, label_value=0, cancer_test=False):
        file_list = glob.glob(pathname)

        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = '/data1/trinh/data/patches_data/prostate_harvard/'
    data_root_dir_train = f'{data_root_dir}/train_validation_patches_750/'
    data_root_dir_test = f'{data_root_dir}/test_patches_750/'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_train)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    return train_set, valid_set, test_set


def prepare_prostate_ubc_data(fold_idx=0):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            label_dict = {2: 0, 3: 1, 4: 2}
            label_list = [label_dict[k] for k in label_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
            label_dict = {0: 0, 2: 1, 3: 2, 4: 3}
            label_list = [label_dict[k] for k in label_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    data_root_dir = '/data1/trinh/data/patches_data/'
    data_root_dir_train_ubc = f'{data_root_dir}/prostate_miccai_2019_patches_690_80_step05_test/'
    test_set_ubc = load_data_info('%s/*/*.jpg' % data_root_dir_train_ubc)
    return test_set_ubc


def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size




