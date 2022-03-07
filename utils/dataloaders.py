import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import time

'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/label/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/label/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset


def full_path_loader_for_txt(train_txt_path, val_txt_path):
    train_dataset = {}
    val_dataset = {}

    with open(train_txt_path, "r") as f1:
        lines = f1.read().splitlines()
    train_A_path = []
    train_B_path = []
    train_label_path = []
    for index, line in enumerate(lines):
        path = line.split(' ')
        imageA = path[0]
        imageB = path[1]
        label = path[2]
        assert os.path.isfile(imageA)
        assert os.path.isfile(imageB)
        assert os.path.isfile(label)
        train_A_path.append(imageA)
        train_B_path.append(imageB)
        train_label_path.append(label)
        train_dataset[index] = {'imageA': train_A_path[index],
                                'imageB': train_B_path[index],
                                'label': train_label_path[index]}
    # print("train_ok")
    with open(val_txt_path, "r") as f2:
        lines = f2.read().splitlines()
    val_A_path = []
    val_B_path = []
    val_label_path = []
    for index, line in enumerate(lines):
        path = line.split(' ')
        imageA = path[0]
        imageB = path[1]
        label = path[2]
        assert os.path.isfile(imageA)
        assert os.path.isfile(imageB)
        assert os.path.isfile(label)
        val_A_path.append(imageA)
        val_B_path.append(imageB)
        val_label_path.append(label)
        val_dataset[index] = {'imageA': val_A_path[index],
                              'imageB': val_B_path[index],
                              'label': val_label_path[index]}
    # print("val_ok")
    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/label/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset


def full_test_loader_for_txt(test_txt_path):
    test_dataset = {}
    with open(test_txt_path, "r") as f:
        lines = f.read().splitlines()
    test_A_path = []
    test_B_path = []
    test_label_path = []
    for index, line in enumerate(lines):
        path = line.split(' ')
        imageA = path[0]
        imageB = path[1]
        label = path[2]
        # assert os.path.isfile(imageA)
        # assert os.path.isfile(imageB)
        # assert os.path.isfile(label)
        test_A_path.append(imageA)
        test_B_path.append(imageB)
        test_label_path.append(label)
        test_dataset[index] = {'imageA':test_A_path[index],
                               'imageB':test_B_path[index],
                               'label':test_label_path[index]}
    # print("test_ok")
    return test_dataset


def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


def cdd_loader_for_txt(imageA_path, imageB_path, label_path, opt, aug):
    img1 = Image.open(imageA_path)
    img2 = Image.open(imageB_path)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    # print(imageA_path)
    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    if opt.visual:
        name = imageA_path.split('/')[-1][0:-4]
        return sample['image'][0], sample['image'][1], sample['label'], name
    else:
        return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)

class CDDloader_for_txt(data.Dataset):

    def __init__(self, full_load, opt, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader_for_txt
        self.aug = aug
        self.opt = opt

    def __getitem__(self, index):

        imgA_path, imgB_path, label_path = self.full_load[index]['imageA'], self.full_load[index]['imageB'], self.full_load[index]['label']

        return self.loader(imgA_path,
                           imgB_path,
                           label_path,
                           self.opt,
                           self.aug,)

    def __len__(self):
        return len(self.full_load)
