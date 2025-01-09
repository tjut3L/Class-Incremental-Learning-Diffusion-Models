import numpy as np
import os
import sys
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from .autoaugment import CIFAR10Policy
from .ops import Cutout
import json

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./CIFAR10", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./CIFAR10", train=False, download=True)
        self.train_data, self.train_targets, self.train_labels = train_dataset.data, np.array(train_dataset.targets), train_dataset.classes
        self.test_data, self.test_targets, self.test_labels = test_dataset.data, np.array(test_dataset.targets), test_dataset.classes


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        Cutout(n_holes=1, length=16)
    ]
    test_trsf = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
        ]
    common_trsf = []

    class_order = np.arange(100).tolist()

    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
    
    def get_coarselabels(self,entry,coarse_label_names):
        coarse_labels=[]
        for i in range(100):
           fine_index = entry["fine_labels"].index(i)
           coarse_index = entry["coarse_labels"][fine_index]
           coarse_labels.append(coarse_label_names[coarse_index].replace("_"," "))
        return coarse_labels

    def download_data(self):
        path = "/home/HDD2/jskj_taozhe/CIFAR100/"
        train_dataset = datasets.cifar.CIFAR100(path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(path, train=False, download=True)
        entry = self.unpickle(path+"cifar-100-python/train")
        meta = self.unpickle(path+"cifar-100-python/meta")
        self.coarse_labels = self.get_coarselabels(entry,meta["coarse_label_names"])
        # self.train_data, self.train_targets, self.train_labels = train_dataset.data, np.array(
        #     train_dataset.targets
        # )
        # self.test_data, self.test_targets = test_dataset.data, np.array(
        #     test_dataset.targets
        # )
        self.train_data, self.train_targets, self.train_labels = train_dataset.data, np.array(train_dataset.targets), train_dataset.classes
        self.test_data, self.test_targets, self.test_labels = test_dataset.data, np.array(test_dataset.targets), test_dataset.classes


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/HDD2/jskj_taozhe/imagenet_subset/train"
        test_dir = "/home/HDD2/jskj_taozhe/imagenet_subset/val"

        json_file_path = 'data/class_label.json'

        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
        self.train_labels = self.test_labels = json_data['imagenet_subset_classes']
        superclass = json_data['imagenet_subset_superclass']
        coarse_labels = json_data['imagenet_coarse_labels']
        self.coarse_labels = [superclass[idx] for idx in coarse_labels]


class tinyImageNet(object):
    use_path = True
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(64),
        transforms.CenterCrop(56),
    ]
    common_trsf = [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join('/home/HDD2/tjut_sunxiaopeng', 'tiny-imagenet-200', 'train')
        test_dir = os.path.join('/home/HDD2/tjut_sunxiaopeng', 'tiny-imagenet-200', 'val')
        train_dset = datasets.ImageFolder(train_dir)

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
        self.train_data, self.train_targets = np.array(train_images), np.array(train_labels)

        test_images = []
        test_labels = []
        _, class_to_idx = find_classes(train_dir)
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        self.test_data, self.test_targets = np.array(test_images), np.array(test_labels)


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx