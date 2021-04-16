import numpy as np
import os
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from torchvision.datasets.utils import list_dir
import scipy.io as sio
from os.path import join
import pandas as pd

transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class AIRDateset(Dataset):
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True):
        self.train = train
        self.root = root
        self.class_type = 'variant'
        self.split = 'trainval' if self.train else 'test'
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.train:
            sample = transform_train(sample)
        else:
            sample = transform_test(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images

class CARDataSet(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.loader = default_loader
        self.train = train

        loaded_mat = sio.loadmat(os.path.join(self.root, "cars_annos.mat"))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.train:
            image = transform_train(image)
        else:
            image = transform_test(image)
        return image, target

    def __len__(self):
        return len(self.samples)

class CUBDataSet(Dataset):

    def __init__(self, root, train=True):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.train = train

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        img = default_loader(os.path.join(self.root, file_path))
        if self.train:
            img = transform_train(img)
            return img, target
        else:
            img = transform_test(img)
            return img, target

    def __len__(self):
        return len(self.imgs)

class DOGDateSet(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.loader = default_loader
        self.train = train

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.train:
            image = transform_train(image)
        else:
            image = transform_test(image)
        return image, target

    def load_split(self):
        if self.train:
            split = sio.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = sio.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = sio.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts

from config import HyperParams, root_dirs
def get_trainAndtest():
    kind = HyperParams['kind']
    root_dir = root_dirs[kind]
    if kind == 'bird':
        return CUBDataSet(root=root_dir, train=True), CUBDataSet(root=root_dir, train=False)
    elif kind == 'car':
        return CARDataSet(root=root_dir, train=True), CARDataSet(root=root_dir, train=False)
    elif kind == 'air':
        return AIRDateset(root=root_dir, train=True), AIRDateset(root=root_dir, train=False)
    elif kind == 'dog':
        return DOGDateSet(root=root_dir, train=True), DOGDateSet(root=root_dir, train=False)
    else:
        print("unsupported dataset")
        exit(0)