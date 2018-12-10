import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

# Constants
NUM_WORKERS = 8

# Transform for CIFAR10
cifar_transformer = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(28),
    transforms.ToTensor()])  # Tensor

# Transform for FashionMNIST
fashion_transformer = transform = transforms.Compose([
    transforms.ToTensor()])  # Tensor


class Triplet(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, ds):
        self.ds = ds.dataset
        self.train = self.ds.train
        self.transform = self.ds.transform
        self.indices = ds.indices.data.cpu().numpy().tolist() \
            if type(ds.indices) == torch.Tensor else ds.indices

        if self.train:
            self.train_labels = self.ds.train_labels.data.cpu().numpy().tolist() \
                if type(self.ds.train_labels) == torch.Tensor else self.ds.train_labels
            self.train_labels = np.take(self.train_labels, self.indices, axis=0)
            self.train_data = self.ds.train_data.data.cpu().numpy() \
                if type(self.ds.train_data) == torch.Tensor else self.ds.train_data
            self.train_data = np.take(self.train_data, self.indices, axis=0)
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.ds.test_labels.data.cpu().numpy().tolist() \
                if type(self.ds.test_labels) == torch.Tensor else self.ds.test_labels
            self.test_data = self.ds.test_data.data.cpu().numpy() \
                if type(self.ds.test_data) == torch.Tensor else self.ds.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[np.random.choice(
                             list(self.labels_set - set([self.test_labels[i]])))]), self.test_labels[i]] for i in
                        range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label])
            negative_label = np.random.choice(list(self.labels_set - set([label])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
            label = self.test_triplets[index][3]

        img1 = Image.fromarray(img1, mode=None)
        img2 = Image.fromarray(img2, mode=None)
        img3 = Image.fromarray(img3, mode=None)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), label

    def __len__(self):
        return len(self.indices)


def create_dataset(dataset, root, train, download):
    data_set = None
    if dataset == 'cifar':
        data_set = CIFAR10(root=root, train=train, download=download, transform=cifar_transformer)
    elif dataset == 'fashion':
        data_set = FashionMNIST(root=root, train=train, download=download, transform=fashion_transformer)
    else:
        pass
    return data_set


def fetch_train_dataloaders(dataset, data_dir, params, split=0.2):
    """
    Fetches dataloaders for train and test, choosing from the indicated datset.
    :param dataset: datset to be loaded (cifar|fashion).
    :param data_dir: path where images will be saved or, ifa already downloaded, current path of dataset
    :param params: hyperparameters
    :param split: portion of data used for validation (0-1)
    :return: train and validation dataloaders
    """

    # Load datasets, in case data does not exist then download it
    try:
        set = create_dataset(dataset, data_dir, True, False)
    except RuntimeError:
        set = create_dataset(dataset, data_dir, True, True)

    # Split
    val_size = int(split * len(set))
    train_size = len(set) - val_size
    train_set, val_set = random_split(set, [train_size, val_size])

    train_dl = DataLoader(Triplet(train_set), batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=params.cuda)
    val_dl = DataLoader(Triplet(val_set), batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=params.cuda)

    return train_dl, val_dl


def fetch_test_dataloader(dataset, data_dir, params):
    """
    Fetches dataloaders for test, choosing from the indicated dataset.
    :param dataset: datset to be loaded (cifar|fashion).
    :param data_dir: path where images will be saved or, ifa already downloaded, current path of dataset
    :param params: hyperparameters
    :return: test dataloader
    """

    # Load datasets, in case data does not exist then download it
    try:
        test_set = create_dataset(dataset, data_dir, False, False)
    except RuntimeError:
        test_set = create_dataset(dataset, data_dir, False, True)

    test_set, _ = random_split(test_set, [len(test_set), 0])
    test_dl = DataLoader(Triplet(test_set), batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                         pin_memory=params.cuda)

    return test_dl