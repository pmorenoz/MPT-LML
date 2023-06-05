# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import numpy as np
from scipy import sparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, MNIST, FashionMNIST, CIFAR10

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple


def load_dataset(args):
    label_scaler = None  # for regression datasets we return label scaler to be able to untransform data
    if args.dataset == 'mnist':
        ## MNIST // TRAIN=60.000, TEST=10.000
        if args.standarize:
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform=transforms.Compose([transforms.ToTensor()])

        mnist_train = MNIST(root='./data/', train=True, download=True, transform=transform)
        mnist_test = MNIST(root='./data/', train=False, download=True, transform=transform)

        if args.nof_observations < 60000:
            indices = list(range(args.nof_observations))
            mnist_train = torch.utils.data.Subset(mnist_train, indices)
        elif args.nof_observations > 60000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(mnist_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(mnist_test):
            test_loader = DataLoader(mnist_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(mnist_test, batch_size=len(mnist_test), pin_memory=True, shuffle=True)
        data_dimension = 784  # for folding later

    elif args.dataset == 'fmnist':
        ## FashionMNIST // TRAIN=60.000, TEST=10.000
        if args.standarize:
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform=transforms.Compose([transforms.ToTensor()])

        fmnist_train = FashionMNIST(root='./data/', train=True, download=True, transform=transform)
        fmnist_test = FashionMNIST(root='./data/', train=False, download=True, transform=transform)

        if args.nof_observations < 60000:
            indices = list(range(args.nof_observations))
            fmnist_train = torch.utils.data.Subset(fmnist_train, indices)
        elif args.nof_observations > 60000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(fmnist_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(fmnist_test):
            test_loader = DataLoader(fmnist_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(fmnist_test, batch_size=len(fmnist_test), pin_memory=True, shuffle=True)
        data_dimension = 784  # for folding later

    elif args.dataset == 'cifar':
        ## CIFAR10 // TRAIN=50.000, TEST=10.000
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # mean+std normalization
        cifar10_train = CIFAR10(root='./data/', train=True, download=False, transform=transform)
        cifar10_test = CIFAR10(root='./data/', train=False, download=False, transform=transform)

        if args.nof_observations < 50000:
            indices = list(range(args.nof_observations))
            cifar10_train = torch.utils.data.Subset(cifar10_train, indices)
        elif args.nof_observations > 50000:
            raise AssertionError('NOF Observations larger than dataset size')

        # Data Loaders / Train & Test
        data_loader = DataLoader(cifar10_train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        if args.nof_test < len(cifar10_test):
            test_loader = DataLoader(cifar10_test, batch_size=args.nof_test, pin_memory=True, shuffle=True)
        else:
            test_loader = DataLoader(cifar10_test, batch_size=len(cifar10_test), pin_memory=True, shuffle=True)
        data_dimension = 3072  # for folding later // CIFAR10 is 32x32x3

    else:
        raise NotImplementedError

    return data_loader, test_loader, data_dimension

def select_dataset(args):
    if args.dataset == 'mnist':
        data_loader, test_loader = mnist(args)
    elif args.dataset == 'fashion':
        data_loader, test_loader = fashion(args)
    else:
        print("Only mnist and fashion mnist implemented so far")
        raise TypeError("Wrong dataset choice. Use mnist or fashion")

    return data_loader,test_loader

class indexedMNIST(MNIST):
    def __init__(self,dtype,root: str,train: bool = True,transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,download: bool = False) -> None:

        super().__init__('./data/', train=train, download=download, transform=transform)
        self.dtype = dtype

    # Subclass of MNIST for returning the index
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.to(self.dtype), target, index


class indexedFashionMNIST(FashionMNIST):
    def __init__(self,dtype,root: str,train: bool = True,transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,download: bool = False) -> None:

        super().__init__('./data/', train=train, download=download, transform=transform)
        self.dtype = dtype

    # Subclass of MNIST for returning the index
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
   x         tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.to(self.dtype), target, index


def mnist(args):
    train_set = indexedMNIST(args.dtype,'./data/', train=True, download=True, transform=transforms.ToTensor())
    test_set = indexedMNIST(args.dtype,'./data/', train=False, download=True, transform=transforms.ToTensor())
    if args.nof_observations != 60000:

        # select N random numbers between 0 and 60000 without replacement
        indices = list(range(args.nof_observations))#random.sample(range(60000), args.nof_observations)#list(range(0, len(trainset), 2))
        train_set = torch.utils.data.Subset(train_set, indices)

    data_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,drop_last=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=True)
    #data_dimension = 784  # for folding later
    args.data_dim = 784
    return data_loader,test_loader

def fashion(args):
    train_set = indexedFashionMNIST(args.dtype,'./data/', train=True, download=True, transform=transforms.ToTensor())
    test_set = indexedFashionMNIST(args.dtype,'./data/', train=False, download=True, transform=transforms.ToTensor())

    if args.nof_observations != 60000:

        # select N random numbers between 0 and 60000 without replacement
        indices = list(range(args.nof_observations))#random.sample(range(60000), args.nof_observations)#list(range(0, len(trainset), 2))
        train_set = torch.utils.data.Subset(train_set, indices)

    data_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=True)
    args.data_dim = 784
    return data_loader,test_loader
