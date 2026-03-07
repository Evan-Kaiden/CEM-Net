import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def get_dataloader(dataset, batch_size):
    if dataset == "cifar10":
        return Cifar10DataSet(batch_size)
    if dataset == "cifar100":
        return Cifar100DataSet(batch_size)
    else:
        raise NotImplementedError
    

class Cifar10DataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=False
        )

        self.classes = self.trainset.classes


class Cifar100DataSet():
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.mean = [0.5071, 0.4867, 0.4408]
        self.std  = [0.2675, 0.2565, 0.2761]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=False
        )

        self.classes = self.trainset.classes
