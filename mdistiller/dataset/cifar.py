import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar100_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def get_cifar100_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def get_cifar10_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def get_cifar10_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def get_dataset(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        train_transform = get_cifar100_train_transform()
        test_transform = get_cifar100_test_transform()
        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )
        n_classes = 100
    elif cfg.DATASET.TYPE == "cifar10":
        train_transform = get_cifar10_train_transform()
        test_transform = get_cifar10_test_transform()
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        n_classes = 10
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.DATASET.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, test_loader, n_classes 