"""Dataset loaders for CIFAR-10, Fashion-MNIST and MNIST.

MNIST-like datasets are padded from 28x28 to 32x32 (classic LeNet setup) and
stay single-channel. Returns seeded train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

DATASETS = {
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "classes": ("airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"),
    },
    "fashionmnist": {
        "num_classes": 10,
        "in_channels": 1,
        "image_size": 32,
        "mean": (0.2860,),
        "std": (0.3530,),
        "classes": ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"),
    },
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "image_size": 32,
        "mean": (0.1307,),
        "std": (0.3081,),
        "classes": ("digit 0", "digit 1", "digit 2", "digit 3", "digit 4",
                    "digit 5", "digit 6", "digit 7", "digit 8", "digit 9"),
    },
}

_ALIASES = {"fashion_mnist": "fashionmnist", "fashion-mnist": "fashionmnist"}


def dataset_info(name):
    name = _ALIASES.get(name.lower(), name.lower())
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    info = dict(DATASETS[name])
    info["name"] = name
    return info


def _build_transforms(name, train, with_aug):
    info = dataset_info(name)
    tfms = []
    if info["in_channels"] == 1:
        tfms.append(transforms.Pad(2))  # 28x28 -> 32x32
    if train and with_aug:
        if name == "cifar10":
            tfms += [
                transforms.RandomCrop(info["image_size"], padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            tfms.append(
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    tfms += [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    return transforms.Compose(tfms)


def _dataset_cls(name):
    return {
        "cifar10": datasets.CIFAR10,
        "fashionmnist": datasets.FashionMNIST,
        "mnist": datasets.MNIST,
    }[dataset_info(name)["name"]]


def get_dataloaders(cfg, seed=42, batch_size=128, num_workers=2, with_aug=True,
                    pin_memory=None, hw=None):
    """Build train/val/test loaders from the dataset section of a config.

    Config keys (under cfg['dataset']): name, root, val_fraction,
    max_train_samples (optional, for smoke tests / scaling studies).

    hw: optional HardwareManager; when given, it decides the effective
    batch size, worker count and pin_memory from the detected hardware.
    """
    if hw is not None:
        batch_size = hw.batch_size(batch_size)
        num_workers = hw.num_workers
        pin_memory = hw.pin_memory
    dcfg = cfg["dataset"] if "dataset" in cfg else cfg
    name = dataset_info(dcfg["name"])["name"]
    root = dcfg.get("root", "./data")
    val_fraction = float(dcfg.get("val_fraction", 0.1))
    max_train = dcfg.get("max_train_samples", None)

    cls = _dataset_cls(name)
    train_full = cls(root, train=True, download=True,
                     transform=_build_transforms(name, True, with_aug))
    test_set = cls(root, train=False, download=True,
                   transform=_build_transforms(name, False, False))

    n_total = len(train_full)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=gen)

    if max_train is not None and int(max_train) < n_train:
        gen2 = torch.Generator().manual_seed(seed + 1)
        idx = torch.randperm(n_train, generator=gen2)[: int(max_train)].tolist()
        train_set = Subset(train_set, idx)

    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    eval_bs = max(512, batch_size)
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin,
                            drop_last=False),
        "val": DataLoader(val_set, batch_size=eval_bs, shuffle=False,
                          num_workers=num_workers, pin_memory=pin),
        "test": DataLoader(test_set, batch_size=eval_bs, shuffle=False,
                           num_workers=num_workers, pin_memory=pin),
    }
    return loaders
