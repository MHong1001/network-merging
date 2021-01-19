from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, KMNIST
from config import DATA_DIR

fmnist_classes = [
    "0 - T-shirt/top",
    "1 - Trouser",
    "2 - Pullover",
    "3 - Dress",
    "4 - Coat",
    "5 - Sandal",
    "6 - Shirt",
    "7 - Sneaker",
    "8 - Bag",
    "9 - Ankle boot",
]

kmnist_classes = [
    "0 - お",
    "1 - き",
    "2 - す",
    "3 - つ",
    "4 - な",
    "5 - は",
    "6 - ま",
    "7 - や",
    "8 - れ",
    "9 - を",
]


class ExtendedFMNIST(FashionMNIST):
    """
    FashionMNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedFMNIST, self).__init__(root, **kwargs)


class ExtendedKMNIST(KMNIST):
    """
    KMNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedKMNIST, self).__init__(root, **kwargs)
        extended_class_len = len(extended_classes)
        self.targets = [t + extended_class_len for t in self.targets]


# Rgb transform
transform_rgb = transforms.Lambda(lambda img: img.convert("RGB"))


fmnist_train_dataset = FashionMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
fmnist_test_dataset = FashionMNIST(
    DATA_DIR,
    train=False,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
kmnist_train_dataset = KMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
kmnist_test_dataset = KMNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


# Dataset for training PAN purpose
extended_fmnist_train_dataset = ExtendedFMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    extended_classes=kmnist_classes,
)

extended_kmnist_train_dataset = ExtendedKMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    extended_classes=fmnist_classes,
)


extended_fmnist_test_dataset = ExtendedFMNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    extended_classes=fmnist_classes,
)

extended_kmnist_test_dataset = ExtendedKMNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    extended_classes=kmnist_classes,
)


# Concat the datasets
fmnist_kmnist_train_dataset = ConcatDataset(
    [extended_fmnist_train_dataset, extended_kmnist_train_dataset]
)
fmnist_kmnist_test_dataset = ConcatDataset(
    [extended_fmnist_test_dataset, extended_kmnist_test_dataset]
)
