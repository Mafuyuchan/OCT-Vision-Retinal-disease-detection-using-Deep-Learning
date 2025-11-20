# src/data.py
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_transforms(img_size=224, augment=False):
    """
    Returns transforms for training and validation.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


def load_datasets(data_dir, img_size=224):
    """
    Loads train/val/test datasets using ImageFolder structure:
    
    data/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
        test/  (optional)
            class1/
            class2/
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augment=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(img_size, augment=False))

    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(img_size, augment=False))
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Returns DataLoaders for training, validation, and optional test set.
    """
    train_dataset, val_dataset, test_dataset = load_datasets(
        data_dir, img_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
