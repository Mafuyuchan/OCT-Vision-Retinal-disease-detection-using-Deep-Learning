import os


def __init__(self, root, img_size=224, batch_size=32, num_workers=4):
self.root = root
self.img_size = img_size
self.batch_size = batch_size
self.num_workers = num_workers


def _pil_loader(self, path):
return Image.open(path).convert('RGB')


def get_transforms(self, split="train"):
if split == "train":
aug = A.Compose([
A.Resize(self.img_size, self.img_size),
A.HorizontalFlip(p=0.5),
A.RandomBrightnessContrast(p=0.2),
A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.2),
A.Normalize(),
ToTensorV2()
])
else:
aug = A.Compose([
A.Resize(self.img_size, self.img_size),
A.Normalize(),
ToTensorV2()
])
return aug


def _albumentations_loader(self, dataset_root):
# We will create a simple ImageFolder wrapper that applies albumentations
# Create torchvision transform stub to satisfy ImageFolder, then we apply alb augmentation separately in collate
transform = None
return ImageFolder(dataset_root, transform=transform)


def make_loaders(self):
train_root = os.path.join(self.root, 'train')
val_root = os.path.join(self.root, 'val')


def pil_to_tensor(img_path, aug):
img = self._pil_loader(img_path)
transformed = aug(image=np.array(img))
return transformed['image']


# Use ImageFolder but with custom loader using a lambda -> simpler: use torchvision transforms
from torchvision import transforms
train_transform = transforms.Compose([
transforms.Resize((self.img_size, self.img_size)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
transforms.Resize((self.img_size, self.img_size)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


train_ds = ImageFolder(train_root, transform=train_transform)
val_ds = ImageFolder(val_root, transform=val_transform)


train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class_names = train_ds.classes
return train_loader, val_loader, class_names
