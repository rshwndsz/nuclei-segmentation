# Custom dataset for the kidney dataset
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from utils.PIL_helpers import pil_loader
from config import config as cfg


class KidneyDataset(Dataset):

    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.labels_dir = os.path.join(root_dir, mode, 'labels') if self.mode != 'test' else None

        self.sample_paths = self.make_dataset(self.images_dir, self.labels_dir)
        self.transform = transform

    def __getitem__(self, idx):
        # TODO: Use OpenCV instead of PIL
        if self.mode == 'test':
            image = Image.open(self.sample_paths[idx]['image'])
            if self.transform is not None:
                image = self.transform(image)
            return {'image': image, 'label': None}
        elif self.mode == 'train' or self.mode == 'val':
            image = pil_loader(self.sample_paths[idx]['image'])
            label = pil_loader(self.sample_paths[idx]['label'], mode='L')
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(label)
            return {'image': image, 'label': label[0, :, :]}

    def __len__(self):
        return len(self.sample_paths)

    @staticmethod
    def make_dataset(images_dir, labels_dir=None):
        """
        Create (image, label) path pairs
        :param images_dir: Directory of images
        :param labels_dir: Directory of labels (None if 'test')
        :return: list of (image, label) paths (image paths if 'test')
        """
        sample_paths = []
        if labels_dir:
            for image, label in zip(os.listdir(images_dir), os.listdir(labels_dir)):
                sample_paths.append({
                    'image': os.path.join(images_dir, image),
                    'label': os.path.join(labels_dir, label)
                })
        else:
            for image in os.listdir(images_dir):
                sample_paths.append({
                    'image': os.path.join(images_dir, image),
                    'label': None
                })
        return sample_paths


# TODO: Add data augmentation
train_transforms = T.Compose([T.ToTensor()])
train_set = KidneyDataset(root_dir=cfg.dataset_root,
                          mode='train',
                          transform=train_transforms)
train_loader = DataLoader(train_set,
                          batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers,
                          shuffle=True)

val_transforms = T.Compose([T.ToTensor()])
val_set = KidneyDataset(root_dir=cfg.dataset_root,
                        mode='val',
                        transform=val_transforms)
val_loader = DataLoader(val_set,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        shuffle=False)

test_transforms = T.Compose([T.ToTensor()])
test_set = KidneyDataset(root_dir=cfg.dataset_root,
                         mode='test',
                         transform=test_transforms)
test_loader = DataLoader(test_set,
                         batch_size=cfg.batch_size,
                         num_workers=cfg.num_workers,
                         shuffle=False)
