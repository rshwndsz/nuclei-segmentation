# Custom dataset for the kidney dataset
import os
from torch.utils.data import Dataset
from PIL import Image


class KidneyDataset(Dataset):

    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.labels_dir = os.path.join(root_dir, mode, 'labels') if self.mode != 'test' else None

        self.data_names = make_dataset(self.images_dir, self.labels_dir)
        self.transform = transform

    def __getitem__(self, idx):
        if self.mode == 'test':
            image = Image.open(self.data_names[idx]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image
        elif self.mode == 'train' or self.mode == 'val':
            image = Image.open(self.data_names[idx]).convert('RGB')
            label = Image.open(self.data_names[idx]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(label)
            return image, label

    def __len__(self):
        return len(self.data_names)


def make_dataset(images_dir, labels_dir=None):
    """
    Create (image, label) path pairs
    :param images_dir: Directory of images
    :param labels_dir: Directory of labels (None if 'test')
    :return: list of (image, label) paths (image paths if 'test')
    """
    items = []
    if labels_dir:
        for image, label in zip(os.listdir(images_dir), os.listdir(labels_dir)):
            items.append((os.path.join(images_dir, image), os.path.join(labels_dir, label)))
    else:
        for image in os.listdir(images_dir):
            items.append(os.path.join(images_dir, image))
    return items
