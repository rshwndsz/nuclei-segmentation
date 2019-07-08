import torchvision.transforms as T

augmentation = T.Compose([
    T.RandomRotation(30),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# TODO Find better mean, std_dev for Normalization
# TODO See: https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/src/pre_processing.py
