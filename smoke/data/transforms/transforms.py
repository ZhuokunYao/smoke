import torch
from torchvision.transforms import functional as F


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
# 3
class To255():
    def __call__(self, image, target):
        image = image * 255.0
        return image, target
# 2
class ToBGR():
    def __call__(self, image, target):
        image = image[[2, 1, 0]]
        return image, target
# 1
class ToTensor():
    def __call__(self, image, target):
        return F.to_tensor(image), target

class NormalizeMeanStd():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class NormalizeMean():
    def __init__(self, factor, mean):
        self.mean = mean
        self.factor = factor

    def __call__(self, image, target):
        image *= self.factor
        image -= 0.5

        return image, target
