import random

import cv2
import numpy as np
import torch


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img


class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(
            np.expand_dims(cvimage.astype(np.float32), 2) / 255.0
        ).permute(2, 0, 1)


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))

        return image


class Binarize(object):
    def __call__(self, image):
        image = np.where(image >= 128, 255, 0)

        return image


class Quantization(object):
    def __init__(self, step=4):
        self.step = step

    def __call__(self, image):
        randNum = random.uniform(0, 100)
        if randNum >= 75:
            step_pix = 1.0 // self.step
            for i in range(self.step):
                image = np.where(
                    (image >= step_pix * i) & (image < step_pix * (i + 1)),
                    step_pix * i,
                    image,
                )

        return image
