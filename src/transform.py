from src.augmentation import *


class InputTransform:
    def __init__(self, input_size):
        self.data_transform = {
            "train": Compose(
                [
                    Resize(input_size),
                    Binarize(),
                    ToTensor(),
                ]
            ),
            "val": Compose(
                [
                    Resize(input_size),
                    Binarize(),
                    ToTensor(),
                ]
            ),
            "test": Compose(
                [
                    Binarize(),
                    ToTensor(),
                ]
            ),
        }

    def __call__(self, img, phase):
        out = self.data_transform[phase](img)

        return out


class OutputTransform:
    def __init__(self, input_size):
        self.data_transform = {
            "train": Compose(
                [
                    Resize(input_size),
                    ToTensor(),
                ]
            ),
            "val": Compose(
                [
                    Resize(input_size),
                    ToTensor(),
                ]
            ),
            "test": Compose(
                [
                    Resize(input_size),
                    ToTensor(),
                ]
            ),
        }

    def __call__(self, img, phase):
        out = self.data_transform[phase](img)

        return out
