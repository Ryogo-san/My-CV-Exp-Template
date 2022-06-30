import cv2
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self, input_list, heatmap_list, mode, input_transform, output_transform
    ):
        self.input_list = input_list
        self.heatmap_list = heatmap_list
        self.mode = mode
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_path = self.input_list[idx]
        input_img = cv2.imread(input_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = np.abs(255 - input_img)

        if self.mode == "train" or self.mode == "val":
            input_img = self.input_transform(input_img, self.mode)

            heatmap_path = self.heatmap_list[idx]
            heatmap = cv2.imread(heatmap_path)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            heatmap = self.output_transform(heatmap, self.mode)

            return input_img, heatmap

        elif self.mode == "test":
            input_img = self.input_transform(input_img, self.mode)

            return input_img

        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
