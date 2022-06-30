import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.dataset import MyDataset
from src.transform import *
from src.utils import *


class MyDataModule(pl.LightningDataModule):
    def __init__(self, input_list, heatmap_list, cfg):
        super().__init__()

        self.input_list = input_list
        self.heatmap_list = heatmap_list
        self.cfg = cfg
        self.input_transform = InputTransform(input_size=self.cfg.image_size)
        self.output_transform = OutputTransform(input_size=self.cfg.image_size)

    def setup(self, stage=None):
        """
        assign train/val datasets for use in dataloaders
        """
        if stage == "fit" or stage is None:
            num_of_images = len(self.input_list)
            train_size = int(0.8 * num_of_images)
            val_size = num_of_images - train_size

            dataset_full = MyDataset(
                self.input_list,
                self.heatmap_list,
                mode="train",
                input_transform=self.input_transform,
                output_transform=self.output_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                dataset_full, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = MyDataset(
                self.input_list,
                self.heatmap_list,
                mode="test",
                input_transform=self.input_transform,
                output_transform=None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset, 1, shuffle=False, num_workers=self.cfg.num_workers
        )
