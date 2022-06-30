import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import pytorch_lightning as pl
from hydra.utils import instantiate

from utils import add_heatmap_to_orig, save_image


class SaveOutput:
    """Forward Hook"""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach())

    def clear(self):
        self.outputs = []


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        x, y_hat = batch
        y = self(x)
        loss = self.criterion(y, y_hat)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_hat = batch
        y = self(x)
        loss = self.criterion(y, y_hat)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch  # image
        y = self(x)
        orig = save_image(
            x[0, :, :, :].cpu(),
            self.cfg.output_dir + "input/" + f"{batch_idx}.png",
            "input",
        )
        _ = save_image(
            y[0, :, :, :].cpu(),
            self.cfg.output_dir + "heatmap_gray/" + f"{batch_idx}.png",
            "heatmap_gray",
        )
        heatmap = save_image(
            y[0, :, :, :].cpu(),
            self.cfg.output_dir + "heatmap/" + f"{batch_idx}.png",
            "heatmap",
        )
        add_heatmap_to_orig(
            orig,
            heatmap,
            os.path.join(self.cfg.output_dir, "blend", f"{batch_idx}.png"),
        )

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        """
        optimizer=get_optimizer(
                self.cfg.optimizer,self.parameters(),self.cfg.learning_rate
        )
        scheduler=get_scheduler(optimizer,self.cfg)
        """
        optimizer = instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)

        return [optimizer], [scheduler]
