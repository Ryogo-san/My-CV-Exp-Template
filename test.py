import os
import warnings

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from src.config import CFG
from src.datamodule import MyDataModule
from src.models.unet import *
from src.utils import get_img_list


def main(cfg):
    seed_everything(cfg.seed)

    input_list = get_img_list(CFG.data_dir, "test")

    os.makedirs("./output/input", exist_ok=True)
    os.makedirs("./output/heatmap_gray", exist_ok=True)
    os.makedirs("./output/heatmap", exist_ok=True)
    os.makedirs("./output/blend", exist_ok=True)

    model = UNet(cfg)
    data = MyDataModule(input_list, None, cfg)

    ckpt_path = cfg.ckpt_path
    model = model.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=cfg)

    trainer = pl.Trainer(gpus=cfg.gpus, deterministic=True)

    trainer.test(model, data)


if __name__ == "__main__":
    main(CFG)
