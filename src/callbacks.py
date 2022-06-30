from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class MyPrintCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_epoch_end(self, trainer, pl_module):
        pass


class MyProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar


def get_my_callbacks(cfg):
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.patience, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_dir,
        filename="best_model",
    )

    print_callback = MyPrintCallback()

    bar = MyProgressBar(refresh_rate=5, process_position=1)

    return [early_stopping_callback, checkpoint_callback, print_callback, bar]
