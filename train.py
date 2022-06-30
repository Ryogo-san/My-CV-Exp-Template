import warnings

warnings.filterwarnings("ignore")
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.seed import seed_everything

from src.callbacks import get_my_callbacks
from src.datamodule import MyDataModule
from src.model_dispatcher import model_dispatcher
from src.utils import get_img_list


def log_artifact(logger, artifact_path):
    logger.experiment.log_artifact(logger.run_id, artifact_path)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    seed_everything(seed=config.seed)
    logger = instantiate(config.logger)
    trainer = instantiate(
        config.trainer, logger=logger, callbacks=get_my_callbacks(config)
    )

    # list
    input_list = get_img_list(config.data_dir, "input_whole")
    heatmap_list = get_img_list(config.data_dir, "heatmap_gray_whole")

    model = model_dispatcher(config.model_name, config)
    data = MyDataModule(input_list, heatmap_list, config)

    trainer.fit(model, data)

    logger.log_hyperparams(
        {
            "batch_size": config.batch_size,
            "lr": config.lr,
        }
    )

    log_artifact(logger, ".hydra/config.yaml")
    log_artifact(logger, ".hydra/hydra.yaml")
    log_artifact(logger, ".hydra/overrides.yaml")


if __name__ == "__main__":
    main()
