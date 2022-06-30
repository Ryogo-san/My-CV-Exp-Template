from src.models.custom_unet import *


def model_dispatcher(model_name, config):
    if model_name == "custom_unet":
        model = CustomUNet(config)
    elif model_name == "custom_unet_32":
        model = CustomUNet32(config)

    return model
