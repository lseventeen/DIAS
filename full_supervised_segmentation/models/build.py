import models

model_2d = {
    "UNet",
    "FR_UNet",
    "Att_UNet",
    "CSNet",
    "UNet_Nested",
    "MAA_Net",
    "Res_UNet"
}

model_3d = {
    "UNet_3D",
    "FR_UNet_3D",
    "CSNet_3D",
    "Att_UNet_3D",
    "Res_UNet_3D",
    "UNet_Nested_3D"
}

model_3dto2d = {
    "IPN",
    "PSC"

}


def build_model(config):
    if config.MODEL.TYPE in model_2d:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=2,
            num_channels=8
        ), True
    elif config.MODEL.TYPE in model_3d or config.MODEL.TYPE in model_3dto2d:
        return getattr(models, config.MODEL.TYPE)(
            num_classes=2,
            num_channels=1
        ), False
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.TYPE}")
