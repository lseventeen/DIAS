import models

def build_model(config):
    
    return getattr(models, config.MODEL.TYPE)(
        num_classes =  2,
        num_channels = 8
        )


 