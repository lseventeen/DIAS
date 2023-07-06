import models
def build_model(config):
    
    return getattr(models, config.MODEL.TYPE)(
        num_classes =  3,
        num_channels = 8
        )

