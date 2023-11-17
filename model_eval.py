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




def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def model_build(model_name):
    if model_name in model_2d:
        return getattr(models, model_name)(
            num_classes=2,
            num_channels=8
        )
    else:
        return getattr(models, model_name)(
            num_classes=2,
            num_channels=1
        )
  


if __name__ == '__main__':

    print_model_parm_nums(model_build("VSS_Net"))
