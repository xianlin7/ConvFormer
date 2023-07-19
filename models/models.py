from imp import IMP_HOOK
from .SETR import Setr_ConvFormer, Setr, Setr_deepvit, Setr_cait, Setr_refiner



def get_model(modelname="Unet", img_size=256, img_channel=1, classes=9, assist_slice_number=4):
    if modelname == "SETR":
        model = Setr(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_deepvit":
        model = Setr_deepvit(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_cait":
        model = Setr_cait(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_refiner":
        model = Setr_refiner(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_ConvFormer":
        model = Setr_ConvFormer(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model