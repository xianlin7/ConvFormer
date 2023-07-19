# This file is used to configure the training parameters for each task

class Config_ACDC:
    data_path = "../../dataset/cardiac/"
    save_path = "./checkpoints/ACDC/"
    result_path = "./result/ACDC/"
    tensorboard_path = "./tensorboard/ACDC/"
    visual_result_path = "./Visualization/SEGACDC"
    load_path = "xxx"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 4                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "trainofficial"        # the file name of training set
    val_split = "valofficial"
    test_split = "testofficial"           # the file name of testing set
    crop = (256, 256)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ConvFormer"


class Config_ICH:
    data_path = "../../dataset/ICH/"
    save_path = "./checkpoints/ICH/"
    result_path = "./result/ICH/"
    tensorboard_path = "./tensorboard/ICH/"
    visual_result_path = "./Visualization/SEGICH"
    load_path = "./xxxx"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"        # the file name of training set
    val_split = "val"
    test_split = "test"           # the file name of testing set
    crop = (256, 256)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "ConvFormer"


class Config_ISIC:
    data_path = "../../dataset/ISIC/"
    save_path = "./checkpoints/ISIC/"
    result_path = "./result/ISIC/"
    tensorboard_path = "./tensorboard/ISIC/"
    visual_result_path = "./Visualization/SEGISIC"
    load_path = "./xxx"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "test"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "no"                 # the type of input image
    img_channel = 3              # the channel of input image
    eval_mode = "slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    

# ==================================================================================================
def get_config(task="Synapse"):
    if task == "ACDC":
        return Config_ACDC()
    elif task == "ISIC":
        return Config_ISIC()
    elif task == "ICH":
        return Config_ICH()
