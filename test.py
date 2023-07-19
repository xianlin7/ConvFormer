import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval

def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='MsRed', type=str, help='type of model')
    parser.add_argument('--task', default='ISIC', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    #opt.eval_mode = "patient_record"
    opt.save_path_code = "_plane_"
    opt.modelname = "TransUnet_CF"
    opt.mode = "eval"
    opt.load_path = "./checkpoints/ISIC/MsRed_+_03151848_4_0.902537761437633.pth"
    print(opt.load_path)

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================
    tf_test = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_test, opt.classes)  # return image, mask, and filename
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_path))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1)
    #dices, mean_dice, val_losses, rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3 = get_eval(testloader, model, criterion, opt)
    #dices, mean_dice, val_losses = get_eval(testloader, model, criterion, opt)
    if opt.mode == "train":
        dices, mean_dice, hds, val_losses = get_eval(testloader, model, criterion, opt)
        print(dices, mean_dice, hds)
    else:
        dice, iou, acc, se, sp = get_eval(testloader, model, criterion, opt)
        print(dice, iou, acc, se, sp)

    # ----------- keep the sesult into txt -------------------
    '''
    timestr0 = time.strftime('%m%d%H%M')
    record_path = "./records/ACDC/" + args.modelname + opt.save_path_code + timestr0 + '/'
    if not os.path.exists(record_path):
        os.mkdir(record_path)
        for i in range(6):
            os.mkdir(record_path + "sample" + str(i) + "/")
    for i in range(6): # the sample id 
        with open(record_path + "sample" + str(i) + "/" + "rtoken1.txt", 'a') as f:
            for j in range (rtoken1.shape[1]): 
                f.write(str(rtoken1[i, j]) + " ")
            f.write('\n')
        with open(record_path + "sample" + str(i) + "/" + "rtoken2.txt", 'a') as f:
            for j in range (rtoken2.shape[1]): 
                f.write(str(rtoken2[i, j]) + " ")
            f.write('\n')
        with open(record_path + "sample" + str(i) + "/" + "rtoken3.txt", 'a') as f:
            for j in range (rtoken3.shape[1]): 
                f.write(str(rtoken3[i, j]) + " ")
            f.write('\n')
        with open(record_path + "sample" + str(i) + "/" + "rmap1.txt", 'a') as f:
            for j in range (rmap1.shape[1]): 
                f.write(str(rmap1[i, j]) + " ")
            f.write('\n')
        with open(record_path + "sample" + str(i) + "/" + "rmap2.txt", 'a') as f:
            for j in range (rmap2.shape[1]): 
                f.write(str(rmap2[i, j]) + " ")
            f.write('\n')
        with open(record_path + "sample" + str(i) + "/" + "rmap3.txt", 'a') as f:
            for j in range (rmap3.shape[1]): 
                f.write(str(rmap3[i, j]) + " ")
            f.write('\n')
    '''


if __name__ == '__main__':
    main()
            


