import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
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
    parser.add_argument('--modelname', default='SETR_ConvFormer', type=str, help='type of model')
    parser.add_argument('--task', default='ISIC', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.save_path_code = "_"

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + timestr
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 30  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================
    tf_train = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, 
                                color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes)
    model.to(device)
    if opt.pre_trained:
        model.load_state_dict(torch.load(opt.load_path))

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # if torch.cuda.device_count() > 1:  # distributed parallel training
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model, device_ids = [0,1]).cuda()

    #  ========================================== begin to train the model =============================================

    best_dice, loss_log = 0.0, np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (input_image, ground_truth, *rest) in enumerate(trainloader):
            input_image = Variable(input_image.to(device=opt.device))
            ground_truth = Variable(ground_truth.to(device=opt.device))
            # ---------------------------------- forward ----------------------------------
            output = model(input_image)
            train_loss = criterion(output, ground_truth)
            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)
        #  ---------------------------- log the train progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
        loss_log[epoch] = train_losses / (batch_idx + 1)
        #  ----------------------------------- evaluate -----------------------------------
        if epoch % opt.eval_freq == 0:
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion, opt)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            print("dice of each class:", dices)
            TensorWriter.add_scalar('val_loss', val_losses, epoch)
            TensorWriter.add_scalar('dices', mean_dice, epoch)
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
            


