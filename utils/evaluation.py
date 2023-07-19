# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from utils.record_tools import get_records, get_records2
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary

def eval_2d_slice(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        with torch.no_grad():
            predict = model(input_image)

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for i in range(0, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            dices[i] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[i] += iou
            accs[i] += acc
            ses[i] += se
            sps[i] += sp
            hds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        if opt.visual:
            visual_segmentation(seg, image_filename, opt)
    dices = dices / (batch_idx + 1)
    hds = hds / (batch_idx + 1)
    ious, accs, ses, sps = ious/(batch_idx + 1), accs/(batch_idx + 1), ses/(batch_idx + 1), sps/(batch_idx + 1)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp

def eval_slice_record(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    layer = 12
    num_head = 8
    rtoken1, rtoken2, rtoken3 = np.zeros((6, layer+1)), np.zeros((6, layer+1)), np.zeros((6, layer+1))
    rmap1, rmap2, rmap3 = np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head))
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        if batch_idx < 6:
            with torch.no_grad():
                _, ftokens, attmaps = model.infere(input_image)
            rtoken1[batch_idx, :], rtoken2[batch_idx, :], rtoken3[batch_idx, :], rmap1[batch_idx, :], rmap2[batch_idx, :], rmap3[batch_idx, :] = get_records(ftokens, attmaps, layer, num_head)
        with torch.no_grad():
            predict = model(input_image)   

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for i in range(0, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            dices[i] += metrics.dice_coefficient(pred_i, gt_i)
            del pred_i, gt_i
    dices = dices / (batch_idx + 1)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    return dices, mean_dice, val_losses, rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3

def eval_slice_visual(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    layer = 12
    num_head = 8
    rtoken1, rtoken2, rtoken3 = np.zeros((6, layer+1)), np.zeros((6, layer+1)), np.zeros((6, layer+1))
    rmap1, rmap2, rmap3 = np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head))
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        test_img_path = os.path.join(opt.data_path + '/img', image_filename)
        from utils.imgname import keep_img_name
        keep_img_name(test_img_path)

        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        if batch_idx < 6:
            with torch.no_grad():
                _, ftokens, attmaps = model.infere(input_image)
            rtoken1[batch_idx, :], rtoken2[batch_idx, :], rtoken3[batch_idx, :], rmap1[batch_idx, :], rmap2[batch_idx, :], rmap3[batch_idx, :] = get_records(ftokens, attmaps, layer, num_head)
        with torch.no_grad():
            predict = model(input_image)   

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for i in range(0, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            dices[i] += metrics.dice_coefficient(pred_i, gt_i)
            del pred_i, gt_i
    dices = dices / (batch_idx + 1)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    return dices, mean_dice, val_losses, rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3

def eval_2d_patient(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 2000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        with torch.no_grad():
            predict = model(input_image)
        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        patientid = int(image_filename[:4])
        flag[patientid] = 1
        for i in range(1, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            if hd > hds[patientid, i]:
                hds[patientid, i] = hd
            tps[patientid, i] += tp
            fps[patientid, i] += fp
            tns[patientid, i] += tn
            fns[patientid, i] += fn
        if opt.visual:
            visual_segmentation(seg, image_filename, opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :]
    patient_dices = 2 * tps / (2 * tps + fps + fns + 1e-6)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        smooth = 0.0001
        iou = (tps + smooth) / (fps + tps + fns + smooth)
        iou = np.mean(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc = np.mean(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se = np.mean(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp = np.mean(sp, axis=0)
        return mean_dice, np.mean(iou[1:], axis=0), np.mean(acc[1:], axis=0), np.mean(se[1:], axis=0), np.mean(sp[1:], axis=0)


def eval_patient_record(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    layer = 12
    num_head = 8 # 8
    rtoken1, rtoken2, rtoken3 = np.zeros((6, layer+1)), np.zeros((6, layer+1)), np.zeros((6, layer+1))
    rmap1, rmap2, rmap3 = np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head)), np.zeros((6, layer*num_head))
    patientnumber = 2000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        test_img_path = os.path.join(opt.data_path + '/img', image_filename)
        from utils.imgname import keep_img_name
        keep_img_name(test_img_path)
        if batch_idx < 6:
            with torch.no_grad():
                _, ftokens, attmaps = model.infere(input_image)
            rtoken1[batch_idx, :], rtoken2[batch_idx, :], rtoken3[batch_idx, :], rmap1[batch_idx, :], rmap2[batch_idx, :], rmap3[batch_idx, :] = get_records2(ftokens, attmaps, layer, num_head)
        with torch.no_grad():
            predict = model(input_image)   

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        patientid = int(image_filename[:4])
        flag[patientid] = 1
        for i in range(1, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            tps[patientid, i] += tp
            fps[patientid, i] += fp
            tns[patientid, i] += tn
            fns[patientid, i] += fn
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    patient_dices = 2 * tps / (2 * tps + fps + fns + 1e-6)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    return dices, mean_dice, val_losses, rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3


def eval_2d_patient2p5(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 200  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    for batch_idx, (input_image, ground_truth, mask_mini, assist_slices, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        with torch.no_grad():
            predict, _ = model(input_image)
        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        patientid = int(image_filename[:3])
        flag[patientid] = 1
        for i in range(1, opt.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            tps[patientid, i] += tp
            fps[patientid, i] += fp
            tns[patientid, i] += tn
            fns[patientid, i] += fn
    patients = np.sum(flag)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    patient_dices = 2 * tps / (2 * tps + fps + fns + 1e-6)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    return dices, mean_dice, val_losses

def get_eval(valloader, model, criterion, opt):
    if opt.eval_mode == "slice":
        return eval_2d_slice(valloader, model, criterion, opt)
    elif opt.eval_mode == "slice_record":
        return eval_slice_record(valloader, model, criterion, opt)
    elif opt.eval_mode == "slice_visual":
        return eval_slice_visual(valloader, model, criterion, opt)
    elif opt.eval_mode == "patient":
        return eval_2d_patient(valloader, model, criterion, opt)
    elif opt.eval_mode == "patient_record":
        return eval_patient_record(valloader, model, criterion, opt)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)