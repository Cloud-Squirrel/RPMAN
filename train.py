import datetime
import os
from math import sqrt
import numpy as np
import random
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


working_path = os.path.abspath('.')

import time
from skimage import io

#####################################################
from datasets import CTimg as RS
from models.Unet import UNet as Net   #import PSPNet/ SegNet/ deeplabv3_plus/ RPMAN...
NET_NAME = 'UNet'
#####################################################

# from utils.loss import CrossEntropyLoss2d
# from torch.nn.modules.loss import CrossEntropyLoss as CEloss
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter

args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'lr': 0.1,
    'epochs': 5000,
    'gpu': True,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 10,
    'predict_step': 5,
    'pred_dir': working_path + '/results/UNet/',
    'chkpt_path': working_path + '/checkpoints/UNet/' + NET_NAME,
    'log_dir': working_path + '/logs/UNet/' + NET_NAME + '/',
    'load_path': working_path + '/checkpoints/UNet/xxx.pth' ###
}
torch.cuda.empty_cache()
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    #net = Net(3, num_classes=1).cuda()
    net = Net(num_classes=1).cuda()
    train_set = RS.RS('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True,
                              drop_last=True)
    val_set = RS.RS('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    weight = torch.tensor([2.0], device='cuda')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
                                               gamma=0.7)

    train(train_loader, net, criterion, optimizer, scheduler, 0, args, val_loader)
    writer.close()
    print('Training finished.')
def train(train_loader, net, criterion, optimizer, scheduler, curr_epoch, train_args, val_loader):
    bestaccT = 0
    bestaccV = 0
    bestF_V = 0
    bestloss = 1
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_s = torch.nn.MSELoss().cuda()

    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        F1_meter = AverageMeter()
        train_main_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_learning_rate(optimizer, running_iter, all_iters, args)
            imgs, labels = data
            #print(imgs.shape)
            #print(labels.shape)

            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.unsqueeze(1).cuda().float()
            labels_s = F.interpolate(labels, scale_factor=1 / 8, mode='area')

            optimizer.zero_grad()
            #outputs, aux_s = net(imgs) #DAMM_ASPP_Resnet
            outputs= net(imgs)
            loss_main = criterion(outputs, labels)
            #loss_aux = criterion_s(F.sigmoid(aux_s), labels_s)
            #loss = loss_main + loss_aux * 0.5
            loss = loss_main
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            probs = torch.sigmoid(outputs)
            preds = outputs.cpu().detach().numpy()
            F1_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, _ = accuracy(pred, label)
                if F1 > 0: F1_curr_meter.update(F1)
            if F1_curr_meter.avg is not None:
                F1_meter.update(F1_curr_meter.avg)
            else:
                F1_meter.update(0)
            train_main_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f F1 %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_main_loss.val, F1_meter.avg * 100))
                writer.add_scalar('train loss', train_main_loss.val, running_iter)
                loss_rec = train_main_loss.val
                writer.add_scalar('train F1', F1_meter.avg, running_iter)
                # writer.add_scalar('train_aux_loss', train_aux_loss.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        acc_v, F_v, loss_v = validate(val_loader, net, criterion, curr_epoch, train_args)
        if F1_meter.avg > bestaccT: bestaccT = F1_meter.avg
        #if F_v > bestF_V:
        if F_v > bestF_V:
            bestF_V = F_v
            bestaccV = acc_v
            bestloss = loss_v
        if curr_epoch%3==0:
            torch.save(net.state_dict(), args['chkpt_path'] + '_%de_acc%.2f_F%.2f_loss%.4f.pth' % (
            curr_epoch, acc_v * 100, F_v * 100, loss_v))
        print('Total time: %.1fs Best rec: Train %.2f, Val acc %.2f, F1 %.2f, Val_loss %.4f' % (
        time.time() - begin_time, bestaccT * 100, bestaccV * 100, bestF_V * 100, bestloss))
        curr_epoch += 1
        # scheduler.step()
        if curr_epoch >= train_args['epochs']:
            return

from torchvision.utils  import save_image
def seeaspng(seeaspngi,outputs):
    #preds = torch.argmax(outputs, dim=1)
    #pred_first = preds[0].unsqueeze(0).float()     # 转换为4D张量 (1, C, H, W)
    # 1. logits -> probability
    probs = torch.sigmoid(outputs)
    # 2. threshold to binary mask (for visualization only)
    pred_first = (probs[0] > 0.5).float()   # shape: [1, H, W]
    print("save image")
    save_image(pred_first, 
           "outputs\output_image%d.png" %seeaspngi,  
           normalize=True,  # 若输出值范围不在[0,1]，需启用归一化
           nrow=1,          # 每行显示1张图片
           padding=0)
    return

def validate(val_loader, net, criterion, curr_epoch, train_args):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    Acc_meter = AverageMeter()
    seeaspngi=0

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if train_args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.unsqueeze(1).cuda().float()
        labels_s = F.interpolate(labels, scale_factor=1/8, mode='area')

        with torch.no_grad():
            #outputs, aux = net(imgs)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            if seeaspngi%10 == 0:  #这里
                seeaspng(seeaspngi,outputs)
            seeaspngi+=1
        val_loss.update(loss.cpu().detach().numpy())

        labels = labels.cpu().detach().numpy()
        # _, preds = torch.max(outputs, dim=1)
        outputs = torch.sigmoid(outputs)
        preds = outputs.cpu().detach().numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, _ = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_color = RS.Index2Color(preds[0, 0, :, :] > 0.5)
            io.imsave(args['pred_dir'] + NET_NAME + '.png', pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f acc: %.2f F1: %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100))
    # print('%.1fs Val  acc: %.2f F1: %.2f'%(curr_time,  Acc_meter.average()*100, F1_meter.average()*100))

    print('')
    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_F1', F1_meter.average(), curr_epoch)
    writer.add_scalar('val_acc', Acc_meter.average(), curr_epoch)

    return Acc_meter.avg, F1_meter.avg, val_loss.avg


def adjust_learning_rate(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 1.5)
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
