import datetime
import os
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import torch
from tensorboardX import SummaryWriter
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import stats
import time


from skimage import io
from utils.transform import depadding
#################################
from datasets import Road_DX as RS
from models.RPMAN import RPMAN as Net
#from models.Unet import UNet as Net...
#################################
NET_NAME = 'RPMAN' #UNet/PSPNet/...
from utils.loss import CrossEntropyLoss2d
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter, CaclTP

working_path = os.path.abspath('.')
args = {
    'gpu': True,
    'batch_size': 1,
    #'net_name':'Unet',
    'net_name':'RPMAN',
    'load_path': working_path+'/checkpoints/RPMAN/best.pth'
}

def main():
    #net = Net(3, num_classes=1).cuda()   'DiResSeg'
    net = Net(num_classes=1).cuda()
    net.load_state_dict(torch.load(args['load_path']))
    net.eval()
    print('Model loaded.')
    pred_path = os.path.join(RS.root, 'pred', args['net_name'])
    pred_name_list = RS.get_file_name('val')
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')
        
    val_set = RS.RS('val')
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    predict(net, val_loader, pred_path, pred_name_list, f)

def predict(net, pred_loader, pred_path, pred_name_list, f_out=None):
    output_info = f_out is not None
    
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    
    total_iter = len(pred_name_list)
    for vi, data in enumerate(pred_loader):
        with torch.no_grad():
            imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
            outputs = net(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        outputs = F.sigmoid(outputs)
        outputs = outputs.detach().cpu().numpy()
        #_, pred = torch.max(output, dim=1)
        for i in range(args['batch_size']):
            idx = vi*args['batch_size']+i
            if (idx>=total_iter): break
            pred = outputs[i, 0]
            label = labels[i].detach().cpu().numpy()
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)
            pred_color = RS.Index2Color(pred)
            pred_name = os.path.join(pred_path, pred_name_list[idx]+'.png')
            print(pred_name)
            io.imsave(pred_name, pred_color)
            
            print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f'%(idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
            if output_info:
                f_out.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n'%(idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
    
    print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, mIoU %.2f'%(acc_meter.avg*100, precision_meter.avg*100, recall_meter.avg*100, F1_meter.avg*100, IoU_meter.avg*100))
    
    if output_info:
        f_out.write('Acc %.2f\n'%(acc_meter.avg*100))
        f_out.write('Avg Precision %.2f\n'%(precision_meter.avg*100))
        f_out.write('Avg Recall %.2f\n'%(recall_meter.avg*100))
        f_out.write('Avg F1 %.2f\n'%(F1_meter.avg*100))
        f_out.write('mIoU %.2f\n'%(IoU_meter.avg*100))
    return F1_meter.avg


import torch.nn as nn
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    main()
