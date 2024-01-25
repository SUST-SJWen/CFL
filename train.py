import argparse
from asyncore import write
from copy import deepcopy
from decimal import ConversionSyntax
import logging
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFilter
from scipy.fft import fft2,fftshift
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from networks.vision_transformer import SwinUnet as ViT_seg

from val import test_myHiFormer_single_volume, test_single_volume
from dataloaders.dataset import (BaseDataSets_FLARE22, RandomGenerator_slice, TwoStreamBatchSampler, BaseDataSets)
from networks.net_factory import net_factory,BCP_net
import losses, ramps
from config import get_config
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/BCP_train_my_twoUnet', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--model1', type=str, default='unet', help='model_name')
parser.add_argument('--model2', type=str, default='enet', help='model_name')
parser.add_argument('--model_encoder', type=str,
                    default='myHiFormer', help='model_name')
parser.add_argument('--model_decoder', type=str,
                    default='unet_decoder', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

args = parser.parse_args()
config = get_config(args)
dice_loss = losses.DiceLoss(n_classes=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state)

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def generate_mask_part(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/3), int(img_y*1/3)
    w1 = np.random.randint(0, img_x - patch_x)
    h1 = np.random.randint(0, img_y - patch_y)
    w2 = np.random.randint(0, img_x - patch_x)
    h2 = np.random.randint(0, img_y - patch_y)
    w3 = np.random.randint(0, img_x - patch_x)
    h3 = np.random.randint(0, img_y - patch_y)
    w4 = np.random.randint(0, img_x - patch_x)
    h4 = np.random.randint(0, img_y - patch_y)
    mask[w1:w1+patch_x, h1:h1+patch_y] = 0
    loss_mask[:, w1:w1+patch_x, h1:h1+patch_y] = 0
    mask[w2:w2 + patch_x, h2:h2 + patch_y] = 0
    loss_mask[:, w2:w2 + patch_x, h2:h2 + patch_y] = 0
    mask[w3:w3 + patch_x, h3:h3 + patch_y] = 0
    loss_mask[:, w3:w3 + patch_x, h3:h3 + patch_y] = 0
    mask[w4:w4 + patch_x, h4:h4 + patch_y] = 0
    loss_mask[:, w4:w4 + patch_x, h4:h4 + patch_y] = 0
    return mask.long(), loss_mask.long()

def generate_mask_2x(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/2), int(img_y*1/2)
    w1 = np.random.randint(0, int(img_x - patch_x)/2)
    w2 = np.random.randint(int(img_x - patch_x)/2, img_x - patch_x)
    h1 = 0
    h2 = img_x - patch_x
    mask[w1:w1+patch_x, h1:h1+patch_y] = 0
    mask[w2:w2 + patch_x, h2:h2 + patch_y] = 0
    loss_mask[:, w1:w1+patch_x, h1:h1+patch_y] = 0
    loss_mask[:, w2:w2 + patch_x, h2:h2 + patch_y] = 0
    return mask.long(), loss_mask.long()
def generate_mask_4x(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/4), int(img_y*1/2)
    w1 = np.random.randint(int(img_x - patch_x)/4, int(img_x - patch_x)*3/4)
    w2 = np.random.randint(int(img_x - patch_x) / 4, int(img_x - patch_x) * 3 / 4)
    w3 = np.random.randint(int(img_x - patch_x) / 4, int(img_x - patch_x) * 3 / 4)
    w4 = np.random.randint(int(img_x - patch_x) / 4, int(img_x - patch_x) * 3 / 4)
    h1 = 0
    h2 = int(img_x*1/4)
    h3 = int(img_x * 1 / 4)*2
    h4 = int(img_x * 1 / 4)*3
    mask[w1:w1+patch_x, h1:h1+patch_y] = 0
    mask[w2:w2 + patch_x, h2:h2 + patch_y] = 0
    mask[w3:w3 + patch_x, h3:h3 + patch_y] = 0
    mask[w4:w4 + patch_x, h4:h4 + patch_y] = 0
    loss_mask[:, w1:w1+patch_x, h1:h1+patch_y] = 0
    loss_mask[:, w2:w2 + patch_x, h2:h2 + patch_y] = 0
    loss_mask[:, w3:w3 + patch_x, h3:h3 + patch_y] = 0
    loss_mask[:, w4:w4 + patch_x, h4:h4 + patch_y] = 0
    return mask.long(), loss_mask.long()
def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask

    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


# def patients_to_slices(dataset, patiens_num):
#     ref_dict = {"-1": 0.01, "0": 0.05, "1": 0.1, "2": 0.2,
#                 "3": 0.3, "4": 0.4, "5": 0.5, "6": 0.6, "7": 0.7, "8": 0.8, "9": 0.9, "10": 1}
#     return ref_dict[str(patiens_num)]
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    def create_model(ema=False, net_type=args.model):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = BCP_net(in_chns=1, class_num=num_classes)
    model2 = BCP_net(in_chns=1, class_num=num_classes)
    # model1 = create_model(net_type=args.model1)
    # model2 = create_model(net_type=args.model2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator_slice(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labeled_num)
    # labeled_slice = int(total_slices * patients_to_slices(args.root_path, args.labeled_num))
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                                  momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                                   momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model1.train()
    model2.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_sa = deepcopy(img_a)
            for b in range(img_sa.size(0)):
                img_s = transforms.ToPILImage()(img_sa[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_sa[b] = transforms.ToTensor()(img_s)

            img_sb = deepcopy(img_b)
            for b in range(img_sb.size(0)):
                img_s = transforms.ToPILImage()(img_sb[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_sb[b] = transforms.ToTensor()(img_s)

                img_maska, loss_maska = generate_mask_part(img_a)
                img_maskb, loss_maskb = generate_mask_part(img_b)
                gt_mixla = lab_a * img_maska + lab_b * (1 - img_maska)
                gt_mixlb = lab_b * img_maskb + lab_a * (1 - img_maskb)
                img_mask_sa, loss_mask_sa = generate_mask_part(img_sa)
                img_mask_sb, loss_mask_sb = generate_mask_part(img_sb)
                gt_mixl_sa = lab_a * img_mask_sa + lab_b * (1 - img_mask_sa)
                gt_mixl_sb = lab_b * img_mask_sb + lab_a * (1 - img_mask_sb)

            #-- original
            net_inputa = img_a * img_maska + img_b * (1 - img_maska)
            net_inputb = img_b * img_maskb + img_a * (1 - img_maskb)
            out_mixla1 = model1(net_inputa)
            out_mixlb1 = model1(net_inputb)
            out_mixla2 = model2(net_inputa)
            out_mixlb2 = model2(net_inputb)

            net_input_sa = img_sa * img_mask_sa + img_sb * (1 - img_mask_sa)
            net_input_sb = img_sb * img_mask_sb + img_sa * (1 - img_mask_sb)
            out_mixl_sa1 = model1(net_input_sa)
            out_mixl_sb1 = model1(net_input_sb)
            out_mixl_sa2 = model2(net_input_sa)
            out_mixl_sb2 = model2(net_input_sb)

            loss_dicea, loss_cea = mix_loss(out_mixla1, lab_a, lab_b, loss_maska, u_weight=1.0, unlab=True)
            loss_diceb, loss_ceb = mix_loss(out_mixlb1, lab_b, lab_a, loss_maskb, u_weight=1.0, unlab=True)
            loss_dicea_T2, loss_cea_T2 = mix_loss(out_mixla2, lab_a, lab_b, loss_maska, u_weight=1.0, unlab=True)
            loss_diceb_T2, loss_ceb_T2 = mix_loss(out_mixlb2, lab_b, lab_a, loss_maskb, u_weight=1.0, unlab=True)

            loss_dice_sa, loss_ce_sa = mix_loss(out_mixl_sa1, lab_a, lab_b, loss_mask_sa, u_weight=1.0, unlab=True)
            loss_dice_sb, loss_ce_sb = mix_loss(out_mixl_sb1, lab_b, lab_a, loss_mask_sb, u_weight=1.0, unlab=True)
            loss_dice_sa_T2, loss_ce_sa_T2 = mix_loss(out_mixl_sa2, lab_a, lab_b, loss_mask_sa, u_weight=1.0, unlab=True)
            loss_dice_sb_T2, loss_ce_sb_T2 = mix_loss(out_mixl_sb2, lab_b, lab_a, loss_mask_sb, u_weight=1.0, unlab=True)

            lossa_T2 = (loss_dicea_T2 + loss_cea_T2 + loss_dice_sa_T2 + loss_ce_sa_T2) / 4
            lossb_T2 = (loss_diceb_T2 + loss_ceb_T2 + loss_dice_sb_T2 + loss_ce_sb_T2) / 4

            lossa = (loss_dicea + loss_cea + loss_dice_sa + loss_ce_sa) / 4
            lossb = (loss_diceb + loss_ceb + loss_dice_sb + loss_ce_sb) / 4

            loss=(lossa+lossb+lossa_T2+lossb_T2)/4

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', (loss_dicea+loss_diceb)/2, iter_num)
            writer.add_scalar('info/mix_ce', (loss_cea+loss_ceb)/2, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, (loss_dicea+loss_diceb)/2, (loss_cea+loss_ceb)/2))
                
            if iter_num % 20 == 0:
                imagea = net_inputa[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Imagea', imagea, iter_num)
                outputsa = torch.argmax(torch.softmax(out_mixla1, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Predictiona', outputsa[1, ...] * 50, iter_num)
                outputsa_T2 = torch.argmax(torch.softmax(out_mixla2, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Predictiona_T2', outputsa_T2[1, ...] * 50, iter_num)
                labsa = gt_mixla[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTrutha', labsa, iter_num)

                imageb = net_inputb[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Imageb', imageb, iter_num)
                outputsb = torch.argmax(torch.softmax(out_mixlb1, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Predictionb', outputsb[1, ...] * 50, iter_num)
                outputsb_T2 = torch.argmax(torch.softmax(out_mixlb2, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Predictionb_T2', outputsb_T2[1, ...] * 50, iter_num)
                labsb = gt_mixlb[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruthb', labsb, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"],
                                                             model1, classes=num_classes,
                                                             patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best_model1 = os.path.join(snapshot_path,
                                                      '{}_best_model1.pth'.format(args.model1))
                    torch.save(model1.state_dict(), save_best_model1)
                    torch.save(model1.state_dict(), save_mode_path)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2,
                        classes=num_classes,patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2,4)))
                    save_best_model2 = os.path.join(snapshot_path,
                                                      '{}_best_model2.pth'.format(args.model2))
                    torch.save(model2.state_dict(), save_best_model2)
                    torch.save(model2.state_dict(), save_mode_path)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # pre_trained1 = os.path.join(pre_snapshot_path,
    #                                   '{}_best_model1.pth'.format(args.model1))
    # pre_trained2 = os.path.join(pre_snapshot_path,
    #                                     '{}_best_model2.pth'.format(args.model2))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    def create_model(ema=False, net_type=args.model):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = BCP_net(in_chns=1, class_num=num_classes)
    model2 = BCP_net(in_chns=1, class_num=num_classes)
    # model1 = create_model(net_type=args.model1)
    # model2 = create_model(net_type=args.model2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator_slice(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labeled_num)
    # labeled_slice = int(total_slices * patients_to_slices(args.root_path, args.labeled_num))
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                                  momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                                   momentum=0.9, weight_decay=0.0001)
    # load_net(model1, pre_trained1)
    # load_net(model2, pre_trained2)
    # logging.info("Loaded from {}".format(pre_trained1))
    # logging.info("Loaded from {}".format(pre_trained2))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model1.train()
    model2.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img = volume_batch[:args.labeled_bs]
            uimg = volume_batch[args.labeled_bs:]
            ulab = label_batch[args.labeled_bs:]
            lab = label_batch[:args.labeled_bs]

            img_s1 = deepcopy(volume_batch)
            for b in range(img_s1.size(0)):
                img_s = transforms.ToPILImage()(img_s1[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_s1[b] = transforms.ToTensor()(img_s)

            img_s2 = deepcopy(volume_batch)
            for b in range(img_s2.size(0)):
                img_s = transforms.ToPILImage()(img_s2[b])
                if random.random() < 0.8:
                    img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
                img_s = transforms.RandomGrayscale(p=0.2)(img_s)
                img_s = blur(img_s, p=0.5)
                img_s2[b] = transforms.ToTensor()(img_s)

            with torch.no_grad():
                outputs1_l = model1(img)
                outputs2_l = model2(img)
                outputs_soft1_l = torch.softmax(outputs1_l, dim=1)
                outputs_soft2_l = torch.softmax(outputs2_l, dim=1)

                outputs1_l_s1 = model1(img_s1)
                outputs2_l_s1 = model2(img_s1)
                outputs_soft1_l_s1 = torch.softmax(outputs1_l_s1, dim=1)
                outputs_soft2_l_s1 = torch.softmax(outputs2_l_s1, dim=1)

                outputs1_l_s2 = model1(img_s2)
                outputs2_l_s2 = model2(img_s2)
                outputs_soft1_l_s2 = torch.softmax(outputs1_l_s2, dim=1)
                outputs_soft2_l_s2 = torch.softmax(outputs2_l_s2, dim=1)

                outputs1 = model1(uimg)
                outputs2 = model2(uimg)
                # outputs_soft1 = torch.softmax(outputs1, dim=1)
                # outputs_soft2 = torch.softmax(outputs2, dim=1)
                # uncertainty_map1 = -1.0 * torch.sum(outputs_soft1 * torch.log(outputs_soft1 + 1e-6), dim=1,
                #                                     keepdim=True)
                # uncertainty_map2 = -1.0 * torch.sum(outputs_soft2 * torch.log(outputs_soft2 + 1e-6), dim=1,
                #                                     keepdim=True)
                # pre=torch.where(uncertainty_map1 < uncertainty_map2, outputs1, outputs2)
                pre=(outputs1 + outputs2)/2
                plab = get_ACDC_masks(pre, nms=1)
                img_mask, loss_mask = generate_mask_part(img)
                uimg_mask, uloss_mask = generate_mask_part(uimg)
                uimg_mask_s1, uloss_mask_s1 = generate_mask_part(img_s1[args.labeled_bs:])
                uimg_mask_s2, uloss_mask_s2 = generate_mask_part(img_s2[args.labeled_bs:])
                unl_label = ulab * uimg_mask + lab * (1 - uimg_mask)
                l_label = lab * img_mask + ulab * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num//150)

            net_input_unl = uimg * uimg_mask + img * (1 - uimg_mask)
            net_input_l = img * img_mask + uimg * (1 - img_mask)
            out_unl = model1(net_input_unl)
            out_l = model2(net_input_l)

            net_input_unl_s1 = img_s1[args.labeled_bs:] * uimg_mask_s1 + img * (1 - uimg_mask_s1)
            net_input_l_s1 = img * img_mask + img_s1[args.labeled_bs:] * (1 - img_mask)
            out_unl_s1 = model1(net_input_unl_s1)
            out_l_s1 = model2(net_input_l_s1)

            net_input_unl_s2 = img_s2[args.labeled_bs:] * uimg_mask_s2 + img * (1 - uimg_mask_s2)
            net_input_l_s2 = img * img_mask + img_s2[args.labeled_bs:] * (1 - img_mask)
            out_unl_s2 = model1(net_input_unl_s2)
            out_l_s2 = model2(net_input_l_s2)

            loss1 = 0.5 * (ce_loss(outputs1_l[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1_l[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2_l[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2_l[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss1_s1 = 0.5 * (ce_loss(outputs1_l_s1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1_l_s1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2_s1 = 0.5 * (ce_loss(outputs2_l_s1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2_l_s1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss1_s2 = 0.5 * (ce_loss(outputs1_l_s2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1_l_s2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2_s2 = 0.5 * (ce_loss(outputs2_l_s2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2_l_s2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            unl_dice, unl_ce = mix_loss(out_unl, plab, lab, uloss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_T2, l_ce_T2 = mix_loss(out_l, lab, plab, loss_mask, u_weight=args.u_weight)
            unl_dice_s1, unl_ce_s1 = mix_loss(out_unl_s1, plab, lab, uloss_mask_s1, u_weight=args.u_weight, unlab=True)
            l_dice_T2_s1, l_ce_T2_s1 = mix_loss(out_l_s1, lab, plab, loss_mask, u_weight=args.u_weight)
            unl_dice_s2, unl_ce_s2 = mix_loss(out_unl_s2, plab, lab, uloss_mask_s2, u_weight=args.u_weight, unlab=True)
            l_dice_T2_s2, l_ce_T2_s2 = mix_loss(out_l_s2, lab, plab, loss_mask, u_weight=args.u_weight)

            supervised_loss1=(loss1+loss1_s1+loss1_s2)/3
            supervised_loss2 = (loss2 + loss2_s1 + loss2_s2) / 3
            loss_ce = (unl_ce + l_ce_T2) /2+(unl_ce_s1 + l_ce_T2_s1+unl_ce_s2 + l_ce_T2_s2)/4
            loss_dice = (unl_dice + l_dice_T2)/2+(unl_dice_s1 + l_dice_T2_s1+unl_dice_s2 + l_dice_T2_s2)/4

            loss = (loss_dice + loss_ce + supervised_loss1 + supervised_loss2) / 4

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            # if iter_num % 20 == 0:
            #     image = net_input_unl[1, 0:1, :, :]
            #     writer.add_image('train/Un_Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(out_unl1, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = unl_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/Un_GroundTruth', labs, iter_num)
            #
            #     image_l = net_input_l[1, 0:1, :, :]
            #     writer.add_image('train/L_Image', image_l, iter_num)
            #     outputs_l = torch.argmax(torch.softmax(out_l1, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
            #     labs_l = l_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"],
                                                             model1, classes=num_classes,
                                                             patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best_model1 = os.path.join(snapshot_path,
                                                      '{}_best_model1.pth'.format(args.model1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best_model1)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2,
                        classes=num_classes,patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2,4)))
                    save_best_model2 = os.path.join(snapshot_path,
                                                      '{}_best_model2.pth'.format(args.model2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best_model2)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "../model/{}_{}_labeled/pre_train".format(args.exp, args.labeled_num)
    self_snapshot_path = "../model/{}_{}_labeled/self_train".format(args.exp, args.labeled_num)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    # #Pre_train
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


