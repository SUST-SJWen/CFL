import torch
import torch.nn as nn
from glob import glob
import SimpleITK as sitk
import torch
import numpy as np
import os
import nibabel
import skimage.measure as measure
from skimage.morphology import skeletonize_3d
from utils import get_parsing
import math

EPSILON = 1e-32


def compute_binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def evaluation_branch_metrics(fid,label, pred,refine=False):
    """
    :return: iou,dice, detected length ratio, detected branch ratio,
     precision, leakages, false negative ratio (airway missing ratio),
     large_cd (largest connected component)
    """
    # compute tree sparsing
    parsing_gt = get_parsing(label, refine)
    # find the largest component to locate the airway prediction
    cd, num = measure.label(pred, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    iou = compute_binary_iou(label, large_cd)
    flag=-1
    while iou < 0.1:
        print(fid," failed cases, require post-processing")
        large_cd = (cd == (volume_sort[flag-1] + 1)).astype(np.uint8)
        iou = compute_binary_iou(label, large_cd)
    skeleton = skeletonize_3d(label)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    DLR = (large_cd * skeleton).sum() / skeleton.sum()
    precision = (large_cd * label).sum() / large_cd.sum()
    leakages = ((large_cd - label)==1).sum() / label.sum()

    num_branch = parsing_gt.max()
    detected_num = 0
    for j in range(num_branch):
        branch_label = ((parsing_gt == (j + 1)).astype(np.uint8)) * skeleton
        if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
            detected_num += 1
    DBR = detected_num / num_branch
    return iou, DLR, DBR, precision, leakages
def dice(predict, soft_y):
    """
    get dice scores for each class in predict and soft_y
    """
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if (tensor_dim == 5):
        soft_y = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        soft_y = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    soft_y = torch.reshape(soft_y, (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))

    y_vol = torch.sum(soft_y, dim=0)
    p_vol = torch.sum(predict, dim=0)
    intersect = torch.sum(soft_y * predict, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    return dice_score


if __name__ == "__main__":
    infer_path = "/opt/data/private/SSL4MIS-master/model/AbdomenCT/Cross_pseudo_supervision_0_labeled/unet_predictions/*_pred.nii.gz"  # 推理结果地址
    label_path = "/opt/data/private/SSL4MIS-master/model/AbdomenCT/Cross_pseudo_supervision_0_labeled/unet_predictions/*_gt.nii.gz"  # 测试集label地址
    infer = sorted(glob(infer_path))
    label = sorted(glob(label_path))
    iou_avg1 = 0
    DLR_avg1 = 0
    DBR_avg1 = 0
    precision_avg1 = 0
    iou_avg2 = 0
    DLR_avg2 = 0
    DBR_avg2 = 0
    precision_avg2 = 0
    iou_avg3 = 0
    DLR_avg3 = 0
    DBR_avg3 = 0
    precision_avg3 = 0
    iou_avg4 = 0
    DLR_avg4 = 0
    DBR_avg4 = 0
    precision_avg4 = 0
    iou_avg_all = 0
    DLR_avg_all = 0
    DBR_avg_all = 0
    precision_avg_all = 0
    for i in range(len(label)):
        inf, lab = infer[i], label[i]
        inf, lab = sitk.ReadImage(inf, sitk.sitkFloat32), sitk.ReadImage(lab, sitk.sitkFloat32)
        inf, lab = sitk.GetArrayFromImage(inf), sitk.GetArrayFromImage(lab)
        # inf, lab = torch.from_numpy(inf), torch.from_numpy(lab)
        # inf, lab = inf.unsqueeze(0).unsqueeze(0), lab.unsqueeze(0).unsqueeze(0)
        iou1, DLR1, DBR1, precision1, leakages1 = evaluation_branch_metrics(i,inf==1, lab==1)
        iou2, DLR2, DBR2, precision2, leakages2 = evaluation_branch_metrics(i, inf == 2, lab == 2)
        iou3, DLR3, DBR3, precision3, leakages3 = evaluation_branch_metrics(i, inf == 3, lab == 3)
        iou4, DLR4, DBR4, precision4, leakages4 = evaluation_branch_metrics(i, inf == 4, lab == 4)
        iou_avg1 += iou1
        DLR_avg1 += DLR1
        DBR_avg1 += DBR1
        precision_avg1 += precision1
        iou_avg2 += iou2
        DLR_avg2 += DLR2
        DBR_avg2 += DBR2
        precision_avg2 += precision2
        iou_avg3 += iou3
        DLR_avg3 += DLR3
        DBR_avg3 += DBR3
        precision_avg3 += precision3
        iou_avg4 += iou4
        DLR_avg4 += DLR4
        DBR_avg4 += DBR4
        precision_avg4 += precision4
    iou_avg1 /= len(label)
    print("avg iou1 is ", iou_avg1)
    DLR_avg1 /= len(label)
    print("avg DLR1 is ", DLR_avg1)
    DBR_avg1 /= len(label)
    print("avg DBR1 is ", DBR_avg1)
    precision_avg1 /= len(label)
    print("avg precision1 is ", precision_avg1)
    print("************"*20)
    iou_avg2 /= len(label)
    print("avg iou2 is ", iou_avg2)
    DLR_avg2 /= len(label)
    print("avg DLR2 is ", DLR_avg2)
    DBR_avg2 /= len(label)
    print("avg DBR2 is ", DBR_avg2)
    precision_avg2 /= len(label)
    print("avg precision2 is ", precision_avg2)
    print("************" * 20)
    iou_avg3 /= len(label)
    print("avg iou3 is ", iou_avg3)
    DLR_avg3 /= len(label)
    print("avg DLR3 is ", DLR_avg3)
    DBR_avg3 /= len(label)
    print("avg DBR3 is ", DBR_avg3)
    precision_avg3 /= len(label)
    print("avg precision3 is ", precision_avg3)
    print("************" * 20)
    iou_avg4 /= len(label)
    print("avg iou4 is ", iou_avg4)
    DLR_avg4 /= len(label)
    print("avg DLR4 is ", DLR_avg4)
    DBR_avg4 /= len(label)
    print("avg DBR4 is ", DBR_avg4)
    precision_avg4 /= len(label)
    print("avg precision4 is ", precision_avg4)
    print("************" * 20)
    iou_avg_all = (iou_avg1+iou_avg2+iou_avg3+iou_avg4)/4
    print("avg iou_all is ", iou_avg_all)
    DLR_avg_all = (DLR_avg1+DLR_avg2+DLR_avg3+DLR_avg4)/4
    print("avg DLR_all is ", DLR_avg_all)
    DBR_avg_all = (DBR_avg1+DBR_avg2+DBR_avg3+DBR_avg4)/4
    print("avg DBR_all is ", DBR_avg_all)
    precision_avg_all = (precision_avg1+precision_avg2+precision_avg3+precision_avg4)/4
    print("avg precision_all is ", precision_avg_all)

    # iou = 0.873114965258233
    #
    # DLR = 0.873702703306051
    #
    # DBR = 0.862606831959138
    #
    # precision = 0.9217716734179241
    #
    # leakages = 0.08139449103475939
    # Overall_score = (iou+precision+DBR+DLR)*0.25*0.7 + (1-leakages)*0.3
    # print("Overall_score is ", Overall_score)