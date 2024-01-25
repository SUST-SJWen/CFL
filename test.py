import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks1.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--model_encoder', type=str,
                    default='myHiFormer', help='model_name')
parser.add_argument('--model_decoder', type=str,
                    default='unet_decoder', help='model_name')
parser.add_argument('--model_decoder_T', type=str,
                    default='unet_decoder_T', help='model_name')
parser.add_argument('--root_path', type=str,
                    default='../data/FLARE22Train', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='FLARE22/Cross_pseudo_supervision', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=14,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=1,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File('..\data\FLARE22Train\data\FLARE22_Tr_0002.h5', 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
        print(ind)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return 0, 0, 0
def test_single_volume_CTCT(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net[0].eval()
        net[1].eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main1 = net[0](input)
                out_main2 = net[1](input)
                out_main=(out_main1+out_main2)/2
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
        if 1 in prediction:
            first_metric = calculate_metric_percase(prediction == 1, label == 1)
        else:
            first_metric =0
        if 2 in prediction:
            second_metric = calculate_metric_percase(prediction == 2, label == 2)
        else:
            second_metric =0
        if 3 in prediction:
            third_metric = calculate_metric_percase(prediction == 3, label == 3)
        else:
            third_metric =0

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric
def test_myHiFormer_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda(1)
        net[0].eval()
        net[1].eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                encoder_out=net[0](input)
                out_main = net[1](encoder_out[1],encoder_out[3],encoder_out[5])
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def Inference(FLAGS):
    # with open(FLAGS.root_path + '/test.list', 'r') as f:
    #     image_list = f.readlines()
    # image_list = sorted([item.replace('\n', '').split(".")[0]
    #                      for item in image_list])
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    case='FLARE22_Tr_0001.h5'
    # for case in tqdm(image_list):
    first_metric, second_metric, third_metric = test_single_volume(
        case, net, test_save_path, FLAGS)
    first_total += np.asarray(first_metric)
    second_total += np.asarray(second_metric)
    third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    # with open(test_save_path+'tags_best.txt', 'w') as f:
    #     f.truncate(0)
    #     f.write("First_dice:" + str(avg_metric[0][0]) + "  First_hd95:" + str(avg_metric[0][1]) + "  First_asd:" + str(avg_metric[0][2]) + '\n')
    #     f.write("Second_dice:" + str(avg_metric[1][0]) + "  Second_hd95:" + str(avg_metric[1][1]) + "  Second_asd:" + str(avg_metric[1][2]) + '\n')
    #     f.write("Third_dice:" + str(avg_metric[2][0]) + "  Third_hd95:" + str(avg_metric[2][1]) + "  Third_asd:" + str(avg_metric[2][2]) + '\n')
    #     f.write("Avg_dice:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[0]) + "  Avg_hd95:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[1]) + "  Avg_asd:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[2]) + '\n')
    # return avg_metric

def myHiFormer_Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net1 = net_factory(net_type=FLAGS.model_encoder, in_chns=1,
                      class_num=FLAGS.num_classes)
    net2 = net_factory(net_type=FLAGS.model_decoder, in_chns=1,
                       class_num=FLAGS.num_classes)
    save_mode_path1 = os.path.join(
        snapshot_path, '{}_best_model_encoder1.pth'.format(FLAGS.model_decoder))
    save_mode_path2 = os.path.join(
        snapshot_path, '{}_best_model_decoder1.pth'.format(FLAGS.model_decoder))
    net1.load_state_dict(torch.load(save_mode_path1))
    net2.load_state_dict(torch.load(save_mode_path2))
    print("init weight from {}".format(save_mode_path1))
    print("init weight from {}".format(save_mode_path2))
    net1.eval()
    net2.eval()
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_myHiFormer_single_volume(
            case, [net1,net2], test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    with open(test_save_path+'tags_best.txt', 'w') as f:
        f.truncate(0)
        f.write("First_dice:" + str(avg_metric[0][0]) + "  First_hd95:" + str(avg_metric[0][1]) + "  First_asd:" + str(avg_metric[0][2]) + '\n')
        f.write("Second_dice:" + str(avg_metric[1][0]) + "  Second_hd95:" + str(avg_metric[1][1]) + "  Second_asd:" + str(avg_metric[1][2]) + '\n')
        f.write("Third_dice:" + str(avg_metric[2][0]) + "  Third_hd95:" + str(avg_metric[2][1]) + "  Third_asd:" + str(avg_metric[2][2]) + '\n')
        f.write("Avg_dice:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[0]) + "  Avg_hd95:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[1]) + "  Avg_asd:" + str(((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)[2]) + '\n')
    return avg_metric
# def test_myHiFormer_single_volume1(case, net, test_save_path, FLAGS):
#     h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     slice = image[5, :, :]
#     x, y = slice.shape[0], slice.shape[1]
#     slice = zoom(slice, (224 / x, 224 / y), order=0)
#     input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda(0)
#     net[0].eval()
#     net[1].eval()
#     with torch.no_grad():
#         if FLAGS.model == "unet_urds":
#             out_main, _, _, _ = net(input)
#         else:
#             encoder_out=net[0](input)
#             out_main = net[1](encoder_out[1],encoder_out[3],encoder_out[5])
#         out = torch.argmax(torch.softmax(
#             out_main, dim=1), dim=1).squeeze(0)
#         out = out.cpu().detach().numpy()
#         pred = zoom(out, (x / 224, y / 224), order=0)
#         prediction[5] = pred
#
#     # first_metric = calculate_metric_percase(prediction == 1, label == 1)
#     # second_metric = calculate_metric_percase(prediction == 2, label == 2)
#     # third_metric = calculate_metric_percase(prediction == 3, label == 3)
#     #
#     # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     # img_itk.SetSpacing((1, 1, 10))
#     # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     # prd_itk.SetSpacing((1, 1, 10))
#     # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     # lab_itk.SetSpacing((1, 1, 10))
#     # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
    # print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
    # net1 = net_factory(net_type=FLAGS.model_encoder, in_chns=1,
    #                    class_num=FLAGS.num_classes)
    # net2 = net_factory(net_type=FLAGS.model_decoder, in_chns=1,
    #                    class_num=FLAGS.num_classes)
    # save_mode_path1 = './weights/unet_decoder_best_model_encoder1.pth'
    # save_mode_path2 = './weights/unet_decoder_best_model_decoder1.pth'
    # net1.load_state_dict(torch.load(save_mode_path1, map_location='cpu'))
    # net2.load_state_dict(torch.load(save_mode_path2, map_location='cpu'))
    # first_metric, second_metric, third_metric = test_myHiFormer_single_volume1(
    #     'patient013_frame01', [net1, net2], './weights', FLAGS)