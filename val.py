import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

from dataloaders.dataset import BaseDataSets_Synapse, BaseDataSets, BaseDataSets_BraTS, BaseDataSets_FLARE22, \
    BaseDataSets_ISIC
from torch.utils.data import DataLoader


from networks1.net_factory import net_factory


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_myHiFormer_single_volume(image, label, net, classes, patch_size=[224, 224],option=[1,3,5]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda(1)
        net[0].eval()
        net[1].eval()
        with torch.no_grad():
            encoder_out=net[0](input)
            out = torch.argmax(torch.softmax(
                net[1](encoder_out[option[0]],encoder_out[option[1]],encoder_out[option[2]]), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
def test_ISIC_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
def test_ISIC_myHiFormer_single_volume(image, label, net, classes, patch_size=[224, 224],option=[1,3,5]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda(1)
    net[0].eval()
    net[1].eval()
    with torch.no_grad():
        encoder_out=net[0](input)
        out = torch.argmax(torch.softmax(
            net[1](encoder_out[option[0]],encoder_out[option[1]],encoder_out[option[2]]), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
def test_ISIC_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[1], slice.shape[2]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        output_main, _, _, _ = net(input)
        out = torch.argmax(torch.softmax(
            output_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction= pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda(0)
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

if __name__ == "__main__":
    def create_model(ema=False,net_type='myHiFormer'):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=9)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    num=0
    model_encoder = create_model()
    model_decoder1 = create_model(net_type="unet_decoder")
    model_decoder2 = create_model(net_type="unet_decoder")
    db_val = BaseDataSets_ISIC(base_dir='E:/skin/Skin_Cancer_dataset/big_data', split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    print(len(db_val), len(valloader))
    for i_batch, sampled_batch in enumerate(valloader):
        print(sampled_batch["image"].shape,sampled_batch["label"].shape)
        metric_i= test_myHiFormer_single_volume(
            sampled_batch["image"], sampled_batch["label"], [model_encoder, model_decoder1], classes=sampled_batch["label"].max()+1,
            option=[1, 3, 5])