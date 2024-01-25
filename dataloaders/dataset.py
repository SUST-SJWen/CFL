import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
class BaseDataSets_LA(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/2018LA_Seg_Training_Set/{}/mri_norm2.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/2018LA_Seg_Training_Set/{}/mri_norm2.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
class BaseDataSets_FLARE22(Dataset):
        def __init__(
                self,
                base_dir=None,
                split="train",
                num=None,
                transform=None,
                ops_weak=None,
                ops_strong=None,
        ):
            self._base_dir = base_dir
            self.sample_list = []
            self.split = split
            self.transform = transform
            self.ops_weak = ops_weak
            self.ops_strong = ops_strong

            assert bool(ops_weak) == bool(
                ops_strong
            ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

            if self.split == "train":
                with open(self._base_dir + "/train_slice.txt", "r") as f1:
                    self.sample_list = f1.readlines()
                self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            elif self.split == "val":
                with open(self._base_dir + "/val.txt", "r") as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            if num is not None and self.split == "train":
                self.sample_list = self.sample_list[:num]
            print("total {} samples".format(len(self.sample_list)))

        def __len__(self):
            return len(self.sample_list)

        def __getitem__(self, idx):
            case = self.sample_list[idx]
            if self.split == "train":
                h5f = h5py.File(self._base_dir + "/data/slice/{}".format(case), "r")
            else:
                h5f = h5py.File(self._base_dir + "/data/{}".format(case), "r")
            image = h5f["image"][:].astype(np.float32)
            label = h5f["label"][:].astype(np.uint8)
            sample = {"image": image, "label": label}
            if self.split == "train":
                if None not in (self.ops_weak, self.ops_strong):
                    sample = self.transform(sample, self.ops_weak, self.ops_strong)
                else:
                    sample = self.transform(sample)
            sample["idx"] = idx
            return sample

class BaseDataSets_KiTS23(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slice.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}".format(case), "r")
        image = h5f["image"][:].astype(np.float32)
        label = h5f["label"][:].astype(np.uint8)
        # image = np.clip(image, 0.4, 0.6)
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
class BaseDataSets_BraTS(Dataset):
        def __init__(
                self,
                base_dir=None,
                split="train",
                num=None,
                transform=None,
                ops_weak=None,
                ops_strong=None,
        ):
            self._base_dir = base_dir
            self.sample_list = []
            self.split = split
            self.transform = transform
            self.ops_weak = ops_weak
            self.ops_strong = ops_strong

            assert bool(ops_weak) == bool(
                ops_strong
            ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

            if self.split == "train":
                with open(self._base_dir + "/train_slice.txt", "r") as f1:
                    self.sample_list = f1.readlines()
                self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            elif self.split == "val":
                with open(self._base_dir + "/val.txt", "r") as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            if num is not None and self.split == "train":
                self.sample_list = self.sample_list[:num]
            print("total {} samples".format(len(self.sample_list)))

        def __len__(self):
            return len(self.sample_list)

        def __getitem__(self, idx):
            case = self.sample_list[idx]
            if self.split == "train":
                h5f = h5py.File(self._base_dir + "/data/train_slice/{}".format(case), "r")
            else:
                h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
            image = h5f["image"][:].astype(np.float32)
            label = h5f["label"][:].astype(np.uint8)
            sample = {"image": image, "label": label}
            if self.split == "train":
                if None not in (self.ops_weak, self.ops_strong):
                    sample = self.transform(sample, self.ops_weak, self.ops_strong)
                else:
                    sample = self.transform(sample)
            sample["idx"] = idx
            return sample
class BaseDataSets_Synapse(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/lists/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/lists/test_vol.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = np.load(self._base_dir + "/train_npz/{}.npz".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/test_vol_h5/{}.npy.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map
def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)
    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, transform=None):
        self.norm_x_transform = transform
        self.norm_y_transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)
        print("total {} samples".format(len(self.sample_list)))
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image, label = augment_seg(self.img_aug, image, label)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/val.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/data/2018LA_Seg_Training_Set/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image = torch.from_numpy(image).float()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        label = torch.from_numpy(label).float()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image_old, label_old = sample["image"], sample["label"]
        image_old = self.resize(image_old)
        label_old = self.resize(label_old)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image_old = torch.from_numpy(image_old.astype(np.float32)).unsqueeze(0)
        label_old = torch.from_numpy(label_old.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image_old), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label_old), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image":image_old,
            "label":label_old,
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
class REGATransform(object):
    """returns weakly and strongly augmented images

        Args:
            object (tuple): output size of network
        """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_old, label_old = sample["image"], sample["label"]
        # weak augmentation is rotation / flip
        print(image_old.shape,label_old.shape)

        image_weak, label = random_rot_flip(image_old, label_old)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image_old = torch.from_numpy(image_old.astype(np.float32))
        image_weak = torch.from_numpy(image_weak.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {
            "image": image_old,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
            "label": label_old
        }
        return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        ind = random.randrange(0, image.shape[0])
        image = image[ind, ...]
        label = label[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
class RandomGenerator_slice(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

#_*_coding:utf-8_*_
# 作者     ：10571
# 创建时间 : 2021/3/25 20:52
# 文件     : dataset.py
# IDE     : PyCharm
# 此文件用来加载、预处理数据集

import torch.utils.data as data
import PIL.Image as Image
import os  # 系统读取库
import cv2
import glob

# 定义一个加载图片数据的函数
def train_val_dataset(root1):   # 传入两个路径（原图 和 label）
    imgs = []  # 这个是我们定义的存储一个一个原图+label的数组
    n = len(os.listdir(root1))  # 获取该文件夹下图片总数
    # print("图片数量：",n)
    # print("图片路径：", root1)
    for i in range(n):  # 循环读取文件夹中的每个图片，建议对照着文件夹中的图片名称阅读此代码
        img = os.path.join(root1,"%04d.jpg"%i)
        mask = os.path.join(root1.replace('image','label'),"%04d.png"%i)
        imgs.append((img,mask))  # 将读取到的原图+label一一对应，存储在imgs中
    return imgs

class BaseDataSets_ISIC(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=transforms.Compose([
    transforms.ToTensor(),#将数据转为tensor类型，方便pytorch进行自动求导，优化之类的操作
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#数据归一化，两个参数，一个为均值，一个为方差，均设置为0.5，每个参数里三个0.5表示有三个通道
    # transforms.Normalize([0.5], [0.5])  # 单通道
]),
        target_transform= transforms.ToTensor(),
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.target_transform=target_transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            imgs = train_val_dataset(os.path.join(self._base_dir,'train/image'))  # 调用make_dataset函数对root地址中的图片进行读取，函数定义在上方，一起去看看，好了此时的imgs是存储着一一对应的原图+label图
            self.imgs = imgs

        elif self.split == "val":
            imgs = train_val_dataset(os.path.join(self._base_dir, 'test/image'))  # 调用make_dataset函数对root地址中的图片进行读取，函数定义在上方，一起去看看，好了此时的imgs是存储着一一对应的原图+label图
            self.imgs = imgs
        print("total {} samples".format(len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x_path, y_path = self.imgs[idx]
        image = Image.open(x_path)
        label = Image.open(y_path)
        # 更改图片尺寸 size=(256,256)
        image = image.resize((224,224))
        label = label.resize((224,224))
        sample = {"image": image, "label":label}
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform is not None:
            sample['label'] = self.target_transform(sample['label']).squeeze()
        if None not in (self.ops_weak, self.ops_strong):
            sample['image_weak'],sample['label_aug']=self.ops_weak(sample['image'],sample['label'])
            sample['image_strong']=self.ops_strong(sample['image'])
        sample["idx"] = idx
        return sample

import torchvision.transforms.functional as transforms_f
def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            #  color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  For PyTorch 1.9/TorchVision 0.10 users
            color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

class BaseDataSets_pascal(Dataset):
    def __init__(self, root,  split="train", crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.split = split
        if self.split == "train":
            file_name = root + '/train_aug.txt'
        elif self.split == "val":
            file_name = root + '/val.txt'
        with open(file_name) as f:
            self.idx_list = f.read().splitlines()
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed

    def __getitem__(self, index):
        image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
        if self.apply_partial is None:
            label_root = Image.open(self.root + '/SegmentationClassAug_p0_0/{}.png'.format(self.idx_list[index]))
        else:
            label_root = Image.open(self.root + '/SegmentationClassAug_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index],))

        image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
        sample = {"image": image, "label": label.squeeze()}
        sample["idx"] = index
        return sample

    def __len__(self):
        return len(self.idx_list)
class RandomGenerator_w(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
