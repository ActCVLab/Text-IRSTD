import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from utils import Normalized, PadImg, random_crop, get_img_norm_cfg
import random
from torch.utils.data import DataLoader
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.patch_size = patch_size
        self.text_dir = os.path.join(self.dataset_dir, 'prompt')

        with open(os.path.join(self.dataset_dir, f'img_idx/train_{dataset_name}.txt'), 'r') as f:
            self.train_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            #单通道
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.train_list[idx] + '.png')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.train_list[idx] + '.png'))
        except:
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.train_list[idx] + '.bmp')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.train_list[idx] + '.bmp'))

        text_path = os.path.join(self.text_dir, self.train_list[idx] + '.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch, text

    def __len__(self):
        return len(self.train_list)

class TestSetLoader_inTrain(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader_inTrain, self).__init__()
        self.dataset_dir = os.path.join(dataset_dir, test_dataset_name)
        self.text_dir = os.path.join(self.dataset_dir, 'prompt')

        with open(os.path.join(self.dataset_dir, f'img_idx/val_{test_dataset_name}.txt'), 'r') as f:
            self.test_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.test_list[idx] + '.png')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.test_list[idx] + '.png'))
        except:
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.test_list[idx] + '.bmp')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.test_list[idx] + '.bmp'))

        text_path = os.path.join(self.text_dir, self.test_list[idx] + '.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, text, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)

class TestSetLoader_inTest(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader_inTest, self).__init__()
        self.dataset_dir = os.path.join(dataset_dir, test_dataset_name)
        self.text_dir = os.path.join(self.dataset_dir, 'prompt')

        with open(os.path.join(self.dataset_dir, f'img_idx/test_{test_dataset_name}.txt'), 'r') as f:
            self.test_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.test_list[idx] + '.png')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.test_list[idx] + '.png'))
        except:
            img = Image.open(os.path.join(self.dataset_dir, 'images', self.test_list[idx] + '.bmp')).convert('I')
            mask = Image.open(os.path.join(self.dataset_dir, 'masks', self.test_list[idx] + '.bmp'))

        text_path = os.path.join(self.text_dir, self.test_list[idx] + '.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, text, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)
    
class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        self.text_dir = os.path.join(self.dataset_dir, 'prompt')

        with open(os.path.join(self.dataset_dir, f'img_idx/test_{test_dataset_name}.txt'), 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open(os.path.join(self.mask_pred_dir, self.test_dataset_name, self.model_name,
                                            self.test_list[idx] + '.png'))
        mask_gt = Image.open(os.path.join(self.dataset_dir, 'masks', self.test_list[idx] + '.png'))

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        text_path = os.path.join(self.text_dir, self.test_list[idx] + '.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, text, [h, w]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target


