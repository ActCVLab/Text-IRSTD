from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from skimage.feature.tests.test_orb import img
import clip
from model.ISTDUNet.Text import ISTDU_Net_Text_NewFFNimplaceMerge as ISTDU_Net_Text
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.cal_loss = SoftIoULoss()
        self.model_clip, self.preprocess = clip.load("ViT-B/32", device=device)

        if model_name == 'Text_model':
            self.model = Text_model(self.model_clip)
        if model_name == 'ISTDU_Net_Text':
            self.model = ISTDU_Net_Text(self.model_clip)
        if model_name == 'ISTDU_Net_Text_cat':
            self.model == ISTDU_Net_Text_cat(self.model_clip)
        for param in self.model.model_clip.parameters():
            param.requires_grad = False

    def forward(self, img, text):
        return self.model(img, text)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss

    def get_clip_model(self):
        return self.model_clip
