import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import sys

from model.ISTDUNet.resnet2020 import ResNet, Bottleneck, ResNetCt, BottleneckMode
# from resnet2020 import ResNet, Bottleneck, ResNetCt, BottleneckMode

import clip

sys.path.append('/home/ZSY/BasicIRSTD-text/')

##网络结构：把解码器的每一个进行拼接后进行预测
#把merge模块换成SCTrans的FFN


##编码器
class Down(nn.Module):
    def __init__(self,
                 inp_num = 1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 **kwargs
                 ):
        super(Down, self).__init__()

        # stemWidth = int(channels[0])
        stemWidth = int(8)
        ##经过stem，图像尺寸不变，通道数从1——》stemWidth*2
        self.stem = nn.Sequential(
            normLayer(1, affine=False),
            nn.Conv2d(1, stemWidth*2, kernel_size=3, stride=1, padding=1, bias=False),
            normLayer(stemWidth*2),
            activate()
        )
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                       radix=2, groups=4, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.down(x)

        ret=x
        return ret


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x

class UPCt_each(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normLayer=nn.BatchNorm2d, activate=nn.ReLU):
        super(UPCt_each, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            normLayer(out_channels),
            activate()
        )      

    def forward(self, x):
        out = self.conv(x)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')

        return out


class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 # Dropout = 0.1
                 ):
        super(Head, self).__init__()
        interChannel = inpChannel // 4
        # interChannel = 1
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,
                      kernel_size=3, padding=1,
                      bias=False),
            normLayer(interChannel),
            activate(),
            # nn.Dropout(),
            nn.Conv2d(interChannel, oupChannel,
                      kernel_size=1, padding=0,
                      bias=True)
        )

    def forward(self, x):
        return self.head(x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor) #隐藏层
        self.bottleneck_channels = int(dim // 4)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        #self.eca = eca_layer_2d(dim) #原本的GAP+1D卷积+sigmoid
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x_new = torch.cat([x1_3, x1_5], dim=1)
        x_new = self.project_out(x_new) #1×1卷积
        x_weight = self.topdown(x_new)

        return x_new * x_weight + x
    

class AsymBiChaFuse(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, text_channel=512):
        super(AsymBiChaFuse, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor) #隐藏层
        self.bottleneck_channels = int(dim // 4)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        #self.eca = eca_layer_2d(dim) #原本的GAP+1D卷积+sigmoid
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

        self.MLP = nn.Sequential(
            nn.Linear(text_channel, text_channel * 2),
            nn.LeakyReLU(),
            nn.Linear(text_channel * 2, dim* 2)
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=dim, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(dim, momentum=0.9),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
        nn.Conv2d(in_channels=dim,out_channels=dim, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(dim,momentum=0.9),
        nn.ReLU(inplace=True)
        )

    def forward(self, x_l,x_h, text_embed):
        #文本处理部分
        batch = x_l.shape[0]

        gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
        x_l = (1 + gamma) * x_l
        x_h = (1 + beta) * x_h

        x_3,x_5 = self.project_in(x_l).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x_new = torch.cat([x1_3, x1_5], dim=1)
        x_new = self.project_out(x_new) #1×1卷积
        x_l_weight = self.bottomup(x_new)
        x_l_new = x_new * x_l_weight

        x_h_new_weight = self.topdown(x_h) #1×1卷积
        x_h_new = x_h * x_h_new_weight
        
        x_fusion = x_l_new + x_h_new
        return x_l + x_h + x_fusion

class ISTDU_Net_Text_NewFFNimplaceMerge(nn.Module):
    def __init__(self,  model_clip):
        super(ISTDU_Net_Text_NewFFNimplaceMerge, self).__init__()
        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder = None
        self.decoder = None

        self.down = Down(channels=[16, 32, 64, 128])
        self.headSeg = Head(inpChannel=64, oupChannel=1)
        # self.DN = EDN_FFN(channels=[64, 128, 256, 512])

        # 实例化 4 个 UPCt，每个对应不同尺度
        self.up1 = UPCt_each(in_channels=512, out_channels=256)  # x4 -> x3
        self.up2 = UPCt_each(in_channels=256, out_channels=128)  # x3 -> x2
        self.up3 = UPCt_each(in_channels=128, out_channels=64)   # x2 -> x1

        # 仿射变换（文本调制）
        self.text_image_1 = FeatureWiseAffine(in_channels=512, out_channels=256)  
        self.text_image_2 = FeatureWiseAffine(in_channels=512, out_channels=128) 
        self.text_image_3 = FeatureWiseAffine(in_channels=512, out_channels=64) 

        ##跨层特征融合
        self.Fusion_1 = FeedForward(512, ffn_expansion_factor=2.66, bias=False)
        self.Fusion_2 = AsymBiChaFuse(256, ffn_expansion_factor=2.66, bias=False)
        self.Fusion_3 = AsymBiChaFuse(128, ffn_expansion_factor=2.66, bias=False)
        self.Fusion_4 = AsymBiChaFuse(64, ffn_expansion_factor=2.66, bias=False)
    def forward(self, x, text):

        text = self.get_text_feature(text)

        x1, x2, x3, x4 = self.down(x)
        x4 = self.Fusion_1(x4)
        # 依次上采样
        out_3 = self.up1(x4)
        out_3 = self.Fusion_2(x3, out_3,text)
        out_3 = self.text_image_1(out_3,text)

        out_2 = self.up2(out_3)
        out_2 = x2 + out_2
        out_2 = self.Fusion_3(x2, out_2,text)
        out_2 = self.text_image_2(out_2,text)

        out_1 = self.up3(out_2)
        out_1 = self.Fusion_4(x1, out_1,text)
        out_1 = self.text_image_3(out_1,text)
        


        seg_out = torch.sigmoid(self.headSeg(out_1))  # 分割输出
        #print("Segmentation output shape: ", seg_out.shape)
        return seg_out

    def get_text_feature(self, text):
        with torch.no_grad():
            text_feature = self.model_clip.encode_text(text).float()
        return text_feature

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("clip加载完成")

    image_input = torch.rand((3, 1, 512, 512), device=device) #[B,C,H,W]
    text_input = torch.rand((3, 512), device=device)

    model = ISTDU_Net_Text_NewFFNimplaceMerge(clip_model).to(device)
    output = model(image_input, text_input)

    print("Model output shape:", output.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    total_params_in_million = pytorch_total_params / 1_000_000
    print("Total_params: {:.2f}M".format(total_params_in_million))
