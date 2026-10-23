import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import sys

# #from minet import miNet
from model.ISTDUNet.resnet2020 import ResNet, Bottleneck, ResNetCt, BottleneckMode

# from resnet2020 import ResNet, Bottleneck, ResNetCt, BottleneckMode
import clip

sys.path.append('/home/ZSY/BasicIRSTD-text/')

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

        stemWidth = int(8)
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

class UPCt(nn.Module):
    def __init__(self, channels=[],
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU
                 ):
        super(UPCt, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0],
                      channels[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[1]),
            activate()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1],
                      channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[2]),
            activate()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2],
                      channels[3],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[3]),
            activate()
        )

        self.text_image_1 = FeatureWiseAffine(512,channels[1])
        self.text_image_2 = FeatureWiseAffine(512, channels[2])
        self.text_image_3 = FeatureWiseAffine(512, channels[3])

    def forward(self, x, text):
        x1, x2, x3, x4 = x

        out = self.up1(x4)
        out = x3 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out_3 = self.text_image_1(out,text)

        out = self.up2(out_3)
        out = x2 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out_2 = self.text_image_2(out, text)

        out = self.up3(out_2)
        out = x1 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out_1 = self.text_image_3(out, text)
        return out_1



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
        x_new = self.project_out(x_new)
        x_weight = self.topdown(x_new)

        return x_new * x_weight + x
    
    

class EDN_FFN(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(EDN_FFN, self).__init__()

        self.X1 = FeedForward(channels[0], ffn_expansion_factor=2.66, bias=False)
        self.X2 = FeedForward(channels[1], ffn_expansion_factor=2.66, bias=False)
        self.X3 = FeedForward(channels[2], ffn_expansion_factor=2.66, bias=False)
        self.X4 = FeedForward(channels[3], ffn_expansion_factor=2.66, bias=False)

    def forward(self, x):
        x1 ,x2, x3, x4 = x
        x1 = self.X1(x1)
        x2 = self.X2(x2)
        x3 = self.X3(x3)
        x4 = self.X4(x4)
        return [x1, x2, x3, x4]

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class Text_model(nn.Module):
    def __init__(self,  model_clip):
        super(Text_model, self).__init__()
        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder = None
        self.decoder = None

        self.down = Down(channels=[16, 32, 64, 128])
        self.up = UPCt(channels=[512, 256, 128, 64])
        self.headSeg = Head(inpChannel=64, oupChannel=1)
        self.DN = EDN_FFN(channels=[64, 128, 256, 512])

    def forward(self, x, text):

        text = self.get_text_feature(text)

        x = self.down(x)
        x = self.DN(x)
        out = self.up(x, text)

        seg_out = torch.sigmoid(self.headSeg(out))  # 分割输出
        #print("Segmentation output shape: ", seg_out.shape)
        return seg_out

    def get_text_feature(self, text):
        with torch.no_grad():
            text_feature = self.model_clip.encode_text(text).float()
        return text_feature

if __name__ == '__main__':
    device = "cuda"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    image_input = torch.rand((3, 1, 512, 512), device=device) #[B,C,H,W]
    text_input = torch.rand((3, 512), device=device)

    model = Text_model(clip_model).to(device)
    output = model(image_input, text_input)

    print("Model output shape:", output.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))