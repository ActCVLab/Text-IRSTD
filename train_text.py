import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
#from dataset import *
from dataset_text import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os

from utils import *
import clip

###数据集
#6：2：2 /8 ： 2
#ISTD-1K 600:200:201 /800：201
##NUAA 341:86 /256:85:86
##NUDT 1061:266 / 796:265:266



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['ISTDU_Net_Text'], type=list,
                    help="model_name:'ISTDU_Net_Text_NewFFNimplaceMerge','ISTDU_Net_ab_Text','ISTDU_Net_Text'")
parser.add_argument("--dataset_names", default=['NUDT-SIRST'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='/home/ZSY/BasicIRSTD-text/dataset', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='/home/ZSY/BasicIRSTD-text/log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--optimizer_name", default='AdamW', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 0.005}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict,
                    help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=3407, help="Threshold for test,原本设置是42")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)


def train():
    print(f"当前设置的 batch size: {opt.batchSize}")
    print(f"当前设置的 optimizer name: {opt.optimizer_name}")

    ##训练数据集加载
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    net = Net(model_name=opt.model_name, mode='train').to(device)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    # resume
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']

    ### Default settings
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs': 400, 'step': [200, 300], 'gamma': 0.1}

    ### Default settings of DNANet
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 1500, 'min_lr': 1e-5}

    if opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': 0.005, 'weight_decay': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 500, 'min_lr': 1e-6}
        

    opt.nEpochs = opt.scheduler_settings['epochs']
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask, texts) in enumerate(train_loader):
            img, gt_mask = Variable(img).to(device), Variable(gt_mask).to(device)
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(device)
            if img.shape[0] == 1:  # 如果batch大小为1，跳过这个batch
                continue

            pred = net.forward(img, text_tokens)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            # 在第一轮训练时打印 texts 信息
            if idx_epoch == 0 and idx_iter == 0:
                print(f"Epoch {idx_epoch + 1}, Batch {idx_iter + 1}: texts = {texts}")

        scheduler.step()  # 更新学习率

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        if (idx_epoch + 1) % 10 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, current_lr---%f'
                  % (idx_epoch + 1, total_loss_list[-1], current_lr))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, current_lr---%f\n'
                        % (idx_epoch + 1, total_loss_list[-1], current_lr))
            total_loss_epoch = []  # 清空total_loss_epoch列表，为下一个epoch记录做准备

        if (idx_epoch + 1) % 50 == 0:
            # 保存路径
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)

        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)


def test(save_pth):
    # 创建测试数据集加载器
    test_set = TestSetLoader_inTrain(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # 打印测试集的数量
    print(f"Number of test samples: {len(test_set)}")

    # 加载网络
    net = Net(model_name=opt.model_name, mode='test').to(device)
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()  # 设置网络为评估模式

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    # 遍历测试数据集
    for idx_iter, (img, gt_mask,texts, size, _) in enumerate(test_loader):
        img = Variable(img).to(device)
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)

        pred = net.forward(img, text_tokens)
        pred = pred[:, :, :size[0], :size[1]]  # 将预测结果裁剪到原图大小
        gt_mask = gt_mask[:, :, :size[0], :size[1]]  # 将真实标签裁剪到原图大小
        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)  # 更新mIoU评估器
        eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)  # 更新PD和FA评估器

        # 在第一轮验证时打印 texts 信息
        if idx_iter == 0:
            print(f"Validation Batch {idx_iter + 1}: texts = {texts}")

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            # 打开一个日志文件，用于记录训练过程中的信息
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()