import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
##导入数据集
from dataset_text import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time

#clip
import clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##模型预测阶段，即从模型推理到评估的一体化过程
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
#模型名称
parser.add_argument("--model_names", default=['ab_MSFEM_without_CA'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
# 模型权重文件路径
parser.add_argument("--pth_dirs", default=['IRSTD-1K/ISTDU_Net_Text_NewFFNimplaceMerge_600.pth.tar',]

, type=list, help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
# 数据集目录
parser.add_argument("--dataset_dir", default='/home/ZSY/BasicIRSTD-text/dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
# 是否保存预测图像
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
# 预测图像保存目录
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='/home/ZSY/BasicIRSTD-text/log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

# 全局变量
global opt
opt = parser.parse_args()

def test():
    # 加载测试数据集
    test_set = TestSetLoader_inTest(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # 打印测试集的数量
    print(f"Number of test samples: {len(test_set)}")

    # 加载预训练模型
    net = Net(model_name=opt.model_name, mode='test').to(device)
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()

    # 初始化评估指标
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()

    # 对测试集逐个样本进行预测和评估
    for idx_iter, (img, gt_mask, texts, size, img_dir) in enumerate(test_loader):

        # 在第一轮验证时打印 texts 信息
        if idx_iter == 0:
            print(f"Validation Batch {idx_iter + 1}: texts = {texts}")

        img = Variable(img).to(device)
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)
        pred = net.forward(img, text_tokens)
        # 裁剪预测结果以匹配原始图像大小
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
        
        ### save img保存预测图像
        if opt.save_img == True:
            img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
            if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
            img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')

            # 获取评估结果并打印
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    # 如果没有指定模型权重路径，则依次加载每个模型和数据集进行测试
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        # 如果指定了权重路径，则根据路径进行测试
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    #if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
