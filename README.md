## Text-IRSTD:
Paper: https://ieeexplore.ieee.org/document/10937752, 
https://ieeexplore.ieee.org/document/11667722.
## Contributions
1.We introduce a novel approach leveraging semantic text to guide infrared small target detection, called Text-IRSTD. It innovatively expands classical IRSTD to text-guided IRSTD, providing a new research idea.

2.We devise a novel fuzzy semantic text prompt to accommodate ambiguous target categories.

3.We propose a progressive cross-modal semantic interaction decoder (PCSID) to facilitate information fusion between texts and images.

4.We construct a new benchmark consisting of 2,755 infrared images of different scenarios with fuzzy semantic textual annotations, called FZDT.
## Requirements
conda create -n text python=3.8

conda activate text

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

##pip install packaging
#pip install timm==0.4.12
#pip install pytest chardet yacs termcolor
#pip install submitit tensorboardX
#pip install triton==2.0.0
#pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
#pip install mamba_ssm==1.0.1  # mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
#pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
pip install openai-clip==1.0.1
pip install opencv-python
## Datasets
NUAA-SIRST [download](https://github.com/YimianDai/sirst) [paper](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html?ref=https://githubhelp.com)
NUDT-SIRST [download](https://github.com/YeRen123455/Infrared-Small-Target-Detection) [paper](https://ieeexplore.ieee.org/document/9864119)
IRSTD-1K [download](https://github.com/RuiZhang97/ISNet) [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html)
SIRST-UAVB [download](https://github.com/JN-Yang/PConv-SDloss-Data) [paper](https://ojs.aaai.org/index.php/AAAI/article/view/32996)
IRSTD-Air [download](https://github.com/Heron0625/PBT) [paper](https://ieeexplore.ieee.org/abstract/document/10559654)
Our project has the following structure:
|--./dataset/
|   |--IRSTD-1K
|   |    |--images
|   |    |    |--XDU0.png
|   |    |    |--XDU1.png
|   |    |    |--...
|   |    |--masks
|   |    |    |--XDU0.png
|   |    |    |--XDU1.png
|   |    |    |--...
|   |    |--img_idx
|   |    |    |--train_IRSTD-1K.txt
|   |    |    |--test_IRSTD-1K.txt
|   |    |    |--val_IRSTD-1K.txt
|   |    |--prompt
|   |    |    |--XDU0.txt
|   |    |    |--XDU1.txt
|   |    |    |--...
## Training
python train_text.py
## Test
python test_text.py
## Citation
If you use this code, please kindly cite our paper:
@ARTICLE{huang2025ICCV,
  author={Huang, Feng and Zheng, Shuyuan and Qiu, Zhaobin and Liu, Huanxian and Bai, Huanxin and Chen, Liqiong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Text-IRSTD: Leveraging Semantic Text to Promote Infrared Small Target Detection in Complex Scenes},
  year={2025},
  pages={10635-10644},
  publisher={IEEE}
}

@ARTICLE{chen2026TGRS,
  author={Chen, Liqiong and Liu, Yuhao and Qiu, Zhaobin and Zheng, Shuyuan and Zou, Jiaxin and Yuan, Yangping and Huang, Feng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Text-IRSTD: Leveraging Fuzzy Text to Promote Infrared Small Target Detection in Complex Scenes},
  year={2026},
  volume={64},
  pages={5010216-5010216},
  publisher={IEEE}
}
## Contact
If you have any questions, please contact:  
Author: Zhaobing Qiu  
Email: qiuzhaobing@fzu.edu.cn  
Copyright: Fuzhou University 
## License
This code is only freely available for non-commercial research use.

If you find some help for you, star is a good reward ^_^. 
