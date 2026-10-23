import torch

def load_model_weights(filepath):
    try:
        # 尝试加载模型权重
        model_weights = torch.load(filepath, map_location='cpu')
        print("Model weights loaded successfully.")
        return model_weights
    except Exception as e:
        print("Failed to load model weights. Error:", str(e))
        return None

def print_model_info(model_weights):
    if model_weights is not None:
        # 打印权重中的第一个层的参数，作为示例
        first_param_key = next(iter(model_weights))
        print("First parameter key:", first_param_key)
        print("Shape of the first parameter:", model_weights[first_param_key].shape)

        # 打印所有层的名称和形状
        print("\nAll parameters in the model:")
        for param_key in model_weights:
            print(param_key, model_weights[param_key].shape)
    else:
        print("No model weights to display.")

# 调用函数
model_path = "C:/Users\Administrator\Desktop\FZU/1\ISTD/2021/2021-ACM/acm-pytorch-main/acm-pytorch-main/result/2024.4.8-ISTD-1K-weight/Epoch-294_IoU-0.6237_nIoU-0.5934.pkl"  # 替换为你的.pkl文件路径
model_weights = load_model_weights(model_path)
print_model_info(model_weights)
