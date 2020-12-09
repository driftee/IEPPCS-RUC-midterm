import os

# 训练时batch的大小
BATCH_SIZE = 100

# 网络默认输入图像的大小
INPUT_SIZE = 84

# 训练最多的epoch
MAX_EPOCH = 100

# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

# 优化器超参数
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# 初始学习率
LR = 1e-3

# 使用的gpu列表
USING_GPU = ["0"]

DATASET_PATH = 'dataset/'

SAVE_FOLDER = "save/"



