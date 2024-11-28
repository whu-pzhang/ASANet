import os
import random
import numpy as np

# 设置随机种子，以确保划分一致
random_seed = 42
random.seed(random_seed)

# 输入文件夹
root_folder = '/data1/pengbc/code/dataset/DDHR-DATA/korea/'

img_folder = root_folder + 'label'
# 获取文件夹中的所有文件
files = os.listdir(img_folder)

# 随机打乱文件顺序
# random.shuffle(files)

# 划分数据集
split_index = len(files) // 100
train_files, val_files = [], []
for j in range(1,split_index+1):
    if j%2 == 0:
        for i in range(100*j,100*(j+1)):
            val_files.append(str(i))
    # else:
    #     for i in range(100*j,100*(j+1)):
    #         train_files.append(str(i))
    
for i in range(1, len(files)):
    if str(i) not in val_files:
        train_files.append(str(i))
# train_files = files[:split_index]
# val_files = files[split_index:]

# 输出文件路径
train_txt = root_folder+ 'train.txt'
val_txt = root_folder+ 'val.txt'

# 写入训练集文件名到train.txt
with open(train_txt, 'w') as train_file:
    for filename in train_files:
        train_file.write(filename + '\n')

# 写入验证集文件名到validation.txt
with open(val_txt, 'w') as val_file:
    for filename in val_files:
        val_file.write(filename + '\n')

print(f"数据集划分完成，并保存到 {train_txt} 和 {val_txt}")
