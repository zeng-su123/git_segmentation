import torch
import numpy as np
result = np.load(r'H:\other_files\Heart_OpenDataset\OpenDataset\Training\Training-corrected\Labeled_npy\A0S9V9/A0S9V9_slice5_phase9.npy')
# # shape = (216,256)
# 从此处可得知 此时的npy 数据格式为元组，非三维图像数据  和原始的nii.gz的数据值一样
mask = np.load(r'H:\other_files\Heart_OpenDataset\OpenDataset\Training\Training-corrected-contours\Labeled_npy\A0S9V9/A0S9V9_slice5_phase9.npy')
# # shape = （216，256) # 虽然通过软件打开标签文件，可以很清晰的看到图像的样子

print("")