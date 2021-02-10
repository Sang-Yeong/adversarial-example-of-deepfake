import pandas as pd
import numpy as np
from natsort import natsorted
import glob
import os
from PIL import Image


image_path = glob.glob('C:/Users/mmclab1/Desktop/AI training data set/data/sampled_face/train/*.*', recursive=False)
image_path = natsorted(image_path)
# 'C:/Users/mmclab1/Desktop/AI training data set/data/sampled_face/train\\id32_id35_0002_frame-2.jpg'
# id1_id3_0001_frame-0_0_FGSM_eps2_547

# ===================================== make csv file =======================================
image_name = []
base_id = []
target_id = []
method = []
epsilon = []

for i in image_path:
    file_components = i.split("\\")[1:5]
    file_com = ''.join(file_components)  #'id1_id3_0001_frame-0_0_FGSM_eps2_547.jpg'

    file_com_image = file_com
    file_com_base_id = file_com.split('_')[0]+'_'+file_com.split('_')[2]
    file_com_target_id = file_com.split('_')[1]+'_'+file_com.split('_')[2]
    file_com_method = file_com.split('_')[-3]
    file_com_epsilon = (file_com.split('_')[-2])[3::]

    image_name.append(file_com_image)
    base_id.append(file_com_base_id)
    target_id.append(file_com_target_id)
    method.append(file_com_method)
    epsilon.append(file_com_epsilon)


dataFrame = pd.DataFrame(image_name, columns=['image_name'])
dataFrame['base_id'] = base_id
dataFrame['target_id'] = target_id
dataFrame['method'] = method
dataFrame['epsilon'] = epsilon

dataFrame.to_csv('C:/Users/mmclab1/Desktop/AI training data set/data/sampled_face/train.csv')
print('success')

# method = []         # 0 ~ 절반: 0, 이후: 1
# id = []             # base image
# swap_id = []        # swap target
# target = []         # default = 0
# image_name = []     # image full name
#
# cnt = 1
#
# for i in image_path:
#     file_components = i.split("\\")[1:5]
#     file_components = ''.join(file_components)  #'id32_id35_0002_frame-2.jpg'
#
#     file_components_id = file_components.split('_')[0]+'_'+file_components.split('_')[2]
#     file_components_swap_id = file_components.split('_')[1]+'_'+file_components.split('_')[2]
#     file_components_image = file_components
#
#     id.append(file_components_id)
#     swap_id.append(file_components_swap_id)
#     target.append(0)
#     image_name.append(file_components_image)
#
#     if cnt >= 1270 // 2:
#         method.append(1)
#     else:
#         method.append(0)
#
#     cnt += 1
#
#
# dataFrame = pd.DataFrame(image_name, columns=['image_name'])
# dataFrame['method'] = method
# dataFrame['id'] = id
# dataFrame['target'] = target
#
# dataFrame.to_csv('C:/Users/mmclab1/Desktop/AI training data set/data/sampled_face/train.csv')
# print('success')
