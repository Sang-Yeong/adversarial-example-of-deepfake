import cv2
import pandas as pd
import numpy as np
from natsort import natsorted
import glob
import os
from PIL import Image


image_path = glob.glob('C:/Users/mmclab1/Desktop/AI training data set/data/Celeb-synthesis/*.mp4', recursive=False)
image_path = natsorted(image_path) # 'C:/Users/mmclab1/Desktop/AI training data set/data/Celeb-synthesis\\id1_id3_0001.mp4'


for path in image_path:
    vidcap = cv2.VideoCapture(path)
    count = 0
    while (vidcap.isOpened()):
        retval, image = vidcap.read()

        if (int(vidcap.get(1)) % 30 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            img_name = path.split('\\')[1]
            cv2.imwrite("C:/Users/mmclab1/Desktop/AI training data set/data/train/%s_frame-%d.jpg" % (
            img_name.split('.')[0], count), image)
            print('Saved frame%d.jpg' % count)
            count += 1

        if count == 3:
            break
    vidcap.release()

print('success')