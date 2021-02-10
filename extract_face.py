from random import randint
import cv2
import sys
import os
import traceback
import pandas as pd
import numpy as np
from natsort import natsorted
import glob

import numpy as np
import cv2 as cv


# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드
face_cascade = cv.CascadeClassifier('haarcascade_frontface.xml')


# 얼굴과 눈을 검출할 그레이스케일 이미지를 준비
image_path = glob.glob('C:/Users/mmclab1/Desktop/AI training data set/data/video_2images/*.*', recursive=False)
image_path = natsorted(image_path) #'C:/Users/mmclab1/Desktop/AI training data set/data/video_2images\\id32_id35_0002_frame-2.jpg'


for path in image_path:
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 이미지에서 얼굴을 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    # 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴
    padding = 30
    for (x,y,w,h) in faces:
        sub_img = img[y - padding:y + h + padding, x - padding:x + w + padding]
        fin_img = cv2.resize(sub_img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img_name = path.split('\\')[1]
        cv2.imwrite("C:/Users/mmclab1/Desktop/AI training data set/data/extracted face/%s.jpg" %
            img_name.split('.')[0], fin_img)

print('success')


