# 로보플로우
# 화재생성
#%%
## 학습 데이터셋 구성
import numpy as np
import pandas as pd
import os
import glob
import torch
import cv2 as cv
#%%
# 학습 데이터셋 구성
# 학습 데이터셋은 3개의 폴더로 구성
# train, test, valid
# train 폴더에는 학습에 사용할 이미지 파일과 라벨 파일이 저장
# test 폴더에는 학습에 사용하지 않을 이미지 파일과 라벨 파일이 저장
# valid 폴더에는 학습에 사용하지 않을 이미지 파일과 라벨 파일이 저장

train_img_list = glob.glob('C:/yolov5/smorke/train/images/*.jpg')
test_img_list  = glob.glob('C:/yolov5/smorke/test/images/*.jpg')
valid_img_list = glob.glob('C:/yolov5/smorke/valid/images/*.jpg')

len(train_img_list), len(test_img_list), len(valid_img_list)

# %%

with open('C:/yolov5/smorke/train.txt', 'w') as f:
    for img in train_img_list:
        f.write(img+'\n')

with open('C:/yolov5/smorke/test.txt', 'w') as f:
    for img in test_img_list:
        f.write(img+'\n')

with open('C:/yolov5/smorke/valid.txt', 'w') as f:
    for img in valid_img_list:
        f.write(img+'\n')


# %%        

# data.yaml 파일 수정(파일 경로)
# yolov5s.yaml 파일 수정(nc detection class 수)


# python train.py --img 480 --batch 2 --epochs 30 --data ./smoke/data.yaml --cfg ./models/yolov5s.yaml --name ./smoke/smoke_res --cache
# python detect.py --weights ./runs/train/smoke/smoke_res/weights/best.pt --img 480 --conf 0.4 --source ./smoke/forest_fire.mp4 --save-txt --save-conf

# opencv 를활용하여 동영상을 읽어서 이미지를 추출

vedio_path = 'C:/yolov5/smoke/forest_fire.mp4'
cap = cv.VideoCapture(vedio_path)

# 프레임 스탭수
frame_step = 10

# 프레임 번호
frame_num = 0

# 이미지 번호
img_num = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    if frame_num % frame_step == 0:
        img_name = 'C:/yolov5/smoke/images/img_%04d.jpg' % img_num
        cv.imwrite(img_name, frame)
        img_num += 1

    frame_num += 1
    
cap.release()

# %%

import sklearn as sk
import sklearn.model_selection  as skms

# 이미지 파일 리스트
img_list = glob.glob('C:/yolov5/smoke/images/*.jpg')

# 학습 데이터셋과 테스트 데이터셋으로 분리
train_img_list, test_img_list = skms.train_test_split(img_list, test_size=0.2, random_state=42)

# 학습 데이터셋과 검증 데이터셋으로 분리
train_img_list, valid_img_list = skms.train_test_split(train_img_list, test_size=0.2, random_state=42)

len(train_img_list), len(test_img_list), len(valid_img_list)

# %%

with open('C:/yolov5/smoke/train.txt', 'w') as f:
    for img in train_img_list:
        f.write(img+'\n')
        
with open('C:/yolov5/smoke/test.txt', 'w') as f:
    for img in test_img_list:
        f.write(img+'\n')

with open('C:/yolov5/smoke/valid.txt', 'w') as f:
    for img in valid_img_list:
        f.write(img+'\n')

# %%

# newtrain/labels 폴더에 있는 라벨 파일을 읽어서 첫번째 줄에 있는 클래스 번호를 0으로 변경
# 라벨 파일의 첫번째 줄은 클래스 번호, 중심점 x, 중심점 y, 너비, 높이
    
label_list = glob.glob('C:/yolov5/smoke/new_train/labels/*.txt')

for label in label_list:
    with open(label, 'r') as f:
        lines = f.readlines()
        
    with open(label, 'w') as f:
        for line in lines:
            line = line.replace('05', '0')
            f.write(line)            
# %%

# python train.py --img 480 --batch 2 --epochs 30 --data ./smoke/data.yaml --cfg ./models/yolov5s.yaml --name ./smoke/smoke_res --cache --weights ./runs/train/smoke/smoke_res/weights/best.pt
# python detect.py --weights ./runs/train/smoke/smoke_res14/weights/best.pt --img 480 --conf 0.4 --source ./smoke/forest_fire.mp4 --save-txt --save-conf