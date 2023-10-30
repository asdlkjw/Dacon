import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import _LRScheduler
from torchcontrib.optim import SWA

from sklearn.metrics import recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import math
from glob import glob

import joblib
from tqdm.auto import tqdm
from natsort import natsorted
print('cuda on:', torch.cuda.is_available())

# config

index = 0
img_size = 224
title = 'TW_eva_224_AdamW'

data_dir = "./data/"

# train, validation 데이터 분할
X_train = np.array(natsorted(glob(f"{data_dir}train_crop/*")))
y_train = []
for i in range(len(X_train)):
    y_train.append(X_train[i].split("/")[-1].split("_")[0])
y_train = np.array(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01, stratify=y_train, random_state=42)

class Custom_Dataset(Dataset):
    def __init__(self, path, img_height, img_width, transform, is_test=False):
        self.img_ids = path
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        if is_test == False:
            is_test = 'train'
        else:
            is_test = 'test'
        self.is_test = is_test

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = pd.read_pickle(f'{img_id}')
        if self.transform is not None:
            img = self.transform(image=img)['image']
        label = int(img_id.split("/")[-1].split("_")[0])
        return img, label, img_id

# cutmix or cutout을 추가하여 실험할 예정

train_aug = A.Compose([
    A.Resize(img_size, img_size),
    A.OneOf([
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.VerticalFlip(p=0.3)
    ], p=0.3),
    A.OneOf([
        A.MotionBlur(p=0.3),
        A.OpticalDistortion(p=0.3),
        A.GaussNoise(p=0.3)
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.pytorch.transforms.ToTensorV2(),
])

valid_aug = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.pytorch.transforms.ToTensorV2(),
])

trn_dataset = Custom_Dataset(path=X_train,
                             img_height=img_size,
                             img_width=img_size,
                             transform=train_aug,
                             is_test=False,
                             )

vld_dataset = Custom_Dataset(path=X_val,
                             img_height=img_size,
                             img_width=img_size,
                             transform=valid_aug,
                             is_test=False,
                             )

trn_loader = DataLoader(trn_dataset,
                        shuffle=True,
                        num_workers=16,
                        batch_size=64,
                        )

vld_loader = DataLoader(vld_dataset,
                        num_workers=16,
                        batch_size=64,
                        )

class EVA(nn.Module):
    def __init__(self, class_n=34):
        super().__init__()
        self.model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=34)
        # self.model = timm.create_model('tf_efficientnetv2_b3.in21k', pretrained=True, num_classes=34, in_chans=3)
        
    def forward(self, x):
        x = self.model(x)
        return x

model = EVA()
model = model.cuda()
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       patience=5,
                                                       factor=0.1,
                                                       verbose=True,
                                                       )

best_score = -1
final_score = []

lrs = []
early_stop = np.inf
for ep in range(30):
    train_loss = []
    val_loss = []
    val_true = []
    val_pred = []

    print(f'======================== {ep} Epoch train start ========================')

    model.train()
    for inputs, targets, img_path in tqdm(trn_loader):
        targets = torch.from_numpy(np.array(targets))
        inputs = inputs.cuda()  # GPU 환경에서 돌아가기 위해 cuda() 사용
        targets = targets.cuda()  # 정답 데이터

        # 변화도(Gradient) 매개변수를 0으로 설정
        optimizer.zero_grad()
        logits = model(inputs.float())  # 모델의 결과값

        # 순전파 + 역전파 + 최적화
        loss = loss_fn(logits, targets.long())  # 정답과 예측값의 오차 계산
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        for inputs, targets, img_path in tqdm(vld_loader):
            targets = torch.from_numpy(np.array(targets))
            inputs = inputs.cuda()
            targets = targets.cuda()

            logits = model(inputs.float())

            loss = loss_fn(logits, targets.long())

            val_loss.append(loss.item())

            # 정답 비교 코드
            logits = logits.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            # # 소프트맥스 함수를 적용하여 확률 값으로 변환
            # probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

            # # 가장 확률이 높은 클래스 선택
            # pred = np.argmax(probabilities, axis=1)

            pred = np.argmax(logits, axis=1)
            F1_score = f1_score(targets, pred, average='macro')
            final_score.append(F1_score)

    Val_loss_ = np.mean(val_loss)
    Train_loss_ = np.mean(train_loss)
    Final_score_ = np.mean(final_score)
    print(f'train_loss: {Train_loss_:.5f}; val_loss: {Val_loss_:.5f}; f1_score: {Final_score_:.5f}')

    if Final_score_ > best_score and early_stop > Val_loss_:
        best_score = Final_score_
        early_stop = Val_loss_
        early_count = 0
        state_dict = model.cpu().state_dict()
        model = model.cuda()
        torch.save(state_dict, f"./weights/{title}_{ep}.pt")

        print('\n SAVE MODEL UPDATE \n\n')

    elif early_stop < Val_loss_ or Final_score_ < best_score:
        early_count += 1

    if early_count == 20:
        print('early stop!!!')
        break   