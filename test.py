# -*- coding:utf-8 -*-

import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
import cv2
from data import tta_test_transform, get_test_transform

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict_on_large_img(model, img_path):
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    if torch.cuda.is_available():
        model.cuda()
    # img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    img_w, img_h, _ = img.shape
    map_ = np.zeros((img_w, img_h))
    base = np.zeros((img_w, img_h))
    ii = 0
    jj = 0
    while ii + 84 < img_w:
        while jj + 84 < img_h:
            base[ii:ii+84, jj:jj+84] += 1
            to_pred = img[ii:ii+84, jj:jj+84]
            to_pred = Image.fromarray(to_pred).convert('RGB')
            to_pred = get_test_transform(size=cfg.INPUT_SIZE)(to_pred).unsqueeze(0)
            if torch.cuda.is_available():
                to_pred = to_pred.cuda()
            with torch.no_grad():
                out = model(to_pred)
            class_ = torch.argmax(out, dim=1).cpu().item()
            map_[ii:ii+84, jj:jj+84] += class_
            jj += 84
            print("{}, {}".format(ii, jj), end = '\r')
        ii += 84
        jj = 0
    result = map_ / base > 0.5
    img = cv2.imread(img_path)
    for i in tqdm(range(img_w)):
        for j in range(img_h):
            if result [i][j]:
                cv2.circle(img, (j, i), radius = 5, color = (0, 255, 0), thickness = -1)
    cv2.imwrite('result.png', img)


def predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split()[0]
        # print(img_path)
        _id.append(os.path.basename(img_path))
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list


def tta_predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img1 = Image.open(img_path).convert('RGB')
        # print(type(img))
        pred = []
        for i in range(8):
            img = tta_test_transform(size=cfg.INPUT_SIZE)(img1).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred.append(prediction)
        res = Counter(pred).most_common(1)[0][0]
        pred_list.append(res)
    return _id, pred_list


if len(cfg.USING_GPU) > 1:
    model = nn.DataParallel(model, device_ids = cfg.GPUS)
else:
    print("Using single GPU {}".format(cfg.USING_GPU[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.USING_GPU[0]


if __name__ == "__main__":

    trained_model = cfg.TRAINED_MODEL
    model_name = cfg.model_name
    # with open(cfg.TEST_LABEL_DIR,  'r')as f:
    #     imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    # _id, pred_list = predict(trained_model)
    predict_on_large_img(trained_model, "/home/huangjiawen/file/lgx_trial/data_cls/big_img/43.tif")

    # submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    # submission.to_csv(cfg.BASE + '{}_submission.csv'
    #                   .format(model_name), index=False, header=False)



