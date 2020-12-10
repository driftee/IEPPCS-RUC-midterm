import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
import cv2
from copy import copy
from dataset import MyDataset, get_transform

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  
    tmp_dict = checkpoint["class_dict"]
    model.load_state_dict(checkpoint['model_state_dict']) 
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, tmp_dict


res_path = 'dataset/result'
os.makedirs(res_path, exist_ok = True)
def predict_on_extra(model, path, class_dict):
    if torch.cuda.is_available():
        model.cuda()
    file_list = os.listdir(path)
    points = []
    transfrom = get_transform(input_size = cfg.INPUT_SIZE, imgset = "test")
    for i in tqdm(file_list):
        img_path = os.path.join(path, i)
        save_path = os.path.join(res_path, i)
        img = cv2.imread(img_path)
        ii = 0
        jj = 0
        img_w, img_h, _ = img.shape
        while ii + 84 < img_w:
            while jj + 84 < img_h:
                to_pred = img[ii:ii+84, jj:jj+84]
                to_pred = Image.fromarray(to_pred).convert('RGB')
                to_pred = transfrom(to_pred)
                to_pred = to_pred.unsqueeze(0)
                if torch.cuda.is_available():
                    to_pred = to_pred.cuda()
                with torch.no_grad():
                    out = model(to_pred)
                class_ = torch.argmax(out, dim=1).cpu().item()
                if class_dict[class_] == 'garbage':
                    points.append((jj + 42, ii + 42))
                jj += 84
            ii += 84
            jj = 0
        for point in points:
            cv2.circle(img, point, radius = 15, color = (0, 0, 255), thickness = -1)
        cv2.imwrite(save_path, img)




if len(cfg.USING_GPU) > 1:
    devices = [int(i) for i in cfg.USING_GPU]
    str_ = ""
    for i in devices:
        str_ += str(i) + " "
    print("========= Using multi GPU {}=========".format(str_))
    model = nn.DataParallel(model, device_ids = devices)
else:
    print("========= Using single GPU {} =========".format(cfg.USING_GPU[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.USING_GPU[0]


if __name__ == "__main__":

    trained_model = os.path.join(cfg.SAVE_FOLDER, "best.pth")
    model, class_dict = load_checkpoint(trained_model)

    test_dataset = MyDataset('test', class_dict=class_dict)

    acc_per_class = {i:0 for i in class_dict}
    base_per_class = {i:0 for i in class_dict}
    acc_all = 0
    for i in tqdm(range(len(test_dataset))):
        img, label = test_dataset[i]
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        if prediction == label:
            acc_all += 1
            acc_per_class[label.item()] += 1
        base_per_class[label.item()] += 1

    print("Global accuracy {}%".format(acc_all / len(test_dataset) * 100))
    for i in class_dict:
        print("Class: {}, accuracy: {}%".format(class_dict[i], acc_per_class[i] / base_per_class[i] * 100))

    predict_on_extra(model, 'dataset/extra', class_dict)




