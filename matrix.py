import torch
import seaborn as sns
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from MosoBamboo.AgeDetect.model.Classify.MyDataset import data_cfg
from MosoBamboo.AgeDetect.model.Classify.Mytrans import tran_method
from MosoBamboo.AgeDetect.model.Classify.pro.MyDataset import BambooDataset
from MosoBamboo.AgeDetect.model.Classify.pro.model import MBADNet
import matplotlib.pyplot as plt




# 示例用法
if __name__ == "__main__":
    import os
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MBADNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(
        r"E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\model\Classify\pro\checkpoints\best_model.pth",
        map_location=device,
        weights_only=True  # 解决 FutureWarning
    ))

    weak_trans = tran_method['non_extra_trans']
    y_true = []
    y_pred = []
    model.eval()
    eval_data = data_cfg(
        r"E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\eval_data",
        is_labeled=True,
        trans_methods=tran_method["non_extra_trans"])
    with torch.inference_mode(), \
            tqdm(DataLoader(
                BambooDataset(
                    root_dir='E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\eval_data',
                    is_labeled=True,
                    weak_transform=tran_method['non_extra_trans']
                ),
            batch_size=8,
            num_workers=2,
            pin_memory=True
        ), desc="Validating", leave=False) as val_pbar:

        for inputs in val_pbar:
            lab_img = inputs[0].to(device, non_blocking=True)
            gray_img = inputs[1].to(device, non_blocking=True)
            labels = inputs[2].to(device, non_blocking=True)

            outputs = model(lab_img, gray_img)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 转换为 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(cm)

    # 可视化混淆矩阵
    class_names = ['I', 'II', 'III', 'IV']  # 替换为你的类别名称
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('predict class')
    plt.ylabel('ground-truth')
    plt.title('混淆矩阵')
    plt.show()

    # 打印分类报告
    print(classification_report(y_true, y_pred, target_names=class_names))

