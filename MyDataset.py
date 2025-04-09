import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

def generate_mask(image):
    """
    从图像中动态生成掩膜（背景为0，前景为1）
    :param image: 输入图像（H, W, C）
    :return: 掩膜（H, W）
    """
    mask = (image.sum(axis=-1) > 0)  # 对RGB通道求和，大于0为前景
    return mask.astype(np.uint8)  # 转换为0-1掩膜

class BambooDataset(Dataset):
    def __init__(self, root_dir, weak_transform, strong_transform=None, is_labeled=True):
        self.root_dir = root_dir
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.is_labeled = is_labeled
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if self.is_labeled:
            for label in sorted(os.listdir(self.root_dir)):
                label_dir = os.path.join(self.root_dir, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_name)
                        samples.append((img_path, int(label)))
        else:
            for img_name in os.listdir(self.root_dir):
                img_path = os.path.join(self.root_dir, img_name)
                samples.append(img_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_labeled:
            img_path, label = self.samples[idx]
            image = cv2.imread(img_path)

            new_img = self.weak_transform(image=image)["image"]
            # 转换为灰度纹理图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = self.weak_transform(image=gray_image)["image"]
            return new_img, gray_image, label
        else:
            img_path = self.samples[idx]
            image = cv2.imread(img_path)

            new_img = self.weak_transform(image=image)["image"]
            # 转换为灰度纹理图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = self.weak_transform(image=gray_image)["image"]

            return new_img, gray_image


class SemiDataLoader:
    def __init__(self, labeled_ds, unlabeled_ds, batch_size=16, num_workers=2):
        self.labeled_loader = DataLoader(
            labeled_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        self.unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=batch_size * 2,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )

        self.max_batches = len(self.labeled_loader)
    def __iter__(self):
        self.current_batch = 0
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)

        return self

    def __next__(self):
        if self.current_batch >= self.max_batches:
            raise StopIteration
        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(self.labeled_iter)

        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch = next(self.unlabeled_iter)
        self.current_batch += 1
        return labeled_batch, unlabeled_batch