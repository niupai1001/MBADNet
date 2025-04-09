import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
import random
from albumentations.core.transforms_interface import ImageOnlyTransform

class RandAugment(ImageOnlyTransform):
    def __init__(self, num_ops=2, magnitude=10, always_apply=False, p=1.0):
        """
        RandAugment 实现。
        :param num_ops: 每次应用的增强操作数量。
        :param magnitude: 增强强度，范围通常为 [0, 10]。
        :param always_apply: 是否总是应用。
        :param p: RandAugment 的总体概率。
        """
        super().__init__(always_apply)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.augment_pool = [
            A.RandomBrightnessContrast(brightness_limit=magnitude / 10, contrast_limit=magnitude / 10, p=1.0),
            A.HueSaturationValue(hue_shift_limit=int(magnitude), sat_shift_limit=int(magnitude), val_shift_limit=int(magnitude), p=1.0),
            A.Affine(scale=(1.0 - magnitude / 50, 1.0 + magnitude / 50),translate_percent=(0.2, 0.2),rotate=(-15, 15), p=1.0),  # 仿射变换
            A.MotionBlur(blur_limit=7, p=1.0),  # 运动模糊
            A.OpticalDistortion(distort_limit=0.5, p=1.0),  # 光学畸变
            A.GaussNoise(p=1.0),  # 高斯噪声
            A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(1,int(40 * magnitude / 10)), hole_width_range=(1,int(40 * magnitude / 10)), p=1.0),  # 随机遮挡
            A.Perspective(scale=(0.05, 0.1), p=1.0)  # 透视变换
        ]

    def apply(self, img, **params):
        # 随机选择 num_ops 个增强操作
        augmentations = random.sample(self.augment_pool, self.num_ops)
        pipeline = A.Compose(augmentations)
        return pipeline(image=img)["image"]


class SelectiveAugment:
    def __init__(self, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, image, **kwargs):
        # 生成前景掩码
        mask = (image.sum(axis=2) > 0).astype(np.uint8)

        # 提取前景区域
        foreground = image * mask[..., np.newaxis]

        # 对前景应用亮度对比度增强
        transform = A.RandomBrightnessContrast(
            brightness_limit=self.brightness_limit,
            contrast_limit=self.contrast_limit,
            p=1.0
        )
        augmented_foreground = transform(image=foreground)['image']

        # 合并前景和背景
        result = augmented_foreground + image * (1 - mask[..., np.newaxis])
        return result
tran_method = {
    'extra_trans':
    A.Compose([
        A.Resize(height=448, width=448, p=1.0),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    'non_extra_trans':
    A.Compose([
        A.Resize(height=448, width=448, p=1.0),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2()])
    # 'unlabel':A.Compose([
    #     A.Resize(height=448, width=448, p=1.0),
    #     A.CenterCrop(height=448, width=448, p=1.0),
}

def get_tran_methods(magnitude=1):
    """
    根据传入的 magnitude 动态生成数据增强配置。
    :param magnitude: RandAugment 的增强强度。
    :return: 数据增强方法字典。
    """
    return {
        'extra_trans':
        A.Compose([
            A.Resize(height=448, width=448, p=1.0),
            # A.HorizontalFlip(p=0.4),
            # A.VerticalFlip(p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.1),
            # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),  # 随机明亮对比度
            A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'non_extra_trans': A.Compose([
            A.Resize(height=448, width=448, p=1.0),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    }


def get_transforms():
    # 弱增强（用于有标签数据和教师模型预测）
    non =  A.Compose([
            A.Resize(height=448, width=448, p=1.0),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    weak = A.Compose([
        A.Resize(448, 448),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 强增强（用于学生模型训练）
    strong = A.Compose([
        A.Resize(448, 448),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.5, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return {'weak': weak, 'strong': strong, 'non': non}