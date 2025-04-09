import torch
import torch.nn.functional as F
import cv2
import numpy as np
from MosoBamboo.AgeDetect.model.Classify.Mytrans import tran_method
from MosoBamboo.AgeDetect.model.Classify.pro.model import MBADNet
import matplotlib.pyplot as plt


def get_gradcam(model, lab_img, gray_img, target_class=None):
    """
    生成颜色流和纹理流的Grad-CAM热图
    :param model: 训练好的FastMSAGNet模型
    :param lab_img: LAB颜色图像，形状为[1, 3, H, W]
    :param gray_img: 灰度图像，形状为[1, 1, H, W]
    :param target_class: 目标类别（默认使用模型预测的类别）
    :return: 颜色流的CAM、纹理流的CAM
    """
    model.eval()
    device = next(model.parameters()).device

    # 为颜色流和纹理流分别定义存储空间
    color_activations = {}
    color_gradients = {}
    texture_activations = {}
    texture_gradients = {}

    # 定义独立的钩子函数
    def color_forward_hook(module, input, output):
        color_activations['value'] = output

    def color_backward_hook(module, grad_input, grad_output):
        color_gradients['value'] = grad_output[0]

    def texture_forward_hook(module, input, output):
        texture_activations['value'] = output

    def texture_backward_hook(module, grad_input, grad_output):
        texture_gradients['value'] = grad_output[0]

    # 选择目标层
    # 颜色流：保持使用最后一个 ResidualSEBlock 的 conv 层
    color_target_layer = model.color_stream.blocks[-1].conv[-1]  # 颜色流最后一个卷积层

    # 纹理流：选择 FPN 的输出卷积层或 feature_extractors 的最后一个卷积层
    # 假设选择 FPN 的 output_conv 层以捕捉融合后的多尺度特征
    texture_target_layer = model.texture_stream.fpn.output_conv  # 纹理流 FPN 的输出卷积层

    # 注册钩子
    color_handle = color_target_layer.register_forward_hook(color_forward_hook)
    texture_handle = texture_target_layer.register_forward_hook(texture_forward_hook)
    color_handle_grad = color_target_layer.register_full_backward_hook(color_backward_hook)
    texture_handle_grad = texture_target_layer.register_full_backward_hook(texture_backward_hook)

    # 前向传播
    output = model(lab_img, gray_img)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    output[:, target_class].backward()

    # 生成颜色流的CAM
    color_activations_value = color_activations['value']  # [B, C, H, W]
    color_gradients_value = color_gradients['value']  # [B, C, H, W]
    color_weights = F.adaptive_avg_pool2d(color_gradients_value, 1)  # [B, C, 1, 1]
    color_cam = (color_weights * color_activations_value).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    color_cam = F.relu(color_cam)
    color_cam = F.interpolate(color_cam, size=lab_img.shape[2:], mode='bilinear', align_corners=False)
    color_cam = color_cam.squeeze().cpu().detach().numpy()
    color_cam = (color_cam - color_cam.min()) / (color_cam.max() - color_cam.min() + 1e-8)  # 归一化

    # 生成纹理流的CAM
    texture_activations_value = texture_activations['value']  # [B, C, H, W]
    texture_gradients_value = texture_gradients['value']  # [B, C, H, W]
    texture_weights = F.adaptive_avg_pool2d(texture_gradients_value, 1)  # [B, C, 1, 1]
    texture_cam = (texture_weights * texture_activations_value).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    texture_cam = F.relu(texture_cam)
    texture_cam = F.interpolate(texture_cam, size=gray_img.shape[2:], mode='bilinear', align_corners=False)
    texture_cam = texture_cam.squeeze().cpu().detach().numpy()
    texture_cam = (texture_cam - texture_cam.min()) / (texture_cam.max() - texture_cam.min() + 1e-8)

    # 移除钩子
    color_handle.remove()
    texture_handle.remove()
    color_handle_grad.remove()
    texture_handle_grad.remove()

    return color_cam, texture_cam, output


def visualize_cam(image, cam, alpha=0.5):
    """
    将CAM叠加到原始图像上
    :param image: 原始图像（RGB或灰度，H, W, C 或 H, W）
    :param cam: Grad-CAM生成的热图（H, W）
    :param alpha: 叠加透明度
    :return: 可视化结果
    """
    # 调整原始图像大小以匹配热图
    if image.ndim == 2:  # 灰度图像 [H, W]
        image_resized = cv2.resize(image, (cam.shape[1], cam.shape[0]))  # [448, 448]
        image_resized = image_resized[..., np.newaxis]  # [448, 448, 1]
        image_resized = np.repeat(image_resized, 3, axis=2)  # [448, 448, 3]
    elif image.ndim == 3 and image.shape[-1] == 1:  # 灰度图像 [H, W, 1]
        image_resized = cv2.resize(image[..., 0], (cam.shape[1], cam.shape[0]))  # [448, 448]
        image_resized = image_resized[..., np.newaxis]  # [448, 448, 1]
        image_resized = np.repeat(image_resized, 3, axis=2)  # [448, 448, 3]
    else:  # RGB 图像 [H, W, 3]
        image_resized = cv2.resize(image, (cam.shape[1], cam.shape[0]))  # [448, 448, 3]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # 转换为热图 [448, 448, 3]
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image_resized / 255.0)  # 叠加
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)


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
    dir_path = r'E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\label_data\1'
    for img_path in os.listdir(dir_path):
        abs_path = os.path.join(dir_path, img_path)
        print(img_path)
        image = cv2.imread(abs_path)

        # 应用变换
        new_img = weak_trans(image=image)["image"]  # 假设返回 [H, W, 3] 的 NumPy 数组
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # [H, W]
        gray_image = weak_trans(image=gray_image)["image"]  # 假设返回 [H, W] 或 [H, W, 1]

        # 转换为 PyTorch 张量并调整维度

        # 处理颜色图像
        if isinstance(new_img, np.ndarray):
            if new_img.shape[-1] == 3:  # [H, W, 3]
                new_img = np.transpose(new_img, (2, 0, 1))  # [3, H, W]
            new_img = torch.from_numpy(new_img).float()  # 转换为张量
        new_img = new_img.unsqueeze(0).to(device)  # [1, 3, H, W]

        # 处理灰度图像
        if isinstance(gray_image, np.ndarray):
            if gray_image.ndim == 2:  # [H, W]
                gray_image = gray_image[np.newaxis, ...]  # [1, H, W]
            elif gray_image.shape[-1] == 1:  # [H, W, 1]
                gray_image = np.transpose(gray_image, (2, 0, 1))  # [1, H, W]
            gray_image = torch.from_numpy(gray_image).float()  # 转换为张量
        gray_image = gray_image.unsqueeze(0).to(device)  # [1, 1, H, W]
        #


        # 生成Grad-CAM
        color_cam, texture_cam, output = get_gradcam(model, new_img, gray_image)

        # 可视化
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_img_np = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)  # [H, W]

        color_overlay = visualize_cam(rgb_img, color_cam)
        texture_overlay = visualize_cam(gray_img_np, texture_cam)
        print(output)
        # 显示结果
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title("Color Stream Heatmap")
        plt.imshow(color_overlay)
        plt.axis("off")
        plt.subplot(122)
        plt.title("Texture Stream Heatmap")
        plt.imshow(texture_overlay)
        plt.axis("off")
        plt.show()