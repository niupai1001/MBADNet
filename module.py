import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import models

def mask_background(image_tensor):
    """
    对纯黑背景进行掩膜，避免模型关注黑色背景
    """
    # image_tensor 是形状为 (C, H, W) 的输入图像
    mask = (image_tensor.sum(dim=0)!= 0).float()  # 所有通道的像素值加起来为0的部分视为背景
    masked_image = image_tensor * mask  # 应用掩膜
    return masked_image
#         return self.attn(x)  # 使用轻量注意力
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # 应用通道注意力
        x = x * self.sa(x)  # 应用空间注意力
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# 深度残差SE模块
class ResidualSEBlock(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        self.se = SEModule(in_c, reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),  # 深度可分离卷积
            nn.Conv2d(in_c, in_c, 1),  # Pointwise卷积
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.se(x)
        return x + residual  # 残差连接


# 空洞残差模块
class DilatedResBlock(nn.Module):
    def __init__(self, in_c, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, 1),
            nn.BatchNorm2d(in_c)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        return self.relu(x + residual)


# 层级注意力机制
class HierarchicalAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # 低层特征：空间注意力聚焦竹节边缘
        self.low_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid()
        )
        # 高层特征：通道注意力强化年龄相关通道
        self.high_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c // 8, in_c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        # 空间注意力
        avg_out = torch.mean(low_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(low_feat, dim=1, keepdim=True)
        spatial_attn = self.low_attn(torch.cat([avg_out, max_out], dim=1))
        low_feat = low_feat * spatial_attn

        # 通道注意力
        channel_attn = self.high_attn(high_feat)
        high_feat = high_feat * channel_attn

        return low_feat + high_feat


# 颜色流：深层残差结构
class DeepColorStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *[ResidualSEBlock(64) for _ in range(6)]  # 6层残差
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.dropout(x)
        return self.pool(x)

# 标准残差块（保持不变）
class StandardResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + residual)  # 残差连接
        return x

class FPN(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.lat_layer1 = nn.Conv2d(out_channels, out_channels, 1)
        self.lat_layer2 = nn.Conv2d(out_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c2, c1):
        # c3: [B, 64, H/4, W/4], c2: [B, 64, H/8, W/8], c1: [B, 64, H/16, W/16]
        # Build FPN from bottom up
        p5_fpn = c1  # Lowest resolution
        p4_fpn = c2 + self.lat_layer2(self.upsample(p5_fpn))  # Upsample p5_fpn to match c2
        p3_fpn = c3 + self.lat_layer1(self.upsample(p4_fpn))  # Upsample p4_fpn to match c3
        p = self.output_conv(p3_fpn)
        return p
    
class EnhancedTextureStream(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.feature_extractors =  nn.Sequential(
            *[ResidualSEBlock(128) for _ in range(3)]
        )
        self.fpn = FPN(in_channels=128, out_channels=128)
        self.cbam = CBAM(128)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.attn = HierarchicalAttention(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.downsample(x)
        features = []
        current = x
        for extractor in self.feature_extractors:
            current = extractor(current)
            features.append(current)
            current = F.max_pool2d(current, 2)
        c3, c2, c1 = features  # c3: [B, 64, H/4, W/4], c2: [B, 64, H/8, W/8], c1: [B, 64, H/16, W/16]
        fused_feature = self.fpn(c3, c2, c1)
        fused_feature = self.cbam(fused_feature)
        fused_feature = self.dropout(fused_feature)
        low_feat = c3  
        high_feat = fused_feature
        x = self.attn(low_feat, high_feat)
        x = self.dropout(x)
        return self.pool(x)


# 门控多尺度融合
class GateFusion(nn.Module):
    def __init__(self, color_dim=64, texture_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(color_dim + texture_dim, 2),
            nn.Softmax(dim=1)
        )
        self.color_proj = nn.Linear(color_dim, 128)  # 提升融合维度
        self.texture_proj = nn.Linear(texture_dim, 128)

    def forward(self, color_feat, texture_feat):
        gate_weights = self.gate(torch.cat([color_feat, texture_feat], dim=1))
        projected_color = self.color_proj(color_feat)
        projected_texture = self.texture_proj(texture_feat)
        return gate_weights[:, 0:1] * projected_color + gate_weights[:, 1:2] * projected_texture


# 改进后的双流网络
class MBADNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.color_stream = DeepColorStream()
        self.texture_stream = EnhancedTextureStream()
        self.fusion = GateFusion()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, lab_img, gray_img):
        # 颜色流
        lab_img = mask_background(lab_img)
        gray_img = mask_background(gray_img)
        color_feat = self.color_stream(lab_img)
        color_feat = color_feat.view(color_feat.size(0), -1)  # [B, 64]

        # 纹理流
        texture_feat = self.texture_stream(gray_img)
        texture_feat = texture_feat.view(texture_feat.size(0), -1)  # [B, 64]

        # 门控融合
        fused = self.fusion(color_feat, texture_feat)  # [B, 128]
        return self.fc(fused)


class FocalLoss(nn.Module):
    """Age-adaptive focal loss with class-specific gamma values"""

    def __init__(self, alpha=0.25, gamma_base=2.0, class_gamma_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma_base = gamma_base
        self.class_gamma_weights = class_gamma_weights if class_gamma_weights else [1.0, 1.3, 1.2, 1.1]
        self.reduction = reduction

    def forward(self, input, target):
        # Regular cross entropy
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Get probabilities
        pt = torch.exp(-ce_loss)

        # Apply class-specific gamma values
        batch_gamma = torch.tensor([self.class_gamma_weights[t.item()] for t in target],
                                   device=input.device) * self.gamma_base

        # Calculate focal loss with class-specific gamma
        focal_loss = self.alpha * (1 - pt) ** batch_gamma.unsqueeze(1) * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



class model_cfg:
    def __init__(self):
        self.model_name = None
        self.model = None

        self.num_classes = 4
        self.pretrained = True

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.save_path = r'E:\pycharmProject\MosoBamboo\AgeDetect\model\Classify\weights'

    def mobilenet_v2_init(self, freeze=False):
        print('Using MobileNetV2, model initializing')
        # 使用预训练权重加载MobileNetV2
        self.model = mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # 根据freeze参数决定是否冻结特征提取层
        if freeze:
            for name, param in self.model.named_parameters():
                if "classifier" in name:  # 只解冻分类器层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        # 修改分类器以适应你的类别数（4个类别）
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        # 将模型移动到GPU
        self.model = self.model.to(self.device)
        print('model init done')

if __name__ == "__main__":
    net = MBADNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params / 1)