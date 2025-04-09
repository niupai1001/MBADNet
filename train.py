import copy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import torch.cuda.amp
from sklearn.metrics import classification_report
# 自定义模块
from Mytrans import tran_method
from module import MBADNet, FocalLoss

from MyDataset import BambooDataset, SemiDataLoader


# 配置参数
CONFIG = {
    "data_paths": {
        "labeled": r"E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\label_data",
        "unlabeled": r"E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\unlabeled_data",
        "eval": r"E:\pycharm\Project\Bamboo\MosoBamboo\AgeDetect\Data\TS_Data\eval_data"
    },
    "training": {
        "batch_size": {
            "labeled": 16,
            "unlabeled": 16,
            "eval": 16
        },
        "epochs": 300,
        "lr": 1e-3,
        "lambda_max": 10,
        "save_path": "./checkpoints"
    }
}

last_thr = [1.,1.,1.,1.]
class TrainingLogger:
    """结构化训练日志记录器"""

    def __init__(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.log_file = os.path.join(save_path, "training.log")
        self.f1_log_file = os.path.join(save_path, "f1_scores.log")  # 新增F1-score日志文件
        self._setup_header()
        self._setup_f1_header()  # 新增F1-score日志表头

    def _setup_header(self):
        header = ("\n{:<7} | {:>8} | {:>10} | {:>9} | {:>10} | {:>9} | {:>7} | {}"
                  .format("Epoch", "LR", "Train Loss", "Train Acc", "Val Loss",
                          "Val Acc", "Time", "Best Val (Epoch)"))
        separator = "-" * 98
        with open(self.log_file, "w") as f:
            f.write(header + "\n" + separator + "\n")

    def _setup_f1_header(self):
        header = ("\n{:<7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}"
                  .format("Epoch", "F1-I", "F1-II", "F1-III", "F1-IV", "F1-Avg"))
        separator = "-" * 55
        with open(self.f1_log_file, "w") as f:
            f.write(header + "\n" + separator + "\n")

    def log_f1_scores(self, epoch, f1_scores):
        """
        记录每个类别的F1-score
        :param epoch: 当前epoch
        :param f1_scores: 包含四个类别F1-score的列表 [f1_i, f1_ii, f1_iii, f1_iv]
        """
        f1_avg = sum(f1_scores) / len(f1_scores)  # 计算平均F1-score
        log_str = (f"{epoch + 1:03d} | "
                   f"{f1_scores[0]:.4f} | "
                   f"{f1_scores[1]:.4f} | "
                   f"{f1_scores[2]:.4f} | "
                   f"{f1_scores[3]:.4f} | "
                   f"{f1_avg:.4f}")
        with open(self.f1_log_file, "a") as f:
            f.write(log_str + "\n")
        print("F1 Scores: " + log_str)

    def log_epoch(self, epoch, total_epochs, lr, train_loss, train_acc,
                  val_loss, val_acc, time_used, best_acc, best_epoch):
        log_str = (
            f"{epoch + 1:03d}/{total_epochs} | "
            f"{lr:.2e} | "
            f"{train_loss:.4f} | "
            f"{train_acc:.2%} | "
            f"{val_loss:.4f} | "
            f"{val_acc:.2%} | "
            f"{time_used:.1f}s | "
            f"{best_acc:.2%} ({best_epoch})"
        )
        if val_acc >= best_acc:
            log_str += " *"

        with open(self.log_file, "a") as f:
            f.write(log_str + "\n")
        print(log_str)


class DynamicThresholdTrainer:
    """半监督训练管理器"""

    def __init__(self, model, optimizer, lambda_max=10):
        self.model = model
        self.optimizer = optimizer
        self.lambda_max = lambda_max
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        self.scaler = torch.amp.GradScaler()
        self.sup_loss_fun = FocalLoss()

    def update_teacher(self, epoch):
        """EMA更新教师模型"""
        alpha = min(0.999, 1 - 1 / (epoch + 10))
        for t_param, s_param in zip(self.teacher_model.parameters(),
                                    self.model.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)

    def get_threshold(self, age_class, epoch, last_t):

        """动态阈值计算"""
        if last_t < 0.05:
            alpha = 1
        else:
            alpha = 0
        tau0 = 0.95 - 0.05 * age_class
        beta = 0.1 * (1 + epoch / 100)
        new_t = tau0 * np.exp(-beta * epoch)
        return new_t - alpha * last_t[age_class]

    def train_step(self, labeled_data, unlabeled_data, epoch):
        """混合精度训练步骤"""
        self.model.train()
        lab_images, gray_images, labels = labeled_data
        unlab_images, _ = unlabeled_data

        # 混合精度上下文
        with torch.amp.autocast('cuda', dtype=torch.float16):
            # 监督损失
            outputs = self.model(lab_images, gray_images)
            sup_loss = self.sup_loss_fun(outputs, labels)
            if epoch > 50:
                # 无监督损失
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(*unlab_images)
                    pseudo_probs = F.softmax(teacher_outputs, dim=1)
                    max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
                global last_thr
                thresholds = torch.tensor(
                    [self.get_threshold(i, epoch, last_thr) for i in range(4)],
                    device=max_probs.device
                )
                top2_probs, _ = torch.topk(pseudo_probs, 2, dim=1)
                mask = (max_probs > thresholds[pseudo_labels]) & \
                       ((max_probs - top2_probs[:, 1]) > 0.15)
                if mask.sum() > 0:
                    student_outputs = self.model(*unlab_images)

                    temperature = 0.7  # 可调参数
                    student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
                    unsup_loss = F.kl_div(
                        student_log_probs,
                        F.softmax(teacher_outputs / temperature, dim=1).detach(),
                        reduction='batchmean'
                    ) * (temperature ** 2) * mask.float().mean()

                    # 计算每个类别的伪标签占比
                    selected_pseudo_labels = pseudo_labels[mask]

                    class_counts = torch.bincount(selected_pseudo_labels, minlength=4)

                    total_pseudo_labels = mask.sum().item()

                    last_thr = class_counts.float() / total_pseudo_labels

                else:
                    unsup_loss = torch.tensor(0.0, device='cuda')
            else:
                unsup_loss = torch.tensor(0.0, device='cuda')
            if epoch < 20:
                lambda_current = 0
            else:
                lambda_current = min(self.lambda_max, epoch / 30 * self.lambda_max)
            total_loss = sup_loss + lambda_current * unsup_loss

        # 梯度缩放和更新
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return total_loss.item(), outputs , sup_loss, unsup_loss


def setup_dataloaders(config):
    """初始化数据加载器"""

    def _init_dataset(path, is_labeled, trans=tran_method["non_extra_trans"]):
        return BambooDataset(
            root_dir=path,
            is_labeled=is_labeled,
            weak_transform=trans
        )

    datasets = {
        "labeled": _init_dataset(config["data_paths"]["labeled"], True, trans=tran_method["extra_trans"]),
        "unlabeled": _init_dataset(config["data_paths"]["unlabeled"], False, tran_method["extra_trans"]),
        "eval": _init_dataset(config["data_paths"]["eval"], True)
    }

    return {
        "labeled": DataLoader(
            datasets["labeled"],
            batch_size=config["training"]["batch_size"]["labeled"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        ),
        "unlabeled": DataLoader(
            datasets["unlabeled"],
            batch_size=config["training"]["batch_size"]["unlabeled"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        ),
        "eval": DataLoader(
            datasets["eval"],
            batch_size=config["training"]["batch_size"]["eval"],
            num_workers=2,
            pin_memory=True
        )
    }


def train_pipeline(config):
    # 硬件配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 初始化系统组件
    dataloaders = setup_dataloaders(config)
    # m_cfg = model_cfg()
    # # m_cfg.resnet50_init()
    # m_cfg.mobilenet_v2_init(freeze=False)
    # model = m_cfg.model
    model = MBADNet(num_classes=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=0.05)

    # 配置ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
    )
    logger = TrainingLogger(config["training"]["save_path"])

    # 训练状态跟踪
    history = {
        "best_train": {"epoch": 0, "acc": 0.0, "loss": float("inf")},
        "best_valid": {"epoch": 0, "acc": 0.0, "loss": float("inf")},
        "metrics": []
    }

    # 初始化训练器
    trainer = DynamicThresholdTrainer(
        model=model,
        optimizer=optimizer,
        lambda_max=config["training"]["lambda_max"]
    )

    # 训练循环
    for epoch in range(config["training"]["epochs"]):
        print("Epoch: {}/{}".format(epoch + 1, config["training"]["epochs"]))
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss, train_acc, total_samples = 0.0, 0.0, 0
        Sup_loss, UNSup_loss = 0.0, 0.0
        semi_loader = SemiDataLoader(
            dataloaders["labeled"].dataset,
            dataloaders["unlabeled"].dataset
        )

        with tqdm(semi_loader, desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
            for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(pbar):
                # 数据转移到设备
                lab_images = labeled_batch[0].to(device, non_blocking=True)
                gray_images = labeled_batch[1].to(device, non_blocking=True)
                labels = labeled_batch[2].to(device, non_blocking=True)
                unlab_images = (
                    unlabeled_batch[0].to(device, non_blocking=True),
                    unlabeled_batch[1].to(device, non_blocking=True)
                )

                # 训练步骤
                loss, outputs, sup, unsup = trainer.train_step(
                    (lab_images, gray_images, labels),
                    (unlab_images, None),
                    epoch
                )

                # 计算指标
                with torch.no_grad():
                    _, preds = torch.max(outputs, 1)
                    acc = (preds == labels).float().mean().item()

                # 累计统计量
                batch_size = lab_images.size(0)
                train_loss += loss * batch_size
                Sup_loss += sup * batch_size
                UNSup_loss += unsup * batch_size
                train_acc += acc * batch_size
                total_samples += batch_size

                # 更新进度条
                pbar.set_postfix({
                    "Loss": f"{train_loss / total_samples:.4f}",
                    "Sup": f"{Sup_loss:.4f}",
                    "Unsup": f"{UNSup_loss:.7f}",
                    "Acc": f"{train_acc / total_samples:.2%}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
                })

        # 更新教师模型
        trainer.update_teacher(epoch)

        # 验证阶段
        # 在验证阶段开始前初始化存储变量
        all_preds = []
        all_labels = []

        model.eval()
        valid_loss, valid_acc = 0.0, 0
        with torch.inference_mode(), \
                tqdm(dataloaders["eval"], desc="Validating", leave=False) as val_pbar:

            for inputs in val_pbar:
                lab_img = inputs[0].to(device, non_blocking=True)
                gray_img = inputs[1].to(device, non_blocking=True)
                labels = inputs[2].to(device, non_blocking=True)

                outputs = model(lab_img, gray_img)
                loss = F.cross_entropy(outputs, labels)

                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).sum().item()

                # 累积预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                valid_loss += loss.item() * lab_img.size(0)
                valid_acc += acc
                val_pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{acc / lab_img.size(0):.2%}"
                })

        # 计算平均指标
        avg_train_loss = train_loss / total_samples
        avg_train_acc = train_acc / total_samples
        avg_valid_loss = valid_loss / len(dataloaders["eval"].dataset)
        avg_valid_acc = valid_acc / len(dataloaders["eval"].dataset)

        # 更新历史记录
        history["metrics"].append(
            (avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc)
        )
        # 计算f1
        report = classification_report(
            all_labels, all_preds, target_names=["I", "II", "III", "IV"], output_dict=True
        )
        f1_scores = [
            report["I"]["f1-score"],
            report["II"]["f1-score"],
            report["III"]["f1-score"],
            report["IV"]["f1-score"]
        ]

        # 记录F1-score
        logger.log_f1_scores(epoch, f1_scores)
        # 跟踪最佳结果
        if avg_valid_acc >= history["best_valid"]["acc"]:
            history["best_valid"].update({
                "epoch": epoch + 1,
                "acc": avg_valid_acc,
                "loss": avg_valid_loss
            })
            torch.save(model.state_dict(),
                       os.path.join(config["training"]["save_path"], "best_model.pth"))

        if avg_train_acc >= history["best_train"]["acc"]:
            history["best_train"].update({
                "epoch": epoch + 1,
                "acc": avg_train_acc,
                "loss": avg_train_loss
            })

        # 学习率调度
        scheduler.step(avg_valid_loss)

        # 记录日志
        logger.log_epoch(
            epoch=epoch,
            total_epochs=config["training"]["epochs"],
            lr=optimizer.param_groups[0]["lr"],
            train_loss=avg_train_loss,
            train_acc=avg_train_acc,
            val_loss=avg_valid_loss,
            val_acc=avg_valid_acc,
            time_used=time.time() - epoch_start,
            best_acc=history["best_valid"]["acc"],
            best_epoch=history["best_valid"]["epoch"]
        )

        # 保存周期检查点
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(config["training"]["save_path"],
                             f"model_epoch{epoch + 1:03d}.pt")
            )

    # 最终处理
    torch.save(model.state_dict(),
               os.path.join(config["training"]["save_path"], "final_model.pt"))
    torch.save(history,
               os.path.join(config["training"]["save_path"], "training_history.pt"))

    # 生成训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot([m[0] for m in history["metrics"]], label="Train Loss")
    plt.plot([m[2] for m in history["metrics"]], label="Val Loss")
    plt.legend()

    plt.subplot(122)
    plt.plot([m[1] for m in history["metrics"]], label="Train Acc")
    plt.plot([m[3] for m in history["metrics"]], label="Val Acc")
    plt.legend()

    plt.savefig(os.path.join(config["training"]["save_path"], "training_curves.png"))
    plt.close()

    return model, history


if __name__ == "__main__":
    train_pipeline(CONFIG)