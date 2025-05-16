# 使用 PIL 库操作图片，torch 是 PyTorch 主库
from PIL import Image
import torch
# 导入 torch.utils.data 下的 Dataset 抽象类，用于自定义数据集类
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义数据集类，用于加载图像数据及对应标签"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        构造函数，用于初始化数据集

        参数:
            images_path (list): 包含所有图片路径的列表
            images_class (list): 包含对应标签的列表，和 images_path 一一对应
            transform: 对图片进行处理的函数或 transforms 组合，默认为 None
        """
        # 保存图片路径列表
        self.images_path = images_path
        # 保存对应图片的类别列表
        self.images_class = images_class
        # 保存图片预处理变换，若有提供则进行预处理
        self.transform = transform

    def __len__(self):
        """
        返回数据集中图片的数量

        返回:
            数据集中图片的总数量（列表长度）
        """
        return len(self.images_path)

    def __getitem__(self, item):
        """
        根据索引获取对应的数据项，包括图片和标签

        参数:
            item: 数据项对应的索引

        返回:
            (img, label): 经过预处理后的图片和对应的标签

        异常:
            如果加载的图片模式不是 RGB，则抛出 ValueError 异常
        """
        # 使用 Image.open 打开图片文件
        img = Image.open(self.images_path[item])
        # 检查图片是否为 RGB 模式，否则抛出异常
        if img.mode != 'RGB':
            raise ValueError("image: {} is not RGB mode.".format(self.images_path[item]))
        # 获取对应的标签
        label = self.images_class[item]

        # 如果定义了 transform 则对图片进行预处理操作
        if self.transform is not None:
            img = self.transform(img)

        # 返回预处理后的图片与标签
        return img, label

    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn 函数，用于 dataloader 批处理数据
        此函数可以将每个批次的图片和标签整理为 tensor 格式

        参数:
            batch (list): 包含多个 (img, label) 组成的元组

        返回:
            images: 将所有图片合并为一个 tensor (批处理维度为第 0 维)
            labels: 将所有标签合成为一个 tensor
        """
        # 使用 zip 函数拆分批次中的多个元组数据
        images, labels = tuple(zip(*batch))
        # 将图片列表转换为 tensor，并在第 0 维度合并
        images = torch.stack(images, dim=0)
        # 将标签列表转换为 tensor
        labels = torch.as_tensor(labels)
        return images, labels