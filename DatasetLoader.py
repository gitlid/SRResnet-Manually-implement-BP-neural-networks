import gc
import math
from collections import OrderedDict
from pathlib import Path
import random
import torch
from PIL import Image
from utils import to_tensor, ToTensor
import os
import multiprocessing as mp
from multiprocessing import Process, Queue


class Dataset:
    def __init__(self, data_dir, transform=None, scale=2, crop_size=64):
        self.data_dir = Path(data_dir)
        self.cls = [x.name for x in self.data_dir.glob("./*")]
        self.transform = transform
        self.crop = crop_size
        self.scale = scale
        self._get_all_img()

    def __len__(self):
        return len(self.all_img)


    def __getitem__(self, idx):
        image_file = self.all_img[idx]
        img_pil = Image.open(image_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        # print(image_file)
        out = OrderedDict()
        out["HR"] = img_pil
        lr = img_pil.resize((img_pil.size[0] // self.scale, img_pil.size[1] // self.scale), Image.BICUBIC)
        out["LR"] = lr
        if self.transform is not None:
            out = self.transform(out)
        else:
            out = to_tensor(out)
        return tuple(out.values())

    def _get_all_img(self):
        self.all_img = []
        for d in self.data_dir.glob("./*"):
            for img in d.glob("*.JPEG"):
                im = Image.open(img)
                if im.size[0] > self.crop and im.size[1] > self.crop:
                    self.all_img.append(str(img))


# class DataLoader:
#     def __init__(self, dataset, batch_size=1, shuffle=False):
#         """
#         自定义数据加载器，支持批量加载和打乱数据。
#
#         Args:
#             dataset (Dataset): 自定义的数据集类实例。
#             batch_size (int): 每个批次的样本数量。
#             shuffle (bool): 是否在每个epoch前打乱数据。
#         """
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = list(range(len(dataset)))  # 数据索引列表
#         self.current_index = 0
#
#     def __iter__(self):
#         """返回迭代器自身，重置当前索引到0。"""
#         self.current_index = 0
#         # 打乱索引
#         if self.shuffle:
#             random.shuffle(self.indices)
#         return self
#
#     def __next__(self):
#         """获取下一个批次的数据。"""
#         if self.current_index >= len(self.indices):
#             raise StopIteration  # 结束迭代
#
#         # 获取当前批次的索引范围
#         batch_indices = self.indices[self.current_index: self.current_index + self.batch_size]
#         self.current_index += self.batch_size
#
#         # 收集批次数据
#         batch_x = []
#         for idx in batch_indices:
#             x = self.dataset[idx]
#
#             batch_x.append(x)
#
#         # 堆叠张量为批次（假设所有张量形状一致）
#         if batch_x:
#             if isinstance(batch_x[0], tuple):
#                 batch_x = tuple(torch.stack(x) for x in zip(*batch_x))
#             elif isinstance(batch_x[0], torch.Tensor):
#                 batch_x = torch.stack(batch_x)
#         else:
#             batch_x = None
#
#         return batch_x
#
#
#     def __len__(self):
#         return math.ceil(len(self.dataset) / self.batch_size)




def worker_fn(task_queue, result_queue, dataset):
    """
    工作进程函数，处理数据加载任务。
    """
    while True:
        task = task_queue.get()
        if task is None:  # 接收到终止信号
            break
        batch_idx, batch_indices = task
        batch_data = []
        for idx in batch_indices:
            sample = dataset[idx]
            batch_data.append(sample)
        # 处理样本堆叠
        if batch_data:
            if isinstance(batch_data[0], tuple):
                # 假设每个样本是元组（如输入和标签）
                stacked = []
                for i in range(len(batch_data[0])):
                    stacked.append(torch.stack([sample[i] for sample in batch_data]))
                batch_data = tuple(stacked)
            elif isinstance(batch_data[0], torch.Tensor):
                batch_data = torch.stack(batch_data)
        else:
            batch_data = None
        result_queue.put((batch_idx, batch_data))


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        """
        自定义数据加载器，支持多进程预加载。

        Args:
            dataset (Dataset): 自定义数据集实例。
            batch_size (int): 批大小。
            shuffle (bool): 是否打乱数据顺序。
            num_workers (int): 工作进程数，0表示不使用多进程。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = list(range(len(dataset)))
        self.current_index = 0  # 单进程模式下使用

    def __iter__(self):
        """初始化迭代器，打乱数据（如果需要），并启动工作进程（多进程模式）。"""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)

        if self.num_workers > 0:
            # 多进程模式初始化
            # 生成所有批次索引
            self.batches = []
            for i in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[i:i + self.batch_size]
                self.batches.append((len(self.batches), batch_indices))
            self.total_batches = len(self.batches)
            self.current_batch = 0
            self.received_batches = {}  # 存储已接收的批次数据

            # 创建任务队列和结果队列
            self.task_queue = Queue()
            self.result_queue = Queue()

            # 填充任务队列
            for batch in self.batches:
                self.task_queue.put(batch)
            # 启动工作进程
            self.workers = []
            for _ in range(self.num_workers):
                p = Process(target=worker_fn, args=(self.task_queue, self.result_queue, self.dataset))
                p.start()
                self.workers.append(p)
        return self

    def __next__(self):
        """获取下一批数据。"""
        if self.num_workers == 0:
            # 单进程模式
            if self.current_index >= len(self.indices):
                raise StopIteration()
            # 获取当前批次的索引
            batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
            self.current_index += self.batch_size
            # 加载数据
            batch_data = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                batch_data.append(sample)
            # 堆叠数据
            if batch_data:
                if isinstance(batch_data[0], tuple):
                    stacked = tuple(torch.stack(samples) for samples in zip(*batch_data))
                elif isinstance(batch_data[0], torch.Tensor):
                    stacked = torch.stack(batch_data)
                else:
                    stacked = batch_data
            else:
                stacked = None
            return stacked
        else:
            # 多进程模式
            if self.current_batch >= self.total_batches:
                # 添加终止信号（每个工作进程一个）
                for _ in range(self.num_workers):
                    self.task_queue.put(None)
                gc.collect()
                # 等待所有工作进程结束
                for p in self.workers:
                    p.join()
                raise StopIteration()
            # 等待当前批次的数据
            while True:
                if self.current_batch in self.received_batches:
                    data = self.received_batches.pop(self.current_batch)
                    self.current_batch += 1
                    return data
                else:
                    # 从结果队列获取数据
                    batch_idx, batch_data = self.result_queue.get()
                    self.received_batches[batch_idx] = batch_data

    def __len__(self):
        """返回批次数量。"""
        return math.ceil(len(self.dataset) / self.batch_size)

    def __del__(self):
        # 确保子进程终止
        if hasattr(self, 'workers'):
            for p in self.workers:
                if p.is_alive():
                    p.terminate()


class DatasetDIV2K(Dataset):
    def __init__(self, data_dir, transform=None, LR_type='bicubic', scale=2):
        self.LR_type = LR_type
        # self.scale = scale
        super(DatasetDIV2K, self).__init__(data_dir=data_dir, transform=transform, scale=scale)

    def __getitem__(self, idx):
        image_file = self.all_img[idx]
        img_pil = Image.open(image_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        # print(image_file)
        data = OrderedDict()
        data["HR"] = img_pil
        if self.LR_type in ['bicubic', 'unknown']:
            image_file = image_file.replace("HR", f"LR_{self.LR_type}")
            image_file = image_file.replace(".png", f"x{self.scale}.png")
            dir_name, file_name = os.path.split(image_file)
            image_file = os.path.join(dir_name, f"x{self.scale}", file_name)
            img_pil = Image.open(image_file)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            data["LR"] = img_pil
        else:
            raise ValueError("LR_type must be bicubic or unknown")
        if self.transform is not None:
            img_tensor = self.transform(data)
        else:
            img_tensor = to_tensor(data)
        return tuple(img_tensor.values())

    def _get_all_img(self):
        self.all_img = []
        for d in self.data_dir.glob("./*HR"):
            for img in d.glob("*.png"):
                # im = Image.open(img)
                # if im.size[0] > 64 and im.size[1] > 64:
                self.all_img.append(str(img))


class Dataset14(Dataset):
    def __init__(self, data_dir, transform=None, scale=2):
        super(Dataset14, self).__init__(data_dir=data_dir, transform=transform, scale=scale)

    def __getitem__(self, idx):
        image_file = self.all_img[idx]
        img_pil = Image.open(image_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        data = OrderedDict()
        data["HR"] = img_pil
        image_file = image_file.replace("HR", "LR")
        img_pil = Image.open(image_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        data["LR"] = img_pil
        if self.transform is not None:
            img_tensor = self.transform(data)
        else:
            img_tensor = to_tensor(data)
        return tuple(img_tensor.values())

    def _get_all_img(self):
        self.all_img = []
        for d in self.data_dir.glob(f"./*SRF_{self.scale}"):
            for img in d.glob("*HR.png"):
                self.all_img.append(str(img))


if __name__=='__main__':
    from utils import DynamicCompose, RandomCrop
    import torch.nn.functional as F
    import torch.nn as nn
    import cv2
    import numpy as np
    import time
    transform = DynamicCompose([
        ToTensor(),
        RandomCrop(64)
    ])
    dataset_name = "DIV2K"
    dataset = DatasetDIV2K(data_dir=f"./{dataset_name}/train", transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=8)
    for i, (x, y) in enumerate(data_loader):
        print(i, x.shape, y.shape)
        time.sleep(0.5)
    print()
        # print(x.shape, y.shape)
        # print()
    # transform = DynamicCompose([
    #     to_tensor,
    #     RandomCrop(256)
    # ])
    # data_loader = DataLoader(dataset=Dataset(data_dir="./imagenette2/train", transform=transform), batch_size=16, shuffle=True)
    # layers = nn.Sequential(
    #     nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
    #     nn.ReLU(),
    #     nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
    #     nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
    #     nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
    # )
    #
    # for batch_t in data_loader:
    #     print(batch_t.shape)
    #     batch_x = F.interpolate(batch_t, scale_factor=(0.5, 0.5), mode='nearest') # 模拟模糊图片
    #     print(batch_x.shape)
    #     output_tensor = layers(batch_x)
    #     print(output_tensor.shape)
    # data_loader = DataLoader(dataset=DatasetDIV2K(data_dir="./DIV2K/train", transform=transform), batch_size=1,
    #                          shuffle=True)
    # for batch_t, batch_x in data_loader:
        # print(batch_t.shape)
        # print(batch_x.shape)
        # img1 = ((batch_t[0]+1)/2).permute(1, 2, 0).numpy()
        # img2 = ((batch_x[0]+1)/2).permute(1, 2, 0).numpy()
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        # cv2.waitKey(0)

    # conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    #
    # init = math.sqrt(2 / ((3) * 3 * 3))
    # # 对卷积层进行 He 初始化
    # nn.init.kaiming_normal_(conv1.weight, nonlinearity='tanh')
    # nn.init.constant_(conv1.bias, 0.0)
    # print(conv1.weight)
    # print(conv1.bias)