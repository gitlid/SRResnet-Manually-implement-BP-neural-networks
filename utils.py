import random
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import pad as torch_pad
from PIL import Image


def im2col(input_data, filter_h, filter_w, stride=1, pad=0, output_padding=0):
    """

    Parameters
    ----------
    input_data : (数据数, 通道, 高度, 宽度)的四维数组组成的输入数据
    filter_h : 滤波器高度
    filter_w : 滤波器宽度
    stride : 步长
    pad : 填充

    Returns
    -------
    col : 二维数组
    """
    device = input_data.device
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1 + output_padding
    out_w = (W + 2 * pad - filter_w) // stride + 1 + output_padding

    # img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    img = torch_pad(input_data, (pad, pad + output_padding, pad, pad + output_padding), 'reflect').cpu()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w), device=device)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride].to(device=device)

    # col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col = col.permute(0, 4, 5, 1, 2, 3).contiguous().view(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0, device='cpu'):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例如：(10, 1, 28, 28)）
    filter_h
    filter_w
    stride
    pad

    Returns
    -------

    """
    # device = col.device
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    col = col.view(N, out_h, out_w, C, filter_h, filter_w).permute(0, 3, 4, 5, 1, 2).contiguous()

    # img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    img = torch.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), device=device)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :].to(device)

    return img[:, :, pad:H + pad, pad:W + pad]


# def PSNR(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = torch.sqrt(torch.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * torch.log10(255.0 / rmse)


def psnr(img1, img2, data_range=255.0, shave_border=0):
    """
    计算PSNR，支持灰度和RGB图像。

    Args:
        img1: 原始图像
        img2: 重建图像
        data_range: 像素最大差值（如255或1）
    Returns:
        PSNR值（dB）
    """
    height, width = img1.shape[-2:]
    img2 = img2[:, shave_border:height - shave_border, shave_border:width - shave_border]
    img1 = img1[:, shave_border:height - shave_border, shave_border:width - shave_border]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse)



# 示例：3channel图像处理
def psnr_3channel(img1, img2, data_range=255.0, shave_border=0):
    # 分通道计算PSNR后取平均
    psnr_c1 = psnr(img1[:, 0, :, :], img2[:, 0, :, :], data_range, shave_border)
    psnr_c2 = psnr(img1[:, 1, :, :], img2[:, 1, :, :], data_range, shave_border)
    psnr_c3 = psnr(img1[:, 2, :, :], img2[:, 2, :, :], data_range, shave_border)
    return (psnr_c1 + psnr_c2 + psnr_c3) / 3


def rgb_to_ycbcr(im, max_val=1.):
    """
    将 rgb 图像转换为 YCbCr 图像
    :param batch_X: N, C, H, W
    :return: batch_YCbCr N, C, H, W
    """
    r, g, b = im.split(split_size=1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = max_val/2 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = max_val/2 + 0.5 * r - 0.418688 * g - 0.081312 * b
    # ycbcr_image = cv2.merge([y, cb, cr]).astype(np.uint8)
    ycbcr_image = torch.cat([y, cb, cr], dim=1)
    return ycbcr_image


class ToTensor:
    def __init__(self, last_layer=''):
        self.last_layer = last_layer

    def __call__(self, data):
        return to_tensor(data, self.last_layer)


def to_tensor(data, last_layer=''):
    out = OrderedDict()
    for k in data.keys():
        if isinstance(data[k], np.ndarray):
            # handle numpy array
            img = torch.from_numpy(data[k])
        elif isinstance(data[k], Image.Image):
            pic = data[k]
            # handle PIL Image
            mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
            img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

            if pic.mode == "1":
                img = 255 * img
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()

        if isinstance(img, torch.ByteTensor):
            if last_layer.lower() == 'tanh':
                out[k] = img.to(dtype=torch.float32).div(127.5) - 1.0 # 归一化
            else:
                out[k] = img.to(dtype=torch.float32).div(255.0)
        else:
            out[k] = img
    return out


class DynamicCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        for t in self.transforms:
            img = t(img, *args, **kwargs)
        return img


class RandomCrop:
    def __init__(self, output_size, scale=2):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.scale = scale

    def __call__(self, data):
        out = OrderedDict()
        # for k in data.keys():

        img = data['HR']
        h, w = img.shape[1:]
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = img[:, top: top + new_h, left: left + new_w]

        out["HR"] = image

        if 'LR' in data.keys():
            img = data['LR']
            top, left = top // self.scale, left // self.scale
            new_h, new_w = new_h // self.scale, new_w // self.scale
            image = img[:, top: top + new_h, left: left + new_w]
            out["LR"] = image

        return out


class CenterCrop:
    def __init__(self, output_size, scale=2):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.scale = scale

    def __call__(self, data):
        out = OrderedDict()
        img = data['HR']
        h, w = img.shape[1:]  # 假设图像格式为 (C, H, W)
        new_h, new_w = self.output_size

        # 计算中心裁剪的起始坐标（确保向下取整，避免越界）
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # 裁剪HR图像
        image_hr = img[:, top: top + new_h, left: left + new_w]
        out["HR"] = image_hr

        # 处理LR图像（假设LR尺寸是HR的1/scale）
        if 'LR' in data.keys():
            img_lr = data['LR']
            lr_h, lr_w = img_lr.shape[1:]
            # LR的中心裁剪位置应与HR对应（尺寸为HR的1/scale）
            new_h_lr, new_w_lr = new_h // self.scale, new_w // self.scale
            top_lr = (lr_h - new_h_lr) // 2
            left_lr = (lr_w - new_w_lr) // 2
            image_lr = img_lr[:, top_lr: top_lr + new_h_lr, left_lr: left_lr + new_w_lr]
            out["LR"] = image_lr

        return out



class BICUBIC_LR:
    def __init__(self, output_size=None, scale=2):
        self.output_size = output_size
        self.scale = scale

    def __call__(self, data):
        out = OrderedDict()
        out["HR"] = data["HR"]
        if self.output_size == None:
            if isinstance(data["LR"], Image.Image):
                out["LR"] = data["LR"].resize((data["LR"].size[0]*self.scale, data["LR"].size[1]*self.scale), Image.BICUBIC)
            elif isinstance(data["LR"], np.ndarray):
                out["LR"] = cv2.resize(data["LR"], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            elif isinstance(data["LR"], torch.Tensor):
                from torch.nn import functional as F
                out["LR"] = F.interpolate(data["LR"].unsqueeze(0), scale_factor=self.scale, mode='bicubic').squeeze(0)
            else:
                raise TypeError("LR must be Image , np.ndarray or torch.Tensor")
        else:
            if isinstance(data["LR"], Image.Image):
                out["LR"] = data["LR"].resize((self.output_size, self.output_size), Image.BICUBIC)
            elif isinstance(data["LR"], np.ndarray):
                out["LR"] = cv2.resize(data["LR"], (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
            elif isinstance(data["LR"], torch.Tensor):
                from torch.nn import functional as F
                out["LR"] = F.interpolate(data["LR"].unsqueeze(0), (self.output_size, self.output_size), mode='bicubic').squeeze(0)
            else:
                raise TypeError("LR must be Image , np.ndarray or torch.Tensor")

        return out


# def draw_img(train_loss_list, valid_loss_list, loss_type="mse loss", img_name="SRCNN_net", save_path='SRCNN_net.png'):
#     x = np.arange(max(len(train_loss_list), len(valid_loss_list)))
#     if train_loss_list:
#         plt.plot(x, train_loss_list, label='train mse')
#     if valid_loss_list:
#         plt.plot(x, valid_loss_list, label='valid mse', linestyle='--')
#         plt.text(x[len(valid_loss_list) - 1], valid_loss_list[len(valid_loss_list) - 1],
#                  f"{valid_loss_list[len(valid_loss_list) - 1]:.4f}", fontdict={'weight': 'bold', 'size': 9})
#     plt.xlabel("epochs")
#     plt.ylabel(loss_type)
#     # plt.ylim(0, max(train_loss_list+valid_loss_list))
#     plt.legend(loc='center right')
#     plt.title(img_name)
#     plt.savefig(save_path, dpi=300)
#     plt.clf()
#     # plt.show()


# def draw_img(train_loss_list, valid_loss_list, loss_type="mse loss", img_name="SRCNN_net", save_path='SRCNN_net.png'):
#     x = np.arange(max(len(train_loss_list), len(valid_loss_list)))
#     all_losses = []
#     if train_loss_list:
#         plt.plot(x[:len(train_loss_list)], train_loss_list, label='train mse')
#         train_max = max(train_loss_list)
#         train_min = min(train_loss_list)
#         train_max_idx = train_loss_list.index(train_max)
#         train_min_idx = train_loss_list.index(train_min)
#         plt.axhline(y=train_max, color='r', linestyle='--', label=f'Train Max: {train_max:.4f}')
#         plt.axhline(y=train_min, color='g', linestyle='--', label=f'Train Min: {train_min:.4f}')
#         plt.text(train_max_idx, train_max, f"{train_max:.4f}", fontdict={'weight': 'bold', 'size': 9})
#         plt.text(train_min_idx, train_min, f"{train_min:.4f}", fontdict={'weight': 'bold', 'size': 9})
#         all_losses.extend(train_loss_list)
#     if valid_loss_list:
#         plt.plot(x[:len(valid_loss_list)], valid_loss_list, label='valid mse', linestyle='--')
#         valid_max = max(valid_loss_list)
#         valid_min = min(valid_loss_list)
#         valid_max_idx = valid_loss_list.index(valid_max)
#         valid_min_idx = valid_loss_list.index(valid_min)
#         plt.axhline(y=valid_max, color='m', linestyle='--', label=f'Valid Max: {valid_max:.4f}')
#         plt.axhline(y=valid_min, color='c', linestyle='--', label=f'Valid Min: {valid_min:.4f}')
#         plt.text(valid_max_idx, valid_max, f"{valid_max:.4f}", fontdict={'weight': 'bold', 'size': 9})
#         plt.text(valid_min_idx, valid_min, f"{valid_min:.4f}", fontdict={'weight': 'bold', 'size': 9})
#         plt.text(x[len(valid_loss_list) - 1], valid_loss_list[len(valid_loss_list) - 1],
#                  f"{valid_loss_list[len(valid_loss_list) - 1]:.4f}", fontdict={'weight': 'bold', 'size': 9})
#         all_losses.extend(valid_loss_list)
#
#     if all_losses:
#         # 动态调整 Y 轴范围
#         min_loss = min(all_losses)
#         max_loss = max(all_losses)
#         buffer = (max_loss - min_loss) * 0.1  # 上下各留 10% 的空白
#         plt.ylim(max(0, min_loss - buffer), max_loss + buffer)
#
#         # 局部放大图
#         if len(all_losses) > 10:
#             last_10_losses = all_losses[-10:]
#             min_last_10 = min(last_10_losses)
#             max_last_10 = max(last_10_losses)
#             buffer_last_10 = (max_last_10 - min_last_10) * 0.1
#
#             # 获取当前的 Axes 对象
#             ax = plt.gca()
#             axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
#             if train_loss_list:
#                 axins.plot(x[-10:len(train_loss_list)], train_loss_list[-10:])
#             if valid_loss_list:
#                 axins.plot(x[-10:len(valid_loss_list)], valid_loss_list[-10:], linestyle='--')
#             axins.set_ylim(min_last_10 - buffer_last_10, max_last_10 + buffer_last_10)
#             axins.set_xlim(x[-10], x[-1])
#             axins.set_xticklabels([])
#             axins.set_yticklabels([])
#             ax.indicate_inset_zoom(axins, edgecolor="black")
#
#     plt.xlabel("epochs")
#     plt.ylabel(loss_type)
#     plt.legend(loc='center right')
#     plt.title(img_name)
#     plt.savefig(save_path, dpi=300)
#     plt.clf()


def draw_img(train_loss_list, valid_loss_list, loss_type="mse loss", img_name="SRCNN_net", save_path='SRCNN_net.png'):
    x = np.arange(max(len(train_loss_list), len(valid_loss_list)))
    all_losses = []

    # 创建一个包含两个子图的布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

    if train_loss_list:
        ax1.plot(x[:len(train_loss_list)], train_loss_list, label=f'train {loss_type}')
        train_max = max(train_loss_list)
        train_min = min(train_loss_list)
        train_max_idx = train_loss_list.index(train_max)
        train_min_idx = train_loss_list.index(train_min)
        ax1.axhline(y=train_max, color='r', linestyle='--', label=f'Train Max: {train_max:.4f}')
        ax1.axhline(y=train_min, color='g', linestyle='--', label=f'Train Min: {train_min:.4f}')
        ax1.text(train_max_idx, train_max, f"{train_max:.4f}", fontdict={'weight': 'bold', 'size': 9})
        ax1.text(train_min_idx, train_min, f"{train_min:.4f}", fontdict={'weight': 'bold', 'size': 9})
        all_losses.extend(train_loss_list)

    if valid_loss_list:
        ax1.plot(x[:len(valid_loss_list)], valid_loss_list, label=f'valid {loss_type}', linestyle='--')
        valid_max = max(valid_loss_list)
        valid_min = min(valid_loss_list)
        valid_max_idx = valid_loss_list.index(valid_max)
        valid_min_idx = valid_loss_list.index(valid_min)
        ax1.axhline(y=valid_max, color='m', linestyle='--', label=f'Valid Max: {valid_max:.4f}')
        ax1.axhline(y=valid_min, color='c', linestyle='--', label=f'Valid Min: {valid_min:.4f}')
        ax1.text(valid_max_idx, valid_max, f"{valid_max:.4f}", fontdict={'weight': 'bold', 'size': 9})
        ax1.text(valid_min_idx, valid_min, f"{valid_min:.4f}", fontdict={'weight': 'bold', 'size': 9})
        ax1.text(x[len(valid_loss_list) - 1], valid_loss_list[len(valid_loss_list) - 1],
                 f"{valid_loss_list[len(valid_loss_list) - 1]:.4f}", fontdict={'weight': 'bold', 'size': 9})
        all_losses.extend(valid_loss_list)

    if all_losses:
        # 动态调整 Y 轴范围
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        buffer = (max_loss - min_loss) * 0.1  # 上下各留 10% 的空白
        ax1.set_ylim(max(0, min_loss - buffer), max_loss + buffer)

        # 局部放大图放到第二个子图
        if len(all_losses) > 10:
            last_10_losses = all_losses[-10:]
            min_last_10 = min(last_10_losses)
            max_last_10 = max(last_10_losses)
            buffer_last_10 = (max_last_10 - min_last_10) * 0.1

            if train_loss_list:
                ax2.plot(x[-10:len(train_loss_list)], train_loss_list[-10:])
            if valid_loss_list:
                ax2.plot(x[-10:len(valid_loss_list)], valid_loss_list[-10:], linestyle='--')
            ax2.set_ylim(min_last_10 - buffer_last_10, max_last_10 + buffer_last_10)
            ax2.set_xlim(x[-10] if len(x) > 10 else x[0], x[-1])
            ax2.set_xlabel("epochs")
            ax2.set_ylabel(loss_type)

    ax1.set_xlabel("epochs")
    ax1.set_ylabel(loss_type)
    ax1.set_title(img_name)
    # 获取第一个子图的图例元素
    handles, labels = ax1.get_legend_handles_labels()
    # 将图例添加到第二个子图
    ax2.legend(handles, labels, loc='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_img_2(lr_list, type="lr_evolution", img_name="SRCNN_net", save_path='SRCNN_net.png'):
    x = np.arange(len(lr_list))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(x[:len(lr_list)], lr_list, label=f'train {type}')
    lr_max = max(lr_list)
    lr_min = min(lr_list)
    lr_max_idx = lr_list.index(lr_max)
    lr_min_idx = lr_list.index(lr_min)
    ax1.axhline(y=lr_max, color='r', linestyle='--', label=f'lr Max: {lr_max:.4f}')
    ax1.axhline(y=lr_min, color='g', linestyle='--', label=f'lr Min: {lr_min:.8f}')
    ax1.text(lr_max_idx, lr_max, f"{lr_max:.4f}", fontdict={'weight': 'bold', 'size': 9})
    ax1.text(lr_min_idx, lr_min, f"{lr_min:.8f}", fontdict={'weight': 'bold', 'size': 9})



    if lr_list:
        # 动态调整 Y 轴范围
        min_lr = min(lr_list)
        max_lr = max(lr_list)
        buffer = (max_lr - min_lr) * 0.1  # 上下各留 10% 的空白
        ax1.set_ylim(min_lr - buffer, max_lr + buffer)

        # 局部放大图
        if len(lr_list) > 20:
            last_20_lr = lr_list[-20:]
            min_last_20 = min(last_20_lr)
            max_last_20 = max(last_20_lr)
            buffer_last_20 = (max_last_20 - min_last_20) * 0.1
            mean_last_20 = np.mean(last_20_lr)
            # 获取当前的 Axes 对象
            # ax = plt.gca()
            # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
            ax2.plot(x[-20:len(lr_list)], lr_list[-20:])
            ax2.set_ylim(min_last_20 - buffer_last_20, max_last_20 + buffer_last_20)
            ax2.set_xlim(x[-20], x[-1])
            ax2.axhline(y=mean_last_20, color='m', linestyle='--', label=f'lr Mean: {mean_last_20:.8f}')
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel(type)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(type)
    ax1.set_title(img_name)
    # 获取第一个子图的图例元素
    handles, labels = ax1.get_legend_handles_labels()
    # 将图例添加到第二个子图
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2), labels.extend(labels2)
    ax2.legend(handles, labels, loc='upper center')
    # ax2.legend(loc='lower center')

    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_img_3(v_lists, x_lim, y_label, loss_type, img_name, save_path):
    x = np.arange(x_lim)
    min_v = np.min([min(v[:x_lim]) for v in v_lists])
    max_v = np.max([max(v[:x_lim]) for v in v_lists])
    for idx, v_list in enumerate(v_lists):
        x_l = min(x_lim, len(v_list))
        plt.plot(x[:x_l], v_list[:x_l], label=loss_type[idx])
    plt.axhline(y=min_v, color='r', linestyle='--', label=f'Min: {min_v:.4f}')
    plt.axhline(y=max_v, color='g', linestyle='--', label=f'Max: {max_v:.4f}')
    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.legend(loc='center right')
    plt.title(img_name)
    plt.savefig(save_path, dpi=300)
    plt.close()