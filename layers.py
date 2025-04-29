import torch
from utils import *


class Relu:
    def __init__(self):
        self.mask = None
        self.train = True

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.clone()
        out[self.mask] = 0
        if not self.train:
            self.mask = None

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        self.mask = None
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重·偏置参数的梯度
        self.dW = None
        self.db = None

        self.train = True

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        # col_W = self.W.reshape(FN, -1).T
        col_W = self.W.reshape(FN, -1).t()

        # out = np.dot(col, col_W) + self.b
        col = im2col(x, FH, FW, self.stride, self.pad)
        out = torch.matmul(col, col_W)
        # del col
        if self.b is not None:
            out += self.b

        # out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        out = out.reshape(N, out_h, out_w, -1).permute(0, 3, 1, 2)

        if self.train:
            self.x = x
            self.col = col.cpu()
            self.col_W = col_W
        else:
            self.x = None
            self.col = None
            self.col_W = None

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # col_W = self.W.reshape(FN, -1).t()

        # dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        dout = dout.permute(0, 2, 3, 1).reshape(-1, FN)

        # self.db = np.sum(dout, axis=0)
        # self.dW = np.dot(self.col.T, dout)
        # self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        if self.b is not None:
            self.db = torch.sum(dout, dim=0)

        # 减少显存消耗，重新计算
        # col = im2col(self.x, FH, FW, self.stride, self.pad)
        self.dW = torch.matmul(self.col.t().to(self.x.device), dout)
        # self.dW = torch.matmul(col.t(), dout)
        # del col
        self.dW = self.dW.t().reshape(FN, C, FH, FW)

        # dcol = np.dot(dout, self.col_W.T)
        dcol = torch.matmul(dout, self.col_W.t()).cpu()
        # dcol = torch.matmul(dout, col_W.t())

        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad, device=self.x.device)
        self.x = None
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        self.train = True

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # arg_max = np.argmax(col, axis=1)
        # out = np.max(col, axis=1)
        # out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        out, arg_max = torch.max(col, dim=1)
        out = out.reshape(N, out_h, out_w, C).permute(0, 3, 1, 2)

        if self.train:
            self.x = x
            self.arg_max = arg_max
        else:
            self.x = None
            self.arg_max = None

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # dmax = np.zeros((dout.size, pool_size))
        # dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = torch.zeros((dout.numel(), pool_size))
        dmax[torch.arange(self.arg_max.numel(), device=self.arg_max.device), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad, device=self.x.device)
        self.x = None
        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.1, epsilon=1e-7, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.train = True


    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4:
            # 卷积层输入 (N, C, H, W)
            self.mode = 'conv'
            N, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
            x = x.reshape(-1, C)  # 保持通道分离
        elif x.ndim == 2:
            # 全连接层输入 (N, D)
            self.mode = 'linear'
            C = None
        else:
            raise ValueError

        out = self.__forward(x, train_flg, C)

        if self.mode == 'conv':
            out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return out

    def __forward(self, x, train_flg, C):
        if self.running_mean is None:
            if self.mode == 'conv':
                self.running_mean = torch.zeros(C, device=x.device)
                self.running_var = torch.ones(C, device=x.device)
            else:
                D = x.shape[1]
                self.running_mean = torch.zeros(D, device=x.device)
                self.running_var = torch.ones(D, device=x.device)

        if train_flg:

            # 计算均值和方差（按通道）
            mu = x.mean(dim=0)  # (C,)
            var = x.var(dim=0, unbiased=False)  # (C,)

            xc = x - mu
            std = torch.sqrt(var + self.epsilon)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * mu + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            self.batch_size = None
            self.xc = None
            self.xn = None
            self.std = None
            xc = x - self.running_mean
            xn = xc / ((torch.sqrt(self.running_var + self.epsilon)))

        out = self.gamma * xn + self.beta
        return out


    def backward(self, dout):
        # 保存原始梯度形状
        original_shape = dout.shape

        # 针对卷积输入的特殊处理
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.permute(0, 2, 3, 1)  # [N, H, W, C]
            dout = dout.reshape(-1, C)  # [N*H*W, C]

        dx = self.__backward(dout)

        # 恢复卷积输入的形状
        if len(original_shape) == 4:
            dx = dx.reshape(N, H, W, C)  # [N, H, W, C]
            dx = dx.permute(0, 3, 1, 2)  # [N, C, H, W]

        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        return dx

    def __backward(self, dout):
        batch_size = dout.shape[0]  # 注意：对于卷积这里是 N*H*W

        # gamma/beta梯度计算（按通道聚合）
        dbeta = dout.sum(dim=0)  # [C,]
        dgamma = torch.sum(self.xn * dout, dim=0)  # [C,]

        # 反向传播归一化操作
        dxn = self.gamma * dout  # [N*H*W, C] 或 [N, D]
        dxc = dxn / self.std  # [N*H*W, C]

        # 计算std梯度
        dstd = -torch.sum(dxn * self.xc / (self.std ** 2), dim=0)  # [C,]
        dvar_factor = 0.5 / self.std
        # if batch_size > 1:
        #     dvar_factor *= batch_size / (batch_size - 1)  # 无偏估计调整
        dvar = dvar_factor * dstd

        # 累加方差梯度
        dxc += (2.0 / batch_size) * self.xc * dvar  # [N*H*W, C]

        # 计算均值梯度
        dmu = torch.sum(dxc, dim=0)  # [C,]
        dx = dxc - dmu / batch_size  # [N*H*W, C]

        # 保存参数梯度
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class MSELoss:
    def __init__(self, size_average=True):
        self.y = None  # 预测值
        self.t = None  # 真实值
        self.loss = None
        self.size_average = size_average
        self.train = True

    def forward(self, y, t):
        """
        前向传播：计算 MSE 损失
        Args:
            y (torch.Tensor): 预测值，形状为 (batch_size, ...)
            t (torch.Tensor): 真实值，形状与 y 相同
        Returns:
            loss (torch.Tensor): 均方误差损失
        """
        if self.train:
            self.y = y
            self.t = t
        else:
            self.y = None
            self.t = None
        self.loss = ((y - t) ** 2).sum()  # 计算所有元素的平方差的和
        if self.size_average:
            self.loss /= self.y.numel()

        return self.loss

    def backward(self, dout=1):
        """
        反向传播：计算损失对输入的梯度
        Returns:
            grad (torch.Tensor): 梯度，形状与输入 y 相同
        """
        # 梯度计算公式：2*(y - t) / (batch_size * ...其他维度的尺寸)
        # batch_size = self.y.shape[0]
        grad = 2 * (self.y - self.t)
        if self.size_average:
            grad /= grad.numel()  # numel() 获取总元素数
        self.t = None
        # self.y = None
        return grad * dout


class PixelShuffle:
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def forward(self, x):
        N, C, H, W = x.shape
        r = self.upscale_factor
        # 重组通道为高分辨率特征图
        C_new = C // (r**2)
        x = x.reshape(N, C_new, r, r, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(N, C_new, H * r, W * r)
        return x

    def backward(self, dout):
        N, C, H, W = dout.shape
        r = self.upscale_factor
        # 反向重组（假设反向传播的实现需要展开）
        H_new = H // r
        W_new = W // r
        dout = dout.reshape(N, C, H_new, r, W_new, r)
        dout = dout.permute(0, 1, 3, 5, 2, 4)
        dout = dout.reshape(N, C * r * r, H_new, W_new)
        return dout


class Deconvolution:
    def __init__(self, W, b, stride=1, pad=0, output_padding=0):
        """
        Args:
            W (Tensor): 权重，形状为 (out_channels, in_channels, kernel_h, kernel_w)
            b (Tensor): 偏置，形状为 (out_channels,)
            stride (int or tuple): 步长，默认 1
            pad (int or tuple): 填充，默认 0
            output_padding (int or tuple): 输出填充，默认 0，必须满足 output_padding < stride
        """
        self.W = W
        self.b = b

        # 处理 stride/pad/output_padding 参数格式
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.pad = pad if isinstance(pad, tuple) else (pad, pad)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)

        # 验证 output_padding 合法性（必须小于 stride）
        if (self.output_padding[0] >= self.stride[0]) or (self.output_padding[1] >= self.stride[1]):
            raise ValueError("output_padding must be smaller than stride")

        self.x = None
        self.col = None
        self.col_W = None
        self.W_rotated = None  # 旋转后的权重

        self.dW = None
        self.db = None

        self.train = True

    def forward(self, x):
        self.x = x
        out_channels, in_channels, kernel_h, kernel_w = self.W.shape
        N, C, H, W = x.shape
        assert C == in_channels, "输入通道数与权重输入通道数不匹配"

        # 插入零以处理步长 > 1
        if self.stride[0] > 1 or self.stride[1] > 1:
            x = self.insert_zeros(x, self.stride)

        # 旋转权重 180 度（反卷积的核心操作）
        self.W_rotated = torch.flip(self.W, (2, 3))

        # 计算输出尺寸（包含 output_padding）
        out_h = (H - 1) * self.stride[0] + kernel_h - 2 * self.pad[0] + self.output_padding[0]
        out_w = (W - 1) * self.stride[1] + kernel_w - 2 * self.pad[1] + self.output_padding[1]

        # 调整 im2col 参数以对齐梯度计算
        pad_h_im2col = kernel_h - 1 - self.pad[0]
        pad_w_im2col = kernel_w - 1 - self.pad[1]
        self.pad_im2col = pad_h_im2col
        self.stride_im2col = 1

        # 执行 im2col
        col = im2col(x, kernel_h, kernel_w, stride=self.stride_im2col, pad=self.pad_im2col, output_padding=self.output_padding[0])

        # 将权重展平为 (in_channels * kernel_h * kernel_w, out_channels)
        self.col_W = self.W_rotated.reshape(out_channels, -1).t()

        # 矩阵乘法 + 偏置
        out = torch.matmul(col, self.col_W) + self.b
        out = out.view(N, out_h, out_w, out_channels).permute(0, 3, 1, 2)

        # 保存中间数据供反向传播使用
        if self.train:
            self.x = x
            self.col = col
        else:
            self.x = None
            self.col = None
            self.col_W = None
        return out

    def backward(self, dout):
        out_channels, in_channels, kernel_h, kernel_w = self.W.shape
        N, C, H, W = self.x.shape
        H, W = H + self.output_padding[0], W + self.output_padding[1]

        dout_reshaped = dout.permute(0, 2, 3, 1).reshape(-1, out_channels)
        # 计算 db
        self.db = torch.sum(dout_reshaped, dim=0)

        # 计算 dW_rotated（梯度未旋转）
        dW_rotated = torch.matmul(self.col.t(), dout_reshaped)
        dW_rotated = dW_rotated.t().view(out_channels, in_channels, kernel_h, kernel_w)
        # 旋转梯度以匹配原始权重方向
        self.dW = torch.flip(dW_rotated, (2, 3))

        # 计算输入梯度 dcol
        dcol = torch.matmul(dout_reshaped, self.col_W.t())

        # 反向 im2col 得到 dx
        dx = col2im(dcol, (N, C, H, W), kernel_h, kernel_w,
                    stride=self.stride_im2col, pad=self.pad_im2col, device=self.x.device)

        # 下采样（过滤插入零位置的梯度）
        if self.stride[0] > 1 or self.stride[1] > 1:
            dx = dx[:, :, ::self.stride[0], ::self.stride[1]]
        self.x = None
        self.col = None
        self.col_W = None
        return dx

    def insert_zeros(self, x, stride):
        """在输入中插入零以模拟步长 > 1 的上采样"""
        N, C, H, W = x.shape
        H_up = H * stride[0] - (stride[0] - 1)
        W_up = W * stride[1] - (stride[1] - 1)
        x_up = torch.zeros((N, C, H_up, W_up), device=x.device)
        x_up[:, :, ::stride[0], ::stride[1]] = x
        return x_up


class Tanh:
    def __init__(self):
        self.out = None
        self.train = True

    def forward(self, x):
        # 手动计算tanh
        upper_mask = x >= 0
        lower_mask = x < 0

        exp_pos = torch.exp(2*x)
        exp_neg = torch.exp(-2*x)
        out = torch.zeros_like(x)
        out[upper_mask] = ((1 - exp_neg) / (1 + exp_neg))[upper_mask]
        out[lower_mask] = ((exp_pos - 1) / (exp_pos + 1))[lower_mask]
        if self.train:
            self.out = out
        else:
            self.out = None

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out**2)
        self.out = None
        return dx


class PReLU:
    def __init__(self, num_parameters=1, init_alpha=0.25, device='cpu'):
        """
        Args:
            num_parameters (int): 可学习参数 alpha 的数量（通道维度）。
                默认 1（共享同一个 alpha）。
                若为卷积层，通常设置为通道数 C。
            init_alpha (float): alpha 的初始值，默认为 0.25。
        """
        self.alpha = torch.full((num_parameters,), init_alpha, dtype=torch.float32, device=device)

        self.mask = None  # 记录输入中 <=0 的位置
        self.x = None  # 保存输入用于反向传播
        self.train = True

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.clone()

        # 应用 alpha 到负数区域（自动广播）
        if out.ndim == 4:
            # 卷积输入 (B, C, H, W): alpha 形状需为 (C, 1, 1)
            alpha = self.alpha.view(1, -1, 1, 1)
        elif out.ndim == 2:
            # 全连接输入 (B, D): alpha 形状需为 (1, D)
            alpha = self.alpha.view(1, -1)
        else:
            raise ValueError("Unsupported input dimension")

        out[self.mask] *= alpha.expand_as(out)[self.mask]
        if self.train:
            self.x = x
        else:
            self.x = None
            self.mask = None
        return out

    def backward(self, dout):
        dx = dout.clone()
        dalpha = torch.zeros_like(self.alpha)

        # 计算输入梯度 dx
        if dx.ndim == 4:
            # 卷积层梯度处理
            alpha = self.alpha.view(1, -1, 1, 1)
            # 对于 x > 0 的区域，梯度为 1；对于 x <=0 的区域，梯度为 alpha
            dx[self.mask] *= alpha.expand_as(dx)[self.mask]

            # 计算 alpha 的梯度：sum(dout * x 在 x <=0 的区域)
            # masked_dout = dout[self.mask]  # (N_neg,)
            # masked_x = self.x[self.mask]  # (N_neg,)

            # 按通道分组求和（假设 alpha 形状为 (C,)）
            C = self.alpha.shape[0]
            for c in range(C):
                # 获取第 c 个通道的 mask
                channel_mask = (self.x[:, c, :, :] <= 0)  # (B, H, W)
                # 计算该通道的梯度：sum(dout_c * x_c)
                dalpha[c] = (dout[:, c, :, :][channel_mask] *
                             self.x[:, c, :, :][channel_mask]).sum()

        elif dx.ndim == 2:
            # 全连接层梯度处理
            alpha = self.alpha.view(1, -1)
            dx[self.mask] *= alpha.expand_as(dx)[self.mask]

            # 按特征维度分组求和（假设 alpha 形状为 (D,)）
            D = self.alpha.shape[0]
            for d in range(D):
                feature_mask = (self.x[:, d] <= 0)  # (B,)
                dalpha[d] = (dout[:, d][feature_mask] *
                             self.x[:, d][feature_mask]).sum()

        # 保存 alpha 的梯度（需外部优化器访问）
        self.dalpha = dalpha
        self.x = None
        self.mask = None
        return dx


if __name__ == '__main__':
    # conv_tanh = nn.Conv2d
    from torch import nn    # 仅用于验证手动实现

    conv_param = {'pad': 1, 'stride': 2, 'output_padding': 1}
    input_tensor = torch.randn(1, 3, 32, 32)
    t_tensor = torch.randn(1, 3, 32, 32)
    epsilon = 1e-7
    # txt = t_tensor
    #
    # txt -= 1

    # 定义反卷积层参数
    W = 0.01 * torch.randn(6, 3, 3, 3)  # 输出通道3，输入通道1，卷积核3×3
    b = torch.zeros(6)
    # layer = Deconvolution(W, b, stride=conv_param['stride'], pad=conv_param['pad'], output_padding=conv_param['output_padding'])
    # layer = Convolution(W, b, stride=conv_param['stride'], pad=conv_param['pad'])
    # gamma = torch.ones(3)
    # beta = torch.zeros(3)
    # layer = BatchNormalization(gamma, beta, 0.1, epsilon)
    # layer = PixelShuffle(2)
    # layer = PReLU(num_parameters=3, init_alpha=0.25)
    criterion = MSELoss(size_average=False)

    # output_tensor = layer.forward(input_tensor)
    output_tensor = input_tensor
    loss = criterion.forward(output_tensor, t_tensor)
    dout = criterion.backward()
    # dout = layer.backward(dout)

    # dynamic_layer_dW = layer.dgamma
    # dynamic_layer_db = layer.dbeta

    nn_criterion = nn.MSELoss(size_average=False)
    # nn_layer = nn.Conv2d(1, 3, 3, stride=conv_param['stride'], padding=conv_param['pad'])
    # nn_layer = nn.ConvTranspose2d(3, 6, 3, stride=conv_param['stride'],
    #                               padding=conv_param['pad'], output_padding=conv_param['output_padding'])
    # nn_layer = nn.BatchNorm2d(3, eps=epsilon)
    # nn_layer = nn.PixelShuffle(2)
    # nn_layer.weight = nn.Parameter(W.clone().permute(1, 0, 2, 3))
    # nn_layer.weight = nn.Parameter(gamma.clone())
    # nn_layer.bias = nn.Parameter(beta.clone())
    # nn.LeakyReLU
    # nn_layer = nn.PReLU(num_parameters=3, init=0.25)
    nn_input = input_tensor.clone().requires_grad_(True)
    nn_output_tensor = nn_input
    # nn_output_tensor = nn_layer(nn_input)
    nn_loss = nn_criterion(nn_output_tensor, t_tensor)
    nn_loss.backward()

    print(torch.allclose(output_tensor, nn_output_tensor))
    print(torch.allclose(dout, nn_input.grad))
    print(torch.allclose(loss, nn_loss))
    # print(torch.allclose(dynamic_layer_dW, nn_layer.weight.grad))
    # print(torch.allclose(dynamic_layer_db, nn_layer.bias.grad))

