import pickle
from collections import OrderedDict
from typing import Any

from layers import *
import torch
import math
from functools import reduce



class baseModel:
    def __init__(self, device, weight_mode, init_way):
        self.layers: OrderedDict[str, Any] = OrderedDict()
        self.params: dict[str, Any] = {}
        self.params["device"] = device
        if isinstance(weight_mode, str):
            self.weight_mode = weight_mode.lower()
        else:
            self.weight_mode = weight_mode
        self.init_way = init_way.lower()

    def _init_weight(self, layer_key, shape, weight_mode="he", init_way="uniform", bias=True, device="cpu"):
        if isinstance(weight_mode, str):
            fan = lambda z: reduce(lambda x, y: x * y, z)
            std = lambda x: gain / math.sqrt(fan(x))
            W_bound = lambda x: math.sqrt(3.0) * std(x)
            b_bound = lambda x: 1 / math.sqrt(fan(x))
            weight_mode = weight_mode.lower()
            if weight_mode == "":
                weight_mode = "he"
            if weight_mode == "he":
                gain = math.sqrt(2.0)
            elif weight_mode == "xavier":
                gain = 1.0
            elif weight_mode == "prelu":
                negative_slope = 0.25
                gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            elif weight_mode == "tanh":
                gain = 5.0 / 3
            else:
                raise ValueError("Invalid weight initialization mode")

            if init_way == "uniform":
                init_weight = lambda x: torch.empty(*x, device=device).uniform_(-W_bound(x[1:]), W_bound(x[1:]))
                init_bias = lambda x: torch.empty(x[0], device=device).uniform_(-b_bound(x[1:]), b_bound(x[1:]))
            elif init_way == "normal":
                init_weight = lambda x: torch.empty(*x, device=device).normal_(mean=0, std=std(x[1:]))
                init_bias = lambda x: torch.empty(x[0], device=device).normal_(mean=0, std=std(x[1:]))
            else:
                raise ValueError("Invalid initialization method")
        else:
            init_weight = lambda x: weight_mode * torch.randn(*x, device=device)
            init_bias = lambda x: torch.zeros(x[0], device=device)

        if 'bn' in layer_key:
            self.params[f"gamma{layer_key.replace('bn', '')}"] = torch.ones(shape[0], device=device)
            self.params[f"beta{layer_key.replace('bn', '')}"] = torch.zeros(shape[0], device=device)
            self.layers[layer_key].gamma = self.params[f"gamma{layer_key.replace('bn', '')}"]
            self.layers[layer_key].beta = self.params[f"beta{layer_key.replace('bn', '')}"]
        # elif "prelu" in layer_key:
        #     self.params[f"alpha{layer_key.replace('prelu', '')}"] = torch.full((num_parameters,), init_alpha, dtype=torch.float32, device=device)
        #     self.layers[layer_key].alpha = self.params[f"alpha{layer_key.replace('prelu', '')}"]
        else:
            for key in ["deconv", "conv"]:
                if key in layer_key:
                    self.params[f"W{layer_key.replace(key, '')}"] = init_weight(shape)
                    self.layers[layer_key].W = self.params[f"W{layer_key.replace(key, '')}"]
                    if bias:
                        self.params[f"b{layer_key.replace(key, '')}"] = init_bias(shape)
                        self.layers[layer_key].b = self.params[f"b{layer_key.replace(key, '')}"]
                    break

    def _gradient(self):
        grads = {}

        count = 0
        res_count = 0
        bn_count = 0
        prelu_count = 0
        exclude = {"relu", "pix", "res", "bn", "tanh"}
        # 执行操作
        for key in self.layers.keys():
            if not any(sub in key for sub in exclude):
                count += 1
                grads[f"W{count}"] = self.layers[key].dW
                grads[f"b{count}"] = self.layers[key].db
            elif 'bn' in key:
                bn_count += 1
                grads[f"gamma{bn_count}"] = self.layers[key].dgamma
                grads[f"beta{bn_count}"] = self.layers[key].dbeta
            elif "prelu" in key:
                prelu_count += 1
                grads[f"alpha{prelu_count}"] = self.layers[key].dalpha
            elif "res" in key:
                res_count += 1
                res_Wb = 0
                res_gb = 0
                res_prelu = 0
                for res_key in self.layers[key].layers.keys():
                    if not any(sub in res_key for sub in {'relu', "bn"}):
                        res_Wb += 1
                        grads[f"res{res_count}_W{res_Wb}"] = self.layers[key].layers[res_key].dW
                        grads[f"res{res_count}_b{res_Wb}"] = self.layers[key].layers[res_key].db
                    elif 'bn' in res_key:
                        res_gb += 1
                        grads[f"res{res_count}_gamma{res_gb}"] = self.layers[key].layers[res_key].dgamma
                        grads[f"res{res_count}_beta{res_gb}"] = self.layers[key].layers[res_key].dbeta
                    elif 'prelu' in res_key:
                        res_prelu += 1
                        grads[f"res{res_count}_alpha{res_prelu}"] = self.layers[key].layers[res_key].dalpha

        return grads


    def __repr__(self):
        def _repr(obj):
            if isinstance(obj, (Convolution, Deconvolution)):
                return (f"{obj.__class__.__name__}"
                        f"{tuple(obj.W.shape)+(f'pad={obj.pad}',f'stride={obj.stride}',)+((f'output_padding={obj.output_padding}',)if hasattr(obj, 'output_padding') else ())}")
            elif isinstance(obj, PReLU):
                return f"{obj.__class__.__name__}{tuple(obj.alpha.shape)}"
            elif isinstance(obj, PixelShuffle):
                return f"{obj.__class__.__name__}({obj.upscale_factor})"
            elif isinstance(obj, BatchNormalization):
                return f"{obj.__class__.__name__}{f'(momentum={obj.momentum}, epsilon={obj.epsilon})'}"
            elif isinstance(obj, baseModel):
                fun = lambda x: ('\n'+'\t'*2)+("\n"+"\t"*2).join(x.split('\n'))
                return f"{obj.__class__.__name__}({fun(repr(obj))})"
            else:
                return obj.__class__.__name__
        return "\n".join(f"{key}: { _repr(val)}" for key, val in self.layers.items())


    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        params["net"] = list(self.layers.keys())
        res_count = 0
        bn_count = 0
        # prelu_count = 0
        for layer in params["net"]:
            if 'bn' in layer:
                bn_count += 1
                params[f"running_mean{bn_count}"] = self.layers[layer].running_mean
                params[f"running_var{bn_count}"] = self.layers[layer].running_var
                params[f"epsilon{bn_count}"] = self.layers[layer].epsilon
            # elif "prelu" in layer:
            #     prelu_count += 1
            #     params[f"alpha{prelu_count}"] = self.layers[layer].alpha
            elif "res" in layer:
                res_count += 1
                res_gb = 0
                # res_prelu = 0
                for res_key in self.layers[layer].layers.keys():
                    if 'bn' in res_key:
                        res_gb += 1
                        params[f"res{res_count}_running_mean{res_gb}"] = self.layers[layer].layers[res_key].running_mean
                        params[f"res{res_count}_running_var{res_gb}"] = self.layers[layer].layers[res_key].running_var
                        params[f"res{res_count}_epsilon{res_gb}"] = self.layers[layer].layers[res_key].epsilon
                    # elif 'prelu' in res_key:
                    #     res_prelu += 1
                    #     params[f"res{res_count}_alpha{res_prelu}"] = self.layers[layer].layers[res_key].alpha

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        nets = []
        for key, val in params.items():
            if key == "net":
                nets = val
            elif not any(sub in key for sub in {"mean", "var", "epsilon"}):
                self.params[key] = val
        count = 0
        res_count = 0
        bn_count = 0
        prelu_count = 0
        exclude = {"relu", "pix", "res", "bn", "tanh"}
        for key in nets:
            if not any(sub in key for sub in exclude):
                count += 1
                self.layers[key].W = self.params['W' + str(count)]
                self.layers[key].b = self.params['b' + str(count)]
            elif 'bn' in key:
                bn_count += 1
                self.layers[key].gamma = self.params[f"gamma{bn_count}"]
                self.layers[key].beta = self.params[f"beta{bn_count}"]
                self.layers[key].running_mean = params[f"running_mean{bn_count}"]
                self.layers[key].running_var = params[f"running_var{bn_count}"]
                self.layers[key].epsilon = params[f"epsilon{bn_count}"]
            elif 'prelu' in key:
                prelu_count += 1
                self.layers[key].alpha = self.params[f"alpha{prelu_count}"]
            elif "res" in key:
                res_count += 1
                res_Wb = 0
                res_gb = 0
                res_prelu = 0
                for res_key in self.layers[key].layers.keys():
                    if not any(sub in res_key for sub in {'relu', "bn"}):
                        res_Wb += 1
                        self.layers[key].layers[res_key].W = self.params[f"res{res_count}_W{res_Wb}"]
                        self.layers[key].layers[res_key].b = self.params.get(f"res{res_count}_b{res_Wb}", None)
                    elif 'bn' in res_key:
                        res_gb += 1
                        self.layers[key].layers[res_key].gamma = self.params[f"res{res_count}_gamma{res_gb}"]
                        self.layers[key].layers[res_key].beta = self.params[f"res{res_count}_beta{res_gb}"]
                        self.layers[key].layers[res_key].running_mean = params[f"res{res_count}_running_mean{res_gb}"]
                        self.layers[key].layers[res_key].running_var = params[f"res{res_count}_running_var{res_gb}"]
                        self.layers[key].layers[res_key].epsilon = params[f"res{res_count}_epsilon{res_gb}"]
                    elif 'prelu' in res_key:
                        res_prelu += 1
                        self.layers[key].layers[res_key].alpha = self.params[f"res{res_count}_alpha{res_prelu}"]


class SRCNN(baseModel):
    def __init__(self, scale=2, in_dims=(3, 32, 32), last_activation="", usb_bn=False, weight_mode="he", init_way="normal", device="cpu"):
        super().__init__(device=device, weight_mode=weight_mode, init_way=init_way)

        self.layers["conv1"] = Convolution(None, None, stride=1, pad=4)
        conv_shape = (64, in_dims[0], 9, 9)
        self._init_weight("conv1", conv_shape, self.weight_mode, self.init_way, device=device)
        if usb_bn:
            self.layers["bn1"] = BatchNormalization(None, None, 0.1)
            self._init_weight("bn1", (conv_shape[0],), device=device)
        # self.layers["relu1"] = Relu()
        self.layers["conv2"] = Convolution(None, None, stride=1, pad=2)
        conv_shape = (32, 64, 5, 5)
        self._init_weight("conv2", conv_shape, self.weight_mode, self.init_way, device=device)
        count = 2
        count += 1

        self.layers[f"conv{count}"] = Convolution(None, None, stride=1, pad=2)
        conv_shape = (in_dims[0], 32, 5, 5)
        self._init_weight(f"conv{count}", conv_shape, last_activation, self.init_way, device=device)


        count += 1
        if last_activation.lower() == "tanh":
            self.layers[f"tanh1"] = Tanh()
        elif last_activation.lower() == "relu":
            self.layers[f"relu1"] = Relu()
        else:
            pass
        self.last_layer = MSELoss()

    def forward(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if 'bn' in name or 'res' in name:
                x = layer.forward(x, train_flg)
            else:
                layer.train = train_flg
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.forward(x, train_flg)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = self._gradient()
        return grad


class SRCNNWithDeconv(baseModel):
    def __init__(self, in_dims=(3, 32, 32), scale=2, base_channels=64, last_activation="", weight_mode="he", init_way="normal", device="cpu"):
        super().__init__(device=device, weight_mode=weight_mode, init_way=init_way)
        ##(conv - relu)*5 - (deconv - relu)*4 - deconv #
        channel = base_channels
        conv_shape = (channel, in_dims[0], 3, 3)
        self.layers["conv1"] = Convolution(None, None, stride=1, pad=1)
        self._init_weight("conv1", conv_shape, self.weight_mode, self.init_way, device=device)
        for i in range(2, 6):
            if i % 2 == 0:
                self.layers[f"conv{i}"] = Convolution(None, None, stride=scale, pad=1)
                conv_shape = (channel, channel, 3, 3)
                self._init_weight(f"conv{i}", conv_shape, self.weight_mode, self.init_way, device=device)

                self.layers[f"relu{i}"] = Relu()
            else:
                channel *= 2
                self.layers[f"conv{i}"] = Convolution(None, None, stride=1, pad=1)
                conv_shape = (channel, channel//2, 3, 3)
                self._init_weight(f"conv{i}", conv_shape, self.weight_mode, self.init_way, device=device)

                self.layers[f"relu{i}"] = Relu()

        channel = base_channels * 4
        for i in range(6, 10):
            if i % 2 == 0:
                self.layers[f"deconv{i}"] = Deconvolution(None, None, stride=1, pad=1)
                conv_shape = (channel, channel, scale*2, scale*2)
                self._init_weight(f"deconv{i}", conv_shape, self.weight_mode, self.init_way, device=device)

                self.layers[f"relu{i}"] = Relu()
            else:
                self.layers[f"deconv{i}"] = Deconvolution(None, None, stride=scale, pad=scale//2)
                conv_shape = (channel//2, channel, scale*2, scale*2)
                self._init_weight(f"deconv{i}", conv_shape, self.weight_mode, self.init_way, device=device)
                channel = channel//2

                self.layers[f"relu{i}"] = Relu()
        self.layers[f"deconv{i+1}"] = Deconvolution(None, None, stride=scale, pad=scale//2)
        conv_shape = (in_dims[0], channel, scale*2, scale*2)
        self._init_weight(f"deconv{i+1}", conv_shape, last_activation, self.init_way, device=device)


        if last_activation.lower() == "tanh":
            self.layers[f"tanh1"] = Tanh()
        elif last_activation.lower() == "relu":
            self.layers[f"relu{i+1}"] = Relu()
        else:
            pass
        self.last_layer = MSELoss()

    def forward(self, x, train_flg=False):
        for name, layer in self.layers.items():
            if 'bn' in name or 'res' in name:
                x = layer.forward(x, train_flg)
            else:
                layer.train = train_flg
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.forward(x, train_flg)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = self._gradient()

        return grads


class SimpleResidualBlock(baseModel):
    def __init__(self, channels, weight_init="he", device="cpu", use_bn=True, init_way="uniform"):
        super().__init__(device=device, weight_mode=weight_init, init_way=init_way)
        # C, H, W = 3, 32, 32
        FH, FW = 3, 3
        pad, stride = 1, 1
        negative_slope = 0.25

        ## conv-bn-relu-conv-bn ##
        self.layers["conv1"] = Convolution(None, None, stride=stride, pad=pad)
        self._init_weight("conv1", (channels, channels, FH, FW), weight_init, init_way, bias=(not use_bn), device=device)
        if use_bn:
            self.layers["bn1"] = BatchNormalization(None, None, 0.1)
            self._init_weight("bn1", (channels,), device=device)


        if weight_init.lower() == "prelu":
            self.layers["prelu1"] = PReLU(num_parameters=channels, init_alpha=negative_slope, device=device)
            self.params["alpha1"] = self.layers["prelu1"].alpha
        else:
            self.layers["relu1"] = Relu()


        self.layers["conv2"] = Convolution(None, None, stride=stride, pad=pad)
        self._init_weight("conv2", (channels, channels, FH, FW), weight_init, init_way, bias=(not use_bn), device=device)
        if use_bn:
            self.layers["bn2"] = BatchNormalization(None, None, 0.1)
            self._init_weight("bn2", (channels,), device=device)

    def forward(self, x, train_flg=False):
        identity = x.clone()
        for name, layer in self.layers.items():
            if 'bn' in name:
                x = layer.forward(x, train_flg)
            else:
                layer.train = train_flg
                x = layer.forward(x)
        x += identity
        del identity
        return x

    def backward(self, dout):

        # 1. 反向传播卷积层
        dx_conv = dout.clone()
        for layer in reversed(self.layers.values()):
            dx_conv = layer.backward(dx_conv)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dx_conv = dx_conv.to(dout.device)

        # 2. 残差连接的梯度
        dx_residual = dout  # 直接路径的梯度

        # 3. 总梯度是两者的和
        dx_total = dx_conv + dx_residual
        del dx_conv
        return dx_total


class SRResNet(baseModel):
    def __init__(self, weight_init=0.01, in_dims=(3, 32, 32), upscale_factor=4, num_residual_blocks=16, channels=64,
                 last_activation="relu", init_way="uniform", use_res_bn=False,
                 size_average=True,
                 device="cpu"):
        super().__init__(device=device, weight_mode=weight_init, init_way=init_way)
        self.upscale_factor = upscale_factor
        self.channels = channels
        negative_slope = 0.25


        ### conv - relu - res x n - conv - bn - upscale x n - conv ###

        # 初始卷积层
        self.layers['conv1'] = Convolution(None, None, stride=1, pad=4)
        self._init_weight('conv1', (channels, in_dims[0], 9, 9), self.weight_mode, self.init_way, bias=True, device=device)

        if self.weight_mode == "prelu":
            self.layers['prelu1'] = PReLU(num_parameters=channels, init_alpha=negative_slope, device=device)
            self.params["alpha1"] = self.layers['prelu1'].alpha
        else:
            self.layers['relu1'] = Relu()

        # 残差块堆叠
        for i, _ in enumerate(range(num_residual_blocks), 1):
            self.layers[f'res{i}'] = SimpleResidualBlock(channels, use_bn=use_res_bn , weight_init=self.weight_mode, init_way=self.init_way, device=device)
            for key in self.layers[f'res{i}'].params.keys():
                self.params[f'res{i}_{key}'] = self.layers[f'res{i}'].params[key]

        # 残差块后的卷积层
        self.layers['conv2'] = Convolution(None, None, stride=1, pad=1)
        self._init_weight('conv2', (channels, channels, 3, 3), self.weight_mode, self.init_way, bias=True, device=device)


        self.layers['bn1'] = BatchNormalization(None, None, 0.1)
        self._init_weight('bn1', (channels,), device=device)


        # 上采样模块（根据 upscale_factor 决定层数）
        num_upsample = int(math.log2(upscale_factor))
        count = 2

        ### upscale ###
        ## conv - pix - relu ##
        for _ in range(num_upsample):
            # 子像素卷积：通道数乘以 upscale_factor²
            count += 1
            conv_shape = (channels * (2 ** 2), channels, 3, 3)

            self.layers[f'conv{count}'] = Convolution(None, None, stride=1, pad=1)
            self._init_weight(f'conv{count}', conv_shape, self.weight_mode, self.init_way, bias=True, device=device)

            self.layers[f'pix{count - 2}'] = PixelShuffle(upscale_factor=2)

            if self.weight_mode == "prelu":
                self.layers[f'prelu{count - 1}'] = PReLU(num_parameters=channels, init_alpha=negative_slope, device=device)
                self.params[f'alpha{count - 1}'] = self.layers[f'prelu{count - 1}'].alpha
            else:
                self.layers[f'relu{count - 1}'] = Relu()



        count += 1
        conv_shape = (in_dims[0], channels, 9, 9)
        # 最终卷积层
        self.layers[f'conv{count}'] = Convolution(None, None, stride=1, pad=4)
        self._init_weight(f'conv{count}', conv_shape, last_activation, self.init_way, bias=True, device=device)


        if last_activation.lower() in ["he", "relu"]:
            self.layers[f'relu{count - 1}'] = Relu()
        elif last_activation.lower() == "prelu":
            self.layers[f'prelu{count - 1}'] = PReLU(num_parameters=in_dims[0], init_alpha=negative_slope, device=device)
            self.params[f'alpha{count - 1}'] = self.layers[f'prelu{count - 1}'].alpha
        elif last_activation.lower() == "tanh":
            self.layers[f'tanh1'] = Tanh()
        else:
            pass
        self.last_layer = MSELoss(size_average=size_average)
        self.params['device'] = device

    def forward(self, x, train_flg=False):
        for name, layer in self.layers.items():
            x = x.to(self.params['device'])
            if 'bn' in name or 'res' in name:
                x = layer.forward(x, train_flg)
            else:
                layer.train = train_flg
                x = layer.forward(x)
            if name in ['relu1', 'prelu1']:
                identity = x.clone()
            if name == 'bn1':
                x += identity
                del identity
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return x

    def loss(self, x, t, train_flg=False):
        y = self.forward(x, train_flg)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.items())
        layers.reverse()
        for name, layer in layers:
            dout = dout.to(device=self.params['device'])
            if name == 'relu1':
                dout += identity_dout
                del identity_dout
            if name == 'bn1':
                identity_dout = dout.clone()
            dout = layer.backward(dout)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # 设定
        grads = self._gradient()

        return grads




if __name__=='__main__':
    from torch.nn import functional as F
    import torch.nn as nn
    class ResidualBlock(nn.Module):
        def __init__(self, in_channles, num_channles, use_1x1conv=False, strides=1):  # 第三个参数是是否使用1x1卷积
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channles, num_channles, kernel_size=3, stride=strides, padding=1, )  # 默认为311结构 宽高不会变，但把步长改为2，就会变
            self.conv2 = nn.Conv2d(
                num_channles, num_channles, kernel_size=3, padding=1)  # 默认这里宽高也不会变，但把步长改为2，就会变
            if use_1x1conv:
                self.conv3 = nn.Conv2d(
                    in_channles, num_channles, kernel_size=1, stride=strides)  # 这里相当于就是残差连接了
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channles)  # 批归一化
            self.bn2 = nn.BatchNorm2d(num_channles)
            self.relu = nn.ReLU(inplace=True)  # 节省内存

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            if self.conv3:
                x = self.conv3(x)
            y += x
            return y
            # return F.relu(y)  # 在forward里的relu是这样调用的



    conv_param = {'pad': 1, 'stride': 1}
    input_tensor = torch.randn(1, 1, 32, 32)
    t_tensor = torch.randn(1, 1, 32, 32)
    # 定义反卷积层参数
    # W = 0.01 * torch.randn(3, 1, 3, 3)  # 输出通道3，输入通道1，卷积核3×3
    # b = torch.zeros(3)
    # layer = Deconvolution(W, b, stride=conv_param['stride'], pad=conv_param['pad'])
    # layer = Convolution(W, b, stride=conv_param['stride'], pad=conv_param['pad'])
    layer = SimpleResidualBlock(1, 'he')
    criterion = MSELoss()
    output_tensor = layer.forward(input_tensor, True)
    loss = criterion.forward(output_tensor, t_tensor)
    dout = criterion.backward()
    layer.backward(dout)
    # dynamic_layer_dW = layer.dW
    # dynamic_layer_db = layer.db

    nn_criterion = nn.MSELoss()
    # nn_layer = nn.Conv2d(1, 3, 3,
    #                      stride=conv_param['stride'], padding=conv_param['pad'])
    nn_layer = ResidualBlock(1, 1, strides=1)

    nn_layer.conv1.weight = nn.Parameter(layer.layers['conv1'].W.clone())
    nn_layer.conv1.bias = nn.Parameter(layer.layers['conv1'].b.clone())
    nn_layer.conv2.weight = nn.Parameter(layer.layers['conv2'].W.clone())
    nn_layer.conv2.bias = nn.Parameter(layer.layers['conv2'].b.clone())


    # nn_layer.weight = nn.Parameter(W.clone())
    # nn_layer.bias = nn.Parameter(b.clone())

    nn_output_tensor = nn_layer(input_tensor.clone())
    nn_loss = nn_criterion(nn_output_tensor, t_tensor)
    nn_loss.backward()
    print(torch.allclose(output_tensor, nn_output_tensor, atol=1e-4))
    print(torch.allclose(layer.layers['conv1'].dW, nn_layer.conv1.weight.grad, atol=1e-4))
    print(torch.allclose(layer.layers['conv1'].db, nn_layer.conv1.bias.grad))
    print(torch.allclose(layer.layers['conv2'].dW, nn_layer.conv2.weight.grad, atol=1e-4))
    print(torch.allclose(layer.layers['conv2'].db, nn_layer.conv2.bias.grad))
    # print(torch.allclose(output_tensor, nn_output_tensor))
    # print(torch.allclose(dynamic_layer_dW, nn_layer.weight.grad))
    # print(torch.allclose(dynamic_layer_db, nn_layer.bias.grad))