import math
import pickle
from collections import defaultdict
from typing import DefaultDict, Any

import numpy as np
import torch


class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7,):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.eps = eps

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                if isinstance(val, torch.Tensor):
                    # val = val.to(params["device"])
                    self.m[key] = torch.zeros_like(val, device=params["device"])
                    self.v[key] = torch.zeros_like(val, device=params["device"])

        self.iter += 1
        lr_t = self.lr * torch.sqrt(torch.tensor(1.0 - self.beta2 ** self.iter, device=params["device"])) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            if isinstance(params[key], torch.Tensor):
                # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
                # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
                self.m[key] += (1 - self.beta1) * (grads[key].to(params["device"]) - self.m[key])
                self.v[key] += (1 - self.beta2) * (grads[key].to(params["device"]) ** 2 - self.v[key])

                params[key] -= lr_t * self.m[key] / (torch.sqrt(self.v[key]) + self.eps)

                # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
                # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
                # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

    def save_params(self, file_name="params.pkl"):
        params = {}
        params["m"] = self.m
        params["v"] = self.v
        params["iter"] = self.iter
        params["lr"] = self.lr
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        self.m, self.v = params["m"], params["v"]
        self.iter, self.lr = params["iter"], params["lr"]


class SGD:
    """随机梯度下降（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01, momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        # 参数校验
        if lr < 0.0:
            raise ValueError(f"无效学习率: {lr}")
        if momentum < 0.0:
            raise ValueError(f"无效动量值: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"无效权重衰减值: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov动量需要动量且无阻尼")

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # 状态存储（用于动量）
        self.state: DefaultDict[str, Any] = defaultdict(dict)  # 格式：{key: 状态字典}

    def update(self, params, grads):
        """执行参数更新
        Args:
            params (dict): 参数字典 {key: Tensor}
            grads (dict): 梯度字典 {key: Tensor}
        """
        momentum_buffer_list = []
        params_keys = list(filter(lambda x: isinstance(params[x], torch.Tensor), params.keys()))

        for key in params_keys:
            # param = params[key]
            # param_id = id(param)

            state = self.state[key]
            if 'momentum_buffer' not in state:
                momentum_buffer_list.append(None)
            else:
                momentum_buffer_list.append(state['momentum_buffer'])

        for i, key in enumerate(params_keys):
            param = params[key]
            grad = grads[key]

            # 设备一致性处理
            if grad.device != param.device:
                grad = grad.to(param.device)

            # 权重衰减（L2正则化）
            if self.weight_decay != 0:
                # grad = grad.add(param, alpha=self.weight_decay)
                grad += param * self.weight_decay


            # 动量计算
            if self.momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    # 更新动量缓冲区
                    buf = torch.clone(grad)
                    momentum_buffer_list[i] = buf
                else:
                    # buf.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)
                    buf *= self.momentum
                    buf += grad * (1 - self.dampening)

                # Nesterov调整
                if self.nesterov:
                    # grad = grad.add(buf, alpha=self.momentum)
                    grad += buf * self.momentum
                else:
                    grad = buf

            # 执行参数更新
            # param.add_(grad, alpha=-self.lr)
            param -= grad * self.lr

            for p, momentum_buffer in zip(params_keys, momentum_buffer_list):
                # p = id(params[p])
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

    def save_params(self, file_name="params.pkl"):
        params = {"lr": self.lr, "momentum": self.momentum, "dampening": self.dampening,
                  "weight_decay": self.weight_decay, "nesterov": self.nesterov, "state": self.state}
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov, self.state = params.values()
        # self.weight_decay = 0.0001


class HybridOptimizer:
    """Adam与SGD混合优化器（基于动态切换策略）"""

    def __init__(self, adam_lr=1e-3, sgd_lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-4,
                 eps=1e-8, switch_threshold=1e-5, momentum=0.9, switch_epoch=50):
        # Adam 配置
        self.adam = Adam(lr=adam_lr, beta1=beta1, beta2=beta2, eps=eps)
        self.sgd_lr = sgd_lr
        # SGD 配置
        self.sgd = SGD(momentum=momentum, weight_decay=weight_decay)
        self.eps = eps
        self.switch_threshold = switch_threshold

        # 状态跟踪
        self.current_optim = self.adam
        self.lambda_sgd = 0
        self.alpha_history = []
        self.is_switched = False
        self.switch_epoch = switch_epoch

    def update(self, params, grads):
        self.current_optim.update(params, grads)
        # 在Adam阶段持续计算SGD学习率
        if isinstance(self.current_optim, Adam):
            self._compute_adaptive_sgd_lr(params, grads)

    def _compute_adaptive_sgd_lr(self, params, grads):
        """处理字典结构的梯度计算"""
        total_numerator = 0.0
        total_denominator = 0.0

        # 遍历所有参数层
        for key in params.keys():
            if not isinstance(params[key], torch.Tensor):
                continue

            # 获取Adam状态
            device = params[key].device
            m = self.adam.m[key].to(device)
            v = self.adam.v[key].to(device)
            grad = grads[key].to(device)

            # 计算修正后的动量
            m_hat = m / (1 - self.adam.beta1 ** self.adam.iter)
            v_hat = v / (1 - self.adam.beta2 ** self.adam.iter)

            # 计算Adam下降方向
            eta_adam = (self.adam.lr * m_hat) / (torch.sqrt(v_hat) + self.eps)

            # 计算当前层的点积
            layer_numerator = torch.sum(eta_adam * eta_adam).item()
            layer_denominator = torch.sum(eta_adam * grad).item()

            # 累加到全局计算
            total_numerator += layer_numerator
            total_denominator += layer_denominator

        # 防止除零错误
        total_denominator = total_denominator if abs(total_denominator) > self.eps else self.eps

        # 计算全局学习率
        alpha_sgd = total_numerator / (total_denominator + self.eps)

        # 更新移动平均
        self.lambda_sgd = self.adam.beta2 * self.lambda_sgd + (1 - self.adam.beta2) * alpha_sgd
        lambda_corrected = self.lambda_sgd / (1 - self.adam.beta2 ** self.adam.iter)

        # 记录历史值
        self.alpha_history.append(lambda_corrected)

        # 自动切换条件（基于最近20次平均）
        if len(self.alpha_history) > 20:
            recent_mean = np.mean(self.alpha_history[-20:])
            current_diff = abs(lambda_corrected - recent_mean)

            if current_diff < self.switch_threshold and not self.is_switched:
                print(f"\n切换到SGD，最终学习率: {lambda_corrected:.2e}")
                self.sgd.lr = lambda_corrected
                self.current_optim = self.sgd
                self.is_switched = True

    def save_params(self, file_name="params.pkl"):
        params = {"eps": self.eps, "switch_threshold": self.switch_threshold, "lambda_sgd":self.lambda_sgd,
                  "alpha_history": self.alpha_history, "is_switched": self.is_switched}
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        self.adam.save_params(file_name.replace(".pkl", "_adam.pkl"))
        self.sgd.save_params(file_name.replace(".pkl", "_sgd.pkl"))

    def load_params(self, file_name="params.pkl"):
        self.adam.load_params(file_name.replace(".pkl", "_adam.pkl"))
        self.sgd.load_params(file_name.replace(".pkl", "_sgd.pkl"))
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        self.switch_epoch = params["is_switched"]
        if not params["is_switched"]:
            self.current_optim = self.adam
        else:
            self.current_optim = self.sgd
        self.eps, self.switch_threshold, self.lambda_sgd, self.alpha_history, self.is_switched = params.values()