import os.path
import pickle
import socket
import time
import yaml

from SRCNN_net import SRCNN, SRResNet, SRCNNWithDeconv
from DatasetLoader import *
from optimizer import Adam, SGD, HybridOptimizer
from utils import *

if __name__ == "__main__":

    current_dictory = os.getcwd()
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    hostname = socket.gethostname()
    # hostname = "B20220307229"
    experiment_dir = os.path.join(current_dictory, "checkpoints", '_'.join([hostname, cur_time]))
    params_dir = os.path.join(experiment_dir, "params")
    img_dir = os.path.join(experiment_dir, "imgs")
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备（自动检测GPU）
    crop_size = 96              # 输入图像裁剪尺寸
    scale = 2                   # 超分辨率放大倍数
    last_activation = 'tanh'    # 最后一层激活函数
    weight_init = 'prelu'       # 权重初始化方法
    init_way = "uniform"
    batch_size = 16             # 批次大小
    epoch = 200                 # 训练总轮次
    lr0 = 1e-3                  # 初始学习率
    # lr1 = lr0 * 1e-1
    # dataset_name = "imagenette2"
    dataset_name = "DIV2K"      # 数据集名称
    # decay_rate = (lr1 / lr0) ** (1.0 / epoch)
    # decay_rate = 0.95
    LR_type = 'bicubic'         # 低分辨率生成方法
    max_val = 1. if last_activation.lower() != 'tanh' else 2.
    size_average = True         # 损失是否按元素平均
    num_residual_blocks = 32    # 残差块数量
    channels = 128               # 卷积层基础通道数
    use_res_bn = False
    # switch_threshold = 1e-8
    # switch_epoch = 20
    # weight_decay = 1e-4
    net_name = "SRResNet"
    load_pretrained = False
    load_checkpoint_dir = "checkpoints/LAPTOP-33576NBL_2025-04-26-20_48_35"
    checkpoint_imgs = os.path.join(load_checkpoint_dir, "imgs")
    checkpoint_params = os.path.join(load_checkpoint_dir, "params")


    config = {
        "dataset_name": dataset_name,
        "net_name": net_name,
        "scale": scale,
        "last_activation": last_activation,
        "weight_init": weight_init,
        "init_way": init_way,
        "batch_size": batch_size,
        "epoch": epoch,
        "lr0": lr0,
        # "lr1": lr1,
        # "decay_rate": decay_rate,
        "crop_size": crop_size,
        "LR_type": LR_type,
        "size_average": size_average,
        "num_residual_blocks": num_residual_blocks,
        "channels": channels,
        # "switch_epoch": switch_epoch,
        # "switch_threshold": switch_threshold,
        # "weight_decay": weight_decay,
        "load_pretrained": load_pretrained,
        "load_checkpoint_dir": load_checkpoint_dir,
        "use_res_bn": use_res_bn
    }
    config_path = os.path.join(experiment_dir, "config.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config, f)


    train_loss_list = []
    train_psnr_list = []
    train_rmse_list = []
    valid_loss_list = []
    valid_psnr_list = []
    valid_rmse_list = []


    transform_train = DynamicCompose([
        ToTensor(last_layer=last_activation),
        # CenterCrop(512,scale=scale),
        RandomCrop(crop_size, scale=scale)
    ])

    transform_valid = DynamicCompose([
        ToTensor(last_layer=last_activation),
        CenterCrop(crop_size, scale=scale)
    ])

    # train_data = Dataset(data_dir=f"./{dataset_name}/train", transform=transform, scale=scale, crop_size=crop_size)
    # valid_data = Dataset(data_dir=f"./{dataset_name}/val", transform=transform, scale=scale, crop_size=crop_size)

    train_data = DatasetDIV2K(data_dir=f"./{dataset_name}/train", LR_type=LR_type, transform=transform_train, scale=scale)
    valid_data = DatasetDIV2K(data_dir=f"./{dataset_name}/valid", LR_type=LR_type, transform=transform_valid, scale=scale)

    # train_data = Dataset14(data_dir=f"./{dataset_name}", transform=transform)
    # valid_data = Dataset14(data_dir=f"./{dataset_name}", transform=transform)


    data_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=2)
    model = SRResNet(weight_init=weight_init, in_dims=(3, crop_size//scale, crop_size//scale), upscale_factor=scale,
                     num_residual_blocks=num_residual_blocks, channels=channels,last_activation=last_activation, device=device, init_way=init_way,
                     size_average=size_average, use_res_bn=use_res_bn)

    # model = SRCNN(scale=scale, in_dims=(3, crop_size//scale, crop_size//scale),
    #                         last_activation=last_activation, usb_bn=False, device=device)
    # model = SRCNNWithDeconv(in_dims=(3, crop_size // scale, crop_size // scale),
    #                                       last_activation=last_activation, device=device)

    train_iter = len(data_loader)
    val_iter = len(val_loader)
    optim = Adam(lr=lr0)
    # optim = HybridOptimizer(
    #     adam_lr=lr0,
    #     weight_decay=weight_decay,
    #     switch_threshold=switch_threshold,
    #     momentum=0.9
    # )
    strart_epoch = 1
    if load_pretrained:
        params_dir = checkpoint_params
        img_dir = checkpoint_imgs
        model.load_params(os.path.join(checkpoint_params, f"{net_name}{dataset_name}_last.pkl"))
        optim.load_params(os.path.join(checkpoint_params, f"{net_name}{dataset_name}_optim.pkl"))
        history_path = os.path.join(checkpoint_params, f"{net_name}{dataset_name}_history.pkl")
        with open(history_path, "rb") as f:
            history_dict = pickle.load(f)
        train_loss_list, valid_loss_list, train_psnr_list, valid_psnr_list, train_rmse_list, valid_rmse_list = history_dict.values()
        strart_epoch = len(train_loss_list) + 1
        # draw_img(train_loss_list, valid_loss_list, img_name=f"{net_name}{dataset_name}",
        #          save_path=os.path.join(img_dir, f'{net_name}{dataset_name}.png'))
        # draw_img(train_psnr_list, valid_psnr_list, loss_type="psnr", img_name=f"{net_name}{dataset_name}_psnr",
        #          save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_psnr.png'))
        # draw_img(train_rmse_list, valid_rmse_list, loss_type="rmse", img_name=f"{net_name}{dataset_name}_rmse",
        #          save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_rmse.png'))
        #
        # if isinstance(optim, HybridOptimizer):
        #     draw_img_2(optim.alpha_history, type="sgd_lr", img_name=f"{net_name}{dataset_name}_lr_evolution",
        #                save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_lr_evolution.png'))
        #
        # with open(os.path.join(params_dir, f"{net_name}{dataset_name}_history.pkl"), "wb") as f:
        #     pickle.dump(history_dict, f)

    # optim.iter = 100 * 50
    print(model)
    val_loss = 0
    with torch.no_grad():
        for e in range(strart_epoch, epoch+1):
            # 在每个epoch开始时添加

            mse_mean = []
            psnr_mean = []
            rmse_mean = []
            start_time = time.time()  # 记录当前 epoch 的开始时间
            total_elapsed_time = 0  # 记录已完成批次的总耗时

            print(f"epoch: {e}/{epoch}")
            for i, (batch_t, batch_x) in enumerate(data_loader):
                batch_start_time = time.time()  # 记录当前批次的开始时间


                batch_t = batch_t.to(device)
                batch_x = batch_x.to(device)
                grad = model.gradient(batch_x, batch_t)
                # torch.cuda.empty_cache()
                optim.update(model.params, grad)

                # loss = model.loss(batch_x, batch_t)
                # batch_x = model.forward(batch_x)
                # loss = model.last_layer.forward(batch_x, batch_t)
                batch_y = model.last_layer.y
                mse_loss = model.last_layer.loss.cpu()
                # psnr_value = psnr_3channel(batch_t, batch_y, max_val).cpu()
                rmse_loss = torch.sqrt(mse_loss).cpu()

                mse_mean.append(mse_loss)
                # psnr_mean.append(psnr_value)
                rmse_mean.append(rmse_loss)

                # print(f"\r{i}/{train_iter}\ttrain mse:{mse_loss:.6f}\trmse:{rmse_loss:.6f}\tpsnr:{psnr_value:.6f}", end="")
                batch_end_time = time.time()  # 记录当前批次的结束时间
                batch_elapsed_time = batch_end_time - batch_start_time  # 计算当前批次的耗时
                total_elapsed_time += batch_elapsed_time  # 累加已完成批次的总耗时
                average_batch_time = total_elapsed_time / (i + 1)  # 计算已完成批次的平均耗时
                remaining_batches = train_iter - (i + 1)  # 计算剩余批次数量
                estimated_remaining_time = average_batch_time * remaining_batches  # 估算剩余时间


                print(f"\r{i}/{train_iter}\ttrain mse:{mse_loss:.6f}\trmse:{rmse_loss:.6f}\t"
                      # f"psnr:{psnr_value:.6f}\t"
                      f"Estimated remaining time: {estimated_remaining_time:.2f}s", end="")


            mse_loss = float(torch.mean(torch.tensor(mse_mean)))
            psnr_value = float(torch.mean(torch.tensor(psnr_mean))) if psnr_mean else 0
            rmse_loss = float(torch.mean(torch.tensor(rmse_mean)))

            train_loss_list.append(mse_loss)
            train_psnr_list.append(psnr_value)
            train_rmse_list.append(rmse_loss)

            print(f"\tmean mse:{mse_loss:.6f}"
                  f"\trmse:{rmse_loss:.6f}"
                  # f"\tpsnr:{psnr_value:.6f}"
                  )

            mse_mean = []
            psnr_mean = []
            rmse_mean = []
            val_start_time = time.time()  # 记录验证阶段开始时间
            val_total_elapsed = 0  # 验证阶段已完成批次的总耗时

            # optim.lr *= decay_rate
            # 在SGD阶段添加学习率衰减
            # if isinstance(optim.current_optim, SGD):
            #     optim.current_optim.lr *= decay_rate  # 每epoch衰减5%

            for j, (batch_t, batch_x) in enumerate(val_loader):
                batch_start = time.time()  # 记录当前验证批次开始时间
                batch_t = batch_t.to(device)
                batch_x = batch_x.to(device)

                batch_x = model.forward(batch_x)
                mse_loss = model.last_layer.forward(batch_x, batch_t).cpu()
                psnr_value = psnr_3channel(batch_t, batch_x, max_val).cpu()
                rmse_loss = torch.sqrt(mse_loss).cpu()


                mse_mean.append(mse_loss)
                psnr_mean.append(psnr_value)

                rmse_mean.append(rmse_loss)

                # print(f"\r{j}/{val_iter}\tvalid mse:{mse_loss:.6f}\trmse:{rmse_loss:.6f}\tpsnr:{psnr_value:.6f}", end="")
                batch_end = time.time()
                batch_elapsed = batch_end - batch_start  # 当前批次耗时
                val_total_elapsed += batch_elapsed  # 累加验证阶段已用时间
                avg_batch_time = val_total_elapsed / (j + 1)  # 平均每个验证批次耗时
                remaining_batches = val_iter - (j + 1)  # 剩余验证批次
                estimated_val_remaining = avg_batch_time * remaining_batches  # 验证阶段剩余时间
                epoch_elapsed = time.time() - start_time  # 整个 epoch 已用时间（含训练阶段）

                # 打印验证进度 + 验证阶段预计剩余时间 + 整个 epoch 预计总时间
                print(f"\r{j}/{val_iter}\tvalid mse:{mse_loss:.6f}\trmse:{rmse_loss:.6f}\tpsnr:{psnr_value:.6f}\t"
                      f"Val Remaining: {estimated_val_remaining:.2f}s\tEpoch Estimated Total: {epoch_elapsed + estimated_val_remaining:.2f}s",
                      end="")


            mse_loss = float(torch.mean(torch.tensor(mse_mean)))
            psnr_value = float(torch.mean(torch.tensor(psnr_mean)))
            rmse_loss = float(torch.mean(torch.tensor(rmse_mean)))

            valid_loss_list.append(mse_loss)
            valid_psnr_list.append(psnr_value)
            valid_rmse_list.append(rmse_loss)

            text = f"mean mse: {mse_loss:.6f}\trmse: {rmse_loss:.6f}\tpsnr: {psnr_value:.6f}"
            print(f"\t{text}")
            if psnr_value > val_loss:
                val_loss = psnr_value
                model.save_params(os.path.join(params_dir, f"SRResNet{dataset_name}_{psnr_value:.4f}.pkl"))
                optim.save_params(os.path.join(params_dir, f"{net_name}{dataset_name}_optim_best.pkl"))


            draw_img(train_loss_list, valid_loss_list, img_name=f"{net_name}{dataset_name}",
                     save_path=os.path.join(img_dir, f'{net_name}{dataset_name}.png'))
            draw_img(train_psnr_list, valid_psnr_list, loss_type="psnr", img_name=f"{net_name}{dataset_name}_psnr",
                     save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_psnr.png'))
            draw_img(train_rmse_list, valid_rmse_list, loss_type="rmse", img_name=f"{net_name}{dataset_name}_rmse",
                     save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_rmse.png'))

            history_dict = {
                "train_loss": train_loss_list,
                "valid_loss": valid_loss_list,
                "train_psnr": train_psnr_list,
                "valid_psnr": valid_psnr_list,
                "train_rmse": train_rmse_list,
                "valid_rmse": valid_rmse_list,
            }
            # if not optim.is_switched:
            if isinstance(optim, HybridOptimizer):
                draw_img_2(optim.alpha_history, type="sgd_lr", img_name=f"{net_name}{dataset_name}_lr_evolution",
                       save_path=os.path.join(img_dir, f'{net_name}{dataset_name}_lr_evolution.png'))

            with open(os.path.join(params_dir, f"{net_name}{dataset_name}_history.pkl"), "wb") as f:
                pickle.dump(history_dict, f)

            model.save_params(os.path.join(params_dir, f"{net_name}{dataset_name}_last.pkl"))
            optim.save_params(os.path.join(params_dir, f"{net_name}{dataset_name}_optim.pkl"))

    # draw_img(train_loss_list, valid_loss_list, img_name=f"SRResNet{dataset_name}", save_path=os.path.join(img_dir, f'SRResNet{dataset_name}.png'))
    # draw_img(train_psnr_list, valid_psnr_list, loss_type="psnr", img_name=f"SRResNet{dataset_name}_psnr", save_path=os.path.join(img_dir, f'SRResNet{dataset_name}_psnr.png'))
    #