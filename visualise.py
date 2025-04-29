from SRCNN_net import SRCNN, SRResNet, SRCNNWithDeconv
from DatasetLoader import *
from utils import *
import cv2
from skimage.metrics import structural_similarity as ssim


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    crop_size = 64
    scale = 2
    batch_size = 1
    visualize = True
    save = True
    last_activation = ""
    max_val = 1. if last_activation.lower() != 'tanh' else 2.
    transform = DynamicCompose([
        # BICUBIC_LR(scale=scale),
        ToTensor(last_layer=last_activation),
        # CenterCrop(crop_size,scale=scale),
        # RandomCrop(crop_size, scale=scale)
    ])
    dataset_name = "Set5"
    # dataset_name = "DIV2K/valid"
    # valid_data = Dataset14(data_dir=f"./{dataset_name}", transform=transform, scale=scale)
    valid_data = DatasetDIV2K(data_dir=f"./{dataset_name}", transform=transform, scale=scale)
    val_loader = DataLoader(dataset=valid_data, batch_size=batch_size)

    model = SRResNet(weight_init='prelu', in_dims=(3, 32, 32), upscale_factor=scale, num_residual_blocks=16, channels=64, use_res_bn=True, last_activation=last_activation, device=device)
    # model = SRCNN(scale=scale, in_dims=(3, crop_size // scale, crop_size // scale),
    #                         last_activation=last_activation, usb_bn=False, device=device)
    # model = SRCNNWithDeconv(in_dims=(3, crop_size // scale, crop_size // scale), scale=scale,
    #                        last_activation=last_activation, device=device)
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-21-00_20_38/params/SRResNetDIV2K_27.6776.pkl")
    model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-04-14_53_09/params/SRResNetDIV2K_27.5738.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-24-23_27_43/params/SRResNetDIV2K_27.2731.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-26-22_00_17/params/SRResNetDIV2K_24.1830.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-26-20_48_35/params/SRResNetDIV2K_24.5416.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-27-07_41_37/params/SRResNetDIV2K_24.0370.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-27-11_12_58/params/SRResNetDIV2K_23.6828.pkl")
    # model.load_params("checkpoints/LAPTOP-33576NBL_2025-04-27-15_12_23/params/SRResNetDIV2K_23.8232.pkl")
    print(model)
    psnr_mean = []
    psnr1_mean = []
    ssim1_mean = []
    ssim_mean = []
    mse_mean = []
    rmse_mean = []
    for j, (batch_t, batch_x) in enumerate(val_loader):
        # batch_t = batch_t.half()
        # batch_x = batch_x.half()
        t_y = rgb_to_ycbcr(batch_t)[:, 0].permute(1,2,0)
        # 动态设置 win_size
        min_side = min(t_y.shape[:2])
        win_size = min(7, min_side) if min_side % 2 == 1 else min(7, min_side - 1)

        x_y = rgb_to_ycbcr(batch_x)[:, 0].permute(1,2,0)
        # psnr_v = float(psnr(batch_t, batch_x, data_range=max_val))
        # ssim_value = float(ssim(t_y.numpy().squeeze(-1), x_y.numpy().squeeze(-1), data_range=max_val, multichannel=True))
        print()
        # print(psnr_v)
        # print(ssim_value)
        # psnr1_mean.append(psnr_v)
        # ssim1_mean.append(ssim_value)
        # if last_activation == "tanh":
        #     cv2.imshow("img", ((batch_x[0]+1)/2).permute(1, 2, 0).numpy()[:, :, ::-1])
        # else:
        #     cv2.imshow("img", (batch_x[0]).permute(1, 2, 0).numpy()[:, :, ::-1])
        batch_x = batch_x.to(device)
        # batch_x = x_y.tile(1,1,3).permute(2,0,1).unsqueeze(0).to(device)
        batch_out = model.forward(batch_x)
        batch_out = batch_out.cpu()
        # batch_out.clip_(0, 1)
        mse_loss = model.last_layer.forward(batch_out, batch_t)
        mse_mean.append(mse_loss.item())
        rmse = torch.sqrt(mse_loss)
        rmse_mean.append(rmse.item())
        print(rmse)
        out_y = rgb_to_ycbcr(batch_out)[:, 0].permute(1,2,0)
        # out_y = batch_out[:,0]

        # batch_t = t_y.tile(1,1,3).permute(2,0,1).unsqueeze(0)
        psnr_v = float(psnr_3channel(batch_t, batch_out, data_range=max_val))
        ssim_value = float(ssim(t_y.numpy().squeeze(-1), out_y.numpy().squeeze(-1), data_range=max_val, multichannel=True))
        print(psnr_v)
        print(ssim_value)
        ssim_mean.append(ssim_value)
        psnr_mean.append(psnr_v)
        if visualize and batch_size == 1:
            if last_activation == "tanh":
                img1 = ((batch_t[0]+1)/2).permute(1, 2, 0).numpy()[:, :, ::-1]
                img2 = ((batch_out[0]+1)/2).permute(1, 2, 0).numpy()[:, :, ::-1]
            else:
                img1 = (batch_t[0]).permute(1, 2, 0).numpy()[:, :, ::-1]
                img2 = (batch_out[0]).permute(1, 2, 0).numpy()[:, :, ::-1]

            im = np.uint8(np.clip(np.concatenate([img1, img2], axis=1), 0, 1)*255)
            cv2.imshow("HR___SR", im)
            if save:
                cv2.imwrite(f"{j}_HR___SR.png", im)
            cv2.waitKey(0)
    if psnr1_mean:
        print(f"mean BICUBIC {np.mean(psnr1_mean):.4f}")
        print(f"mean BICUBIC {np.mean(ssim1_mean):.4f}")
    print(f"mean {np.mean(psnr_mean):.4f}")
    print(f"mean ssim {np.mean(ssim_mean):.4f}")
    print(f"mean mse {np.mean(mse_mean):.4f}")
    print(f"mean rmse {np.mean(rmse_mean):.4f}")