import pickle

from utils import draw_img_3, draw_img


if __name__ == '__main__':
    history_paths = [
        "checkpoints/LAPTOP-33576NBL_2025-04-27-07_41_37/params/SRResNetDIV2K_history.pkl",
        "checkpoints/LAPTOP-33576NBL_2025-04-27-11_12_58/params/SRResNetDIV2K_history.pkl",
        "checkpoints/LAPTOP-33576NBL_2025-04-27-15_12_23/params/SRResNetDIV2K_history.pkl"
    ]
    lists = []
    types = ["bn and without bias", "without bn and with bias", "more channels, without bn and with bias"]
    for history_path in history_paths:
        with open(history_path, "rb") as f:
            history_dict = pickle.load(f)
        train_loss_list, valid_loss_list, train_psnr_list, valid_psnr_list, train_rmse_list, valid_rmse_list = history_dict.values()
        lists.append(valid_rmse_list)
    draw_img_3(lists, x_lim=25, loss_type=types, y_label="rmse loss", img_name="SRResNet_rmse_comparison", save_path="SRResNet_rmse_comparison.png")

    # history_path = "checkpoints/LAPTOP-33576NBL_2025-04-27-07_41_37/params/SRResNetDIV2K_history.pkl"
    # with open(history_path, "rb") as f:
    #     history_dict = pickle.load(f)
    # train_loss_list, valid_loss_list, train_psnr_list, valid_psnr_list, train_rmse_list, valid_rmse_list = history_dict.values()
    # draw_img(train_rmse_list, valid_rmse_list,  loss_type="rmse_loss", img_name="SRResNet_rmse_loss", save_path="SRResNet_rmse_loss.png")