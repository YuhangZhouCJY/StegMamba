import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import logging
import util
import kornia.losses
from calculate_PSNR_SSIM import calculate_psnr, calculate_mae, calculate_rmse


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def gauss_noise(shape):
        noise = torch.zeros(shape).to(device)
        for i in range(noise.shape[0]):
            noise[i] = torch.randn(noise[i].shape).to(device)
        return noise

    def guide_loss(output, bicubic_image):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(output, bicubic_image)
        return loss.to(device)

    def reconstruction_loss(rev_input, input):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(rev_input, input)
        return loss.to(device)

    def low_frequency_loss(ll_input, gt_input):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(ll_input, gt_input)
        return loss.to(device)

    def computePSNR(origin, pred):
        origin = np.array(origin)
        origin = origin.astype(np.float32)
        pred = np.array(pred)
        pred = pred.astype(np.float32)
        mse = np.mean((origin/1.0 - pred/1.0) ** 2)
        if mse < 1.0e-10:
          return 100
        return 10 * math.log10(255.0**2/mse)

    def load(name):
        state_dicts = torch.load(name)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}

        # Filter out unexpected or unwanted keys from network_state_dict
        filtered_network_state_dict = {}
        for key, value in network_state_dict.items():
            if key in net.state_dict():
                filtered_network_state_dict[key] = value

        net.load_state_dict(filtered_network_state_dict)

        try:
            optim.load_state_dict(state_dicts['opt'])
        except Exception as e:
            print(f'Cannot load optimizer for some reason: {e}')

    #####################
    # Model initialize: #
    #####################
    if c.stage == 'stage1':
        net = Model1()
    if c.stage == 'stage2' or c.stage == 'stage3':
        net = Model2()

    net.to(device)

    if c.train_next == False:
        init_model(net)

    if c.stage == 'stage1':
        net.adjustmodel.requires_grad = False
    if c.stage == 'stage2':
        net.invmodel.requires_grad = False
        net.interact.requires_grad = False
    if c.train_next:
        load(c.MODEL_PATH + c.suffix)

    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    dwt = common.DWT()
    iwt = common.IWT()

    util.setup_logger('train', '/logging/', 'train_', level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger('train')
    logger_train.info(net)

    try:
        writer = SummaryWriter(comment='ours', filename_suffix="stego")

        for i_epoch in range(c.epochs):
            i_epoch = i_epoch + c.trained_epoch + 1
            loss_history = []
            g_loss1_history = []
            g_loss2_history = []
            r_loss1_history = []
            #################
            #     train:    #
            #################

            for i_batch, data in enumerate(datasets.trainloader):
                data = data.to(device)
                cover = data[data.shape[0] // 2:]
                secret = data[:data.shape[0] // 2]

                cover_input = dwt(cover)
                secret_input = dwt(secret)

                input_img = torch.cat((cover_input, secret_input), 1)
                stego_out, r, rcover, rsecret, acover_out = net(input_img)

                stego = iwt(stego_out)
                rcover = iwt(rcover)
                rsecret = iwt(rsecret)
                acover = iwt(acover_out)

                if c.stage == 'stage1':
                    g_loss1 = torch.mean(torch.sum((stego.to(device) - acover.to(device))**2, (1, 2, 3)))
                    g_loss2 = torch.mean(torch.sum((cover.to(device) - acover.to(device))**2, (1, 2, 3)))
                    r_loss1 = torch.mean(torch.sum(torch.abs(rsecret.to(device) - secret.to(device)), (1, 2, 3)))
                    r_loss2 = torch.mean(torch.sum(torch.abs(rcover.to(device) - acover.to(device)), (1, 2, 3)))
                    stego_low = stego_out.narrow(1, 0, c.channels_in)
                    cover_low = cover_input.narrow(1, 0, c.channels_in)
                    l_loss = low_frequency_loss(stego_low, cover_low)
                    total_loss = 1 * r_loss1 + 0.1 * r_loss2 + 32 * g_loss1 + 16 * l_loss

                if c.stage == 'stage2':
                    g_loss1 = torch.mean(torch.sum((stego.to(device) - acover.to(device))**2, (1, 2, 3)))
                    g_loss2 = torch.mean(torch.sum((cover.to(device) - acover.to(device))**2, (1, 2, 3)))
                    r_loss1 = torch.mean(torch.sum(torch.abs(rsecret.to(device) - secret.to(device)), (1, 2, 3)))
                    r_loss2 = torch.mean(torch.sum(torch.abs(rcover.to(device) - acover.to(device)), (1, 2, 3)))
                    stego_low = stego_out.narrow(1, 0, c.channels_in)
                    acover_low = cover_input.narrow(1, 0, c.channels_in)
                    l_loss = low_frequency_loss(stego_low, acover_low)
                    total_loss = 1 * r_loss1 + 0.1 * r_loss2 + 32 * g_loss1 + 32 * g_loss2 + 16 * l_loss

                if c.stage == 'stage3':
                    g_loss1 = torch.mean(torch.sum((stego.to(device) - acover.to(device))**2, (1, 2, 3)))
                    g_loss2 = torch.mean(torch.sum((cover.to(device) - acover.to(device))**2, (1, 2, 3)))
                    r_loss1 = torch.mean(torch.sum(torch.abs(rsecret.to(device) - secret.to(device)), (1, 2, 3)))
                    r_loss2 = torch.mean(torch.sum(torch.abs(rcover.to(device) - acover.to(device)), (1, 2, 3)))
                    stego_low = stego_out.narrow(1, 0, c.channels_in)
                    cover_low = acover_out.narrow(1, 0, c.channels_in)
                    l_loss = low_frequency_loss(stego_low, cover_low)
                    total_loss = 1 * r_loss1 + 0.1 * r_loss2 + 32 * g_loss1 + 32 * g_loss2 + 16 * l_loss

                total_loss.backward()
                optim.step()
                optim.zero_grad()

                loss_history.append([total_loss.item(), 0.])
                r_loss1_history.append([r_loss1.item(), 0.])
                g_loss1_history.append([g_loss1.item(), 0.])
                g_loss2_history.append([g_loss2.item(), 0.])

            epoch_losses = np.mean(np.array(loss_history), axis=0)
            r_epoch_losses1 = np.mean(np.array(r_loss1_history), axis=0)
            g_epoch_losses1 = np.mean(np.array(g_loss1_history), axis=0)
            g_epoch_losses2 = np.mean(np.array(g_loss2_history), axis=0)

            epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
            #################
            #     val:    #
            #################
            if i_epoch % c.val_freq == 0:
                with torch.no_grad():
                    psnr_cover_acover = []
                    ssim_cover_acover = []

                    psnr_acover_stego = []
                    psnr_secret_rsecret = []

                    ssim_acover_stego = []
                    ssim_secret_rsecret = []

                    mae_acover_stego = []
                    mae_secret_rsecret = []

                    rmse_acover_stego = []
                    rmse_secret_rsecret = []

                    net.eval()
                    for x in datasets.validloader:
                        x = x.to(device)
                        cover = x[x.shape[0] // 2:, :, :, :]
                        secret = x[:x.shape[0] // 2, :, :, :]

                        cover_input = dwt(cover)
                        secret_input = dwt(secret)

                        input_img = torch.cat((cover_input, secret_input), 1)
                        stego, r, rcover, rsecret, acover = net(input_img)

                        stego = iwt(stego)
                        rcover = iwt(rcover)
                        rsecret = iwt(rsecret)
                        acover = iwt(acover)

                        rsecret = rsecret.cpu().numpy().squeeze() * 255
                        np.clip(rsecret, 0, 255)
                        secret = secret.cpu().numpy().squeeze() * 255
                        np.clip(secret, 0, 255)
                        cover = cover.cpu().numpy().squeeze() * 255
                        np.clip(cover, 0, 255)
                        stego = stego.cpu().numpy().squeeze() * 255
                        np.clip(stego, 0, 255)
                        acover = acover.cpu().numpy().squeeze() * 255
                        np.clip(acover, 0, 255)

                        # 自然度
                        psnr_temp_c_a = calculate_psnr(cover, acover)
                        psnr_cover_acover.append(psnr_temp_c_a)
                        ssim_temp_c_a = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(acover).unsqueeze(0), torch.tensor(cover).unsqueeze(0), window_size=5, max_val=255.)
                        ssim_cover_acover.append(ssim_temp_c_a)

                        # PSNR
                        psnr_temp_a_t = calculate_psnr(acover, stego)
                        psnr_acover_stego.append(psnr_temp_a_t)
                        psnr_temp_s_r = calculate_psnr(secret, rsecret)
                        psnr_secret_rsecret.append(psnr_temp_s_r)

                        # SSIM
                        ssim_temp_a_t = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(acover).unsqueeze(0), torch.tensor(stego).unsqueeze(0), window_size=5, max_val=255.)
                        ssim_acover_stego.append(ssim_temp_a_t)
                        ssim_temp_s_r = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(secret).unsqueeze(0), torch.tensor(rsecret).unsqueeze(0), window_size=5, max_val=255.)
                        ssim_secret_rsecret.append(ssim_temp_s_r)

                        # MAE
                        mae_temp_a_t = calculate_mae(acover, stego)
                        mae_acover_stego.append(mae_temp_a_t)
                        mae_temp_s_r = calculate_mae(secret, rsecret)
                        mae_secret_rsecret.append(mae_temp_s_r)

                        # RMSE
                        rmse_temp_a_t = calculate_rmse(acover, stego)
                        rmse_acover_stego.append(rmse_temp_a_t)
                        rmse_temp_s_r = calculate_rmse(secret, rsecret)
                        rmse_secret_rsecret.append(rmse_temp_s_r)

                    writer.add_scalars("PSNR_Cover_Acover", {"average psnr": np.mean(psnr_cover_acover)}, i_epoch)
                    writer.add_scalars("PSNR_Acover_Stego", {"average psnr": np.mean(psnr_acover_stego)}, i_epoch)
                    writer.add_scalars("PSNR_Secret_Rsecret", {"average psnr": np.mean(psnr_secret_rsecret)}, i_epoch)

                    logger_train.info(
                        f"TEST:   "
                        f'PSNR_cover_acover: {np.mean(psnr_cover_acover):.4f} | '
                        f'SSIM_cover_acover: {np.mean(ssim_cover_acover):.4f} | '
                        
                        f'PSNR_acover_stego: {np.mean(psnr_acover_stego):.4f} | '
                        f'PSNR_secret_rsecret: {np.mean(psnr_secret_rsecret):.4f} | '

                        f'SSIM_acover_stego: {np.mean(ssim_acover_stego):.4f} | '
                        f'SSIM_secret_rsecret: {np.mean(ssim_secret_rsecret):.4f} | '

                        f'MAE_acover_stego: {np.mean(mae_acover_stego):.4f} | '
                        f'MAE_secret_rsecret: {np.mean(mae_secret_rsecret):.4f} | '

                        f'RMSE_acover_stego: {np.mean(rmse_acover_stego):.4f} | '
                        f'RMSE_secret_rsecret: {np.mean(rmse_secret_rsecret):.4f} | '
                    )

            viz.show_loss(epoch_losses)
            writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

            logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
            logger_train.info(
                f"Train epoch {i_epoch}:   "
                f'Loss: {epoch_losses[0].item():.4f} | '
                f'g_Loss1: {g_epoch_losses1[0].item():.4f} | '
                f'g_Loss2: {g_epoch_losses2[0].item():.4f} | '
                f'r_Loss1: {r_epoch_losses1[0].item():.4f} | '
            )

            if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
                torch.save({'opt': optim.state_dict(),
                            'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
            weight_scheduler.step()

        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
        writer.close()

    except:
        if c.checkpoint_on_error:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
        raise

    finally:
        viz.signal_stop()
