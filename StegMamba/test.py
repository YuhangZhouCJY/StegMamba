import torch
import torch.nn
import torch.optim
from model import *
import config as c
import datasets
import modules.Unet_common as common
import kornia
from calculate_PSNR_SSIM import *
import lpips
import time


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def gauss_noise(shape):
        noise = torch.zeros(shape).cuda()
        for i in range(noise.shape[0]):
            noise[i] = torch.randn(noise[i].shape).cuda()

        return noise

    def computePSNR(origin,pred):
        origin = np.array(origin)
        origin = origin.astype(np.float32)
        pred = np.array(pred)
        pred = pred.astype(np.float32)
        mse = np.mean((origin/1.0 - pred/1.0) ** 2)
        if mse < 1.0e-10:
          return 100
        return 10 * math.log10(255.0**2/mse)


    net = Model2()
    net.cuda()
    init_model(net)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    load(c.MODEL_PATH + c.suffix)
    net.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    num = 0
    test_log = 'test_log.txt'
    print("\nStart Testing : \n\n")
    test_result = {
        "psnr_cover_acover": 0.0,
        "ssim_cover_acover": 0.0,
        "lpips_cover_acover": 0.0,
        "psnr_acover_stego": 0.0,
        "ssim_acover_stego": 0.0,
        "lpips_acover_stego": 0.0,
        "psnr_secret_rsecret": 0.0,
        "ssim_secret_rsecret": 0.0,
        "lpips_secret_rsecret": 0.0
    }
    lpips_model = lpips.LPIPS(net="alex").to(device)
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(datasets.testloader):
            data = data.to(device)
            cover = data[data.shape[0] // 2:, :, :, :]
            secret = data[:data.shape[0] // 2, :, :, :]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            stego, r, rcover, rsecret, acover = net(input_img)

            stego = iwt(stego)
            rcover = iwt(rcover)
            rsecret = iwt(rsecret)
            acover = iwt(acover)

            # LPIPS
            lpips_cover_acover = lpips_model(acover.detach(), cover.detach()).item()
            lpips_acover_stego = lpips_model(stego.detach(), acover.detach()).item()
            lpips_secret_rsecret = lpips_model(rsecret.detach(), secret.detach()).item()

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

            # PSNR
            psnr_cover_acover = calculate_psnr(cover, acover)
            psnr_acover_stego = calculate_psnr(acover, stego)
            psnr_secret_rsecret = calculate_psnr(secret, rsecret)

            # SSIM
            ssim_cover_acover = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(cover).unsqueeze(0), torch.tensor(acover).unsqueeze(0), window_size=5, max_val=255.)
            ssim_acover_stego = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(acover).unsqueeze(0), torch.tensor(stego).unsqueeze(0), window_size=5, max_val=255.)
            ssim_secret_rsecret = 1 - 2 * kornia.losses.ssim_loss(torch.tensor(secret).unsqueeze(0), torch.tensor(rsecret).unsqueeze(0), window_size=5, max_val=255.)

            result = {
                "psnr_cover_acover": psnr_cover_acover,
                "ssim_cover_acover": ssim_cover_acover,
                "lpips_cover_acover": lpips_cover_acover,
                "psnr_acover_stego": psnr_acover_stego,
                "ssim_acover_stego": ssim_acover_stego,
                "lpips_acover_stego": lpips_acover_stego,
                "psnr_secret_rsecret": psnr_secret_rsecret,
                "ssim_secret_rsecret": ssim_secret_rsecret,
                "lpips_secret_rsecret": lpips_secret_rsecret
            }

            for key in result:
                test_result[key] += float(result[key])

            num += 1

            '''
            test results
            '''
            content = "Image " + str(i) + " : \n"
            for key in test_result:
                content += key + "=" + str(result[key]) + ","
            content += "\n"

            with open(test_log, "a") as file:
                file.write(content)

            print(content)

        end_time = time.time()
        total_time = end_time - start_time
        print("Total inference time:", total_time, "seconds")

        '''
        test results
        '''
        content = "Average : \n"
        for key in test_result:
            content += key + "=" + str(test_result[key] / num) + ","
        content += "\n"

        with open(test_log, "a") as file:
            file.write(content)

        print(content)

