import re
import os
import yaml
import torch
import shutil
import numpy as np
from src.Utils import *
import scipy.io as sio
from DataLoader import DataLoaderSL as DL
from src.Unrolled_Network import UnrolledNet

if __name__ == '__main__':

    seed_everything(42)    
    with open("Config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")


    # Saving 
    Saving_Path  = f"./{config['Saving']['path']}/"
    

    os.makedirs(Saving_Path, exist_ok=True)
    os.makedirs(f"{Saving_Path}/iterations", exist_ok=True)
    os.makedirs(f"{Saving_Path}/plots", exist_ok=True)


    CartesianData = DL(config["data_path"])
    Data_sampler  = torch.utils.data.RandomSampler(CartesianData)
    data_loader   = torch.utils.data.DataLoader(dataset=CartesianData,
                                              batch_size=1, 
                                              sampler = Data_sampler,
                                              num_workers = 4)
    
    Omega_Mask   = torch.tensor(sio.loadmat("./data/omega_mask.mat")["mask"].transpose([1,0])).unsqueeze(0).to(device)
    D1           = torch.tensor(sio.loadmat('./data/delta1.mat')['mask'].transpose([1,0]),dtype = torch.complex64).unsqueeze(0).to(device)
    D2           = torch.tensor(sio.loadmat('./data/delta2.mat')['mask'].transpose([1,0]),dtype = torch.complex64).unsqueeze(0).to(device)
    D3           = torch.tensor(sio.loadmat('./data/delta3.mat')['mask'].transpose([1,0]),dtype = torch.complex64).unsqueeze(0).to(device)
    mask_list    = [D1,D2,D3]
    
    
    network = UnrolledNet(mu = 0.1813, Unrolls = 10)
    network.load_state_dict(torch.load("./BestModel/checkpoint.pth"))
    network.to(device)
    
    cg_EEH = CG_EEH()
    cg_EEH.to(device)

    for idx , (ksp, coil, FileName) in enumerate(data_loader):
        
        coil = coil.to(device)
        ksp  = ksp.to(device)

        zero_filled = IFFT(ksp * Omega_Mask)
        label       = IFFT(ksp)            
        
        zero_filled = torch.sum(zero_filled * torch.conj(coil) , axis = 1, keepdims = True)
        label       = torch.sum(label       * torch.conj(coil) , axis = 1, keepdims = True)
        scale       = torch.max(torch.abs(zero_filled))
        zero_filled = zero_filled/scale
        label       = label/scale
        zero_filled.requires_grad = True

    
        # Noise Jittering calculation
        with torch.no_grad():
            us_ksp   = cg_EEH(fft_loss(zero_filled, coil, config["axis"])*Omega_Mask, coil, Omega_Mask)
            recon_0 = network(zero_filled, coil, Omega_Mask)
            Ex1  = fft_loss(recon_0, coil, config["axis"]) * Omega_Mask
            jitt = noise_jiterring(us_ksp, Ex1, Omega_Mask, std_scale=config["Mitigation"]["noise_jittering_std"]).to(device) 

        # Attack generation
        zf_p     = attack_generation(zero_filled, network, device, Omega_Mask, coil, config) 
        us_ksp_p = cg_EEH(fft_loss(zf_p, coil, config["axis"])*Omega_Mask, coil, Omega_Mask)

        with torch.no_grad():
            recon_p = network(zf_p, coil, Omega_Mask)
                
        label  [torch.abs(label)<config["Saving"]["Threshold"]] = 0  
        recon_p[torch.abs(label)<config["Saving"]["Threshold"]] = 0

        ssimm      = getssim(np.abs(recon_p.squeeze().cpu().detach().numpy().transpose([1,0])), np.abs(label.squeeze().cpu().detach().numpy().transpose([1,0])))
        psnrr      = getpsnr(np.abs(recon_p.squeeze().cpu().detach().numpy().transpose([1,0])), np.abs(label.squeeze().cpu().detach().numpy().transpose([1,0])))
        both       = recon_p.squeeze().permute(1,0).cpu().detach()
        both_error = (recon_p- label).squeeze().permute(1,0).cpu().detach()
        Imshow_Custom(both,       f"{Saving_Path}",ssimm, psnrr, name=f"Attacked", window_scale = config["Saving"]["Window_scale"])
        Imshow_Custom(both_error, f"{Saving_Path}",ssimm, psnrr, name=f"Attacked_Error", window_scale = config["Saving"]["Window_scale"])
        

        # Proposed cyclic mitigation
        PSNR, SSIM, LOSS  = Cyclic_Mitigation(config, network, zf_p, label, coil, Omega_Mask, mask_list, device, jitt)
        

        files  = sorted([f for f in os.listdir(f"./{Saving_Path}/iterations/") if f.endswith(".png")], key=get_iter_num)
        frames = [Image.open(os.path.join(f"./{Saving_Path}/iterations/", f)).convert("P") for f in files]
        frames[0].save(f"./{Saving_Path}/mitigation.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)

        # Full iteration index
        all_iters    = np.arange(1, len(PSNR) + 1)

        # Fixed axis limits from final/full curves
        x_min, x_max = 1, len(PSNR)
        loss_margin  = 0.05 * (max(LOSS) - min(LOSS) + 1e-8)
        psnr_margin  = 0.05 * (max(PSNR) - min(PSNR) + 1e-8)
        loss_ylim    = (min(LOSS) - loss_margin, max(LOSS) + loss_margin)
        psnr_ylim    = (min(PSNR) - psnr_margin, max(PSNR) + psnr_margin)

        for iii in range(1, len(PSNR) + 1):

            iters      = np.arange(1, iii + 1)
            fig, ax1   = plt.subplots(figsize=(7, 5))
            color_loss = "tab:red"
            color_psnr = "tab:blue"

            # -------------------------
            # Cyclic loss: left y-axis
            # -------------------------
            ax1.plot(iters, LOSS[:iii], color=color_loss, linewidth=2.5, label="Cyclic Loss")
            ax1.scatter(iii, LOSS[iii - 1], color=color_loss, s=70, marker="o", zorder=5)

            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Cyclic Loss", color=color_loss)
            ax1.tick_params(axis="y", labelcolor=color_loss)
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(loss_ylim)
            ax1.grid(True, alpha=0.25)

            # -------------------------
            # PSNR: right y-axis
            # -------------------------
            ax2 = ax1.twinx()
            ax2.plot(iters, PSNR[:iii], color=color_psnr, linewidth=2.5, label="PSNR")
            ax2.scatter(iii, PSNR[iii - 1], color=color_psnr, s=70, marker="o", zorder=5)
            ax2.set_ylabel("PSNR [dB]", color=color_psnr)
            ax2.tick_params(axis="y", labelcolor=color_psnr)
            ax2.set_ylim(psnr_ylim)

            plt.title(f"Cyclic Loss and PSNR Over Iterations | Iteration {iii}")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(f"{Saving_Path}plots", f"iter_{iii:03d}.png"), dpi=300, bbox_inches="tight")
            plt.close()


        files  = sorted([f for f in os.listdir(f"{Saving_Path}plots") if re.match(r"iter_\d+\.png$", f)], key=get_iter_num)
        frames = [Image.open(os.path.join(f"{Saving_Path}plots", f)).convert("P") for f in files]
        frames[0].save(os.path.join(Saving_Path, "psnr_ssim.gif"), save_all=True, append_images=frames[1:], duration=150, loop=0)

        mit_gif         = os.path.join(Saving_Path, "mitigation.gif")
        plot_gif        = os.path.join(Saving_Path, "psnr_ssim.gif")
        out_gif         = os.path.join(Saving_Path, "combined.gif")

        mit_frames      = read_gif(mit_gif)
        plot_frames     = read_gif(plot_gif)
        target_height   = mit_frames[0].height   
        plot_frames     = [resize_to_height(f, target_height) for f in plot_frames]
        num_frames      = max(len(mit_frames), len(plot_frames))
        pad             = 20
        combined_frames = []

        for i in range(num_frames):
            left  = plot_frames[i % len(plot_frames)]
            right = mit_frames[i % len(mit_frames)]
            canvas = Image.new("RGB", (left.width + right.width + pad, target_height), "white")
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width + pad, 0))
            combined_frames.append(canvas)

        combined_frames[0].save(out_gif, save_all=True, append_images=combined_frames[1:], duration=150, loop=0)

        # Delete folders after GIF generation
        if os.path.exists(f"{Saving_Path}iterations"):
            shutil.rmtree(f"{Saving_Path}iterations")

        if os.path.exists(f"{Saving_Path}plots"):
            shutil.rmtree(f"{Saving_Path}plots")

