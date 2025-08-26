import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import numpy as np
from skimage.io import imsave

from utils import load_dicom_image, custom_downsample, get_image_patches, save_16bit_dicom_image
from diffusion_model import SimpleUnet, linear_beta_schedule
from loss import VGGPerceptualLoss

# グローバル変数を格納するためのコンテナ
class GlobalSchedule:
    pass
schedule = GlobalSchedule()

def get_noisy_image(x_start, t, device):
    """画像にノイズを加える関数"""
    noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

@torch.no_grad()
def p_sample(model, x, t, low_res_condition, device):
    """1ステップのノイズ除去を行う関数"""
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
    betas_t = schedule.betas.to(device)[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
    sqrt_recip_alphas_t = schedule.sqrt_recip_alphas.to(device)[t].reshape(-1, 1, 1, 1)
    predicted_noise = model(x, t_tensor, low_res_condition)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    if t == 0:
        return model_mean
    else:
        posterior_variance_t = schedule.posterior_variance.to(device)[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def run_training(args, device):
    """学習モードで実行される関数"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_conditions = (
        f"us{args.upscale_factor}_ts{args.training_steps}_ps{args.patch_size}_"
        f"bs{args.batch_size}_lr{args.learning_rate}_dr{args.dropout_rate}_"
        f"t{args.timesteps}_interp-{args.interpolation_mode}_"
        f"l1_{args.lambda_l1}_perc_{args.lambda_perceptual}"
    )
    output_dir = f"{args.output_dir_base}/{timestamp}_{run_conditions}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model will be saved to: {output_dir}")

    i_lr, original_range, original_dicom = load_dicom_image(args.input_image_path, device)
    i_llr = custom_downsample(i_lr, args.upscale_factor)
    
    model = SimpleUnet(dropout_rate=args.dropout_rate).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == 'cuda'))

    print("Starting training with UNIFIED image reconstruction loss...")
    pbar = tqdm(range(args.training_steps))
    for step in pbar:
        optimizer.zero_grad(set_to_none=True) # set_to_none=True for performance
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
            lr_patches = get_image_patches(i_lr, args.patch_size, args.batch_size)
            interp_kwargs = {'mode': args.interpolation_mode}
            if args.interpolation_mode == 'bilinear': interp_kwargs['align_corners'] = False
            llr_condition = F.interpolate(i_llr, size=(args.patch_size, args.patch_size), **interp_kwargs)
            llr_condition = llr_condition.repeat(args.batch_size, 1, 1, 1)
            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long()
            noisy_patches, noise = get_noisy_image(lr_patches, t, device)
            
            predicted_noise = model(noisy_patches, t, llr_condition)
            
            sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
            restored_patches = ((noisy_patches - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t).clamp(-1, 1)
            
            loss_l1 = l1_loss_fn(restored_patches, lr_patches)
            loss_perceptual = perceptual_loss_fn(restored_patches, lr_patches)
            total_loss = args.lambda_l1 * loss_l1 + args.lambda_perceptual * loss_perceptual
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix(loss=total_loss.item(), l1=loss_l1.item(), perc=loss_perceptual.item())

    print("\nTraining finished.")
    model_save_path = f"{output_dir}/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, output_dir, i_lr, original_range, original_dicom

@torch.no_grad()
def run_inference(args, device, model=None, output_dir=None, i_lr=None, original_range=None, original_dicom=None):
    if model is None:
        print(f"Loading model from {args.model_path}...")
        model = SimpleUnet(dropout_rate=args.dropout_rate).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    
    if i_lr is None:
        print(f"Loading image from {args.input_image_path}...")
        i_lr, original_range, original_dicom = load_dicom_image(args.input_image_path, device)

    if output_dir is None: output_dir = os.path.dirname(args.model_path)
    
    target_shape = (1, 1, i_lr.shape[2] * args.upscale_factor, i_lr.shape[3] * args.upscale_factor)
    
    h_sr, w_sr = target_shape[2], target_shape[3]
    predictions_sum = np.zeros((h_sr, w_sr), dtype=np.float32)
    predictions_sq_sum = np.zeros((h_sr, w_sr), dtype=np.float32)
    weight_map_np = np.zeros((h_sr, w_sr), dtype=np.float32)
    window = torch.hann_window(args.inf_patch_size, device=device).unsqueeze(1) * torch.hann_window(args.inf_patch_size, device=device).unsqueeze(0)
    window_np = window.cpu().numpy()

    model.train()
    y_coords = range(0, h_sr, args.inf_patch_size - args.inf_overlap)
    x_coords = range(0, w_sr, args.inf_patch_size - args.inf_overlap)
    pbar_patch = tqdm(total=len(y_coords) * len(x_coords), desc="Processing Patches")

    for y in y_coords:
        for x in x_coords:
            y_start, x_start = y, x
            y_end, x_end = min(y + args.inf_patch_size, h_sr), min(x + args.inf_patch_size, w_sr)
            eff_h, eff_w = y_end - y_start, x_end - x_start
            lr_y_start = y_start // args.upscale_factor
            lr_x_start = x_start // args.upscale_factor
            lr_y_end = (y_end + args.upscale_factor - 1) // args.upscale_factor
            lr_x_end = (x_end + args.upscale_factor - 1) // args.upscale_factor
            patch_cond = i_lr[:, :, lr_y_start:lr_y_end, lr_x_start:lr_x_end]
            
            patch_samples = []
            for _ in range(args.n_samples):
                patch_shape = (1, 1, eff_h, eff_w)
                patch_t = torch.randn(patch_shape, device=device)
                
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_amp):
                    patch_cond_up = F.interpolate(patch_cond, size=patch_t.shape[2:], mode='bilinear', align_corners=False)
                    for t in reversed(range(args.timesteps)):
                        patch_t = p_sample(model, patch_t, t, patch_cond_up, device)
                
                patch_samples.append((patch_t.float().cpu().squeeze().numpy() + 1.0) / 2.0)
            
            samples_array = np.stack(patch_samples, axis=0)
            mean_patch, sq_mean_patch = samples_array.mean(axis=0), (samples_array**2).mean(axis=0)
            current_window_np = window_np[:eff_h, :eff_w]
            predictions_sum[y_start:y_end, x_start:x_end] += mean_patch * current_window_np
            predictions_sq_sum[y_start:y_end, x_start:x_end] += sq_mean_patch * current_window_np
            weight_map_np[y_start:y_end, x_start:x_end] += current_window_np
            pbar_patch.update(1)

    pbar_patch.close()
    model.eval()
    
    weight_map_np[weight_map_np == 0] = 1.0
    mean_image_norm = predictions_sum / weight_map_np
    mean_sq_image = predictions_sq_sum / weight_map_np
    uncertainty_map_np = np.sqrt(np.maximum(mean_sq_image - mean_image_norm**2, 0))
    
    print(f"\nSaving final results to {output_dir}...")
    min_val, max_val = original_range
    mean_image_denorm = mean_image_norm * (max_val - min_val) + min_val
    mean_image_uint16 = np.clip(mean_image_denorm, 0, 65535).astype(np.uint16)
    
    save_16bit_dicom_image(mean_image_uint16, original_dicom, f"{output_dir}/inferred_mean_{args.n_samples}samples.dcm", scale_factor=args.upscale_factor)
    
    uncertainty_map_normalized = (uncertainty_map_np - uncertainty_map_np.min()) / (uncertainty_map_np.max() + 1e-9)
    plt.imsave(f"{output_dir}/inferred_uncertainty_{args.n_samples}samples.png", uncertainty_map_normalized, cmap='inferno')
    print("Inference finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified script for ZSSD-SR with advanced losses.")
    parser.add_argument("--mode", type=str, default="full", choices=['train', 'inference', 'full'])
    parser.add_argument("--input_image_path", type=str, default="Input/Images/MMG.dcm")
    parser.add_argument("--upscale_factor", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--output_dir_base", type=str, default="Results")
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=96, help="Patch size for TRAINING.")
    parser.add_argument("--interpolation_mode", type=str, default='bilinear', choices=['bilinear', 'nearest-exact'])
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)
    parser.add_argument("--model_path", type=str, help="Path to the trained model for inference mode.")
    parser.add_argument("--inf_patch_size", type=int, default=256)
    parser.add_argument("--inf_overlap", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--use_amp", action='store_true', help="Use Automatic Mixed Precision for faster training and inference.")
    
    args = parser.parse_args()

    schedule.betas = linear_beta_schedule(timesteps=args.timesteps)
    schedule.alphas = 1. - schedule.betas
    schedule.alphas_cumprod = torch.cumprod(schedule.alphas, axis=0)
    schedule.alphas_cumprod_prev = F.pad(schedule.alphas_cumprod[:-1], (1, 0), value=1.0)
    schedule.sqrt_recip_alphas = torch.sqrt(1.0 / schedule.alphas)
    schedule.sqrt_alphas_cumprod = torch.sqrt(schedule.alphas_cumprod)
    schedule.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - schedule.alphas_cumprod)
    schedule.posterior_variance = schedule.betas * (1. - schedule.alphas_cumprod_prev) / (1. - schedule.alphas_cumprod)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if args.mode == 'train':
        run_training(args, device)
    elif args.mode == 'inference':
        if not args.model_path:
            parser.error("--model_path is required for inference mode.")
        run_inference(args, device)
    elif args.mode == 'full':
        model, output_dir, i_lr, original_range, original_dicom = run_training(args, device)
        args.model_path = os.path.join(output_dir, "model.pth")
        run_inference(args, device, model=model, output_dir=output_dir, i_lr=i_lr,
                      original_range=original_range, original_dicom=original_dicom)
