import os
import csv
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from train import instantiate_from_config
from utils.metrics import Peak_Signal_to_Noise_Rate_2D, Structural_Similarity_slice, to8b
from x2ct_nerf.modules.losses.lpips import LPIPS


def compute_dice_per_class(pred, gt, num_classes, smooth=1.0):
    dice_list = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (gt == c).float()
        inter = (p * t).sum()
        card = p.sum() + t.sum()
        dice_list.append(((2.0 * inter + smooth) / (card + smooth)).item())
    return dice_list


def compute_iou_per_class(pred, gt, num_classes, smooth=1.0):
    iou_list = []
    for c in range(num_classes):
        p = (pred == c)
        t = (gt == c)
        inter = (p & t).float().sum()
        union = (p | t).float().sum()
        iou_list.append(((inter + smooth) / (union + smooth)).item())
    return iou_list


PALETTE = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255],
], dtype=np.uint8)


def colorize_mask_np(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]

@torch.no_grad()
def load_model(config, ckpt_path, device):
    model = instantiate_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema_state_dict" in ckpt:
        print("Using EMA weights for evaluation.")
        sd = ckpt["ema_state_dict"]
        full_sd = ckpt["state_dict"]
        full_sd.update(sd)
        sd = full_sd
    else:
        print("No EMA weights found, using regular state_dict.")
        sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model.eval().to(device)

def fuse_recon(recon, gt):
    """PSNR/SSIM/LPIPS용: axial/coronal/sagittal 재구성을 같은 좌표계로 정렬 후 평균 (연속값이라 평균 가능)."""
    axial    = recon["axial"]
    coronal  = recon["coronal"].permute(1, 0, 2)
    sagittal = recon["sagittal"].permute(1, 2, 0)
    fused = (axial + coronal + sagittal) / 3.0

    gt_vol = gt["axial"]
    return fused, gt_vol


def fuse_seg(seg_recon, seg_gt):
    """Dice/IoU용: axial/coronal/sagittal 재구성을 같은 좌표계로 정렬 후 voxel별 다수결 (정수 라벨이라 평균 대신 투표)."""
    axial    = seg_recon["axial"]
    coronal  = seg_recon["coronal"].permute(1, 0, 2)
    sagittal = seg_recon["sagittal"].permute(1, 2, 0)

    stacked = torch.stack([axial, coronal, sagittal], dim=0)
    fused_pred = torch.mode(stacked, dim=0).values

    gt_vol = seg_gt["axial"]
    return fused_pred, gt_vol


@torch.no_grad()
def calc_metrics_per_patient(recon, gt, seg_recon, seg_gt, num_seg_classes, ct_min_max, perceptual_loss_fn):
    """axial/coronal/sagittal 세 재구성을 정렬 후 융합(fusion)한 단일 3D 볼륨으로 metric 계산."""
    fused_recon, fused_gt = fuse_recon(recon, gt)

    psnr = Peak_Signal_to_Noise_Rate_2D(
        fused_gt.unsqueeze(0), fused_recon.unsqueeze(0),
        PIXEL_MIN=ct_min_max[0], PIXEL_MAX=ct_min_max[1], use_real_max=True
    )
    ssim = Structural_Similarity_slice(
        fused_gt.unsqueeze(1).cpu().numpy(), fused_recon.unsqueeze(1).cpu().numpy(), PIXEL_MAX=1.0
    )

    target_3ch = fused_gt.unsqueeze(1).repeat(1, 3, 1, 1)
    output_3ch = fused_recon.unsqueeze(1).repeat(1, 3, 1, 1)
    lpips_val = perceptual_loss_fn(target_3ch.contiguous(), output_3ch.contiguous()).mean()

    patient_psnr = float(psnr.mean())
    patient_ssim = float(ssim)
    patient_lpips = float(lpips_val)

    patient_dice_fg, patient_iou_fg = None, None
    if "axial" in seg_recon and "coronal" in seg_recon and "sagittal" in seg_recon:
        fused_pred, fused_seg_gt = fuse_seg(seg_recon, seg_gt)
        dice_arr = np.array(compute_dice_per_class(fused_pred, fused_seg_gt, num_seg_classes))
        iou_arr = np.array(compute_iou_per_class(fused_pred, fused_seg_gt, num_seg_classes))
        patient_dice_fg = float(dice_arr[1:].mean())
        patient_iou_fg = float(iou_arr[1:].mean())

    return patient_psnr, patient_ssim, patient_lpips, patient_dice_fg, patient_iou_fg


@torch.no_grad()
def run_test(model, dataloader, device, ct_min_max, num_seg_classes, save_dir):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    perceptual_loss_fn = LPIPS().eval().to(device)

    all_psnr, all_ssim, all_lpips = [], [], []
    all_dice_fg, all_iou_fg = [], []
    all_patient_names = []

    recon, gt, seg_recon, seg_gt = {}, {}, {}, {}
    prev_patient = None

    def flush_patient():
        nonlocal recon, gt, seg_recon, seg_gt, prev_patient
        p_psnr, p_ssim, p_lpips, p_dice_fg, p_iou_fg = calc_metrics_per_patient(
            recon, gt, seg_recon, seg_gt, num_seg_classes, ct_min_max, perceptual_loss_fn
        )
        all_psnr.append(p_psnr)
        all_ssim.append(p_ssim)
        all_lpips.append(p_lpips)
        if p_dice_fg is not None:
            all_dice_fg.append(p_dice_fg)
            all_iou_fg.append(p_iou_fg)
        all_patient_names.append(prev_patient)

        patient_dir = f"{save_dir}/{prev_patient}"
        Path(patient_dir).mkdir(exist_ok=True, parents=True)
        for axis_name in recon:
            output_np = recon[axis_name].cpu().numpy()
            target_np = gt[axis_name].cpu().numpy()
            for slice_idx in range(output_np.shape[0]):
                slice_dir = f"{patient_dir}/{axis_name}_{slice_idx:03d}"
                Path(slice_dir).mkdir(exist_ok=True, parents=True)
                imageio.imwrite(f"{slice_dir}/recon.png", to8b(output_np[slice_idx]))
                imageio.imwrite(f"{slice_dir}/input.png", to8b(target_np[slice_idx]))
                if axis_name in seg_recon:
                    imageio.imwrite(f"{slice_dir}/seg_pred.png",
                                     colorize_mask_np(seg_recon[axis_name][slice_idx].cpu().numpy()))
                    imageio.imwrite(f"{slice_dir}/seg_gt.png",
                                     colorize_mask_np(seg_gt[axis_name][slice_idx].cpu().numpy()))

    for batch in tqdm(dataloader):
        file_path = batch["file_path_"][0].split("/")
        patient = file_path[-3]
        curr_axis = file_path[-1].split("_")[0]

        log = model.log_images(batch, return_raw_seg=True)
        output_img = log["reconstructions"][:, 0].to(device)
        target_img = log["inputs"][:, 0].to(device)
        seg_pred = log.get("seg_pred_raw")
        seg_gt_raw = log.get("seg_gt_raw")

        if prev_patient is None or prev_patient == patient:
            gt[curr_axis], recon[curr_axis] = target_img, output_img
            if seg_pred is not None:
                seg_recon[curr_axis], seg_gt[curr_axis] = seg_pred, seg_gt_raw
            prev_patient = patient
        else:
            flush_patient()
            recon, gt = {curr_axis: output_img}, {curr_axis: target_img}
            seg_recon, seg_gt = {}, {}
            if seg_pred is not None:
                seg_recon[curr_axis], seg_gt[curr_axis] = seg_pred, seg_gt_raw
            prev_patient = patient

    flush_patient()

    print("\n========== Recon Results (3방향 융합 단일 3D 기준, mean ± std) ==========")
    print(f"PSNR: {np.mean(all_psnr):.3f} ± {np.std(all_psnr):.3f}, "
          f"SSIM: {np.mean(all_ssim):.3f} ± {np.std(all_ssim):.3f}, "
          f"LPIPS: {np.mean(all_lpips):.3f} ± {np.std(all_lpips):.3f}")

    if all_dice_fg:
        print(f"\n[Foreground avg Dice] {np.mean(all_dice_fg):.4f} ± {np.std(all_dice_fg):.4f}")
        print(f"[Foreground avg IoU]  {np.mean(all_iou_fg):.4f} ± {np.std(all_iou_fg):.4f}")

    csv_path = f"{save_dir}/metrics_per_patient.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "psnr", "ssim", "lpips", "dice_fg", "iou_fg"])
        for i, name in enumerate(all_patient_names):
            row = [name, f"{all_psnr[i]:.4f}", f"{all_ssim[i]:.4f}", f"{all_lpips[i]:.4f}"]
            row.append(f"{all_dice_fg[i]:.4f}" if i < len(all_dice_fg) else "")
            row.append(f"{all_iou_fg[i]:.4f}" if i < len(all_iou_fg) else "")
            writer.writerow(row)
    print(f"CSV saved: {csv_path}")
    print(f"Images saved under: {save_dir}/<patient>/<axis>_<slice>/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config_path)
    config.data.params.batch_size = config.data.params.validation.params.opt.ct_size

    model = load_model(config, args.ckpt_path, device)

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    ct_min_max = config.data.params.test.params.opt.CT_MIN_MAX
    num_seg_classes = config.model.params.num_seg_classes

    run_test(model, data.test_dataloader(), device, ct_min_max, num_seg_classes, args.save_dir)
    print("\n[Done]")


if __name__ == "__main__":
    main()