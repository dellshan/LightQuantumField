import torch
import torch.nn.functional as F
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure
from pytorch3d.loss import chamfer_distance

class EvaluationMetrics:
    def __init__(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips_metric = lpips.LPIPS(net='alex')  # Can also use 'vgg' or 'squeeze'

    def compute_psnr(self, pred_img, target_img):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
        mse = F.mse_loss(pred_img, target_img)
        psnr = -10 * torch.log10(mse)
        return psnr

    def compute_ssim(self, pred_img, target_img):
        """Calculate Structural Similarity Index (SSIM)."""
        return self.ssim(pred_img, target_img)

    def compute_lpips(self, pred_img, target_img):
        """Calculate Learned Perceptual Image Patch Similarity (LPIPS)."""
        return self.lpips_metric(pred_img, target_img)

    def compute_chamfer_distance(self, pred_points, gt_points):
        """Calculate Chamfer Distance for point clouds."""
        loss_chamfer, _ = chamfer_distance(pred_points, gt_points)
        return loss_chamfer

    def compute_normal_consistency(self, pred_normals, gt_normals):
        """Calculate Normal Consistency between predicted and ground-truth normals."""
        dot_products = (pred_normals * gt_normals).sum(dim=-1)
        normal_consistency = dot_products.mean()
        return normal_consistency

    def compute_fps(self, render_function):
        """Calculate Frames Per Second (FPS) during rendering."""
        import time
        start_time = time.time()
        render_function()
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        return fps
