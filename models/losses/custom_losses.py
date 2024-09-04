# losses/custom_losses.py

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

class CustomLosses:
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def custom_chamfer_loss(self, pred_points, target_points, weights=None):
        if weights is None:
            weights = torch.ones_like(pred_points[:, :, 0])
 
        dist1, dist2 = chamfer_distance(pred_points, target_points)
 
        weighted_dist1 = torch.mean(weights * dist1, dim=1)
        weighted_dist2 = torch.mean(weights * dist2, dim=1)
 
        return torch.mean(weighted_dist1 + weighted_dist2)

    def hybrid_loss(self, pred_points, target_points, normals_pred, normals_target, alpha=0.1):
        chamfer = self.custom_chamfer_loss(pred_points, target_points)
        normal_loss = self.mse_loss(normals_pred, normals_target)
        return chamfer + alpha * normal_loss
