import torch
import numpy as np

class SH2RGB:
    def __init__(self, order=4):
        self.order = order
        self.num_coeffs = order ** 2
    
    def eval_sh(self, l, m, theta, phi):
        """
        Evaluate the spherical harmonics function Y_l^m(theta, phi).
        """
        if m == 0:
            return np.sqrt((2*l+1)/(4*np.pi))
        elif m > 0:
            return np.sqrt(2) * np.cos(m * phi) * np.sqrt((2*l+1)/(4*np.pi))
        else:
            return np.sqrt(2) * np.sin(-m * phi) * np.sqrt((2*l+1)/(4*np.pi))
    
    def sh_to_rgb(self, sh_coeffs, directions):
        """
        Convert SH coefficients to RGB using the evaluated directions.
        """
        assert sh_coeffs.shape[1] == self.num_coeffs, "SH coefficients size mismatch"
        # Initialize RGB colors
        rgb = torch.zeros((sh_coeffs.shape[0], 3), device=sh_coeffs.device)
        
        # Iterate over each SH band and accumulate results
        for l in range(self.order):
            for m in range(-l, l+1):
                coeff_index = l**2 + m + l
                sh_val = self.eval_sh(l, m, directions[:, 0], directions[:, 1])
                rgb += sh_coeffs[:, coeff_index].unsqueeze(1) * sh_val.unsqueeze(0)
        
        return torch.clamp(rgb, 0, 1)

    def __call__(self, sh_coeffs, directions):
        """
        Convert SH coefficients to RGB colors given specific directions.
        """
        return self.sh_to_rgb(sh_coeffs, directions)

# Example usage
sh2rgb = SH2RGB(order=4)
directions = torch.tensor([[0, 0], [np.pi/2, np.pi/2], [np.pi, np.pi]])  # Example directions in radians
sh_coeffs = torch.rand((1, 16))  # Example SH coefficients (1 sample, 16 coefficients for order 4)
rgb_colors = sh2rgb(sh_coeffs, directions)
