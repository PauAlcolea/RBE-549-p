import torch
import torch.nn as nn


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        self.coarse_net = NeRFNetwork(embed_pos_L, embed_direction_L)
        self.fine_net = NeRFNetwork(embed_pos_L, embed_direction_L)
        self.mse_loss = nn.MSELoss()

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        return y

    def forward(self, pos, direction):
        """
        forward for both coarse and fine networks
        """
        return output

    def compute_loss(
        self,
        pred_rgb_coarse: torch.Tensor,
        pred_rgb_fine: torch.Tensor,
        gt_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        total squared error between the rendered and true pixel colors for both the coarse and fine renderings
        """
        loss_coarse = self.mse_loss(pred_rgb_coarse, gt_rgb)
        loss_fine = self.mse_loss(pred_rgb_fine, gt_rgb)
        return loss_coarse + loss_fine


class NeRFNetwork(nn.Module):
    """
    base NeRF network architecture for coarse and fine models
    """

    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFNetwork, self).__init__()
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        self.layers = None  # TODO: define MLP layers

    def forward(self, pos, direction):

        return output
