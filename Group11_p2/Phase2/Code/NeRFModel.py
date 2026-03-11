import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        
        # separate the two network instances so that they can be refined and they can learn separately
        self.coarse_net = NeRFNetwork(embed_pos_L, embed_direction_L)
        self.fine_net = NeRFNetwork(embed_pos_L, embed_direction_L)
    
        # points sampled for the coarse and the fine network
        self.Nc = 64
        self.Nf = 128
        self.t_near, self.t_far = 2.0, 6.0

        self.mse_loss = nn.MSELoss()

    def forward(self, pos, direction):
        """
        forward for both coarse and fine networks
        pos: ray origins
        direction: ray directions
        """
        # go accross a ray and sample some points for the coarse pass
        pos_samples_c, z_vals_c = self.sample_coarse(pos, direction, self.t_near, self.t_far, self.Nc)

        # coarse network forward, encoding occurs within the network
        color_c, density_c = self.coarse_net(pos_samples_c, direction)
        
        # volume rendering from output of the coarse network and sample weights
        C_c, weights = self.volume_rendering(color_c, density_c, z_vals_c, direction)

        # from the weights sample points to "investigate" further in the fine network
        pos_samples_f, z_vals_f = self.importance(z_vals_c, weights, self.Nf)

        # fine network forward encoding also occurs inside of the network
        color_f, density_f = self.fine_net(pos_samples_f, direction)

        # volume rendering for fine network as final
        C_f, _ = self.volume_rendering(color_f, density_f, z_vals_f, direction)

        # return the outputs from the coarse and the fine alpha-composited colors
        # these are for computing the loss against ground truth
        return C_c, C_f 
    
    def sample_coarse(pos, dir, t_near, t_far, Nc):
        """
        based on some set values, sample each ray uniformly for the coarse forward pass
        """
        # we want Nc number of 3d points along each ray (ray denoted by a position and a direction)
        z_vals = torch.linspace(t_near, t_far, Nc, device=pos.device)

        # Stratified sampling, make bins and sample the middle of them. Then the ranges for each bin
        mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)

        # randonmness to each bin to get varied, random numbers from [0,1]
        t_rand = torch.rand(z_vals.shape, device=pos.device)
        z_vals = lower + (upper - lower) * t_rand

        # finally get the points
        pos_samples_c = pos[..., None, :] + dir[..., None, :] * z_vals[..., :, None]
        return pos_samples_c, z_vals
    
    def volume_rendering(color, density, z_vals, direction):
        """
        relate the densty and the color to a rendered image that can then be checked with the ground truth to calculate the real loss
        the color and the density are acquired from the network
        must be differentiable
        This is where the coarse and fine outputs are converted into colors that can be compared to ground truths
        """
        # distance between adjacent samples
        deltas = z_vals[...,1:] - z_vals[...,:-1]

        # last delta is infinity
        delta_inf = torch.full_like(deltas[...,:1], 1e10)
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        # account for ray length
        deltas = deltas * torch.norm(direction[...,None,:], dim=-1)

        sigma = density.squeeze(-1)

        alpha = 1.0 - torch.exp(-sigma * deltas)

        # transmittance
        T = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[...,:1]),
                1.0 - alpha + 1e-10
            ], dim=-1),
            dim=-1
        )[...,:-1]

        weights = T * alpha

        rgb = torch.sum(weights[...,None] * color, dim=-2)

        return rgb, weights
         
    def compute_weights(density):
        return
    
    def importance(z_vals_c, w, Nf=128):
        """
        this function will take the origina and the directions and the weights and return points for fine sampling
        the weights show make a pdf along the ray
        Nf is the number of locations for the fine network to evaluate, 128 as per the paper
        returns
        """
        B, Nc = weights.shape

        # 1. PDF (normalize weights)
        pdf = weights + 1e-5  # avoid division by zero
        pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)  # [B, Nc]

        # 2. CDF (cumulative sum)
        cdf = torch.cumsum(pdf, dim=-1)  # [B, Nc]
        cdf = torch.cat([torch.zeros(B, 1, device=weights.device), cdf], dim=-1)  # [B, Nc+1]

        # 3. Sample uniform numbers in [0,1]
        u = torch.rand(B, Nf, device=weights.device)

        # 4. Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)  # [B, Nf]
        inds_below = torch.clamp(inds - 1, 0, Nc - 1)
        inds_above = torch.clamp(inds, 0, Nc - 1)

        cdf_below = torch.gather(cdf, 1, inds_below)
        cdf_above = torch.gather(cdf, 1, inds_above)
        z_below = torch.gather(z_vals_c, 1, inds_below)
        z_above = torch.gather(z_vals_c, 1, inds_above)

        t = (u - cdf_below) / (cdf_above - cdf_below + 1e-5)
        z_vals_f = z_below + t * (z_above - z_below)  # linear interpolation

        pos_samples_f = pos[..., None, :] + direction[..., None, :] * z_vals_f[..., :, None]

        return pos_samples_f, z_vals_f
    
    def compute_loss(
        self,
        pred_rgb_coarse: torch.Tensor,
        pred_rgb_fine: torch.Tensor,
        gt_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        total squared error between the rendered and true pixel images for both the coarse and fine renderings
        """
        # TODO this is not correct, it needs to compare the images, not the pixels, since the color needs to be related to the density
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
        self.density_length = 1

        # specify the length of the new encoded position and direction set here based on 3 (the dimension of pos and dir vector)
        # 2 is based on the equation
        self.pos_length = 3 + 3 * 2 * embed_pos_L
        self.dir_length = 3 + 3 * 2 * embed_direction_L
        
        # first stage should be 8 fully connected layers, ReLu Activated and 256 channels
        # they only take the position and output the density and a 256 dimensional feature vector
        self.stage1 = nn.Sequential(
            nn.Linear(self.pos_length, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 + self.density_length),   #in the end, outputs the density and  256 feature vector, like the paper
        )

        # stage 2 is 1 fully connected layer (Relu and 128 channels)
        # it takes the dimensional feature vector and the embedded direction and outputs the color
        self.stage2 = nn.Sequential(
            nn.Linear(self.dir_length + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # reduce into 3 for the RGB
        )

    def forward(self, pos, direction):
        #encode the position and the direction to help with high frequency variation
        pos_encoded = self.position_encoding(pos, self.embed_pos_L)
        dir_encoded = self.position_encoding(direction, self.embed_direction_L)

        #the output of the first stage is combined
        #only apply the relu to the density, as it is the only one that needs that restriction on values
        density_feature_raw = self.stage1(pos_encoded)
        density = F.relu(density_feature_raw[:, 0:1])
        feature = density_feature_raw[:, 1:]

        # don't i need to concatenate with a feature vector? doing with density right now until i figure it out think about that
        combined = torch.cat([dir_encoded, feature], dim=1)
        color = F.relu(self.stage2(combined))

        # output should be [B, N, (R, G, B, density)]
        return (color, density)
    
    def position_encoding(self, x, L):
        """
        positional encoding allows the MLP to represent higher frequency functions
        mapping to higher dimensinal space with high freqency functions enables better fitting for high freq variation
        """
        y = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                y.append(fn((2.0 ** i) * math.pi * x))
        return torch.cat(y, dim=-1)


# need to figure out what that feature vector is 
# need to figure out when the volume rendering comes into play

# is the positional encoding not done before this file?
#   no, i don't think that this is the case, regardless, where is the data being inputted? is that done internally?