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
        pos_samples_c, z_vals_c = self.sample_coarse(
            pos, direction, self.t_near, self.t_far, self.Nc
        )

        # coarse network forward, encoding occurs within the network
        color_c, density_c = self.coarse_net(pos_samples_c, direction)

        # volume rendering from output of the coarse network and sample weights
        C_c, weights = self.volume_rendering(color_c, density_c, z_vals_c, direction)

        # from the weights sample points to "investigate" further in the fine network
        pos_samples_f, z_vals_f = self.importance(
            z_vals_c, weights, pos, direction, self.Nf
        )

        # fine network forward encoding also occurs inside of the network
        # fine network should look at the combination of the coarse and the fine points
        z_vals_combined, _ = torch.sort(torch.cat([z_vals_c, z_vals_f], dim=-1), dim=-1)
        pos_samples_combined = (
            pos[..., None, :] + direction[..., None, :] * z_vals_combined[..., :, None]
        )
        color_f, density_f = self.fine_net(pos_samples_combined, direction)

        # volume rendering for fine network as final
        C_f, _ = self.volume_rendering(color_f, density_f, z_vals_combined, direction)

        # return the outputs from the coarse and the fine alpha-composited colors
        # these are for computing the loss against ground truth
        return C_c, C_f

    def sample_coarse(self, pos, dir, t_near, t_far, Nc):
        """
        based on some set values, sample each ray uniformly for the coarse forward pass
        stratified sampling (Section 4):
            partition t_near and t_far into Nc evenly spaced bins
            take one samply uniformly at random from within each bin
            then take those distances and get the 3d positions based on the origin of the ray and the ray-direction
        """
        # we want Nc number of 3d points along each ray (ray denoted by a position and a direction)
        B = pos.shape[0]
        z_vals = torch.linspace(t_near, t_far, Nc, device=pos.device)
        z_vals = z_vals.expand(B, Nc)

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

    def volume_rendering(self, color, density, z_vals, direction):
        """
        relate the densty and the color to a rendered image that can then be checked with the ground truth to calculate the real loss
        the color and the density are acquired from the network
        must be differentiable
        This is where the coarse and fine outputs are converted into colors that can be compared to ground truths
        """
        # distance between adjacent samples
        deltas = z_vals[..., 1:] - z_vals[..., :-1]

        # last delta is infinity
        delta_inf = torch.full_like(deltas[..., :1], 1e10)
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        # account for ray length
        deltas = deltas * torch.norm(direction[..., None, :], dim=-1)

        sigma = density.squeeze(-1)

        alpha = 1.0 - torch.exp(-sigma * deltas)

        # transmittance
        T = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[..., :-1]

        weights = T * alpha

        rgb = torch.sum(weights[..., None] * color, dim=-2)

        return rgb, weights

    def importance(self, z_vals_c, w, pos, direction, Nf=128):
        """
        Section 5.2 of the paper
        this function will take the origina and the directions and the weights and return points for fine sampling
        because the weigths can show what is more likely to be hit, but normalizing and uniformly sampling from that
            the bigger bins will have more "hits", which when changed to depths will mean more points

        :param z_vals_c depths of the coarse sampels along each ray ---> Shape = [B, Nf]
        :param w        the weights from volume rendering, the probability that ray hits something at sample i
        :param Nf       the number of locations for the fine network to evaluate, 128 as per the paper returns.
        """
        w = w[..., 1:-1]
        z_vals_mid = 0.5 * (z_vals_c[..., 1:] + z_vals_c[..., :-1])

        # B numbers of rays in the batch
        # Nc number of coarse samples per ray
        B, Nc = w.shape

        # normalizing the weights produces a piecewise-constant PDF along the ray
        # 1e-5 to avoid the division by zero so that bins correspond to cdf intervals
        pdf = w + 1e-5
        pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)

        # cumulative sum of the pdfs
        # add zero to the cdf, so now cdf = [0 ...]
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros(B, 1, device=w.device), cdf], dim=-1)

        # random number between 0 and 1 to add some variation and improve robustness
        u = torch.rand(B, Nf, device=w.device)

        # find out which bin each sample belongs to based on the value of u, (the cdf is normalized between 0 and 1)
        inds = torch.searchsorted(cdf, u, right=True)
        inds_below = torch.clamp(inds - 1, 0, Nc - 1)
        inds_above = torch.clamp(inds, 0, Nc - 1)

        # once you know what bin, then get the corresponding depth of the value in terms of z-value with linear interpolation
        # that it can then be be considered for the sampling of positions
        cdf_below = torch.gather(cdf, 1, inds_below)
        cdf_above = torch.gather(cdf, 1, inds_above)
        z_below = torch.gather(z_vals_mid, 1, inds_below)
        z_above = torch.gather(z_vals_mid, 1, inds_above)
        t = (u - cdf_below) / (cdf_above - cdf_below + 1e-5)
        z_vals_f = z_below + t * (z_above - z_below)

        pos_samples_f = (
            pos[..., None, :] + direction[..., None, :] * z_vals_f[..., :, None]
        )

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

        # divide stage 1 into two for skip connection as per paper (fig 7)
        self.stage1_first = nn.Sequential(
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
        )
        # skip concatenation
        # in the end, outputs the density and  256 feature vector, like the paper
        self.stage1_second = nn.Sequential(
            nn.Linear(256 + self.pos_length, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 + self.density_length),
        )

        # stage 2 is 1 fully connected layer (Relu and 128 channels)
        # it takes the dimensional feature vector and the embedded direction and outputs the color
        self.stage2 = nn.Sequential(
            nn.Linear(self.dir_length + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # reduce into 3 for the RGB
        )

    def forward(self, pos, direction):
        # reshape for linear layer inputting
        # expand direction to match pos_flat
        B, N, _ = pos.shape
        pos_flat = pos.reshape(B * N, 3)
        dir_expanded = direction.unsqueeze(1).expand(B, N, 3).reshape(B * N, 3)

        # encode the position and the direction to help with high frequency variation
        pos_encoded = self.position_encoding(pos_flat, self.embed_pos_L)
        dir_encoded = self.position_encoding(dir_expanded, self.embed_direction_L)

        # the output of the first stage is combined
        # only apply the relu to the density, as it is the only one that needs that restriction on values
        h = self.stage1_first(pos_encoded)
        density_feature_raw = self.stage1_second(torch.cat([h, pos_encoded], dim=-1))
        density = F.relu(density_feature_raw[..., :1])
        feature = density_feature_raw[..., 1:]

        # don't i need to concatenate with a feature vector? doing with density right now until i figure it out think about that
        # sigmoid activation in the end, like Fig 7 of the paper
        combined = torch.cat([dir_encoded, feature], dim=1)
        color = torch.sigmoid(self.stage2(combined))

        # reshape back to [B, N, C]
        color = color.reshape(B, N, 3)
        density = density.reshape(B, N, 1)

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
                y.append(fn((2.0**i) * math.pi * x))
        return torch.cat(y, dim=-1)


# need to figure out what that feature vector is
# need to figure out when the volume rendering comes into play

# is the positional encoding not done before this file?
#   no, i don't think that this is the case, regardless, where is the data being inputted? is that done internally?
