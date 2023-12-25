from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from typing import *
from torch import Tensor
import utils as loader_utils

# from zuko.utils import odeint as zodeint
from torch.distributions import Normal
from torchdiffeq import odeint
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from absl import logging
from equivariant_diffusion.origin_rotation_icp import icp
import os
import egnn.node_predict as node_predict
from torch import autograd

# TODO: regularization loss and the model generation.


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def inv_cdf(t):  # this is used to reweight the property.
    return 1 - torch.sqrt(1 - t)


def inv_sin(t):
    return 1 - torch.sin(torch.pi * t / 2)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def loss_reduce_mean_except_batch_with_mask(loss, mask):
    """
    Args:
        loss: [b, n, 3]
        mask: [b, n, 1]
    """
    if len(loss.shape) == 3:
        losses = loss.sum(-1)  # [b, n]
    else:
        losses = loss
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
    return (losses * mask).sum(-1) / mask.sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5)


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
            - 0.5
        )
        * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )


def T(t):
    # 0   0, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


def T_hat(t):
    # 0 beta_min, 1 beta_max
    beta_min = 0.1
    beta_max = 20
    return (beta_max - beta_min) * t + beta_min


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


# class PredefinedNoiseSchedule(torch.nn.Module):
#     def __init__(self) -> None:
#        def __init__(self, noise_schedule, precision):
#         super(PredefinedNoiseSchedule, self).__init__()
#         if noise_schedule == 'cosine':
#             alphas2 = cosine_beta_schedule(timesteps)


def VP_path(x, t):
    # t in zeros and ones
    # if noi
    beta_min = 0.1
    beta_max = 20
    # u = 1 - t
    # t = 1 - t # Reverse time, x0 for sample, x1 for noise
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    # log_mean_coeff.to(x.device)
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    # t-->1, log_mean_coeff-->0, mean-->0, std-->1 GAUSSIAN
    # t-->0, log_mean_coeff-->-0.5*beta_min, mean-->1*x, std-->sqrt(1-exp(-beta_min) DATA

    return mean, std


# def SVP_path(x,t):
#     coeff = 1 - torch.sqrt(t)

#     mean = coeff[:,None,None] * x

#     std = torch.sqrt(2*torch.sqrt(t)-t)


#     return mean, std
def VP_field(x0, xt, t):
    M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)  # add epsilon to stable it
    M_para = M_para[:, None, None]
    vector = (
        torch.exp(-T(t))[:, None, None] * xt
        - torch.exp(-0.5 * T(t))[:, None, None] * x0
    )

    return -vector


def polynomial_schedule_(t, s=1e-7, power=2.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    # steps = timesteps + 1
    # x = np.linspace(0, steps, steps)
    alphas2 = (1 - t**power) ** 2
    # alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s  # numerical stability.
    return alphas2


def polynomial_div(t, s=1e-7, power=2.0):
    alpha_t_d = power * t ** (power - 1)  # reverse direction
    precision = 1 - 2 * s
    alpha_t_d = precision * alpha_t_d + s  # numerical stability.

    return alpha_t_d


def poly_path(x, t):
    alpha_squera = polynomial_schedule_(t)
    alpha = torch.sqrt(alpha_squera)
    sigma = torch.sqrt(1 - alpha_squera)
    # x_t = torch.randn_like(x)*sigma[:,None,None] + alpha[:,None,None]*x
    return alpha[:, None, None] * x, sigma


def p_vector_field(x_0, x_t, t):
    # M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t))+1e-5) # add epsilon to stable it
    # M_para = M_para[:,None,None]
    alpha_t = torch.sqrt(polynomial_schedule_(t))
    vector = (polynomial_div(t) * alpha_t)[:, None, None] * x_t - polynomial_div(t)[
        :, None, None
    ] * x_0  # + alpha_t * x_t - x0
    # reparameterize the vector field
    return vector


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class Cnflows(torch.nn.Module):
    """
    The E(n) continous normalizing flows Module.
    """

    def __init__(
        self,
        dynamics: models.EGNN_dynamics_QM9,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 10000,
        parametrization="eps",
        time_embed=False,
        noise_schedule="learned",
        noise_precision=1e-4,
        loss_type="ot",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
        discrete_path="OT_path",
        cat_loss="l2",
        cat_loss_step=-1,
        on_hold_batch=-1,
        sampling_method="vanilla",
        weighted_methods="jump",
        ode_method="dopri5",
        without_cat_loss=False,
        angle_penalty=False,
        extend_feature_dim=0,
        minimize_type_entropy=False,
        node_classifier_model_ckpt=None,
        minimize_entropy_grad_coeff=0.5,
    ):
        super().__init__()

        # assert loss_type in {'ot'}
        self.set_odeint(method=ode_method)
        self.loss_type = loss_type
        self.include_charges = include_charges
        self._eps = 0.0  # TODO: fix the trace computation part
        self.discrete_path = discrete_path
        self.ode_method = ode_method

        self.cat_loss = cat_loss
        self.cat_loss_step = cat_loss_step
        self.on_hold_batch = on_hold_batch
        self.sampling_method = sampling_method
        self.weighted_methods = weighted_methods
        self.without_cat_loss = without_cat_loss
        self.angle_penalty = angle_penalty

        # Only supported parametrization.

        # self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        # if noise_schedule == 'learned':
        #     self.gamma = GammaNetwork()
        # else:
        #     self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
        #                                          precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges - extend_feature_dim
        self.extend_feature_dim = extend_feature_dim

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.time_embed = time_embed
        self.minimize_type_entropy = minimize_type_entropy
        self.node_classifier_model_ckpt = node_classifier_model_ckpt
        self.minimize_entropy_grad_coeff = minimize_entropy_grad_coeff
        self.node_pred_model = None
        self.register_buffer("buffer", torch.zeros(1))
        if extend_feature_dim % 2 != 0:
            raise ValueError("extend_feature_dim must be a multiple of 2")
        if extend_feature_dim > 0:
            self.extend_feature_embedding = SinusoidalPosEmb(dim=extend_feature_dim)

        if time_embed:
            self.register_buffer(
                "frequencies", 2 ** torch.arange(self.frequencies) * torch.pi
            )
        if self.minimize_type_entropy and not os.path.exists(
            self.node_classifier_model_ckpt
        ):
            raise ValueError(
                "node_classifier_model_ckpt must be provided if minimize_type_entropy is True"
            )

        # if noise_schedule != 'learned':
        #     self.check_issues_norm_values()

    def set_odeint(self, method="dopri5", rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def phi(self, t, x, node_mask, edge_mask, context):
        # TODO: check the frequencies buffer. input is embedding to get better performance.
        if self.time_embed:
            t = self.frequencies * t[..., None]
            t = torch.cat((t.cos(), t.sin()), dim=-1)
            t = t.expand(*x.shape[:-1], -1)

        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(
            self.norm_values[0]
        )

        # Casting to float in case h still has long or int type.
        h_cat = (
            (h["categorical"].float() - self.norm_biases[1])
            / self.norm_values[1]
            * node_mask
        )
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):  # Check the unnormalize_z function
        # Parse from z
        x, h_cat = (
            z[:, :, 0 : self.n_dims],
            z[:, :, self.n_dims : self.n_dims + self.num_classes],
        )
        h_int = z[
            :, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1
        ]

        # print("unnormalize_", h_int.size(),x.size(), h_cat.size())
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        if self.extend_feature_dim > 0:
            h_extend = z[:, :, self.n_dims + self.num_classes + 1 :]
            output = torch.cat([x, h_cat, h_int, h_extend], dim=2)
        else:
            output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    # def zero_step_direction(self, xh_0,  node_mask, edge_mask, context):
    #     """Computes the direction of the zero-step flow."""
    #     zeros = torch.zeros(size=(node_mask.size(0), 1), device=node_mask.device)
    #     # gamma_0 = self.gamma(zeros)
    #     net_out = self.phi(zeros, xh_0, node_mask, edge_mask, context)

    #     return

    def sample_p_xh_given_z0(self, dequantizer, z0, node_mask):
        """Samples x ~ p(x|z0)."""

        # print(z0.size(),node_mask.size())
        # if self.cat_loss_step > 0:
        #     #under this case we use the direction of the network output as the categorical sampling results.
        #     predicted_0 = self.phi(0.)

        x = z0[:, :, : self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)

        # if self.include_charges:
        x, h_cat, h_int = self.unnormalize(
            x, z0[:, :, self.n_dims : self.n_dims + self.num_classes], h_int, node_mask
        )
        # else:
        #     x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:], h_int,
        #                                    node_mask)

        tensor = dequantizer.reverse({"categorical": h_cat, "integer": h_int})

        one_hot, charges = tensor["categorical"], tensor["integer"]
        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask
        h = {"integer": charges, "categorical": one_hot}

        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def _rearange_z_optimal_rotation_first_3d(
        self, x: np.ndarray, z: np.ndarray, node_mask: np.ndarray
    ):
        """
        x:  [b, n, 3+5]
        z:  [b, n, 3+5]
        node_mask: [b, n, 1]
        """
        ret_z = deepcopy(z)
        length = node_mask.squeeze().sum(axis=-1).astype(np.int32)  # [b]

        for _idx, l in enumerate(length):
            _, z_rotated, _ = icp(z[_idx, :l, :3], x[_idx, :l, :3])
            ret_z[_idx, :l, :3] = z_rotated
        return ret_z

    def _rearange_z_first3d(self, x: np.ndarray, z: np.ndarray, node_mask: np.ndarray):
        """
        x:  [b, n, 3+5]
        z:  [b, n, 3+5]
        node_mask: [b, n, 1]
        """
        ret_z = deepcopy(z)
        length = node_mask.squeeze().sum(axis=-1).astype(np.int32)  # [b]
        distance_matrices = np.sqrt(
            np.sum(
                (
                    np.expand_dims(x[:, :, :3], axis=2)
                    - np.expand_dims(z[:, :, :3], axis=1)
                )
                ** 2,
                axis=-1,
            )
        )  # [b, n, n]
        for _idx, l in enumerate(length):
            _, col_ind = linear_sum_assignment(
                distance_matrices[_idx, :l, :l], maximize=False
            )
            ret_z[_idx, :l, :] = z[_idx, col_ind, :]
        return ret_z

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        # TODO： add different path track for the categoricak distribution
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        #  def forward(self, x: Tensor) -> Tensor
        # return (self.v(t.squeeze(-1), y) - u).square().mean()

        # Concatenate x, h[integer] and h[categorical].
        b, n, _ = x.size()
        distance2origin = x.square().sum(dim=-1).sqrt()  # [b, n]
        if self.extend_feature_dim > 0:
            extend_feat = self.extend_feature_embedding(
                distance2origin.view(-1)
            )  # [b, n] -> [b, n, dim]
            extend_feat = extend_feat.view(b, n, -1)  # [bxn, dim] -> [b, n, dim]
            xh = torch.cat([x, h["categorical"], h["integer"], extend_feat], dim=2)
        else:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # t = torch.rand_like(xh[..., 0]).unsqueeze(-1)
        # z = torch.randn_like(xh)
        _z = self.sample_combined_position_feature_noise(
            xh.size(0), xh.size(1), node_mask
        )
        _z = self._rearange_z_optimal_rotation_first_3d(
            xh.detach().cpu().numpy(),
            _z.detach().cpu().numpy(),
            node_mask.detach().cpu().numpy(),
        )
        _z = self._rearange_z_first3d(
            xh.detach().cpu().numpy(),
            _z,
            node_mask.detach().cpu().numpy(),
        )
        z = torch.tensor(
            _z,
            dtype=xh.dtype,
            device=xh.device,
        )

        t = torch.rand_like(xh[:, 0, 0]).view(-1, 1, 1)

        if self.on_hold_batch > 0 and self.cat_loss_step > 0:
            t[-self.on_hold_batch :, :, :] = (
                t[-self.on_hold_batch :, :, :] * self.cat_loss_step
            )
        # else:
        #     raise ValueError('on_hold_batch and cat_loss_step should be larger than 0')

        if self.discrete_path == "OT_path":
            # t = torch.rand_like(xh[:,0,0]).view(-1,1,1)
            y = (1 - t) * xh + (1e-4 + (1 - 1e-4) * t) * z
            field_z = (1 - 1e-4) * z - xh

            # Neural net prediction.
            net_out = self.phi(t, y, node_mask, edge_mask, context)
            # Compute the error.
            # loss = sum_except_batch((net_out-u).square())

        elif self.discrete_path == "HB_path":
            # t = torch.rand_like(xh[:,0,0]).view(-1,1,1)
            # t = torch.rand()
            inter_x = (1 - t) * xh[:, :, : self.n_dims] + (1e-4 + (1 - 1e-4) * t) * z[
                :, :, : self.n_dims
            ]
            t_ = t.squeeze()
            inter_h_m, inter_h_std = VP_path(xh[:, :, self.n_dims :], t_)

            inter_h = (
                torch.randn(inter_h_m.size(), device=inter_h_m.device)
                * inter_h_std[:, None, None]
                + inter_h_m
            )

            # print(inter_h.size(),node_mask.size(),inter_h_m.size())
            inter_h = inter_h * node_mask

            inter_z = torch.cat([inter_x, inter_h], dim=2)

            net_out = self.phi(t, inter_z, node_mask, edge_mask, context)

            field_x = (1 - 1e-4) * z[:, :, : self.n_dims] - xh[:, :, : self.n_dims]

            field_h = VP_field(xh[:, :, self.n_dims :], inter_h, t_)

            field_z = torch.cat([field_x, field_h], dim=2)

            # direction
            # loss = sum_except_batch((net_out-field_z).square())

        elif self.discrete_path == "VP_path":
            # TODO: add the VP path on both x and h
            t_ = t.squeeze()
            # torch.rand_like(xh[:,0,0])  # the step issues.
            # *(1 - 1e-5)
            # t_ = t.squeeze()
            inter_z_m, inter_z_std = VP_path(xh, t_)

            inter_z = (
                torch.randn(inter_z_m.size(), device=inter_z_m.device)
                * inter_z_std[:, None, None]
                + inter_z_m
            )

            field_z = VP_field(xh, inter_z, t_)

            net_out = self.phi(
                t_.view(-1, 1, 1), inter_z, node_mask, edge_mask, context
            )

            # loss = sum_except_batch((net_out-field_z).square())

        elif self.discrete_path == "HB_path_poly":
            # A hybrid path with od and poly noise scale

            # *(1 - 1e-5) + 1e-5
            # t = torch.rand()
            inter_x = (1 - t) * xh[:, :, : self.n_dims] + (1e-4 + (1 - 1e-4) * t) * z[
                :, :, : self.n_dims
            ]
            t_ = t.squeeze()

            inter_h_m, inter_h_std = poly_path(xh[:, :, self.n_dims :], t_)
            inter_h = (
                torch.randn(inter_h_m.size(), device=inter_h_m.device)
                * inter_h_std[:, None, None]
                + inter_h_m
            )
            # print(inter_h.size(),node_mask.size(),inter_h_m.size())
            inter_h = inter_h * node_mask
            inter_z = torch.cat([inter_x, inter_h], dim=2)
            net_out = self.phi(t, inter_z, node_mask, edge_mask, context)

            field_x = (1 - 1e-4) * z[:, :, : self.n_dims] - xh[:, :, : self.n_dims]

            field_h = p_vector_field(xh[:, :, self.n_dims :], inter_h, t_)

            field_z = torch.cat([field_x, field_h], dim=2)

            # direction
            # loss = sum_except_batch((net_out-field_z).square())

        elif self.discrete_path == "VP_path_poly":
            t_ = t.squeeze()
            # torch.rand_like(xh[:,0,0])
            # *(1 - 1e-5) + 1e-5
            # t_ = t.squeeze()
            inter_z_m, inter_z_std = poly_path(xh, t_)

            inter_z = (
                torch.randn(inter_z_m.size(), device=inter_z_m.device)
                * inter_z_std[:, None, None]
                + inter_z_m
            )

            field_z = p_vector_field(xh, inter_z, t_)

            net_out = self.phi(
                t_.view(-1, 1, 1), inter_z, node_mask, edge_mask, context
            )

        elif self.discrete_path == "OT_path_compressed":
            t = torch.rand_like(xh[:, 0, 0]).view(-1, 1, 1)

            if self.on_hold_batch > 0 and self.cat_loss_step > 0:
                t[-self.on_hold_batch :, :, :] = (
                    t[-self.on_hold_batch :, :, :] * self.cat_loss_step
                )
            else:
                raise ValueError(
                    "on_hold_batch and cat_loss_step should be larger than 0"
                )
            # t_ = t.squeeze()
            # def get_inter_z(xh,t_):
            y = (1 - t) * xh + (1e-4 + (1 - 1e-4) * t) * z
            field_z = (1 - 1e-4) * z - xh
            t_ = t.squeeze()
            mask = t_ < self.cat_loss_step
            # for masked
            y[~mask][:, self.n_dims : -1] = z[~mask][
                :, self.n_dims : -1
            ]  # The slope is too big.
            y[mask][:, self.n_dims : -1] = (
                xh[mask][:, self.n_dims : -1]
                + (z[mask][:, self.n_dims : -1] - xh[mask][:, self.n_dims : -1])
                * t_[mask][:, None, None]
                / self.cat_loss_step
            )

            net_out = self.phi(t, y, node_mask, edge_mask, context)

        else:
            raise NotImplementedError

            # get position, categorical, integer loss
            # Combining the terms

        def compute_cosine_dist(v1, v2):
            v1_norm = torch.norm(v1, dim=-1, keepdim=True) + 1e-10
            v2_norm = torch.norm(v2, dim=-1, keepdim=True) + 1e-10
            v1_normalized = v1 / v1_norm  # [batch, n_nodes, n_dims]
            v2_normalized = v2 / v2_norm  # [batch, n_nodes, n_dims]
            losses = 1 - torch.matmul(
                v1_normalized[:, :, None, :], v2_normalized[:, :, :, None]
            ).squeeze(-1).squeeze(-1)
            return losses

        if self.cat_loss == "l2_masked_mean":
            logging.debug("Using l2_masked_mean")
            assert (
                net_out.shape[-1]
                == self.n_dims
                + self.num_classes
                + self.include_charges
                + self.extend_feature_dim
            )
            loss = loss_reduce_mean_except_batch_with_mask(
                (net_out[:, :, : self.n_dims] - field_z[:, :, : self.n_dims]).square(),
                node_mask,
            ) + (
                loss_reduce_mean_except_batch_with_mask(
                    (
                        net_out[:, :, self.n_dims + self.num_classes]
                        - field_z[:, :, self.n_dims + self.num_classes]
                    ).square(),
                    node_mask,
                )
                if (not self.without_cat_loss) and self.include_charges
                else 0
            )
            if not self.without_cat_loss:
                cat_loss = loss_reduce_mean_except_batch_with_mask(
                    (
                        net_out[:, :, self.n_dims : self.n_dims + self.num_classes]
                        - field_z[:, :, self.n_dims : self.n_dims + self.num_classes]
                    ).square(),
                    node_mask,
                )
            else:
                cat_loss = 0
            if self.extend_feature_dim > 0:
                extend_feat_loss = (
                    loss_reduce_mean_except_batch_with_mask(
                        (
                            net_out[:, :, -self.extend_feature_dim :]
                            - field_z[:, :, -self.extend_feature_dim :]
                        ).square(),
                        node_mask,
                    )
                    / self.extend_feature_dim
                )
            else:
                extend_feat_loss = 0
            if self.cat_loss_step > 0:
                cat_mask = t.squeeze() < self.cat_loss_step
                cat_loss = cat_loss * cat_mask
            if self.angle_penalty:
                angle_loss = loss_reduce_mean_except_batch_with_mask(
                    compute_cosine_dist(
                        net_out[:, :, : self.n_dims], field_z[:, :, : self.n_dims]
                    ),
                    node_mask,
                )
            else:
                angle_loss = 0
            logging.debug(
                f"loss: {loss}, cat_loss: {cat_loss}, angle_loss: {angle_loss}, extend_feat_loss: {extend_feat_loss}"
            )
            loss = loss + cat_loss + angle_loss + extend_feat_loss
        elif self.cat_loss == "l2":
            loss = sum_except_batch(
                (net_out[:, :, : self.n_dims] - field_z[:, :, : self.n_dims]).square()
            ) + (
                sum_except_batch((net_out[:, :, -1] - field_z[:, :, -1]).square())
                if not self.without_cat_loss
                else 0
            )
            if not self.without_cat_loss:
                cat_loss = sum_except_batch(
                    (
                        net_out[:, :, self.n_dims : self.n_dims + self.num_classes]
                        - field_z[:, :, self.n_dims : self.n_dims + self.num_classes]
                    ).square()
                )
            else:
                cat_loss = 0
            if self.extend_feature_dim > 0:
                extend_feat_loss = (
                    sum_except_batch(
                        (
                            net_out[:, :, -self.extend_feature_dim :]
                            - field_z[:, :, -self.extend_feature_dim :]
                        ).square()
                    )
                    / self.extend_feature_dim
                )
            else:
                extend_feat_loss = 0
            if self.cat_loss_step > 0:
                cat_mask = t.squeeze() < self.cat_loss_step
                cat_loss = cat_loss * cat_mask
            if self.angle_penalty:
                angle_loss = sum_except_batch(
                    compute_cosine_dist(
                        net_out[:, :, : self.n_dims], field_z[:, :, : self.n_dims]
                    )
                )
            else:
                angle_loss = 0
            loss = loss + cat_loss + angle_loss + extend_feat_loss

        elif self.cat_loss == "cse":
            # coord_loss = sum_except_batch((net_out[:,:,self.n_dims]-field_z[:,:,self.n_dims]).square())
            loss = sum_except_batch((net_out - field_z).square())
            cat_loss = sum_except_batch(
                F.cross_entropy(
                    net_out[:, :, self.n_dims : -1],
                    xh[:, :, self.n_dims : -1].argmax(dim=2),
                )
            )
            if self.cat_loss_step > 0:
                cat_mask = t.squeeze() < self.cat_loss_step
                cat_loss = cat_loss * cat_mask

            loss = loss + cat_loss
            # cat_mask = (t.squeeze() < 0.01)  * cat_loss
            # loss = coord_loss + cat_loss + inter_loss
        else:
            raise NotImplementedError
        # loss =

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, {"error": loss}

    def get_slope(self, x, h, node_mask):
        """get the slope of the loss function at the current point,return the first self.n_dims, [self.n_dims:-1], [-1]"""
        x, h, _ = self.normalize(x, h, node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        z = self.sample_combined_position_feature_noise(
            xh.size(0), xh.size(1), node_mask
        )

        u = (1 - 1e-4) * z - xh

        postion_slope = sum_except_batch(u[:, :, : self.n_dims].square())
        categorical_slope = sum_except_batch(u[:, :, self.n_dims : -1].square())
        integer_slope = sum_except_batch(u[:, :, -1:].square())

        return postion_slope, categorical_slope, integer_slope

    def log_prob(
        self, x: Tensor, node_mask: Tensor, edge_mask: Tensor, context: Tensor
    ) -> Tensor:
        def hutch_trace(f, y, e=None):
            """Hutchinson's estimator for the Jacobian trace"""
            e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
            e_dzdx_e = e_dzdx * e
            approx_tr_dzdx = sum_except_batch(e_dzdx_e)
            return approx_tr_dzdx

        def only_frobenius(f, y, e=None):
            """Hutchinson's estimator for the Jacobian trace"""
            e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
            frobenius = sum_except_batch(e_dzdx.pow(2))
            return frobenius

        def hutch_trace_and_frobenius(f, y, e=None):
            """Hutchinson's estimator for the Jacobian trace"""
            e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
            frobenius = sum_except_batch(e_dzdx.pow(2))
            e_dzdx_e = e_dzdx * e
            approx_tr_dzdx = sum_except_batch(e_dzdx_e)
            return approx_tr_dzdx, frobenius

        def exact_trace(f, y):
            """Exact Jacobian trace"""
            dims = y.size()[1:]
            tr_dzdx = 0.0
            dim_ranges = [range(d) for d in dims]
            for idcs in itertools.product(*dim_ranges):
                batch_idcs = (slice(None),) + idcs
                tr_dzdx += torch.autograd.grad(
                    f[batch_idcs].sum(), y, create_graph=True
                )[0][batch_idcs]
            return tr_dzdx

        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t < self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = (
                        dx[:, :, self.n_dims : -1].clone() / self.cat_loss_step
                    )
                else:
                    dx[:, :, self.n_dims : -1] = 0
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx.clone() * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :].clone() * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx.clone() * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :].clone() * M_para
            else:
                pass

            return dx

        def ode_func(t, state):
            method = "hutch"
            x, ldj = state
            with torch.set_grad_enabled(True):
                x.requires_grad_(True)
                t.requires_grad_(True)
                dx = wrapper(t, x)
                if method == "exact":
                    ldj = exact_trace(dx, x)
                elif method == "hutch":
                    ldj = hutch_trace(dx, x, e=self._eps)
                # No regularization terms, set to zero.
                # reg_term = torch.zeros_like(ldj)
            return dx, ldj

        ladj = x.new_zeros(x.shape[0])
        self._eps = torch.rand_like(x)
        z, ladj = odeint(
            ode_func,
            (x, ladj),
            torch.tensor([0, 1.0], dtype=torch.float, device=x.device),
            method=self.method,
            rtol=self._rtol,
            atol=self._atol,
        )

        return self.prior_likelihood(z[-1], node_mask=node_mask) + ladj[-1]

    # def log_prob(self, x: Tensor,node_mask:Tensor,edge_mask:Tensor,context:Tensor) -> Tensor:
    #     I = torch.eye(x.shape[-1]).to(x)
    #     I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

    #     def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
    #         with torch.enable_grad():
    #             x = x.requires_grad_()
    #             dx = self.phi(t, x,node_mask,edge_mask,context)

    #         jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
    #         trace = torch.einsum('i...i', jacobian)

    #         return dx, trace * 1e-2

    #     ladj = torch.zeros_like(x[..., 0])

    #     z, ladj = zodeint(augmented, (x, ladj), 0.0,1.0, phi=self.parameters())

    #     return self.prior_likelihood(z,node_mask=node_mask) + ladj * 1e2

    # def encode(self, x: Tensor) -> Tensor:
    #     return odeint(self, x, 0.0, 1.0, phi=self.parameters())
    # self.register_buffer('inv_int_time', torch.tensor(list(reversed(times)), dtype=torch.float, device=device))
    def encode(self, x, node_mask, edge_mask, context):
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass
            return dx

        # if self.discrete_path == 'VP_path' or self.discrete_path == 'VP_path_poly' or self.discrete_path == 'HB_path_poly':
        return odeint(
            wrapper,
            x,
            torch.tensor([0.0, 1.0], dtype=torch.float, device=x.device),
            method=self.method,
            rtol=self._rtol,
            atol=self._atol,
        )
        # elif self.discrete_path == 'HB_path':
        #     return odeint(wrapper, x, torch.tensor([1e-5, 1.0],dtype=torch.float,device=x.device), method=self.method,rtol=self._rtol,atol=self._atol)
        # else:
        #     return odeint(wrapper, x, torch.tensor([0.0, 1.0],dtype=torch.float,device=x.device), method=self.method,rtol=self._rtol,atol=self._atol)

        # return odeint(wrapper, x, torch.tensor([0.0, 1.0],dtype=torch.float,device=x.device), method=self.method,rtol=self._rtol,atol=self._atol)

    def _gradients_from_node_type_entropy(self, x, node_mask, edge_mask):
        print(
            f"shape of x:{x.shape} shape of node_mask: {node_mask.shape}, shape of edge_mask: {edge_mask.shape}"
        )
        input_h = torch.ones(
            x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype, device=x.device
        )
        input_h = input_h / input_h.shape[-1]
        input_h = input_h.to(x.device, x.dtype)
        input_h.requires_grad = True
        xh = torch.cat([x, input_h], dim=-1)
        output = self.node_pred_model._forward(
            0, xh, node_mask, edge_mask, context=None
        )
        _h = torch.softmax(output[:, :, self.n_dims :], dim=-1)  # [B,N,K]
        print(f"===_h: {_h}")
        h_entropy = -torch.sum(_h * torch.log(_h + 1e-10), dim=-1)  # [B,N]
        print(f"===h_entropy: {h_entropy}")
        h_entropy_loss = loss_reduce_mean_except_batch_with_mask(
            h_entropy, node_mask
        ).mean()
        print(f"===h_entropy_loss: {h_entropy_loss}")
        xh_grad = autograd.grad(h_entropy_loss, xh, create_graph=True)[0]
        print(f"===x_grad: {xh_grad}")

        return xh_grad[:, :, : self.n_dims]

    def decode_minimize_type_entropy(self, z, node_mask, edge_mask, context) -> Tensor:
        if self.node_pred_model is None:
            if self.minimize_type_entropy and os.path.exists(
                self.node_classifier_model_ckpt
            ):
                print(
                    f"Loading node classifier model from {self.node_classifier_model_ckpt}"
                )
                print(
                    f"self.n_dims: {self.n_dims}, self.num_classes: {self.num_classes}"
                )
                print("=============================")
                self.node_pred_model = node_predict.EGNN_dynamics_QM9(
                    in_node_nf=self.num_classes,
                    context_node_nf=0,
                    n_dims=self.n_dims,
                    device="cuda",
                ).cuda()
                loader_utils.load_model(
                    self.node_pred_model, self.node_classifier_model_ckpt
                )
                self.node_pred_model.train()
            else:
                print("sampling without minimizing type entropy")

        def wrapper(t, x):
            dx_ = self.phi(t, x, node_mask, edge_mask, context)
            if t > 0.99:
                dx, dh = (
                    dx_[:, :, : self.n_dims].clone(),
                    dx_[:, :, self.n_dims :].clone(),
                )
                _in_x = x[:, :, : self.n_dims].clone()
                with torch.set_grad_enabled(True):
                    _in_x.requires_grad_(True)
                    _dx = self._gradients_from_node_type_entropy(
                        _in_x, node_mask, edge_mask
                    )
                dx = (
                    (1 - self.minimize_entropy_grad_coeff)
                    * (dx / dx.norm(dim=-1, keepdim=True))
                    + self.minimize_entropy_grad_coeff
                    * (_dx / _dx.norm(dim=-1, keepdim=True))
                ) * dx.norm(dim=-1, keepdim=True)
                dx = torch.cat([dx, dh], dim=-1)
            else:
                dx = dx_
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = 0
                else:
                    dx[:, :, self.n_dims : -1] = dx[:, :, self.n_dims : -1] / (
                        self.cat_loss_step
                    )
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            # if self.cat_loss_step > 0:
            #     if t > self.cat_loss_step:
            #         dx[:,:,self.n_dims:-1] = 0
            #     else:
            #         dx[:,:,self.n_dims:-1] = dx[:,:,self.n_dims:-1] / (self.cat_loss_step)

            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass

            return dx

        if self.method == "euler":
            t_list = np.linspace(1.0, 0, self.T)
            t_list = torch.tensor(t_list, dtype=torch.float, device=z.device)
            t_list = inv_cdf(t_list)
        else:
            t_list = [1.0, 0]
            t_list = torch.tensor(t_list, dtype=torch.float, device=z.device)
        return odeint(
            wrapper, z, t_list, method=self.method, rtol=self._rtol, atol=self._atol
        )

        # if self.discrete_path == 'VP_path':
        #     return odeint(wrapper, z, torch.tensor([1.0, 1e-5],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)
        # elif self.discrete_path == 'HB_path':
        #     return odeint(wrapper, z, torch.tensor([1.0, 1e-5],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)
        # else:
        #     return odeint(wrapper, z, torch.tensor([1.0, 0.0],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)

    def decode(self, z, node_mask, edge_mask, context) -> Tensor:
        self.wrapper_count = 0
        self.time_steps = []

        def wrapper(t, x):
            self.wrapper_count += 1
            self.time_steps.append(t.cpu().numpy().item())
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = 0
                else:
                    dx[:, :, self.n_dims : -1] = dx[:, :, self.n_dims : -1] / (
                        self.cat_loss_step
                    )
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            # if self.cat_loss_step > 0:
            #     if t > self.cat_loss_step:
            #         dx[:,:,self.n_dims:-1] = 0
            #     else:
            #         dx[:,:,self.n_dims:-1] = dx[:,:,self.n_dims:-1] / (self.cat_loss_step)

            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass

            dx = dx * node_mask

            return dx

        if self.method in {"euler", "rk4", "midpoint"}:
            t_list = np.linspace(1.0, 0, self.T)
            t_list = torch.tensor(t_list, dtype=torch.float, device=z.device)
            # t_list = inv_cdf(t_list)
        else:
            t_list = [1.0, 0]
            t_list = torch.tensor(t_list, dtype=torch.float, device=z.device)
        out = odeint(
            wrapper, z, t_list, method=self.method, rtol=self._rtol, atol=self._atol
        )
        print(f"wrapper_count: {self.wrapper_count}")
        # print(f"time_steps: {self.time_steps}")
        return out

        # if self.discrete_path == 'VP_path':
        #     return odeint(wrapper, z, torch.tensor([1.0, 1e-5],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)
        # elif self.discrete_path == 'HB_path':
        #     return odeint(wrapper, z, torch.tensor([1.0, 1e-5],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)
        # else:
        #     return odeint(wrapper, z, torch.tensor([1.0, 0.0],dtype=torch.float,device=z.device), method=self.method,rtol=self._rtol,atol=self._atol)

    def decode_chain(self, z, t, node_mask, edge_mask, context) -> Tensor:
        # here t is all the model which we used to decode
        def wrapper(t, x):
            dx = self.phi(t, x, node_mask, edge_mask, context)
            if self.cat_loss_step > 0:
                if t > self.cat_loss_step:
                    dx[:, :, self.n_dims : -1] = 0
                else:
                    dx[:, :, self.n_dims : -1] = dx[:, :, self.n_dims : -1] / (
                        self.cat_loss_step
                    )
                # cat_mask = t.squeeze() < self.cat_loss_step
                # dx[~cat_mask][:,self.n_dims:-1] = 0
                # dx[cat_mask][:,self.n_dims:-1] = dx[cat_mask][:,self.n_dims:-1] / self.cat_loss_step # align the speed.
            if self.discrete_path == "VP_path":
                M_para = (
                    -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                )  # add epsilon to stable it
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path":
                M_para = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            elif self.discrete_path == "VP_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx = dx * M_para
            elif self.discrete_path == "HB_path_poly":
                alpha_s2 = polynomial_schedule_(t)
                M_para = 1 / (1 - alpha_s2 + 1e-5)  # alpha_div / 1 - alpha_t2
                M_para = M_para.unsqueeze(-1)[:, None, None]
                dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * M_para
            else:
                pass
            return dx

        t = torch.tensor(t, dtype=torch.float, device=z.device)

        return odeint(
            wrapper, z, t, method=self.method, rtol=self._rtol, atol=self._atol
        )

    def prior_likelihood(self, z, node_mask):
        z_x = z[:, :, : self.n_dims]
        z_h = z[:, :, self.n_dims :]
        # def forward(self, z_x, z_h, node_mask=None):
        assert len(z_x.size()) == 3
        assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2]

        assert (z_x * (1 - node_mask)).sum() < 1e-8 and (
            z_h * (1 - node_mask)
        ).sum() < 1e-8, "These variables should be properly masked."

        log_pz_x = utils.center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = utils.standard_gaussian_log_likelihood_with_mask(z_h, node_mask)

        log_pz = log_pz_x + log_pz_h

        return log_pz

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        # log_pN = nodes_dist.log_prob(N)

        # log_prob = self.log_prob(xh,node_mask,edge_mask,context)

        # if self.training:
            # Only 1 forward pass when t0_always is False.
        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)
        neg_log_pxh = loss
        # else:
        #     # Less variance in the estimator, costs two forward passes.

        #     log_px = self.log_prob(xh, node_mask, edge_mask, context) - delta_log_px

        #     neg_log_pxh = -log_px

            # Correct for normalization on x.
            # assert neg_log_pxh.size() == delta_log_px.size()
            # neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_cat_z0(self, xh, node_mask, edge_mask, context):
        """
        get the catgorical distribution according to coordinate and features.
        """
        # whether input use a xh or else.
        t = torch.zeros_like(xh[:, 0, 0]).view(-1, 1, 1)
        net_out = self.phi(0.0, xh, node_mask, edge_mask, context)
        z_h = net_out[
            :, :, self.n_dims : -1
        ]  # use the score function as the sampling direction. Instead of the ode results.
        xh[
            :, :, self.n_dims : -1
        ] = z_h  # replace the original xh with the sampled one.

        return xh

    def training_cat_z0(self, xh, node_mask, edge_mask, context):
        """
        get the categorical distribution on the zeroth term.
        """
        mask = torch.rand_like(xh[:, :, self.n_dims : -1])  # destroy signal for this.
        xh[:, :, self.n_dims : -1] = mask
        net_out = self.phi(0.0, xh, node_mask, edge_mask, context)
        # Get the categorical distribution.
        z_h = net_out[:, :, self.n_dims : -1]
        # z_h = z_h.reshape(z_h.size(0),z_h.size(1),self.n_cat,self.n_cat)
        cat_loss_zero_term = torch.nn.CrossEntropyLoss(
            z_h, xh[:, :, self.n_dims : -1].argmax(dim=2)
        )

        return cat_loss_zero_term

    def compress(self, xh, t, normalize_factor=1.0):
        """
        Compresses the time interval [0, 1] to [0, t]. used for the categorical distribution.
        """
        t = t.view(-1, 1, 1).expand_as(xh)
        t[:, :, self.n_dims : -1] = t[:, :, self.n_dims : -1] / normalize_factor
        return t

    @torch.no_grad()
    def sample(
        self,
        dequantizer,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        fix_noise=False,
    ):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        # def decode(self, z,node_mask,edge_mask,context) -> Tensor:
        if not self.minimize_type_entropy:
            z_ = self.decode(z, node_mask, edge_mask, context)[-1]
        else:
            z_ = self.decode_minimize_type_entropy(z, node_mask, edge_mask, context)[-1]

        if self.sampling_method == "gradient":
            # time_step = [1e-2]
            # for i in range(time_step):
            init = z_[:, :, self.n_dims : -1]
            # print(init.norm(dim=2))
            categorical_steps = np.linspace(0.05, 0, 20)
            for i_ in categorical_steps:
                # slightly perturb
                gradient = self.phi(
                    torch.tensor([i_]), z_, node_mask, edge_mask, context
                )
                init = init + gradient[:, :, self.n_dims : -1] * (0.05 / 20)

            z_[:, :, self.n_dims : -1] = init
        elif self.sampling_method == "vanilla":
            pass
        else:
            raise NotImplementedError
        # dequantization
        # one_hot = z[:, :, 3:8]
        # charges = z[:, :, 8:]
        # tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})
        # one_hot, charges = tensor['categorical'], tensor['integer']
        # z = torch.cat([z[:, :, :3], one_hot, charges], dim=2)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(dequantizer, z_, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(
        self,
        dequantizer,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
        if keep_frames is None:
            keep_frames = 100
        else:
            assert keep_frames <= 1000

        # chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        time_step = list(np.linspace(1, 0, keep_frames))

        chain_z = self.decode_chain(z, time_step, node_mask, edge_mask, context)

        for i in range(len(chain_z) - 1):
            ##fix chain sampling
            chain_z[i] = self.unnormalize_z(chain_z[i], node_mask)
            # chain_z[i] =
            # one_hot = chain_z[i][:, :, 3:8]
            # charges = chain_z[i][:, :, 8:]
            # tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})
            # one_hot, charges = tensor['categorical'], tensor['integer']
            # chain_z[i] = torch.cat([chain_z[i][:, :, :3], one_hot, charges], dim=2)

        chain_z = reversed(chain_z)
        x, h = self.sample_p_xh_given_z0(
            dequantizer, chain_z[-1], node_mask
        )  # TODO this should be the reverse of our flow model
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        # for s in reversed(range(0, self.T)):
        #     s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
        #     t_array = s_array + 1
        #     s_array = s_array / self.T
        #     t_array = t_array / self.T

        #     z = self.sample_p_zs_given_zt(
        #         s_array, t_array, z, node_mask, edge_mask, context)

        #     diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        #     # Write to chain tensor.
        #     write_index = (s * keep_frames) // self.T
        #     chain[write_index] = self.unnormalize_z(z, node_mask)
        # Finally sample p(x, h | z_0).
        diffusion_utils.assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)
        b, n, _ = x.size()
        distance2origin = x.square().sum(dim=-1).sqrt()  # [b, n]
        if self.extend_feature_dim > 0:
            extend_feat = self.extend_feature_embedding(
                distance2origin.view(-1)
            )  # [b, n] -> [b, n, dim]
            extend_feat = extend_feat.view(b, n, -1)  # [bxn, dim] -> [b, n, dim]
            xh = torch.cat([x, h["categorical"], h["integer"], extend_feat], dim=2)
        else:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        # print(chain_z.size(),xh.size(),h['integer'], h['categorical'],chain_z[0])

        chain_z[0] = xh  # Overwrite last frame with the resulting x and h.
        chain_flat = chain_z.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat
