#!/usr/bin/env python3
"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

        # =========================================================================
        # You can add, delete or modify as many functions as you would like
        # =========================================================================

        # Pro tips: If you have a lot of pseudo parameters that you will specify for each
        # model run but will be fixed once you specified them (say in your config),
        # then you can use super().register_buffer(...) for these parameters

        # Pro tips 2: If you need a specific broadcasting for your tensors,
        # it's a good idea to write a general helper function for that

        T = self.num_timesteps

        # betas[1..T], betas[0] unused
        betas = torch.zeros(T + 1)
        betas[1:] = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_var = torch.zeros_like(betas)
        posterior_var[1:] = (  # valid only for t > 0
            betas[1:]
            * (1.0 - alpha_bars[:-1])  # alpha_bar_{t-1}
            / (1.0 - alpha_bars[1:])  # alpha_bar_t
        )  # note that it's 0 for t = 1
        self.register_buffer("posterior_variance", posterior_var)

    @staticmethod
    def _extract(buf: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """buf: (T,), t: (B,) -> (B,1,1,1) broadcastable to x"""
        out = buf.gather(0, t)  # (B,)
        return out.view(x.shape[0], *([1] * (x.ndim - 1)))

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x_0)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments

        Returns:
            loss: Scalar loss tensor for back propagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        B = x_0.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, noise=eps)

        eps_pred = self.model(x_t, t)
        loss = F.mse_loss(eps_pred, eps)
        return loss, {"loss": loss.detach().item()}

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        eps_theta = self.model(x_t, t)

        beta_t = self._extract(self.betas, t, x_t)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t)
        sqrt_omab_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t)

        mu = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_omab_t) * eps_theta)

        var = self._extract(self.posterior_variance, t, x_t)
        z = torch.randn_like(x_t)
        return mu + torch.sqrt(var) * z * (t > 1).float().view(-1, 1, 1, 1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs,
    ) -> torch.Tensor:
        """DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x = torch.randn(batch_size, *image_shape, device=self.device)

        for ti in reversed(range(1, self.num_timesteps + 1)):
            t = torch.full((batch_size,), ti, device=self.device, dtype=torch.long)
            x = self.reverse_process(x, t)

        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
        ).to(device)
