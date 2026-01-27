#!/usr/bin/env python3
"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep

    Encoder (Downsampling path)

    Middle

    Decoder (Upsampling path)

    Output: (batch_size, channels, H, W)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionBlock, Downsample, ResBlock, TimestepEmbedding, Upsample


class UNet(nn.Module):
    """
    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks

    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3,
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # Time embedding: produces (B, time_embed_dim)
        # Here we pick time_embed_dim = base_channels for simplicity.
        self.time_embed_dim = base_channels
        self.time_embed = TimestepEmbedding(time_embed_dim=self.time_embed_dim)

        # Input stem
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Track channels for skip connections (we store outputs after each ResBlock)
        self.skip_channels: List[int] = []

        ch = base_channels
        # Assume input is 64x64 for this homework; if you later generalize,
        # pass image_size into UNet and propagate resolution similarly.
        resolution = 64

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            # ResBlocks at this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(
                                in_channels=ch,
                                out_channels=out_ch,
                                time_embed_dim=self.time_embed_dim,
                                dropout=dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                            ),
                            "attn": (
                                AttentionBlock(out_ch, num_heads=num_heads)
                                if resolution in self.attention_resolutions
                                else nn.Identity()
                            ),
                        }
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)

            # Downsample between levels (except last)
            if level != len(channel_mult) - 1:
                self.downsample_layers.append(Downsample(ch))
                resolution //= 2
                # also save a skip after downsample? (optional; many implementations do)
                # We'll save the tensor in forward anyway, but channel count stays ch.
            else:
                self.downsample_layers.append(nn.Identity())

        # ---------------------------------------------------------------------
        # Middle
        # ---------------------------------------------------------------------
        self.mid_block1 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            time_embed_dim=self.time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attn = (
            AttentionBlock(ch, num_heads=num_heads)
            if resolution in self.attention_resolutions
            else nn.Identity()
        )
        self.mid_block2 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            time_embed_dim=self.time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        # We will pop from skip list in reverse order.
        # Decoder mirrors encoder levels in reverse.
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult

            # At each level, we typically have num_res_blocks + 1 blocks
            # to consume both ResBlock outputs and the skip right after downsample.
            # Here we match the number of skips we recorded: num_res_blocks per level.
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(
                                in_channels=ch + skip_ch,  # concat skip
                                out_channels=out_ch,
                                time_embed_dim=self.time_embed_dim,
                                dropout=dropout,
                                use_scale_shift_norm=use_scale_shift_norm,
                            ),
                            "attn": (
                                AttentionBlock(out_ch, num_heads=num_heads)
                                if resolution in self.attention_resolutions
                                else nn.Identity()
                            ),
                        }
                    )
                )
                ch = out_ch

            # Upsample between levels (except first / highest resolution)
            if level != 0:
                self.upsample_layers.append(Upsample(ch))
                resolution *= 2
            else:
                self.upsample_layers.append(nn.Identity())

        assert len(self.skip_channels) == 0, "Internal error: not all skip channels were consumed."

        # Output head
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Time embedding
        time_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Stem
        h = self.in_conv(x)

        # Encoder forward with skip saves
        skips: List[torch.Tensor] = []
        down_block_idx = 0
        for level in range(len(self.channel_mult)):
            # num_res_blocks at this level
            for _ in range(self.num_res_blocks):
                block = self.down_blocks[down_block_idx]
                down_block_idx += 1
                h = block["res"](h, time_emb)
                h = block["attn"](h)
                skips.append(h)

            # Downsample (or identity at last level)
            h = self.downsample_layers[level](h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder forward
        up_block_idx = 0
        for level in range(len(self.channel_mult)):
            # num_res_blocks at this level
            for _ in range(self.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)

                block = self.up_blocks[up_block_idx]
                up_block_idx += 1
                h = block["res"](h, time_emb)
                h = block["attn"](h)

            # Upsample (or identity at final)
            h = self.upsample_layers[level](h)

        assert len(skips) == 0, "Internal error: not all skip tensors were consumed."

        # Output head
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.

    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters

    Returns:
        Instantiated UNet model
    """
    model_config = config["model"]
    data_config = config["data"]

    return UNet(
        in_channels=data_config["channels"],
        out_channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=model_config["attention_resolutions"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)

    with torch.no_grad():
        out = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
