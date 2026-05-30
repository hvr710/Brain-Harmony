from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.flex_transformer import Block


@dataclass
class PosArgs:
    grid_size: tuple[int, int]
    embed_dim: int
    predictor_embed_dim: int
    cls_token: bool
    grad_dim: int
    gradient: str
    geoh_dim: int
    geo_harm: str
    use_pos_embed_decoder: bool


class DownstreamHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mode: str,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OriginalStyleMLPHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        self.lin3 = nn.Linear(hidden_dim, out_features)
        nn.init.kaiming_uniform_(self.lin1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.lin2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.lin3.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.drop1(self.lin1(x)))
        x = F.relu(self.drop2(self.lin2(x)))
        return self.lin3(x)


class FMRITokenStage2Model(nn.Module):
    """fMRI-only adaptation of the repository's stage2 latent-token downstream model."""

    def __init__(
        self,
        out_dim: int,
        input_tokens: int = 7200,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_latent_tokens: int = 128,
        drop_path_rate: float = 0.1,
        head_type: str = "linear",
        head_dropout: float = 0.0,
        attn_mode: str = "flash_attention_2",
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.input_tokens = input_tokens
        self.num_latent_tokens = num_latent_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, input_tokens + 1, embed_dim), requires_grad=False)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim))
        self.latent_pos_embed = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                )
                for i in range(depth)
            ]
        )
        self.fc_norm = norm_layer(embed_dim)
        if head_type == "mlp":
            self.head = OriginalStyleMLPHead(embed_dim, embed_dim // 2, out_dim, head_dropout)
        else:
            self.head = nn.Linear(embed_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tokens.shape
        pos_embed = self.pos_embed[:, : num_tokens + 1]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + pos_embed
        latent = self.latent_tokens.expand(batch_size, -1, -1) + self.latent_pos_embed
        x = torch.cat([x, latent], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        latent_mask = torch.ones(batch_size, self.num_latent_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask, latent_mask], dim=1)
        for block in self.blocks:
            x = block(x, attention_mask=full_mask)
        x = x[:, -self.num_latent_tokens :].mean(dim=1)
        return self.fc_norm(x)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(tokens, attention_mask))


def build_harmonix_f_encoder(
    checkpoint_path: str = "checkpoints/harmonix-f/model.pth",
    device: torch.device | str = "cuda",
    attn_mode: str = "flash_attention_2",
) -> nn.Module:
    import libs.model as model_lib
    import libs.position_embedding as pos_embeds

    device = torch.device(device)
    pos_args = PosArgs(
        grid_size=(400, 18),
        embed_dim=768,
        predictor_embed_dim=384,
        cls_token=False,
        grad_dim=30,
        gradient="brainharmony_pos_embed/gradient_mapping_400.csv",
        geoh_dim=200,
        geo_harm="brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv",
        use_pos_embed_decoder=True,
    )
    pos_embed = pos_embeds.BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed(
        device, pos_args
    )
    encoder = model_lib.vit_base_flex(
        pos_embed=pos_embed,
        cls_token=None,
        img_size=(400, 48 * 18),
        patch_size=48,
        gradient_checkpointing=False,
        attn_mode=attn_mode,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict):
        prefixed = {
            key[len("encoder_ema.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder_ema.")
        }
        state = prefixed or checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    model_state = encoder.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    msg = encoder.load_state_dict(compatible, strict=False)
    print(f"Loaded Harmonix-F encoder from {checkpoint_path}: {msg}")
    return encoder.to(device)


def pool_tokens(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return tokens.mean(dim=1)
    mask = attention_mask.to(tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denom


class HarmonixFDownstreamModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        out_dim: int,
        head_mode: str,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = DownstreamHead(768, out_dim, head_mode, hidden_dim, dropout)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        patch_size: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        patch = int(patch_size[0].item()) if torch.is_tensor(patch_size) else int(patch_size)
        tokens = self.encoder(x, patch, attention_mask=attention_mask)
        features = pool_tokens(tokens, attention_mask)
        return self.head(features)
