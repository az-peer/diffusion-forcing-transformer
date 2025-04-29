from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat

# pytorches image patchifier
# patcheifies and projects them into an embeddings space
from timm.models.vision_transformer import PatchEmbed

# the backbone class from the other file
from ..base_backbone import BaseBackbone

from .dit_base import DiTBase


class DiT3D(BaseBackbone):
    """
    The DiT3D class is a 3D extension of a diffusion transformer backbone, processing
    sequences of images (or 3D data) by embedding patches and adding noise/external
    conditioning information. It first patchifies the input, adds embeddings based on
    noise level (and optionally external conditions), and passes the result through a
    core transformer (DiTBase). Finally, it reconstructs the spatial structure
    ("unpatchifies") the output, returning it in the original 3D format.

    Note if w ehad the CLIP embeddings here then we would actually have to retrain the
    acutal model.


    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,  # dimension of external conidtional embeddings
        use_causal_mask=True,  # use casual masking useless here
    ):
        if use_causal_mask:
            raise NotImplementedError(
                "Causal masking is not yet implemented for DiT3D backbone"
            )
        # we call the properties from the parent class
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        # grrab the number of hidden input size
        hidden_size = cfg.hidden_size
        # the dimensionality of the pathces
        self.patch_size = cfg.patch_size
        # we only grab the channels and the resolution
        channels, resolution, *_ = x_shape
        assert (
            resolution % self.patch_size == 0
        ), "Resolution must be divisible by patch size."
        # we get the number of patches in both directions .
        # this is why we have to square it to cover height and width
        self.num_patches = (resolution // self.patch_size) ** 2
        # each patch is technically a volumne
        # so we have a patch_height by width which is why we square it
        # then we multiple by the number of channels
        out_channels = self.patch_size**2 * channels

        # creates an instance of the PatchEmbeddder which takes patches and
        # then embedds them
        # this is usually done through a linear layer
        self.patch_embedder = PatchEmbed(
            img_size=resolution,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # and then we create an insance of the transformer that we will be
        # using
        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )
        # calls the self.initalize weights
        self.initialize_weights()

    # allows us to get the number of channels as an attribute
    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

    # a static method is one where we do not need to actually
    # adjust the data within a class object
    @staticmethod
    # here we are only operating on the patch embedder itself
    # not tht Dit Backbone thus we can run this
    # describes how to initialize the weights of the models
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            # simply checks if a module is linear and initializes weights
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # begings the process of the noise embeddings which can include linear
        self.noise_level_pos_embedding.apply(_mlp_init)
        # checks if there is an external condition
        # if there is an extrernal condition then we recursively apply the
        # the linear weight initilization to the entire submodule
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patchified tensor of shape (B, num_patches, patch_size**2 * C)
        Returns:
            unpatchified tensor of shape (B, H, W, C)
        """
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        # if we add CLIP embeddings here then we would actually
        # have to retrain the entire model
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        # this is to treat the beatch dima nd the time as uniform
        x = rearrange(x, "b t c h w -> (b t) c h w")
        # then we actually take our data and put these into patches
        x = self.patch_embedder(x)
        # we reshape this to have patches accross time
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        emb = self.noise_level_pos_embedding(noise_levels)
        # we add th wconidtional to the atual noise embeddings
        if external_cond is not None:
            emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
        # we just do this in a way that maatches the shape that we actually want
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)
        # we then pass this through our actual transformer
        x = self.dit_base(x, emb)  # (B, N, C)
        # then we unpatchify the actual reults
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        # then we rearrange in our original format
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        # we have the actual video!
        return x
