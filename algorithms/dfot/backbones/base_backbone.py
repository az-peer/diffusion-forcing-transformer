# this is a method for using abstract classes
from abc import abstractmethod, ABC
from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from .modules.embeddings import (
    StochasticTimeEmbedding,
    RandomDropoutCondEmbedding,
)


# this is an abstract class
# basically makes a blueprint for all the models
# cannot be called on its own
class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        # typically a config that stores hyperparameters
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        # look at this conditional embeddings
        # seems to define how this condition get embedded
        external_cond_dim: int,
        use_causal_mask=True,
    ):

        super().__init__()

        self.cfg = cfg
        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.x_shape = x_shape

        # we create embeddings from this class
        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            # defines the dimension for the noise
            dim=self.noise_level_dim,
            # for the time embeddings as well
            time_embed_dim=self.noise_level_emb_dim,
            # grab fourier embeddings if it from the config file
            use_fourier=self.cfg.get("use_fourier_noise_embedding", False),
        )
        # if this is called from the config then we have conditional
        # embeddings
        self.external_cond_embedding = self._build_external_cond_embedding()

    # then this defines the function of how to actually use the conditional
    # embeddgins
    # but we only do it if there is a self.external_cond_dim is
    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return (
            RandomDropoutCondEmbedding(
                self.external_cond_dim,
                self.external_cond_emb_dim,
                dropout_prob=self.cfg.get("external_cond_dropout", 0.0),
            )
            if self.external_cond_dim
            else None
        )

    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    # any class that calls the backbone must use the this level
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # same with this
    def external_cond_emb_dim(self):
        raise NotImplementedError

    @abstractmethod
    # we must have a forward class
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError
