from torch import nn, Tensor
from pyiqa.archs.musiq_arch import MUSIQ as _MUSIQ
from algorithms.common.metrics.video.utils import videos_as_images
from utils.print_utils import suppress_print
from .utils import download_model_from_url

MUSIQ_PATH = "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"

"""
    Downloads this fucking model. Actually thid is sick. Basically a github for 
    IQA image quality assesment. Maybe we can use this to evaluate our last frame.
    DEfinately worth going over later.
"""


class MUSIQ(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = download_model_from_url(MUSIQ_PATH, progress=False)
        with suppress_print():
            self.model = _MUSIQ(pretrained_model_path=model_path)

    def train(self, mode: bool) -> "MUSIQ":
        super().train(False)

    @videos_as_images
    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)
