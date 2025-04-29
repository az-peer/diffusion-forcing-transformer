import torch
import torch.nn.functional as F
from algorithms.common.metrics.video.utils import videos_as_images


def warp(img, flow):
    ################### GOAL ############################################
    """
    This function takes an image and then takes in the optical flow as well.
    Remember that optical flow basically takes each pixel and then tells us
    exactly where that pixel will end up. So this function takes an image and
    the flow image and warps the input image.
    """
    #####################################################################
    # grab the batch
    # basically in optical flow this would be like pairs of images
    # we predict the motion or vector field
    # probably height and width of this
    B, _, H, W = flow.shape
    # we then use the next two lines to create the a coordinate system for the
    # entire batch
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    # We then combine this grid together to get the following shape
    # (B, 2, H, W) or the x,y coordinate of every pixel
    grid = torch.cat([xx, yy], 1).to(img)
    # normalizes the optical flow to match the coordinate system we jsut created
    flow_ = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        1,
    )
    # now shift the actual coordinates with the normalized flow
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    # sample the image with the new coordinates to get the warped ones
    # this is the actual warping
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


class InputPadder:
    ##################################### GOAL #############################
    """Pads images such that dimensions are divisible by divisor

    This essentially allows us to pad videos and images such that the
    dimensions are divisible by a certain number.
    """

    ########################################################################
    # it takes in what we want the video or image to be divided by
    # also pass in the dimension of the input channel batch
    def __init__(self, dims, divisor=16):
        # we grab in the height and the width of this
        self.ht, self.wd = dims[-2:]
        # computes how much the height dimension needs to be padded
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        # same with the width
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        # make sure that we pad symmetrically
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    # we then allow this padding to work on a series of videos
    @videos_as_images
    # this is the actual padding
    def pad(self, *inputs):
        # if there is only one image or video do this
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode="replicate")
        # else apply to all the images or video at the same time
        else:
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    # this does the unpadding in the same way as above
    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    # defines how to actually unpad an iamg e along the height and the width
    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
