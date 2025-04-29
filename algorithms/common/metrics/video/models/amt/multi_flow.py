import torch
import torch.nn as nn
from .utils import warp
from .ifrnet import (
    convrelu,
    resize,
    ResBlock,
)

########################### GOAL #############################################
"""
    Get of way of getting intermediate frames using optical flow 
    This file havily uses ifrnet which is a deep learning method of estimating
    the flows .

"""

#############################################################################


def multi_flow_combine(
    comb_block, img0, img1, flow0, flow1, mask=None, img_res=None, mean=None
):
    """
    A parallel implementation of multiple flow field warping
    comb_block: An nn.Seqential object.
    img shape: [b, c, h, w]
    flow shape: [b, 2*num_flows, h, w]
    mask (opt):
        If 'mask' is None, the function conduct a simple average.
    img_res (opt):
        If 'img_res' is None, the function adds zero instead.
    mean (opt):
        If 'mean' is None, the function adds zero instead.

        Basically this is a sick ass function. Call is MULTI-FLOW BABY. Imagine that
        you have two consecutive frames. But sometimes maybe we want to produce a
        frame that is in the middle of these two. How do we do that? This sick ass
        function. We basically take two iamges and two flow maps, then we apply multiple
        approximations for what we believe the flow to be and average them out to get
        a more robutst interpertation.
    """
    b, c, h, w = flow0.shape
    num_flows = c // 2
    flow0 = flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    flow1 = flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    # this is to determien how much of the flow from flow map one to use
    # versu sthe other one
    mask = (
        mask.reshape(b, num_flows, 1, h, w).reshape(-1, 1, h, w)
        if mask is not None
        else None
    )
    img_res = (
        img_res.reshape(b, num_flows, 3, h, w).reshape(-1, 3, h, w)
        if img_res is not None
        else 0
    )
    img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
    img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
    mean = (
        torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1)
        if mean is not None
        else 0
    )

    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)
    # this is where we take the weighted average
    img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
    img_warps = img_warps.reshape(b, num_flows, 3, h, w)
    imgt_pred = img_warps.mean(1) + comb_block(img_warps.view(b, -1, h, w))
    return imgt_pred


# this is the final decoder for the flow
class MultiFlowDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_flows=3):
        super(MultiFlowDecoder, self).__init__()
        # class seems like an encoder that we pass into the function above
        # define the number of estimates for flow that we have
        self.num_flows = num_flows
        # defines a conv block
        self.convblock = nn.Sequential(
            # ti
            convrelu(in_ch * 3 + 4, in_ch * 3),
            ResBlock(in_ch * 3, skip_ch),
            nn.ConvTranspose2d(in_ch * 3, 8 * num_flows, 4, 2, 1, bias=True),
        )

    def forward(self, ft_, f0, f1, flow0, flow1):
        n = self.num_flows
        f0_warp = warp(f0, flow0)
        f1_warp = warp(f1, flow1)
        out = self.convblock(torch.cat([ft_, f0_warp, f1_warp, flow0, flow1], 1))
        delta_flow0, delta_flow1, mask, img_res = torch.split(
            out, [2 * n, 2 * n, n, 3 * n], 1
        )
        mask = torch.sigmoid(mask)

        flow0 = delta_flow0 + 2.0 * resize(flow0, scale_factor=2.0).repeat(
            1, self.num_flows, 1, 1
        )
        flow1 = delta_flow1 + 2.0 * resize(flow1, scale_factor=2.0).repeat(
            1, self.num_flows, 1, 1
        )

        return flow0, flow1, mask, img_res
