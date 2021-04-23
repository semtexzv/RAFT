from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8


# try:
#     _autocast = torch.cuda.amp.autocast
# except ImportError:
# dummy autocast for PyTorch < 1.6
class autocast:
    def __init__(self, enabled: bool = True):
        pass

    def __enter__(self):
        pass

    def __exit__(self, a: Any, b: Any, c: Any):
        pass


class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mixed_precision = args.mixed_precision if 'mixed_precision' in args else False

        if args.tiny:
            self.hidden_dim = hdim = 32
            self.context_dim = cdim = 16
            args.corr_levels = 4
            args.corr_radius = 3

        elif args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args:
            args.dropout = 0

        if 'alternate_corr' not in args or not args.alternate_corr:
            self.corr = CorrBlock(radius=args.corr_radius)
        else:
            self.corr = AlternateCorrBlock(radius=args.corr_radius)

        # feature network, context network, and update block
        if args.small or args.tiny:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(args, hidden_dim=hdim, context_dim=self.context_dim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters: int, flow_init: Optional[torch.Tensor] = None, test_mode: bool =False):
        """ Estimate optical flow between pair of frames """

        image1 = 2.0 * (image1 / 255.0) - 1.0
        image2 = 2.0 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            # Disabled merging in batch dimension here. Caused problems for onnx tracing
            fmap1 = self.fnet(image1)
            fmap2 = self.fnet(image2)
            # 2x 1, 128, 55, 128
            # fmap1, fmap2 = self.fnet.forward_list([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        self.corr.regenerate(fmap1, fmap2)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # TODO
        # if flow_init:
        #     coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = self.corr(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions

        return flow_predictions
