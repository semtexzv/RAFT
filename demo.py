from core.utils.utils import InputPadder
from core.utils import flow_viz
from core.raft import RAFT
from core import DEVICE

from PIL import Image
import torch
import numpy as np
import glob
import cv2
import os
import argparse


def profile(model, *args, **kwargs):
    import torch.autograd.profiler as profiler
    with profiler.profile(record_shapes=True, with_stack=True, with_flops=True) as prof:
        with profiler.record_function("model_inference"):
            res = model(*args, **kwargs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("aa.json")
    return res


def export(model, *args, **kwargs):
    # Grid sampler needs to be defined manually
    def grid_sampler(g, input1, input2, mode, padding_mode, align_corners):
        return g.op("tmp::grid_sampler", input1, input2, mode, padding_mode, align_corners)

    torch.onnx.register_custom_op_symbolic("::grid_sampler", grid_sampler, opset_version=11)

    torch.onnx.export(model, (*args, kwargs), "raft.onnx", verbose=True, opset_version=11)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey(1)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    model = torch.jit.script(model)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # export(
            #     model, image1, image2,
            #     torch.tensor([6], dtype=torch.int),
            #     torch.Tensor([False]),
            #     torch.Tensor([False]),
            #     torch.tensor([False], dtype=torch.bool)
            # )
            flow_low, flow_up = model(image1, image2, iters=1, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--tiny', action='store_true', help='use even smaller model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
