import thop
import yaml
from pathlib import Path
from models.common import *
from utils.general import check_version
from utils.torch_utils import fuse_conv_and_bn


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.tensor([8., 16., 32.])
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) / self.stride.view(-1, 1, 1)
        a = self.anchors.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = self.stride[-1] - self.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            self.anchors[:] = self.anchors.flip(0)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class MyYolo(nn.Module):
    def __init__(self, nc=80, anchors=()):
        super(MyYolo, self).__init__()
        # Backbone
        self.conv1 = Conv(3, 32, k=6, s=2, p=2)  # 0
        self.conv2 = Conv(32, 64, k=3, s=2)  # 1
        self.c31 = C3(64, 64, n=1)  # 2
        self.conv3 = Conv(64, 128, k=3, s=2)  # 3
        self.c32 = C3(128, 128, n=2)  # 4
        self.conv4 = Conv(128, 256, k=3, s=2)  # 5
        self.c33 = C3(256, 256, n=3)  # 6
        self.conv5 = Conv(256, 512, k=3, s=2)  # 7
        self.c34 = C3(512, 512, n=1)  # 8
        self.sppf = SPPF(512, 512)  # 9
        # Head
        self.conv6 = Conv(512, 256, k=1, s=1)  # 10
        self.unsample1 = nn.Upsample(None, 2, 'nearest')  # 11
        self.concat1 = Concat(1)  # 12 # [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        self.c35 = C3(512, 256, n=1, shortcut=False)  # 13

        self.conv7 = Conv(256, 128, k=1, s=1)  # 14
        self.unsample2 = nn.Upsample(None, 2, 'nearest')  # 15
        self.concat2 = Concat(1)  # 16  # [[-1, 4], 1, Concat, [1]],  # cat backbone P4
        self.c36 = C3(256, 128, n=1, shortcut=False)  # 17

        self.conv8 = Conv(128, 128, k=3, s=2)  # 18
        self.concat3 = Concat(1)  # 19  # [[-1, 14], 1, Concat, [1]],  # cat backbone P4
        self.c37 = C3(256, 256, n=1, shortcut=False)  # 20

        self.conv9 = Conv(256, 256, k=3, s=2)  # 21
        self.concat4 = Concat(1)  # 22  # [[-1, 10], 1, Concat, [1]],  # cat backbone P4
        self.c38 = C3(512, 512, n=1, shortcut=False)  # 23

        self.detect = Detect(nc=nc, anchors=anchors, ch=[128, 256, 512])

    def forward(self, x):
        # Backbone
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.c31(x1)
        x3 = self.conv3(x2)
        x4 = self.c32(x3)
        x5 = self.conv4(x4)
        x6 = self.c33(x5)
        x7 = self.conv5(x6)
        x8 = self.c34(x7)
        x9 = self.sppf(x8)
        # Head
        x10 = self.conv6(x9)
        x11 = self.unsample1(x10)
        x12 = self.concat1([x11, x6])
        x13 = self.c35(x12)

        x14 = self.conv7(x13)
        x15 = self.unsample1(x14)
        x16 = self.concat2([x15, x4])
        x17 = self.c36(x16)

        x18 = self.conv8(x17)
        x19 = self.concat3([x18, x14])
        x20 = self.c37(x19)

        x21 = self.conv9(x20)
        x22 = self.concat4([x21, x10])
        x23 = self.c38(x22)

        detect = self.detect([x17, x20, x23])

        return detect

    def fuse(self):
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self


if __name__ == "__main__":
    cfg = 'yolov5s.yaml'
    yaml_file = Path(cfg).name
    with open(cfg, encoding='ascii', errors='ignore') as f:
        d = yaml.safe_load(f)  # model dict
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    myyolo = MyYolo(nc=nc, anchors=anchors)
    myyolo.load_state_dict(torch.load('myyolov5s.pth', map_location=None))

    myyolo = myyolo.to(device)
    myyolo.fuse().eval().half()

    x = torch.randn(1, 3, 32, 32).to(device).half()
    flops, params = thop.profile(myyolo, inputs=(x,))
    flops = flops * 640 / 32 * 640 / 32
    print(params)
    print(flops / 1E9 * 2)
