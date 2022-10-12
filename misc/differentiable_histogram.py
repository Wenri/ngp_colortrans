import math
import sys

import torch
from torch import nn


class HistoBin(nn.Module):
    def __init__(self, locations=torch.arange(0, 1, .1), radius=.2, norm=True):
        super(HistoBin, self).__init__()

        self.locs = locations
        self.r = radius
        self.norm = norm

    def forward(self, x):

        counts = []

        for loc in self.locs:
            dist = torch.abs(x - loc)
            # print dist
            ct = torch.relu(self.r - dist).sum(1)
            counts.append(ct)

        out = torch.stack(counts, 1)

        if self.norm:
            summ = out.sum(1) + 1e-6
            return (out.transpose(1, 0) / summ).transpose(1, 0)
        return out


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        centers = self.delta * (torch.arange(bins, dtype=torch.float) + 0.5) + min
        self.register_buffer('centers', centers, persistent=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * math.sqrt(math.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


class SoftHistogram(GaussianHistogram):
    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


class MultivariateGaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(MultivariateGaussianHistogram, self).__init__()
        self.register_buffer('sigma', torch.as_tensor(sigma, dtype=torch.float), persistent=False)
        delta = float(max - min) / float(bins)
        # self._norm_factor = torch.sqrt(torch.det(math.pi * 2 * torch.diagflat(self.sigma ** 2)))
        k = math.pi * 2
        if len(self.sigma) % 2:
            k **= len(self.sigma) / 2
        k *= torch.prod(self.sigma).item()
        self._norm_factor = delta ** len(self.sigma) / k
        centers = delta * (torch.arange(bins, dtype=torch.float) + 0.5) + min
        centers = torch.cartesian_prod(centers, centers)
        self.register_buffer('centers', centers, persistent=False)

    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.matmul(x ** 2, self.sigma ** -2)
        x = torch.exp(-x / 2) * self._norm_factor
        x = x.sum(dim=1)
        return x


def _testhist():
    if sys.gettrace() is None:
        torch.set_printoptions(precision=0, sci_mode=False)

    # data = torch.randn(1000) / 3

    # print('histc GaussianHistogram SoftHistogram')
    # hist = torch.histogram(data, bins=100, range=(-1., 1.))
    # print(hist)

    # gausshist = GaussianHistogram(bins=100, min=-1, max=1, sigma=1e-2)
    # hist = gausshist(data)
    # print(hist)
    #
    # softhist = SoftHistogram(bins=100, min=-1, max=1, sigma=1e2)
    # hist = softhist(data)
    # print(hist)

    data = torch.randn(100000, 2) / 3

    print('histogramdd GaussianHistogram')
    hist, bin_edges = torch.histogramdd(data, bins=10, range=(-1., 1., -1., 1.))
    print(hist)

    mghist = MultivariateGaussianHistogram(bins=10, min=-1, max=1, sigma=(1e-2, 1e-2))
    hist = mghist(data)
    print(hist)


if __name__ == '__main__':
    _testhist()
