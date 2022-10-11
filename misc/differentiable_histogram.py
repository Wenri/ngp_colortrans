import math

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
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.register_buffer('centers', float(min) + self.delta * (torch.arange(bins).float() + 0.5), persistent=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * math.sqrt(math.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.register_buffer('centers', float(min) + self.delta * (torch.arange(bins).float() + 0.5), persistent=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


def _testhist():
    torch.set_printoptions(precision=0, sci_mode=False)

    data = torch.randn(1000) / 3

    print('histc GaussianHistogram SoftHistogram')
    hist = torch.histc(data, bins=100, min=-1, max=1)
    print(hist)

    gausshist = GaussianHistogram(bins=100, min=-1, max=1, sigma=1e-2)
    hist = gausshist(data)
    print(hist)

    softhist = SoftHistogram(bins=100, min=-1, max=1, sigma=1e2)
    hist = softhist(data)
    print(hist)


if __name__ == '__main__':
    _testhist()
