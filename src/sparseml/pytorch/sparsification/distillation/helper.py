import numpy as np
import torch

def make_center_anchors(anchors_wh, grid_size=80, device='cpu'):
    grid_arange = torch.arange(grid_size)
    xx, yy = torch.meshgrid(grid_arange, grid_arange)  # + 0.5  # grid center, [fmsize*fmsize,2]
    xy = torch.cat((torch.unsqueeze(xx, -1), torch.unsqueeze(yy, -1)), -1) + 0.5

    wh = torch.tensor(anchors_wh)

    xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # centor
    wh = wh.view(1, 1, 5, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # w, h
    center_anchors = torch.cat([xy, wh], dim=3).to(device)
    # cy cx w h

    """
    center_anchors[0][0]
    tensor([[ 0.5000,  0.5000,  1.3221,  1.7314],
            [ 0.5000,  0.5000,  3.1927,  4.0094],
            [ 0.5000,  0.5000,  5.0559,  8.0989],
            [ 0.5000,  0.5000,  9.4711,  4.8405],
            [ 0.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')

    center_anchors[0][1]
    tensor([[ 1.5000,  0.5000,  1.3221,  1.7314],
            [ 1.5000,  0.5000,  3.1927,  4.0094],
            [ 1.5000,  0.5000,  5.0559,  8.0989],
            [ 1.5000,  0.5000,  9.4711,  4.8405],
            [ 1.5000,  0.5000, 11.2364, 10.0071]], device='cuda:0')

    center_anchors[1][0]
    tensor([[ 0.5000,  1.5000,  1.3221,  1.7314],
            [ 0.5000,  1.5000,  3.1927,  4.0094],
            [ 0.5000,  1.5000,  5.0559,  8.0989],
            [ 0.5000,  1.5000,  9.4711,  4.8405],
            [ 0.5000,  1.5000, 11.2364, 10.0071]], device='cuda:0')

    pytorch view has reverse index
    """

    return center_anchors

def center_to_corner(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def corner_to_center(xy):
    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)


def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)