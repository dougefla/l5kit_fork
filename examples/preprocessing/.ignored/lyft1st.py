#!/usr/bin/env python
# coding: utf-8

import yaml
from timm.models.layers.conv2d_same import Conv2dSame
import timm
from typing import Dict
from torch import nn
from torch import Tensor
import torch
from torch.utils.data.dataset import Dataset
from typing import Callable
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torch import nn, optim
import gc
import os
from pathlib import Path
import random
import sys
import math

from tqdm.notebook import tqdm
import numpy as np
import scipy as sp


import matplotlib.pyplot as plt
import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

from matplotlib import animation, rc
from IPython.display import HTML

import time

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# class TransformDataset(Dataset):
#     def __init__(self, dataset: Dataset, transform: Callable):
#         self.dataset = dataset
#         self.transform = transform

#     def __getitem__(self, index):
#         batch = self.dataset[index]
#         return self.transform(batch)

#     def __len__(self):
#         return len(self.dataset)


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(
        pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"  # torch.Size([32, 1, 303])
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len,
                        num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones(
        (batch_size,))), "confidences should sum to 1"
    assert avails.shape == (
        batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(
    ), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    # reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)

    # when confidence is 0 log goes to -inf, but we're fine with it
    with np.errstate(divide="ignore"):
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * \
            torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)
    error = -torch.log(torch.sum(torch.exp(error - max_value),
                       dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error).requires_grad_(True)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)

    batch_size, _, _, _ = pred.shape

    confidences = pred.new_ones((batch_size, 1))

    return pytorch_neg_multi_log_likelihood_batch(gt, pred, confidences, avails)


class LMM(nn.Module):
    def __init__(self, model_architecture, History=30, gem=False):
        super().__init__()
        self.H = History  # 过去3s作为输入帧
        num_history_channels = (self.H + 1) * 2  # 62 (过去3s + 当前帧)*2维XY
        rgb_channels = 3
        num_in_channels = rgb_channels + num_history_channels  # 3 + 62

        # self.num_modes = 3 # 三条轨迹
        self.num_modes = 1  # 一条轨迹

        self.future_len = 50  # 输出未来5s
        num_targets = 2 * self.future_len  # ? *2表示仅有XY二维 100
        self.num_preds = num_targets * self.num_modes  # 轨迹数*轨迹长度*轨迹维度 (2*50)*3

        # timm库提取现有backbone: EfficientNetB3
        self.backbone = timm.create_model(model_architecture, pretrained=False)

#         if gem:
#             self.backbone.global_pool = GeM()

        self.backbone.conv_stem = Conv2dSame(
            num_in_channels,
            self.backbone.conv_stem.out_channels,
            kernel_size=self.backbone.conv_stem.kernel_size,
            stride=self.backbone.conv_stem.stride,  # 滑动步长
            padding=self.backbone.conv_stem.padding,
            bias=False,
        )

        self.backbone_out_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(  # 全连接层
                in_features=self.backbone.classifier.in_features,
                out_features=self.backbone_out_features,
            ),
        )

        self.lin_head = nn.Sequential(  # 由于输入的顺序与构造的结果相关。所以注意邻近层输入输出的size大小
            nn.ReLU(),
            nn.Linear(  # 全连接层 input: [batch_size, input_size]-> output: [batch_size, output_size]
                # 全连接层起到一个矩阵乘法的作用：FC:[input_size, output_size]
                # 输入输出都必须为二维张量，通过.view()来变换
                in_features=self.backbone_out_features,
                #                 out_features=self.num_preds + self.num_modes, # 轨迹数*轨迹长度*轨迹维度 (2*50)*3 + num_modes 3
                out_features=self.num_preds,
            ),
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, image_box, image_sem):
        x = torch.cat((image_box, image_sem), dim=1)
        x = self.backbone(x)
        x = self.lin_head(x)
        x = x.view(-1, self.num_modes, self.future_len, 2)

#         if self.training:
#             loss_nll = pytorch_neg_multi_log_likelihood_single(targets, x, target_availabilities)
# #             print("training model")
#             return loss_nll
#         else:
# #             print("evaluation model")
#             return x
        return x


class LyftMultiRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image_box, image_sem, targets, target_availabilities):
        pred = self.predictor(image_box, image_sem)
#         if self.PARAMS.predict_diffs:
#             pred = torch.cumsum(pred, dim=2)
#         pred_sum = torch.cumsum(pred, dim=2)

        loss_nll = pytorch_neg_multi_log_likelihood_single(
            targets, pred, target_availabilities
        )

        return loss_nll, pred


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com
    /questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        #         'model_architecture': 'resnet50',
        'history_num_frames': 30,
        #         'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        #         'future_step_size': 1,
        'future_delta_time': 0.1,
        'render_ego_history': True,
        'step_time': 0.1
    },

    'raster_params': {
        #         'raster_size': [448, 224],
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'set_origin_to_bottom': True,
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces': False
    },

    'train_data_loader': {
        'key': 'scenes/sample.zarr/train.zarr',
        'batch_size': 6,
        'shuffle': True,
        'num_workers': 2
    },

    'valid_data_loader': {
        'key': 'scenes/sample.zarr/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'max_num_steps': -1,
        'checkpoint_every_n_steps': 100000,

        'eval_every_n_steps': 100000
    }
}

flags_dict = {
    "debug": False,
    # --- Data configs ---
    "l5kit_data_folder": "./l5kit_data",
    # --- Model configs ---
    "pred_mode": "single",
    # --- Training configs ---
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}

flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")

save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)

l5kit_data_folder = "/home/fla/workspace/l5kit_data"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
dm = LocalDataManager(None)

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

train_cfg = cfg['train_data_loader']
train_path = train_cfg['key']
print(f"Loading from {train_path}")
train_zarr = ChunkedDataset(dm.require(train_path)).open()
# print("train_path", type(train_path))

train_dataset = EgoDataset(cfg, train_zarr, rasterizer)
test_loader = DataLoader(
    train_dataset,
    shuffle=train_cfg["shuffle"],
    batch_size=train_cfg["batch_size"],
    num_workers=train_cfg["num_workers"],
    pin_memory=True,
)
print(train_dataset)
print(len(train_dataset))

# tr_it = iter(test_loader)
# data = next(tr_it)
# print(data.keys())

device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
    model = LyftMultiRegressor(predictor)
elif flags.pred_mode == "single":
    print("single mode")
    predictor = LMM("tf_efficientnet_b3_ns")
    model = LyftMultiRegressor(predictor)
else:
    raise ValueError(
        f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

resume = False
load_path = "./model/lyft1st_single/chk1.pt"


my_const_timer = time.time()
def mytimer(my_const_timer, marker=0):
    print("Position {}: {}".format(marker,time.time()-my_const_timer))
    return time.time()

if resume == True:
    pt_file = torch.load(load_path, map_location=lambda storage, loc: storage)
    saved_model_param = pt_file["model_state_dict"]
    optimizer.load_state_dict(pt_file["optimizer_state_dict"])
    losses_train = pt_file["loss_arr"]
    loss = pt_file["loss"].to(device)  # with grad
    trained_steps = pt_file['steps']
    print(trained_steps)
#     trained_steps = 40000
    print(loss)

    state_dict = model.state_dict()
#     for key in saved_model_param.keys():
#         if key in state_dict and (saved_model_param[key].size() == state_dict[key].size()):# 检查完备
#             value = saved_model_param[key]
# #             print(type(value))
# #             print(value,key)
# #             value.requires_grad_(True)
#             if not isinstance(value, torch.Tensor):
#                 value = value.data
#             state_dict[key] = value
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

trained_steps = 0
total_steps = math.ceil(len(train_dataset)/train_cfg['batch_size'])
with torch.set_grad_enabled(True):
    train = True
    if train:
        model.train()
    else:
        model.eval()

    progress_bar = tqdm(test_loader)
    losses_train = []

    start_time = time.time()
    for i, data in enumerate(progress_bar):
        print("Data {} used time {}".format(i, time.time()-start_time))
        start_time = time.time()
        my_const_timer = mytimer(my_const_timer, 1)
        image_box = data['image_box'].to(device).permute(0, 2, 1, 3)
        my_const_timer = mytimer(my_const_timer, 2)
        image_sem = data['image_sem'].to(device)
        my_const_timer = mytimer(my_const_timer, 3)
        targets = data['target_positions'].to(device)
        my_const_timer = mytimer(my_const_timer, 4)
        target_availabilities = data['target_availabilities'].to(device)
        my_const_timer = mytimer(my_const_timer, 5)

        if train:
            loss, pred = model(image_box, image_sem, targets, target_availabilities)
            my_const_timer = mytimer(my_const_timer, 6)
            optimizer.zero_grad()
            loss.backward()
            my_const_timer = mytimer(my_const_timer, 7)
            optimizer.step()
            my_const_timer = mytimer(my_const_timer, 8)
            losses_train.append(loss.item())
            my_const_timer = mytimer(my_const_timer, 9)
            progress_bar.set_description(
                f"loss: {loss.item():.5f} loss(avg): {np.mean(losses_train):.5f}")
            my_const_timer = mytimer(my_const_timer, 10)
        else:
            pred = predictor(
                image, targets, target_availabilities).cpu().numpy()
            print(pred.shape)
        my_const_timer = mytimer(my_const_timer, 11)
        chk_pts = cfg["train_params"]["checkpoint_every_n_steps"]
        eval_pts = cfg["train_params"]['eval_every_n_steps']
        my_const_timer = mytimer(my_const_timer, 12)

        # if i in list(range(chk_pts, total_steps, chk_pts)):
        #     model_index = int((i+trained_steps)/chk_pts)
        #     torch.save({'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'loss_arr': losses_train,
        #                 'loss': loss,
        #                 'steps': i},
        #                f"./model/lyft1st_single/chk{model_index}.pt")


# 尽量减少输入的通道数
# 数据的离线预处理
# 合理选择num_workers:CPU与io的速度，内存容量，GPU处理速度
# CUDA-efficient means “no python control flow”：accessing individual values of GPU tensor may get the job done, but the performance will be awful
# the slow rasterizer

def test_loss_profiling():
    loss = nn.BCEWithLogitsLoss()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        input = torch.randn((8, 1, 128, 128)).cuda()
        input.requires_grad = True

        target = torch.randint(1, (8, 1, 128, 128)).cuda().float()

        for i in range(10):
            l = loss(input, target)
            l.backward()
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
