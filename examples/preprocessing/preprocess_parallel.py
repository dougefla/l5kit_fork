
from re import L
from torch.utils.data import DataLoader
import os
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
import time
import h5py
import cv2
import torch
import numpy as np

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

    # n = 8, b = 1 is the best.
    'train_data_loader': {
        'key': 'scenes/train_full.zarr',
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    },

    'train_params': {
        'max_num_steps': -1,
        'checkpoint_every_n_steps': 100000,

        'eval_every_n_steps': 100000
    }
}
# set env variable for data
l5kit_data_folder = "/home/fla/workspace/l5kit_data"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
dm = LocalDataManager(None)

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)
train_cfg = cfg['train_data_loader']
train_path = train_cfg['key']
print(f"Loading from {train_path}")
train_zarr = ChunkedDataset(dm.require(train_path)).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)
test_loader = DataLoader(
    train_dataset,
    shuffle=train_cfg['shuffle'],
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    pin_memory=True,
)

start_id = 27853100
num_frames = int(train_dataset.dataset.frames.size)
print("Total Number of Frames: {} with {} Skipped".format(num_frames,start_id))

# def mytimer(my_const_timer, marker=0):
#     print("Position {}: {}".format(marker,time.time()-my_const_timer))
#     return time.time()

start_all_time = time.time()
start_100_time = start_all_time

for i, data_batch in enumerate(test_loader,start=start_id):
    if i%100 == 0 and not i==0:
        cur_time = time.time()
        used_100_time = cur_time - start_100_time
        esti_100_time = (used_100_time*(num_frames-i*train_cfg['batch_size'])/(100*train_cfg['batch_size']))/3600
        # used_all_time = cur_time-start_all_time
        # esti_all_time = (used_all_time*(num_frames-i*train_cfg['batch_size'])/(i*train_cfg['batch_size']))/3600
        # Save log
        log_info = "Saved Frame {}/{} | {:.2f}% | ETA {:.2f} h".format(i,num_frames,(train_cfg['batch_size']*i)*100/num_frames,esti_100_time)
        time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        str_1 = log_info + ' '+time_str + '\n'
        file_path = r"/home/fla/workspace/preprocessing/log.txt"
        with open(file_path, 'a') as f:
            f.write(str_1)
        print(log_info)

        start_100_time = time.time()