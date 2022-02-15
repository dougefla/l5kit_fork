
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

history_num_frames = 30
future_num_frames = 50
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
        'key': 'scenes/train.zarr',
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
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
# print("train_path", type(train_path))
batch_size = 8
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)
test_loader = DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
)

num_frames = int(train_dataset.dataset.frames.size)
print("Total Number of Frames: {}".format(num_frames))
data_list = []
save_path = r"/home/fla/workspace/l5kit_data/rasterized"
img_save_path = os.path.join(save_path, "image/")
target_save_path = os.path.join(save_path, "target/")

my_const_timer = time.time()
start_time = time.time()
def mytimer(my_const_timer, marker=0):
    print("Position {}: {}".format(marker,time.time()-my_const_timer))
    return time.time()

# Slow & Simple Solution
for i, data in enumerate(test_loader):
    
    # Get the timestamp as the name
    timestamp = int(data['timestamp'][0])

    # Get the target info: ((pos_x, pos_y, yaw), aval)
    target_pos = (data['target_positions'].numpy())[0]
    target_yaw = (data['target_yaws'].numpy())[0]
    target_aval_half = (data['target_availabilities'].numpy())[0]
    target_aval = [str(target_aval_half[int(i/2)]) for i in range(2*len(target_aval_half))]

    # Rasterize
    image_box = (data['image_box'].numpy())[0]
    image_sem = np.transpose((data['image_sem'].numpy())[0], (1,2, 0))
    image_rgb = rasterizer.to_rgb(image_box,image_sem)
    img_name = os.path.join(img_save_path,'train_{}_{}_{:0>20d}.png'.format(cfg['model_params']['history_num_frames'],cfg['model_params']['future_num_frames'],timestamp))

    # Save the (image, target)
    with open(os.path.join(target_save_path,'train_{}_{}_{:0>20d}.txt'.format(cfg['model_params']['history_num_frames'],cfg['model_params']['future_num_frames'],timestamp)), 'w') as f:
        target_pos_str = " ".join([str(target_pos[i][0])+" "+str(target_pos[i][1])+" "+str(target_yaw[i][0]) for i in range(cfg['model_params']['future_num_frames'])])\
                + '\n' + " ".join(target_aval)
        f.write(target_pos_str)
    cv2.imwrite(img_name,cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

    # Calculate the time comsuption
    used_time = time.time() - my_const_timer
    my_const_timer = time.time()
    if i%100 == 0 and not i==0:
        esti_time = int((used_time*(num_frames-i*batch_size) + (time.time()-start_time)*num_frames/(i*batch_size))/2)
        m, s = divmod(esti_time, 60)
        h, m = divmod(m, 60)
        print("Saved Frame {}/{} at {} | {:.2f}% | ETA {:0>2d}:{:0>2d}:{:0>2d}".format(i,num_frames,img_name, (batch_size*i)/num_frames,h,m,s ))