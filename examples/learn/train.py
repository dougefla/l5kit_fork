import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from models.PlanningNet import PlanningNet

from dataset import ProcessedDataset
from torch.utils.data import DataLoader
from torch import nn, optim
import os
from log import get_logger
from loss import MSELoss
from utils import pytorch_neg_multi_log_likelihood_single

import random

# CUDA Setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# Experiment Setting

use_dataset_id = 1
is_shuffle = False
batch_size = 64
data_batch_size = 64*100
history_num_frames = 30
future_num_frames = 50
resume = False
ck_name = "2022_02_14_10-28-45"

# Path Setting
dataset_root = r"/home/fla/workspace/l5kit_data/rasterized/"
output_root = r"/home/fla/workspace/l5kit_fork/examples/learn/output"
if not os.path.isdir(output_root):
    os.makedirs(output_root)
save_path = os.path.join(output_root,str(time.strftime('%Y_%m_%d_%H-%M-%S',time.localtime())))
if not os.path.isdir(save_path):
    os.makedirs(save_path)
ck_path = os.path.join(output_root, ck_name, "best_model.pth")
if not os.path.exists(ck_path):
    resume = False
image_dir = os.path.join(dataset_root, "image_{}".format(use_dataset_id))
assert os.path.isdir(image_dir)
target_dir = os.path.join(dataset_root, "target_{}".format(use_dataset_id))
assert os.path.isdir(target_dir)
index_dir = os.path.join(dataset_root, "index_{}.txt".format(use_dataset_id))
assert os.path.exists(index_dir)

# Logger Setting
logger = get_logger(os.path.join(save_path,"train.log"))

# Load index of dataset
with open(index_dir, 'r') as f:
    index_list = [index.strip() for index in f.readlines()]
data_num = len(index_list)

# Using to split the dataset by ratio
def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def validate(val_dataset, val_index_list, model, logger, best_loss, optimizer):
    # Set the model to evaluation mode
    model.eval()
    losses_val = []
    batch_size = 32
    #random.shuffle(val_index_list)
    #val_index_list_batch = val_index_list[:val_num]
    val_dataset.reload(val_index_list)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    for i,data in enumerate(val_loader):
        input_image = np.transpose(data[0],(0,3,1,2)).to(device)
        gt_pos = data[1].to(device)
        gt_aval = data[2].to(device)

        pred = model(input_image.float())
        loss = pytorch_neg_multi_log_likelihood_single(
            gt_pos, pred, gt_aval
            )
        losses_val.append(loss.item())
    
        logger.info('[Val] Iter:[{}/{}]\t Loss={:.5f}\t'.format(i , math.ceil(len(val_index_list)/batch_size), loss ))
    loss_avg = np.average(losses_val)
    return loss_avg

def train():
    # Init the model
    myNet = PlanningNet(
            model_arch="resnet_50",
            input_channels=3,
            predict_frames=50,
            pretrained = False
            )
    model = myNet.get_model()
    print(model)
    # Set DPP
    model= nn.DataParallel(model,device_ids=[0,1])
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    with torch.set_grad_enabled(True):
        
        # Split the dataset to train and validation
        # Since the dataset is million-level, splition is done by a constant number, not ratio
        val_data_num = 3000
        train_data_num = data_num-val_data_num
        train_index_list = index_list[:-val_data_num]
        val_index_list = index_list[-val_data_num:]
        train_data_batch_num = int(train_data_num/data_batch_size)

        # Here the dataset will load the first batch of data, refer to ProcessedDataset.init for detail
        train_dataset = ProcessedDataset(
                        image_dir = image_dir,
                        target_dir = target_dir,
                        index_list = train_index_list[0:data_batch_size],
                        history_num_frames = history_num_frames,
                        future_num_frames = future_num_frames
                        )

        val_dataset = ProcessedDataset(
                        image_dir = image_dir,
                        target_dir = target_dir,
                        index_list = val_index_list,
                        history_num_frames = history_num_frames,
                        future_num_frames = future_num_frames
                        )

        # Total number for iteration.
        iter_num = math.ceil(train_data_num/batch_size)
        
        best_loss = 9999999

        if resume == True:
            logger.info("Resume: {}".format(ck_path))
            checkpoint = torch.load(ck_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            losses_train = checkpoint['loss_arr']
            data_id_start = checkpoint['steps']
        else:
            losses_train = []
            data_id_start = 0

        # Timer start
        start_time = time.time()
        data_id = data_id_start
        # Train by batch. Each batch will be pre-loaded.
        for data_batch_id in range(train_data_batch_num):

            # Load new batch into dataset 
            train_dataset.reload(train_index_list[data_batch_id*data_batch_size:(data_batch_id+1)*data_batch_size])
            # Build a new dataloader with the newly loaded dataset
            train_loader = DataLoader(
                train_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
            )

            # Get new mini-batch of data
            for data in train_loader:
                # Set the model to train mode
                model.train()
                input_image = np.transpose(data[0],(0,3,1,2)).to(device)
                gt_pos = data[1].to(device)
                gt_aval = data[2].to(device)

                pred = model(input_image.float())
                loss = pytorch_neg_multi_log_likelihood_single(
                    gt_pos, pred, gt_aval
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_train.append(loss.item())

                # Calculate the ETA
                used_time = time.time() - start_time
                esti_time = (data_num-(data_id+1)*batch_size)*used_time/(((data_id+1)-data_id_start)*batch_size)/3600

                # Get the log info
                logger.info('[Train] Iter:[{}/{}]\t Loss={:.5f}\t Best Loss= {:.5f}\t ETA {:.2f} h'.format(data_id , iter_num, loss, best_loss, esti_time))

                data_id+=1

            # One Batch Over. Save the model
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_arr': losses_train,
                        'loss': loss,
                        'steps': data_id},
                    os.path.join(save_path,"{:0>5d}.pth".format(data_id)))
            
            # Validate & Update the Best Model
            val_loss = validate(
                    val_dataset = val_dataset,
                    val_index_list = val_index_list,
                    model = model,
                    logger = logger,
                    best_loss = best_loss,
                    optimizer = optimizer
                    )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_arr': losses_train,
                            'loss': loss,
                            'steps': data_id},
                        os.path.join(save_path,"best_model.pth".format(data_id)))

if __name__ == "__main__":
    train()